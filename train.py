"""mini-llm 的训练入口脚本。

负责读取数据、构建词表、组装 DataLoader、初始化模型，并执行训练与保存检查点。
"""

import argparse
import importlib.util
import json
import random
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


# 按“特殊标记优先，其余逐字符切分”的方式做最简单分词。
TOKEN_PATTERN = re.compile(r"\[BOS\]|\[SEP\]|\[EOS\]|.")

# 词表中预留的特殊 token。
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[SEP]", "[EOS]"]


def load_model_module():
    """动态加载 `mini-llm.py`。

    文件名里包含连字符，不能直接用普通 `import` 导入，
    因此这里使用 importlib 按路径加载模型定义。
    """
    module_path = Path(__file__).with_name("mini-llm.py")
    spec = importlib.util.spec_from_file_location("mini_llm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    """固定随机种子，尽量让训练结果可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_lines(path, max_samples=None):
    """逐行读取训练文本，跳过空行，并可限制样本数量。"""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if max_samples is not None and len(lines) >= max_samples:
                break
    return lines


def tokenize(text):
    """把文本切成 token 列表。"""
    return TOKEN_PATTERN.findall(text)


def build_vocab(lines):
    """根据训练数据统计词频并构建词表。"""
    counter = Counter()
    for line in lines:
        counter.update(tokenize(line))

    # 先放入特殊 token，确保它们的 id 固定。
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_seq_len):
    """把一条文本编码成 token id，并截断到最大长度。"""
    unk_id = vocab["[UNK]"]
    ids = [vocab.get(token, unk_id) for token in tokenize(text)]
    return ids[:max_seq_len]


class PoetryDataset(Dataset):
    """简单的文本数据集：按行返回编码后的序列。"""

    def __init__(self, lines, vocab, max_seq_len):
        self.lines = lines
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return encode(self.lines[idx], self.vocab, self.max_seq_len)


def make_collate_fn(pad_id):
    """创建 batch 拼接函数，负责补齐并生成训练标签。"""

    def collate_fn(batch):
        batch_size = len(batch)
        max_len = max(len(item) for item in batch)

        # input_ids 用 pad 补齐；labels 中 pad 位置设为 -100，
        # 这样交叉熵损失会自动忽略这些位置。
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for row, ids in enumerate(batch):
            seq = torch.tensor(ids, dtype=torch.long)
            length = seq.numel()
            input_ids[row, :length] = seq
            labels[row, :length] = seq
            attention_mask[row, :length] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


def save_artifacts(output_dir, vocab, config_dict, args):
    """保存训练所需的辅助文件，方便后续推理或继续训练。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def save_loss_history(output_dir, epoch_losses):
    """保存每个 epoch 的平均 loss，便于后续分析。"""
    history = [
        {"epoch": epoch, "avg_loss": avg_loss}
        for epoch, avg_loss in enumerate(epoch_losses, start=1)
    ]
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_pyplot(enable_live_plot):
    """按需加载 matplotlib；实时模式下尽量启用交互式窗口。"""
    try:
        import matplotlib
    except ImportError:
        print("matplotlib is not installed, skipping loss plot export.")
        return None, False

    if not enable_live_plot:
        matplotlib.use("Agg")

    try:
        import matplotlib.pyplot as plt

        return plt, enable_live_plot
    except Exception as exc:
        if enable_live_plot:
            print(f"unable to enable live plot ({exc}), falling back to file-only plotting.")
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            return plt, False

        print(f"unable to initialize matplotlib ({exc}), skipping loss plot export.")
        return None, False


class LossPlotter:
    """训练中刷新 loss 曲线，并可选弹出实时窗口。"""

    def __init__(self, output_dir, enable_live_plot=False):
        self.output_dir = output_dir
        self.plot_path = output_dir / "loss_curve.png"
        self.plt, self.enable_live_plot = load_pyplot(enable_live_plot)
        self.figure = None
        self.axes = None

        if self.plt is not None:
            self.figure, self.axes = self.plt.subplots(figsize=(8, 5))
            if self.enable_live_plot:
                self.plt.ion()
                self.plt.show(block=False)

    def update(self, epoch_losses, live_points=None):
        """刷新图像，并始终把当前快照保存到磁盘。"""
        if self.plt is None or self.figure is None or self.axes is None:
            return

        self.axes.clear()

        if epoch_losses:
            epochs = list(range(1, len(epoch_losses) + 1))
            self.axes.plot(
                epochs,
                epoch_losses,
                marker="o",
                linewidth=2,
                label="Epoch avg loss",
            )

        if live_points is not None:
            live_x, live_y = live_points
            if live_x and live_y:
                self.axes.plot(
                    live_x,
                    live_y,
                    marker=".",
                    linestyle="--",
                    linewidth=1.5,
                    color="orange",
                    label="Running avg loss",
                )

        self.axes.set_title("Training Loss")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.grid(True, linestyle="--", alpha=0.4)

        if epoch_losses or (live_points is not None and live_points[0] and live_points[1]):
            self.axes.legend()

        self.figure.tight_layout()
        self.figure.savefig(self.plot_path, dpi=150)

        if self.enable_live_plot:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.plt.pause(0.001)

    def close(self):
        """训练结束后关闭交互式刷新。"""
        if self.enable_live_plot and self.plt is not None:
            self.plt.ioff()


def make_progress_bar(iterable, epoch, total_epochs, total_steps):
    """创建终端进度条，实时展示训练进度和 loss。"""
    if not TQDM_AVAILABLE:
        return iterable

    return tqdm(
        iterable,
        total=total_steps,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
    )


def get_config_dict(config):
    """把配置对象整理成可序列化的字典。"""
    config_dict = {
        key: value
        for key, value in vars(config.__class__).items()
        if not key.startswith("_") and not callable(value)
    }
    config_dict.update(vars(config))
    return config_dict


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    step,
    avg_loss,
    vocab,
    config_dict,
    args,
    best_loss,
    epoch_losses,
    epoch_complete=True,
):
    """保存训练检查点，包括模型、优化器和训练元信息。"""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "avg_loss": avg_loss,
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
        "epoch_complete": epoch_complete,
        "vocab": vocab,
        "config": config_dict,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path):
    """从磁盘加载训练检查点。"""
    # 这里的 checkpoint 包含优化器状态和训练参数，必须显式保持完整反序列化。
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_config_overrides(config, config_dict):
    """把字典中的配置项写回配置对象。"""
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def resolve_resume_path(args, output_dir):
    """根据命令行参数决定是否从检查点恢复训练。"""
    if args.resume_from:
        return Path(args.resume_from)
    default_resume_path = output_dir / "last.pt"
    if args.resume and default_resume_path.exists():
        return default_resume_path
    return None


def main():
    """解析参数并执行完整训练流程。"""
    parser = argparse.ArgumentParser(description="Train the mini causal language model.")
    parser.add_argument("--data", default="dataset/poetry_train.txt")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.set_defaults(amp=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume-from", default=None)
    parser.set_defaults(live_plot=True)
    parser.add_argument("--live-plot", dest="live_plot", action="store_true")
    parser.add_argument("--no-live-plot", dest="live_plot", action="store_false")
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")

    # 固定随机性，方便复现实验结果。
    seed_everything(args.seed)

    # 支持自动选择设备，也允许手动指定 cpu / cuda / cuda:0 等。
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # CUDA 默认启用混合精度，显著降低激活值和梯度的显存占用。
    use_amp = bool(device.type == "cuda") if args.amp is None else bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    print(
        f"training setup: device={device} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} amp={scaler.is_enabled()}"
    )

    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_path(args, output_dir)
    checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path)
        print(f"resuming from checkpoint: {resume_path}")

    # 读取原始训练文本，每一行视作一个样本。
    lines = read_lines(args.data, max_samples=args.max_samples)
    if not lines:
        raise RuntimeError(f"No training samples found in {args.data}")

    # 优先使用检查点中的词表，确保恢复训练时编码方式完全一致。
    vocab = checkpoint.get("vocab") if checkpoint is not None else build_vocab(lines)

    # 从本地模型文件中加载配置与模型类。
    model_module = load_model_module()
    config = model_module.MiniLLMConfig()

    if checkpoint is not None and "config" in checkpoint:
        config = apply_config_overrides(config, checkpoint["config"])
    else:
        config.vocab_size = len(vocab)
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_layers
        config.num_attention_heads = args.num_heads
        config.max_position_embeddings = args.max_seq_len
        config.pad_token_id = vocab["[PAD]"]

    config.vocab_size = len(vocab)
    config.pad_token_id = vocab["[PAD]"]

    # 多头注意力要求 hidden_size 能被头数整除。
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    model = model_module.MiniModelForCausalLM(config).to(device)

    # DataLoader 在这里完成随机打乱、batch 组装和补齐。
    dataset = PoetryDataset(lines, vocab, config.max_position_embeddings)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(vocab["[PAD]"]),
        pin_memory=(device.type == "cuda"),
    )

    # 使用 AdamW 训练因果语言模型。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config_dict = get_config_dict(config)

    # 先保存词表、配置和训练参数，便于后续复用。
    save_artifacts(output_dir, vocab, config_dict, args)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    epoch_losses = []

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("step", 0)
        epoch_losses = list(checkpoint.get("epoch_losses", []))
        if epoch_losses:
            best_loss = checkpoint.get("best_loss", min(epoch_losses))
        else:
            best_loss = checkpoint.get("best_loss", checkpoint.get("avg_loss", float("inf")))

        last_epoch = checkpoint.get("epoch", 0)
        epoch_complete = checkpoint.get("epoch_complete", True)
        start_epoch = last_epoch + 1 if epoch_complete else max(last_epoch, 1)
        print(
            f"resume state: epoch={last_epoch} step={global_step} "
            f"epoch_complete={epoch_complete}"
        )
        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            print(
                "checkpoint args: "
                f"batch_size={ckpt_args.get('batch_size')} "
                f"max_seq_len={ckpt_args.get('max_seq_len')} "
                f"hidden_size={ckpt_args.get('hidden_size')} "
                f"num_layers={ckpt_args.get('num_layers')}"
            )

    loss_plotter = LossPlotter(output_dir, enable_live_plot=args.live_plot)
    num_batches = len(loader)

    if epoch_losses:
        loss_plotter.update(epoch_losses)

    if not TQDM_AVAILABLE:
        print("tqdm is not installed, falling back to plain logging.")

    if start_epoch > args.epochs:
        print(
            f"checkpoint already reached target epochs: "
            f"start_epoch={start_epoch} total_epochs={args.epochs}"
        )
        loss_plotter.close()
        return

    # 按 epoch 进行标准训练循环。
    current_epoch = start_epoch
    total_loss = 0.0
    total_steps = 0
    live_loss_x = []
    live_loss_y = []
    last_ckpt = output_dir / "last.pt"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss = 0.0
            total_steps = 0
            live_loss_x = []
            live_loss_y = []
            progress_bar = make_progress_bar(loader, epoch, args.epochs, num_batches)
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(progress_bar, start=1):
                # 把一个 batch 的张量搬到目标设备上。
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # 标准训练步骤：前向 -> 反向 -> 按梯度累积步数更新参数。
                autocast_context = nullcontext()
                if device.type == "cuda":
                    autocast_context = torch.cuda.amp.autocast(enabled=use_amp)

                with autocast_context:
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                loss = outputs["loss"]
                scaled_loss = loss / args.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    batch_idx % args.grad_accum_steps == 0 or batch_idx == num_batches
                )

                if should_step:
                    # 梯度裁剪用于降低梯度爆炸风险。
                    if args.grad_clip is not None and args.grad_clip > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                total_steps += 1
                global_step += 1
                running_loss = total_loss / total_steps

                if TQDM_AVAILABLE:
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg_loss=f"{running_loss:.4f}",
                        step=global_step,
                    )

                if total_steps > 0 and (
                    batch_idx == 1
                    or batch_idx == num_batches
                    or global_step % args.log_every == 0
                ):
                    epoch_progress = (epoch - 1) + (batch_idx / max(num_batches, 1))
                    live_loss_x.append(epoch_progress)
                    live_loss_y.append(running_loss)
                    loss_plotter.update(epoch_losses, (live_loss_x, live_loss_y))

                if global_step % args.log_every == 0:
                    message = (
                        f"epoch={epoch} step={global_step} "
                        f"loss={loss.item():.4f} avg_loss={running_loss:.4f}"
                    )
                    if TQDM_AVAILABLE:
                        progress_bar.write(message)
                    else:
                        print(message)

            # 统计当前 epoch 的平均损失。
            avg_loss = total_loss / max(total_steps, 1)
            epoch_losses.append(avg_loss)
            is_best = avg_loss < best_loss
            best_loss = min(best_loss, avg_loss)
            save_loss_history(output_dir, epoch_losses)
            loss_plotter.update(epoch_losses)

            if TQDM_AVAILABLE:
                progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", step=global_step)
                progress_bar.close()

            print(f"epoch={epoch} avg_loss={avg_loss:.4f}")

            # 每轮都保存一个 last checkpoint，便于断点续训。
            save_checkpoint(
                last_ckpt,
                model,
                optimizer,
                epoch,
                global_step,
                avg_loss,
                vocab,
                config_dict,
                args,
                best_loss,
                epoch_losses,
                epoch_complete=True,
            )

            # 如果当前模型更优，再额外保存 best checkpoint。
            if is_best:
                best_ckpt = output_dir / "best.pt"
                save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    avg_loss,
                    vocab,
                    config_dict,
                    args,
                    best_loss,
                    epoch_losses,
                    epoch_complete=True,
                )
    except KeyboardInterrupt:
        interrupted_loss = total_loss / max(total_steps, 1) if total_steps > 0 else float("inf")
        print("\ntraining interrupted, saving resumable checkpoint...")
        if TQDM_AVAILABLE and "progress_bar" in locals() and hasattr(progress_bar, "close"):
            progress_bar.close()
        save_loss_history(output_dir, epoch_losses)
        loss_plotter.update(epoch_losses, (live_loss_x, live_loss_y))

        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            current_epoch,
            global_step,
            interrupted_loss,
            vocab,
            config_dict,
            args,
            best_loss,
            epoch_losses,
            epoch_complete=False,
        )
        print(f"interrupted checkpoint saved to {last_ckpt}")
    finally:
        loss_plotter.close()


if __name__ == "__main__":
    main()
