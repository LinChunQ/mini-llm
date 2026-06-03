"""mini-llm 的文本生成脚本。

负责加载训练好的 checkpoint、词表和模型配置，并基于给定 prompt 做自回归生成。
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

import torch


# 与训练阶段保持一致：优先识别特殊 token，其余按单字符切分。
TOKEN_PATTERN = re.compile(r"\[BOS\]|\[SEP\]|\[EOS\]|.")

# 推理时会用到的特殊 token。
SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[BOS]", "[SEP]", "[EOS]"}


def load_model_module():
    """动态加载 `mini-llm.py`。"""
    module_path = Path(__file__).with_name("mini-llm.py")
    spec = importlib.util.spec_from_file_location("mini_llm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    """读取 UTF-8 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path):
    """加载训练 checkpoint。"""
    # checkpoint 中除了权重还有配置、词表和训练状态，推理时保持完整读取更稳妥。
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_config_overrides(config, config_dict):
    """把配置字典写回配置对象。"""
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def seed_everything(seed):
    """固定随机种子，便于复现生成结果。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    """把文本切成 token 列表。"""
    return TOKEN_PATTERN.findall(text)


def encode(text, vocab, max_seq_len):
    """把文本编码成 token id 序列。"""
    unk_id = vocab["[UNK]"]
    ids = [vocab.get(token, unk_id) for token in tokenize(text)]
    return ids[:max_seq_len]


def decode(token_ids, id_to_token, skip_special_tokens=True):
    """把 token id 序列解码回文本。"""
    pieces = []
    for token_id in token_ids:
        token = id_to_token.get(int(token_id), "[UNK]")
        if skip_special_tokens and token in SPECIAL_TOKENS:
            continue
        pieces.append(token)
    return "".join(pieces)


def resolve_runtime_device(device_arg):
    """根据参数解析运行设备。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_prompt(prompt, add_bos):
    """补齐推理 prompt 的起始标记。"""
    if add_bos and not prompt.startswith("[BOS]"):
        return f"[BOS]{prompt}"
    return prompt


def build_structured_prompt(title, author, content_prefix):
    """把标题、作者和正文前缀拼成训练时同格式的 prompt。"""
    prompt = f"标题:{title} 作者:{author}[SEP]"
    if content_prefix:
        prompt = f"{prompt}{content_prefix}"
    return prompt


def resolve_inference_prompt(args, prompt=None, title=None, author=None, content_prefix=None):
    """优先使用结构化字段生成 prompt，否则回退到原始 prompt。"""
    resolved_title = title if title is not None else args.title
    resolved_author = author if author is not None else args.author
    resolved_content_prefix = (
        content_prefix if content_prefix is not None else args.content_prefix
    )
    if resolved_title is not None and resolved_author is not None:
        return build_structured_prompt(
            resolved_title,
            resolved_author,
            resolved_content_prefix,
        )
    return prompt if prompt is not None else args.prompt


def read_interactive_structured_prompt():
    """在交互模式下逐项读取标题、作者和正文前缀。"""
    title = input("title> ").strip()
    if title.lower() in {"/exit", "/quit", "exit", "quit"}:
        return None
    author = input("author> ").strip()
    if author.lower() in {"/exit", "/quit", "exit", "quit"}:
        return None
    content_prefix = input("content> ").strip()
    return build_structured_prompt(title, author, content_prefix)


def build_chat_session(args):
    """初始化聊天式生成会话状态。"""
    return {
        "title": args.title or "",
        "author": args.author or "",
        "content_prefix": args.content_prefix or "",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }


def print_chat_help():
    """打印聊天模式可用命令。"""
    print("chat commands:")
    print("  /title <文本>       设置标题")
    print("  /author <文本>      设置作者")
    print("  /prefix <文本>      设置正文开头")
    print("  /append <文本>      追加到正文开头")
    print("  /show               查看当前会话状态")
    print("  /clear              清空正文开头")
    print("  /reset              清空标题、作者和正文开头")
    print("  /params k=v ...     修改采样参数，如 temperature=0.8 top_k=20")
    print("  /gen                用当前状态生成")
    print("  /help               查看帮助")
    print("  /exit               退出")
    print("plain text:")
    print("  直接输入一行正文开头，会用当前标题和作者生成")


def print_chat_state(session):
    """打印当前聊天会话状态。"""
    print("current state:")
    print(f"  标题: {session['title'] or '(未设置)'}")
    print(f"  作者: {session['author'] or '(未设置)'}")
    print(f"  正文开头: {session['content_prefix'] or '(空)'}")
    print(
        "  采样参数: "
        f"max_new_tokens={session['max_new_tokens']} "
        f"temperature={session['temperature']} "
        f"top_k={session['top_k']} "
        f"top_p={session['top_p']} "
        f"repetition_penalty={session['repetition_penalty']}"
    )


def build_chat_prompt(session):
    """根据聊天会话状态拼出最终 prompt。"""
    if not session["title"] or not session["author"]:
        raise ValueError("title and author are required before generation")
    return build_structured_prompt(
        session["title"],
        session["author"],
        session["content_prefix"],
    )


def clone_args_with_session(args, session):
    """把聊天会话里的采样参数覆盖到临时 args 上。"""
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.title = session["title"] or None
    runtime_args.author = session["author"] or None
    runtime_args.content_prefix = session["content_prefix"]
    runtime_args.max_new_tokens = session["max_new_tokens"]
    runtime_args.temperature = session["temperature"]
    runtime_args.top_k = session["top_k"]
    runtime_args.top_p = session["top_p"]
    runtime_args.repetition_penalty = session["repetition_penalty"]
    return runtime_args


def update_chat_params(session, params_text):
    """解析并更新聊天模式下的采样参数。"""
    valid_keys = {
        "max_new_tokens": int,
        "temperature": float,
        "top_k": int,
        "top_p": float,
        "repetition_penalty": float,
    }
    assignments = [item for item in params_text.split() if item]
    if not assignments:
        raise ValueError("usage: /params key=value ...")

    # 逐项更新参数，避免一次输入错误把整组会话状态搞乱。
    pending_updates = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"invalid param assignment: {item}")
        key, raw_value = item.split("=", 1)
        if key not in valid_keys:
            raise ValueError(f"unsupported param: {key}")
        pending_updates[key] = valid_keys[key](raw_value)

    for key, value in pending_updates.items():
        session[key] = value


def handle_chat_command(session, command_line):
    """处理聊天模式命令，并返回后续动作。"""
    command, _, value = command_line.partition(" ")
    value = value.strip()

    if command in {"/exit", "/quit"}:
        return "exit"
    if command == "/help":
        print_chat_help()
        return "continue"
    if command == "/show":
        print_chat_state(session)
        return "continue"
    if command == "/clear":
        session["content_prefix"] = ""
        print("正文开头已清空")
        return "continue"
    if command == "/reset":
        session["title"] = ""
        session["author"] = ""
        session["content_prefix"] = ""
        print("标题、作者和正文开头已清空")
        return "continue"
    if command == "/title":
        session["title"] = value
        print(f"标题已更新: {session['title']}")
        return "continue"
    if command == "/author":
        session["author"] = value
        print(f"作者已更新: {session['author']}")
        return "continue"
    if command == "/prefix":
        session["content_prefix"] = value
        print(f"正文开头已更新: {session['content_prefix']}")
        return "continue"
    if command == "/append":
        session["content_prefix"] = f"{session['content_prefix']}{value}"
        print(f"正文开头已更新: {session['content_prefix']}")
        return "continue"
    if command == "/params":
        update_chat_params(session, value)
        print("采样参数已更新")
        print_chat_state(session)
        return "continue"
    if command == "/gen":
        return "generate"

    raise ValueError(f"unknown command: {command}")


def resolve_generation_paths(args):
    """根据命令行参数解析 checkpoint、词表和配置文件位置。"""
    checkpoint_path = Path(args.checkpoint)

    vocab_path = Path(args.vocab_path) if args.vocab_path else checkpoint_path.with_name("vocab.json")
    config_path = Path(args.config_path) if args.config_path else checkpoint_path.with_name("config.json")
    return checkpoint_path, vocab_path, config_path


def load_runtime_artifacts(args):
    """加载推理所需的 checkpoint、词表和配置。"""
    checkpoint_path, vocab_path, config_path = resolve_generation_paths(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)

    if vocab_path.exists():
        vocab = load_json(vocab_path)
    elif "vocab" in checkpoint:
        vocab = checkpoint["vocab"]
    else:
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    if config_path.exists():
        config_dict = load_json(config_path)
    elif "config" in checkpoint:
        config_dict = checkpoint["config"]
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    return checkpoint_path, checkpoint, vocab, config_dict


def build_model_and_tokenizer(args):
    """构建推理模型并准备词表映射。"""
    checkpoint_path, checkpoint, vocab, config_dict = load_runtime_artifacts(args)
    model_module = load_model_module()
    config = model_module.MiniLLMConfig()
    config = apply_config_overrides(config, config_dict)
    config.vocab_size = len(vocab)
    config.pad_token_id = vocab["[PAD]"]

    device = resolve_runtime_device(args.device)
    model = model_module.MiniModelForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_token = {idx: token for token, idx in vocab.items()}
    return checkpoint_path, model, vocab, id_to_token, config, device


def apply_sampling_filters(logits, top_k, top_p):
    """对 logits 应用 top-k 和 top-p 过滤。"""
    filtered_logits = logits.clone()

    if top_k is not None and top_k > 0 and top_k < filtered_logits.numel():
        topk_values, _ = torch.topk(filtered_logits, top_k)
        threshold = topk_values[-1]
        filtered_logits[filtered_logits < threshold] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 保留首个超过阈值的 token，避免整段被过滤空。
        sorted_mask = cumulative_probs > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False

        filtered_logits[sorted_indices[sorted_mask]] = float("-inf")

    return filtered_logits


def sample_next_token(logits, generated_ids, temperature, top_k, top_p, repetition_penalty):
    """从最后一个位置的 logits 中采样下一个 token。"""
    next_token_logits = logits[0, -1, :].float().clone()

    # 对已出现 token 做重复惩罚，降低连续复读概率。
    if repetition_penalty is not None and repetition_penalty > 1.0:
        unique_token_ids = set(generated_ids)
        for token_id in unique_token_ids:
            if next_token_logits[token_id] < 0:
                next_token_logits[token_id] *= repetition_penalty
            else:
                next_token_logits[token_id] /= repetition_penalty

    if temperature <= 0:
        return int(torch.argmax(next_token_logits).item())

    next_token_logits = next_token_logits / temperature
    next_token_logits = apply_sampling_filters(next_token_logits, top_k, top_p)
    probs = torch.softmax(next_token_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_from_prompt(args, model, vocab, id_to_token, config, device, prompt):
    """基于给定 prompt 执行一次文本生成。"""
    prompt = resolve_prompt(prompt, add_bos=args.add_bos)
    input_ids = encode(prompt, vocab, config.max_position_embeddings)
    if not input_ids:
        raise ValueError("Prompt is empty after tokenization.")

    eos_token_id = vocab.get("[EOS]")
    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            current_ids = generated_ids[-config.max_position_embeddings :]
            input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor, dtype=torch.bool)
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)

            next_token_id = sample_next_token(
                outputs["logits"],
                generated_ids=current_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            generated_ids.append(next_token_id)

            if args.stop_at_eos and eos_token_id is not None and next_token_id == eos_token_id:
                break

    generated_text = decode(generated_ids, id_to_token, skip_special_tokens=args.skip_special_tokens)
    prompt_text = decode(input_ids, id_to_token, skip_special_tokens=args.skip_special_tokens)
    return prompt_text, generated_text, len(input_ids), len(generated_ids) - len(input_ids)


def generate_text(args):
    """执行一次完整的文本生成。"""
    checkpoint_path, model, vocab, id_to_token, config, device = build_model_and_tokenizer(args)
    seed_everything(args.seed)
    prompt = resolve_inference_prompt(args)
    prompt_text, generated_text, prompt_token_count, generated_token_count = generate_from_prompt(
        args,
        model,
        vocab,
        id_to_token,
        config,
        device,
        prompt,
    )

    if args.print_meta:
        print(f"checkpoint: {checkpoint_path}")
        print(f"device: {device}")
        print(f"prompt_tokens: {prompt_token_count} generated_tokens: {generated_token_count}")

    if args.only_new_text:
        if generated_text.startswith(prompt_text):
            print(generated_text[len(prompt_text) :])
        else:
            print(generated_text)
    else:
        print(generated_text)


def run_interactive(args):
    """进入交互式生成模式，复用已加载模型连续生成。"""
    checkpoint_path, model, vocab, id_to_token, config, device = build_model_and_tokenizer(args)
    seed_everything(args.seed)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    if args.structured_input:
        print("interactive mode: input title/author/content separately, type /exit to quit.")
    else:
        print("interactive mode: input prompt and press Enter, type /exit to quit.")

    while True:
        try:
            if args.structured_input:
                resolved_prompt = read_interactive_structured_prompt()
                if resolved_prompt is None:
                    break
                user_prompt = resolved_prompt
            else:
                user_prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_prompt:
            continue
        if not args.structured_input and user_prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            break

        try:
            prompt_text, generated_text, prompt_token_count, generated_token_count = generate_from_prompt(
                args,
                model,
                vocab,
                id_to_token,
                config,
                device,
                user_prompt,
            )
        except Exception as exc:
            print(f"generation failed: {exc}")
            continue

        # 交互模式下默认输出本轮 prompt 的 token 统计，便于快速观察长度和截断情况。
        if args.print_meta:
            print(
                f"prompt_tokens={prompt_token_count} "
                f"generated_tokens={generated_token_count}"
            )

        if args.only_new_text and generated_text.startswith(prompt_text):
            print(generated_text[len(prompt_text) :])
        else:
            print(generated_text)
        print()


def run_chat(args):
    """进入仿聊天模式，通过命令维护会话状态并连续生成。"""
    checkpoint_path, model, vocab, id_to_token, config, device = build_model_and_tokenizer(args)
    seed_everything(args.seed)
    session = build_chat_session(args)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print("chat mode: use /help for commands, plain text will be treated as content prefix.")
    print_chat_state(session)

    while True:
        try:
            user_input = input("chat> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        try:
            if user_input.startswith("/"):
                action = handle_chat_command(session, user_input)
                if action == "exit":
                    break
                if action != "generate":
                    continue
            else:
                # 普通输入默认直接替换正文开头，然后立即生成，减少命令负担。
                session["content_prefix"] = user_input

            runtime_args = clone_args_with_session(args, session)
            prompt = build_chat_prompt(session)
            prompt_text, generated_text, prompt_token_count, generated_token_count = generate_from_prompt(
                runtime_args,
                model,
                vocab,
                id_to_token,
                config,
                device,
                prompt,
            )
        except Exception as exc:
            print(f"generation failed: {exc}")
            continue

        if runtime_args.print_meta:
            print(
                f"prompt_tokens={prompt_token_count} "
                f"generated_tokens={generated_token_count}"
            )

        if runtime_args.only_new_text and generated_text.startswith(prompt_text):
            print(generated_text[len(prompt_text) :])
        else:
            print(generated_text)
        print()


def main():
    """解析参数并执行文本生成。"""
    parser = argparse.ArgumentParser(description="Generate text with the trained mini causal language model.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--prompt", default="[BOS]标题:春晓 作者:孟浩然[SEP]")
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--content-prefix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.set_defaults(add_bos=True)
    parser.add_argument("--add-bos", dest="add_bos", action="store_true")
    parser.add_argument("--no-add-bos", dest="add_bos", action="store_false")
    parser.set_defaults(stop_at_eos=True)
    parser.add_argument("--stop-at-eos", dest="stop_at_eos", action="store_true")
    parser.add_argument("--no-stop-at-eos", dest="stop_at_eos", action="store_false")
    parser.set_defaults(skip_special_tokens=True)
    parser.add_argument("--skip-special-tokens", dest="skip_special_tokens", action="store_true")
    parser.add_argument("--keep-special-tokens", dest="skip_special_tokens", action="store_false")
    parser.set_defaults(only_new_text=False)
    parser.add_argument("--only-new-text", dest="only_new_text", action="store_true")
    parser.add_argument("--full-text", dest="only_new_text", action="store_false")
    parser.set_defaults(print_meta=True)
    parser.add_argument("--print-meta", dest="print_meta", action="store_true")
    parser.add_argument("--no-print-meta", dest="print_meta", action="store_false")
    parser.set_defaults(chat=False)
    parser.add_argument("--chat", dest="chat", action="store_true")
    parser.set_defaults(interactive=False)
    parser.add_argument("--interactive", dest="interactive", action="store_true")
    parser.set_defaults(structured_input=False)
    parser.add_argument("--structured-input", dest="structured_input", action="store_true")
    args = parser.parse_args()

    # 结构化单次生成需要标题和作者同时提供，避免只传半套字段时行为不明确。
    if (args.title is None) != (args.author is None):
        raise ValueError("--title and --author must be provided together")

    if args.chat:
        run_chat(args)
    elif args.interactive:
        run_interactive(args)
    else:
        generate_text(args)


if __name__ == "__main__":
    main()
