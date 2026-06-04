#!/usr/bin/env python3
"""简洁的 mini-llm 文本生成 CLI 工具。"""

import argparse
import re
import sys
from pathlib import Path

import torch

from generate import (
    build_model_and_tokenizer,
    generate_from_prompt,
    seed_everything,
    resolve_inference_prompt,
)


def print_banner():
    """打印欢迎信息。"""
    print("=" * 60)
    print("Mini-LLM 文本生成工具")
    print("=" * 60)


def print_generation_result(prompt_text, generated_text, prompt_tokens, generated_tokens, only_new=False):
    """格式化输出生成结果。"""
    print("\n" + "-" * 60)
    if only_new and generated_text.startswith(prompt_text):
        print(generated_text[len(prompt_text):])
    else:
        print(generated_text)
    print("-" * 60)
    print(f"Prompt tokens: {prompt_tokens} | Generated tokens: {generated_tokens}")
    print()


def run_single_generation(args):
    """执行单次生成。"""
    print_banner()
    checkpoint_path, model, vocab, id_to_token, config, device = build_model_and_tokenizer(args)
    seed_everything(args.seed)

    print(f"模型: {checkpoint_path}")
    print(f"设备: {device}")
    print()

    prompt = resolve_inference_prompt(args)
    print(f"输入 Prompt: {prompt}")

    prompt_text, generated_text, prompt_tokens, generated_tokens = generate_from_prompt(
        args, model, vocab, id_to_token, config, device, prompt
    )

    print_generation_result(prompt_text, generated_text, prompt_tokens, generated_tokens, args.only_new_text)


def run_interactive_loop(args):
    """交互式循环生成。"""
    print_banner()
    checkpoint_path, model, vocab, id_to_token, config, device = build_model_and_tokenizer(args)
    seed_everything(args.seed)

    print(f"模型: {checkpoint_path}")
    print(f"设备: {device}")
    print()
    print("⚠️  注意: 此模型使用繁体字训练，请输入繁体字以获得最佳效果")
    print()
    print("提示：")
    print("  - 输入格式: 标题:春曉 作者:孟浩然 正文:春眠不覺曉")
    print("  - 支持中英文冒号（: 或 ：）")
    print("  - 或直接输入任意文本")
    print("  - 输入 /quit 或按 Ctrl+C 退出")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("再见！")
            break

        # 尝试解析结构化输入（支持中英文冒号）
        normalized_input = user_input.replace("：", ":")
        if "标题:" in normalized_input and "作者:" in normalized_input:
            try:
                # 提取各字段：标题和作者遇到空格或下一个字段就停止，正文取到末尾
                title_match = re.search(r'标题:([^\s]+?)(?:\s|$)', normalized_input)
                author_match = re.search(r'作者:([^\s]+?)(?:\s|$)', normalized_input)
                content_match = re.search(r'正文:(.+?)(?:$)', normalized_input)

                title = title_match.group(1) if title_match else ""
                author = author_match.group(1) if author_match else ""
                content = content_match.group(1).strip() if content_match else ""

                if title and author:
                    if content:
                        prompt = f"标题:{title} 作者:{author}[SEP]{content}"
                        print(f"[解析] 引导生成 -> 标题={title} | 作者={author} | 引导句={content}")
                    else:
                        # 如果用户只输了“标题:春晓 作者:孟浩然”，让模型自由发挥写全诗
                        prompt = f"标题:{title} 作者:{author}[SEP]"
                        print(f"[解析] 自由生成 -> 标题={title} | 作者={author}")
                else:
                    prompt = user_input
            except Exception as e:
                print(f"[警告] 解析失败: {e}，使用原始输入")
                prompt = user_input
        else:
            prompt = user_input

        try:
            prompt_text, generated_text, prompt_tokens, generated_tokens = generate_from_prompt(
                args, model, vocab, id_to_token, config, device, prompt
            )
            print_generation_result(prompt_text, generated_text, prompt_tokens, generated_tokens, args.only_new_text)
        except Exception as e:
            print(f"生成失败: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mini-LLM 文本生成 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次生成
  python cli.py --title 春晓 --author 孟浩然

  # 交互模式
  python cli.py --interactive

  # 调整采样参数
  python cli.py --title 春晓 --author 孟浩然 --temperature 0.8 --top-k 50
        """
    )

    # 模型路径
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="模型 checkpoint 路径")
    parser.add_argument("--vocab-path", default=None, help="词表文件路径")
    parser.add_argument("--config-path", default=None, help="配置文件路径")

    # 输入内容
    parser.add_argument("--prompt", default=None, help="直接指定 prompt")
    parser.add_argument("--title", default=None, help="标题")
    parser.add_argument("--author", default=None, help="作者")
    parser.add_argument("--content-prefix", default="", help="正文开头")

    # 生成参数
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度 (0=贪心)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K 采样")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-P (nucleus) 采样")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚系数")

    # 运行模式
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--only-new-text", action="store_true", help="仅输出新生成的文本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto", help="运行设备 (auto/cpu/cuda)")

    args = parser.parse_args()

    # 验证参数
    if not args.interactive and not args.prompt and (not args.title or not args.author):
        parser.error("单次生成模式需要 --prompt 或同时指定 --title 和 --author")

    # 设置默认值
    args.add_bos = True
    args.stop_at_eos = True
    args.skip_special_tokens = True
    args.print_meta = False
    args.chat = False
    args.structured_input = False

    # 检查 checkpoint 是否存在
    if not Path(args.checkpoint).exists():
        print(f"错误: Checkpoint 不存在: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # 运行
    try:
        if args.interactive:
            run_interactive_loop(args)
        else:
            run_single_generation(args)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()