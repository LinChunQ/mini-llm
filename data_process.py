import json
import os
from json import JSONDecodeError
from typing import Iterable, List, Tuple


def prepare_poetry_train_data(json_dir: str, output_file: str) -> Tuple[int, int, int]:
    total_poems = 0
    written_poems = 0
    failed_files = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历下载好的所有唐诗 JSON 文件
        for filename in _iter_tang_files(json_dir):
            file_path = os.path.join(json_dir, filename)
            try:
                data = _load_json_with_fallback(file_path)
            except (OSError, UnicodeDecodeError, JSONDecodeError):
                # 单个分片损坏时不中断整体处理流程，直接跳过并统计失败数量
                failed_files += 1
                continue

            if not isinstance(data, list):
                failed_files += 1
                continue

            for poem in data:
                total_poems += 1
                if not isinstance(poem, dict):
                    continue

                title = str(poem.get("title", "")).strip()
                author = str(poem.get("author", "")).strip()
                paragraphs = _merge_paragraphs(poem.get("paragraphs"))

                if not paragraphs:
                    continue

                # 构造大模型理解的文本范式：
                # [BOS] 规定开始，用 [SEP] 隔开元数据，用 [EOS] 规定结束
                # 这对应了大模型中特殊 Token（special_tokens）的物理意义
                full_text = f"[BOS]标题:{title} 作者:{author}[SEP]{paragraphs}[EOS]\n"
                f_out.write(full_text)
                written_poems += 1

    return total_poems, written_poems, failed_files


def _iter_tang_files(json_dir: str) -> Iterable[str]:
    files: List[str] = []
    for filename in os.listdir(json_dir):
        full_path = os.path.join(json_dir, filename)
        if os.path.isfile(full_path) and filename.startswith("poet.tang.") and filename.endswith(".json"):
            files.append(filename)

    # 按分片编号排序，确保每次生成语料顺序一致，便于复现实验结果
    files.sort(key=_extract_shard_index)
    return files


def _extract_shard_index(filename: str) -> int:
    parts = filename.split(".")
    if len(parts) >= 4 and parts[2].isdigit():
        return int(parts[2])
    return 10**9


def _load_json_with_fallback(file_path: str):
    # 数据集可能混入不同编码，按常见编码顺序尝试读取
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(file_path, 'r', encoding=encoding) as f_in:
                return json.load(f_in)
        except (UnicodeDecodeError, JSONDecodeError):
            continue

    # 最后一次读取用于抛出明确异常，方便定位具体坏文件
    with open(file_path, 'r', encoding='utf-8') as f_in:
        return json.load(f_in)


def _merge_paragraphs(paragraphs) -> str:
    if isinstance(paragraphs, list):
        cleaned_lines = [str(line).strip() for line in paragraphs if str(line).strip()]
        return "".join(cleaned_lines)
    if isinstance(paragraphs, str):
        return paragraphs.strip()
    return ""

# 假设你的 json 文件放在 ./tang_repo 目录下
# prepare_poetry_train_data("./tang_repo", "poetry_train.txt")
if __name__ == "__main__":
    input_dir = "./全唐诗"
    output_path = "poetry_train.txt"
    total, written, failed = prepare_poetry_train_data(input_dir, output_path)
    print(f"处理完成: 扫描诗词 {total} 首，写入 {written} 首，跳过异常分片 {failed} 个。输出文件: {output_path}")
