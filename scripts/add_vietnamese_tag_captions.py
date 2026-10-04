"""Add a Vietnamese caption field to the headerless Danbooru/e621 CSV.

Run from the repository root:
    python scripts/add_vietnamese_tag_captions.py

The captions use the same conservative glossary/rules as the Colab tag browser.
Proper names and uncertain tags are kept intact with a Vietnamese category or
"Chưa có bản dịch" label rather than guessed translations.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colab.studio import (  # noqa: E402
    TAG_CSV_NAME,
    _TAG_VI_CATEGORY_FALLBACKS,
    vietnamese_tag_label,
)


def enrich_csv(input_path, output_path=None):
    source = Path(input_path)
    destination = Path(output_path) if output_path else source
    destination.parent.mkdir(parents=True, exist_ok=True)
    translated = 0
    total = 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            writer = csv.writer(temporary, lineterminator="\n")
            with source.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.reader(stream):
                    if len(row) not in (4, 5):
                        raise ValueError(f"Dòng {total + 1} có {len(row)} cột, cần 4 hoặc 5.")
                    name, category, count, aliases = row[:4]
                    if not name or not count.isdigit():
                        raise ValueError(f"Dòng {total + 1} không có tên thẻ hoặc lượt hợp lệ.")
                    caption = vietnamese_tag_label(name, category)
                    writer.writerow((name, category, count, aliases, caption))
                    total += 1
                    if caption != _TAG_VI_CATEGORY_FALLBACKS.get(category, "Chưa có bản dịch"):
                        translated += 1
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(
        f"Đã thêm nhãn tiếng Việt cho {total:,} thẻ "
        f"({translated:,} có bản dịch từ điển/quy tắc; "
        f"{total - translated:,} giữ tên gốc hoặc ghi rõ chưa dịch)."
    )
    print(f"Tệp: {destination}\nKích thước: {destination.stat().st_size:,} byte\nSHA-256: {digest}")
    return total, translated, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=ROOT / TAG_CSV_NAME,
        help="CSV nguồn (mặc định: catalog trong thư mục gốc)",
    )
    parser.add_argument("--output", type=Path, help="tạo file đích riêng, không ghi đè nguồn")
    args = parser.parse_args()
    enrich_csv(args.input, args.output)


if __name__ == "__main__":
    main()
