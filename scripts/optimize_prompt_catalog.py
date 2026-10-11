"""Filter the merged Danbooru/e621 catalog down to prompt-usable rows.

The source catalog contains ordinary prompt tags plus moderation/source metadata.
The Studio needs the original five columns (name, category, count, aliases,
Vietnamese label), so this script keeps that schema and only removes rows whose
category is metadata or otherwise not recognized by the Studio catalog.

Run from the repository root:
    python scripts/optimize_prompt_catalog.py --in-place

The rewrite is byte-preserving for rows that survive the filter, which makes the
result deterministic and keeps the Vietnamese labels/aliases unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"

# Categories 5 and 14 are Danbooru/e621 metadata. Any category outside this set
# is not a prompt catalog row either (and would be rejected by studio.parse_tag_csv).
PROMPT_CATEGORIES = frozenset({"0", "1", "3", "4", "7", "8", "9", "10", "11", "12", "15"})
EXPECTED_COLUMNS = 5


def prompt_row(row: list[str]) -> bool:
    """Return whether one raw CSV row belongs in the prompt catalog."""
    return (
        len(row) == EXPECTED_COLUMNS
        and bool(row[0].strip())
        and row[1] in PROMPT_CATEGORIES
        and row[2].isdigit()
    )


def optimize_catalog(input_path: Path, output_path: Path) -> tuple[int, int, str]:
    """Write the filtered catalog and return (before, after, sha256)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        raw = source.read()
    kept: list[str] = []
    before = 0
    for line in raw.splitlines(keepends=True):
        if not line.strip():
            continue
        before += 1
        try:
            row = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue
        if prompt_row(row):
            kept.append(line)

    payload = "".join(kept).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)
    return before, len(kept), hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input file atomically (the default output is otherwise a sibling file)",
    )
    args = parser.parse_args()
    if args.in_place and args.output:
        parser.error("dùng --in-place hoặc --output, không dùng cả hai")
    if args.in_place:
        output = args.input
    elif args.output:
        output = args.output
    else:
        output = args.input.with_name(args.input.stem + "_prompt.csv")
    before, after, digest = optimize_catalog(args.input, output)
    print(f"Đã lọc {before:,} → {after:,} dòng; bỏ {before - after:,} dòng metadata/không hợp lệ.")
    print(f"SHA-256: {digest}")
    print(f"Tệp: {output}")


if __name__ == "__main__":
    main()
