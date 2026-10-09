"""Build Vietnamese translation files for selected prompt-catalog groups.

The output follows the existing SAA-compatible format:
``tag,category,translation`` with no header and no comma in the translation.
Only tags that have a real Vietnamese translation in
``danbooru_e621_merged_vi_vn.csv`` are written; untranslated tags are omitted.

Run from the repository root:
    python scripts/build_prompt_catalog_vi.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_DIR = ROOT / "prompt_catalog"
DEFAULT_TRANSLATION = ROOT / "danbooru_e621_merged_vi_vn.csv"
DEFAULT_OUTPUT = ROOT / "prompt_catalog_vi_vn"

SELECTED_GROUPS = (
    "01_chu_the",
    "04_loai_lore",
    "05_trang_phuc_quan_ao",
    "06_phu_kien",
    "07_ngoai_hinh",
    "08_tu_the_bieu_cam",
    "09_boi_canh",
    "10_phong_cach",
    "11_anh_sang_mau_sac",
    "12_bo_cuc_ky_thuat",
    "13_vat_the",
)


def _read_group_names(group_dir: Path, group_id: str) -> set[str]:
    path = group_dir / f"{group_id}.csv"
    with path.open(encoding="utf-8", newline="") as source:
        return {row[0] for row in csv.reader(source) if row}


def _write_csv(path: Path, rows: list[list[str]]) -> tuple[int, int, str]:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
    payload = path.read_bytes()
    return len(rows), len(payload), hashlib.sha256(payload).hexdigest()


def build_group_translations(
    group_dir: Path = DEFAULT_GROUP_DIR,
    translation_path: Path = DEFAULT_TRANSLATION,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    """Create the selected Vietnamese group files and return their manifest."""
    group_dir = Path(group_dir)
    translation_path = Path(translation_path)
    output_dir = Path(output_dir)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    names_by_group = {
        group_id: _read_group_names(group_dir, group_id)
        for group_id in SELECTED_GROUPS
    }
    group_by_name = {
        name: group_id
        for group_id, names in names_by_group.items()
        for name in names
    }
    rows_by_group = OrderedDict((group_id, []) for group_id in SELECTED_GROUPS)
    with translation_path.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 3:
                continue
            group_id = group_by_name.get(row[0])
            if group_id is not None:
                rows_by_group[group_id].append(row)

    groups = []
    for group_id, rows in rows_by_group.items():
        path = temporary / f"{group_id}.csv"
        count, size, digest = _write_csv(path, rows)
        groups.append(
            {
                "id": group_id,
                "file": path.name,
                "rows": count,
                "bytes": size,
                "sha256": digest,
                "source_group_rows": len(names_by_group[group_id]),
                "untranslated_rows": len(names_by_group[group_id]) - count,
            }
        )

    manifest = {
        "source_catalog": "prompt_catalog",
        "source_translation": translation_path.name,
        "schema": "tag,category,translation",
        "header": False,
        "selected_groups": list(SELECTED_GROUPS),
        "translated_rows": sum(group["rows"] for group in groups),
        "groups": groups,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (temporary / "README.md").write_text(_readme(manifest), encoding="utf-8")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.rename(output_dir)
    return manifest


def _readme(manifest: dict) -> str:
    lines = [
        "# File dịch tiếng Việt cho catalog prompt",
        "",
        "Định dạng SAA-compatible: `tag,category,translation`, không header, UTF-8",
        "không BOM, xuống dòng LF. Chỉ các tag có bản dịch thật được ghi; tag chưa",
        "dịch bị bỏ qua, không bịa nhãn.",
        "",
        "| File | Dòng đã dịch | Tag nguồn | Chưa dịch |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in manifest["groups"]:
        lines.append(
            f"| `{group['file']}` | {group['rows']:,} | {group['source_group_rows']:,} | "
            f"{group['untranslated_rows']:,} |"
        )
    lines += [
        "",
        f"Tổng: **{manifest['translated_rows']:,} dòng dịch**.",
        "",
        "Tạo lại bằng:",
        "```bash",
        "python scripts/build_prompt_catalog_vi.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--translation", type=Path, default=DEFAULT_TRANSLATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_group_translations(args.groups, args.translation, args.output_dir)
    print(
        f"Đã tạo {len(manifest['groups'])} file tiếng Việt, "
        f"tổng {manifest['translated_rows']:,} dòng trong `{args.output_dir}`."
    )
    for group in manifest["groups"]:
        print(f"  {group['file']}: {group['rows']:,} dòng")


if __name__ == "__main__":
    main()
