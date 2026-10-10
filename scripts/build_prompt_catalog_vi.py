"""Build Vietnamese translation files for selected prompt-catalog groups.

The output follows the existing SAA-compatible format:
``tag,category,translation`` with no header and no comma in the translation.
The requested groups use real Vietnamese translations from
``danbooru_e621_merged_vi_vn.csv``. Every remaining tag in the selected groups
receives a deterministic machine-style fallback: known words use the local
Vietnamese glossary and underscores are converted to spaces.

Run from the repository root:
    python scripts/build_prompt_catalog_vi.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    "99_khac",
)

# Every requested output group is complete: tags without a curated translation
# receive an automatic label with underscores converted to spaces.
MACHINE_TRANSLATION_GROUPS = frozenset(SELECTED_GROUPS)

_MACHINE_FALLBACK_LABELS = frozenset(
    {"Chưa có bản dịch", "Họa sĩ", "Tác phẩm", "Nhân vật", "Người đóng góp", "Loài", "Lore", "Metadata"}
)

_MACHINE_WORD_OVERRIDES = {
    "after": "sau",
    "against": "chống lại",
    "ambiguous": "mơ hồ",
    "anal": "hậu môn",
    "and": "và",
    "around": "xung quanh",
    "back": "phía sau",
    "big": "lớn",
    "blood": "máu",
    "brown": "nâu",
    "character": "nhân vật",
    "cum": "tinh dịch",
    "double": "đôi",
    "down": "xuống",
    "eating": "ăn",
    "emphasis": "nhấn mạnh",
    "forced": "bị ép buộc",
    "from": "từ",
    "front": "phía trước",
    "gold": "vàng",
    "golden": "vàng",
    "happy": "vui vẻ",
    "head": "đầu",
    "heart": "trái tim",
    "human": "người",
    "ice": "băng",
    "imminent": "sắp xảy ra",
    "in": "trong",
    "inside": "bên trong",
    "large": "lớn",
    "lines": "đường nét",
    "little": "nhỏ",
    "magic": "ma thuật",
    "many": "nhiều",
    "multiple": "nhiều",
    "no": "không có",
    "one": "một",
    "open": "mở",
    "oral": "bằng miệng",
    "over": "trên",
    "peeing": "đi tiểu",
    "playing": "đang chơi",
    "power": "sức mạnh",
    "public": "công cộng",
    "reverse": "ngược",
    "sex": "quan hệ tình dục",
    "shaking": "run rẩy",
    "side": "bên",
    "single": "một",
    "small": "nhỏ",
    "star": "ngôi sao",
    "straight": "thẳng",
    "tan": "nâu rám",
    "the": "",
    "too": "quá",
    "two": "hai",
    "under": "bên dưới",
    "up": "lên",
    "vaginal": "âm đạo",
    "with": "với",
    "without": "không có",
}


def _read_group_rows(group_dir: Path, group_id: str) -> list[list[str]]:
    path = group_dir / f"{group_id}.csv"
    with path.open(encoding="utf-8", newline="") as source:
        return [row for row in csv.reader(source) if len(row) >= 2]


def _is_real_machine_seed(label: str, _category: str) -> bool:
    return bool(label and label not in _MACHINE_FALLBACK_LABELS)


def _is_curated_source_label(tag: str, category: str, label: str) -> bool:
    if not _is_real_machine_seed(label, category):
        return False
    return label.replace("_", " ").casefold() != tag.casefold()


def _curated_tags_from_group(group_dir: Path, group_id: str, source_rows: list[list[str]]) -> set[str]:
    """Find real labels using the same Studio dictionary as the root builder."""
    if not source_rows:
        return set()
    try:
        from colab import studio

        parsed = studio.parse_tag_csv(
            (group_dir / f"{group_id}.csv").read_text(encoding="utf-8")
        )
        return {
            name
            for name, category, _count, _index, _themes, label in parsed
            if _is_curated_source_label(name, category, label)
        }
    except (ImportError, ValueError):
        return {
            row[0]
            for row in source_rows
            if len(row) >= 5 and _is_curated_source_label(row[0], row[1], row[4])
        }


def _machine_translation_tools(translation_rows: list[list[str]]) -> tuple[dict[str, str], object | None]:
    """Load the local Vietnamese word rules used for automatic fallback labels.

    The builder remains usable without importing the Studio module. When it is
    available, its maintained word tables improve the automatic gloss; otherwise
    the checked-in translation rows and the small common-word table still provide
    a deterministic fallback.
    """
    words = dict(_MACHINE_WORD_OVERRIDES)
    for row in translation_rows:
        tag, category, translation = row
        if (
            re.fullmatch(r"[A-Za-z0-9]+", tag)
            and _is_real_machine_seed(translation, category)
        ):
            words.setdefault(tag.casefold(), translation)

    studio = None
    try:
        from colab import studio as studio_module

        studio = studio_module
        for table_name in ("_TAG_VI_WORDS", "_TAG_VI_COLORS", "_TAG_VI_COMPOSITE_HEADS", "TAG_VI_LABELS"):
            for tag, translation in getattr(studio_module, table_name, {}).items():
                if re.fullmatch(r"[A-Za-z0-9]+", tag):
                    words.setdefault(tag.casefold(), str(translation).split("|", 1)[0])
    except (ImportError, AttributeError):
        pass
    return words, studio


def _machine_translate_tag(tag: str, category: str, words: dict[str, str], studio: object | None) -> str:
    """Create a deterministic machine-style gloss and always remove underscores.

    Curated labels remain preferred. For a missing tag, known English words use
    the local Vietnamese glossary; unknown words/proper names are retained as
    readable English, with ``_`` converted to a space instead of being left as an
    untranslated opaque identifier.
    """
    if studio is not None:
        label = studio.vietnamese_tag_label(tag, category)
        if studio._is_translated_tag_label(label, category):
            return _sanitize_machine_label(label)

    pieces = []
    for piece in re.split(r"_+", tag):
        key = piece.casefold()
        replacement = words.get(key)
        if replacement is None and key.endswith("s") and len(key) > 3:
            replacement = words.get(key[:-1])
        pieces.append(replacement if replacement is not None else piece)
    label = " ".join(pieces)
    if not re.search(r"[A-Za-zÀ-ỹ]", label):
        label = f"Ký hiệu {label}".strip()
    return _sanitize_machine_label(label[:1].upper() + label[1:] if label else "Nhãn tự động")


def _sanitize_machine_label(label: str) -> str:
    label = re.sub(r"[\r\n]+", " ", str(label or ""))
    label = label.replace(",", ";").replace('"', "'").replace("_", " ")
    words = re.sub(r"\s+", " ", label).strip().split()
    compact = []
    for word in words:
        if compact and word.casefold() == compact[-1].casefold():
            continue
        compact.append(word)
    label = " ".join(compact)
    return (label[:1].upper() + label[1:]) if label else "Nhãn tự động"


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

    source_rows_by_group = {
        group_id: _read_group_rows(group_dir, group_id)
        for group_id in SELECTED_GROUPS
    }
    names_by_group = {
        group_id: {row[0] for row in rows}
        for group_id, rows in source_rows_by_group.items()
    }
    group_by_name = {
        name: group_id
        for group_id, names in names_by_group.items()
        for name in names
    }
    translation_rows = []
    rows_by_group = OrderedDict((group_id, []) for group_id in SELECTED_GROUPS)
    with translation_path.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 3:
                continue
            translation_rows.append(row)
            group_id = group_by_name.get(row[0])
            if group_id is not None:
                rows_by_group[group_id].append(row)

    # The root translate file may itself contain machine-generated rows. Curated
    # status therefore comes from the original five-column source label, not from
    # mere presence in ``translation_path``.
    curated_tags_by_group = {
        group_id: _curated_tags_from_group(
            group_dir, group_id, source_rows_by_group[group_id]
        )
        for group_id in SELECTED_GROUPS
    }
    curated_translation_rows = [
        row
        for row in translation_rows
        if group_by_name.get(row[0]) is not None
        and row[0] in curated_tags_by_group[group_by_name[row[0]]]
    ]
    curated_by_tag = {row[0]: row for row in curated_translation_rows}
    machine_words, studio = _machine_translation_tools(curated_translation_rows)
    groups = []
    for group_id in SELECTED_GROUPS:
        source_rows = source_rows_by_group[group_id]
        real_rows = [
            curated_by_tag[row[0]]
            for row in source_rows
            if row[0] in curated_by_tag
        ]
        machine_count = 0
        if group_id in MACHINE_TRANSLATION_GROUPS:
            # Walk the canonical group order so autocomplete keeps the source
            # popularity order while adding labels for every missing tag.
            rows = []
            for source_row in source_rows:
                tag = source_row[0]
                translated = curated_by_tag.get(tag)
                if translated is None:
                    translated = [
                        tag,
                        source_row[1],
                        _machine_translate_tag(tag, source_row[1], machine_words, studio),
                    ]
                    machine_count += 1
                else:
                    # Keep the canonical tag/category, but apply the same SAA-safe
                    # separator cleanup to curated labels in this fully normalized file.
                    translated = [translated[0], translated[1], _sanitize_machine_label(translated[2])]
                rows.append(translated)
        else:
            rows = real_rows
        path = temporary / f"{group_id}.csv"
        count, size, digest = _write_csv(path, rows)
        groups.append(
            {
                "id": group_id,
                "file": path.name,
                "rows": count,
                "bytes": size,
                "sha256": digest,
                "source_group_rows": len(source_rows),
                "real_translated_rows": len(real_rows),
                "machine_translated_rows": machine_count,
                "untranslated_rows": len(source_rows) - len(real_rows),
            }
        )

    manifest = {
        "source_catalog": "prompt_catalog",
        "source_translation": translation_path.name,
        "schema": "tag,category,translation",
        "header": False,
        "selected_groups": list(SELECTED_GROUPS),
        "translated_rows": sum(group["rows"] for group in groups),
        "real_translated_rows": sum(group["real_translated_rows"] for group in groups),
        "machine_translated_rows": sum(group["machine_translated_rows"] for group in groups),
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
        "không BOM, xuống dòng LF. Mọi tag trong 12 nhóm được chọn đều có nhãn.",
        "Tag thiếu bản dịch thật nhận nhãn dịch máy dự phòng; dấu `_` được đổi thành",
        "khoảng trắng, từ chưa biết được giữ nguyên để không bịa nghĩa.",
        "",
        "| File | Tổng dòng | Dịch thật | Dịch máy | Tag nguồn | Chưa có bản dịch thật |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in manifest["groups"]:
        lines.append(
            f"| `{group['file']}` | {group['rows']:,} | {group['real_translated_rows']:,} | "
            f"{group['machine_translated_rows']:,} | {group['source_group_rows']:,} | "
            f"{group['untranslated_rows']:,} |"
        )
    lines += [
        "",
        f"Tổng: **{manifest['translated_rows']:,} dòng** — "
        f"{manifest['real_translated_rows']:,} dịch thật + "
        f"{manifest['machine_translated_rows']:,} dịch máy.",
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
