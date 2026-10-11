"""Split the prompt catalog into exclusive, prompt-oriented groups.

The source and every generated file keep the same five-column, headerless CSV
schema used by ``colab.studio.parse_tag_csv``. A row is written to exactly one
group; ``99_khac.csv`` is the lossless fallback.

Run from the repository root:
    python scripts/split_prompt_catalog.py

To choose another destination:
    python scripts/split_prompt_catalog.py --output-dir /tmp/prompt_catalog
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"
DEFAULT_OUTPUT = ROOT / "prompt_catalog"
EXPECTED_COLUMNS = 5

# Category-specific groups are checked before lexical groups. This prevents an
# artist named with a clothing word from leaving the artist catalog.
GROUPS = OrderedDict(
    [
        ("01_chu_the", "Chủ thể / số lượng nhân vật"),
        ("02_hoa_si", "Họa sĩ"),
        ("03_tac_pham_nhan_vat", "Tác phẩm / nhân vật"),
        ("04_loai_lore", "Loài / lore"),
        ("05_trang_phuc_quan_ao", "Trang phục / quần áo"),
        ("06_phu_kien", "Phụ kiện"),
        ("07_ngoai_hinh", "Ngoại hình / bộ phận cơ thể"),
        ("08_tu_the_bieu_cam", "Tư thế / hành động / biểu cảm"),
        ("09_boi_canh", "Bối cảnh / thiên nhiên"),
        ("10_phong_cach", "Phong cách / chất liệu / loại hình"),
        ("11_anh_sang_mau_sac", "Ánh sáng / màu sắc"),
        ("12_bo_cuc_ky_thuat", "Bố cục / kỹ thuật / chất lượng"),
        ("13_vat_the", "Vật thể / đồ vật"),
        ("99_khac", "Khác"),
    ]
)

# Patterns are intentionally broad and are applied to the canonical name plus
# aliases. Order is the documented precedence, not a quality score.
_PATTERNS = OrderedDict(
    [
        (
            "01_chu_the",
            r"(?:^|\s)(?:\d+girls?|\d+boys?|solo|multiple_girls|multiple_boys|couple|duo|group|no_humans|female|male|male/female|female/male|female_focus|male_focus|furry|anthro|wolf|dragon|rabbit|canine|feline|pokemon)(?:$|\s)",
        ),
        (
            "05_trang_phuc_quan_ao",
            r"(?:^|\s)(?:clothing|clothes|dress|shirt|blouse|skirt|pants?|trousers|jeans|shorts|underwear|pant(?:y|ies)|bra|bikini|swimsuit|uniform|robe|kimono|yukata|coat|jacket|hoodie|sweater|cardigan|vest|bodysuit|leotard|apron|cloak|cape|armor|boots?|shoes?|socks?|stockings?|tights?|sneakers?|heels?|sandals?|footwear|garter|gloves?|sleeves?|thighhighs?|pantyhose|outfit|costume|wearing)(?:$|\s)",
        ),
        (
            "06_phu_kien",
            r"(?:^|\s)(?:accessory|accessories|hat|cap|beanie|beret|glasses|eyewear|sunglasses|goggles|earrings?|necklace|jewelry|jewellery|bracelet|ring|choker|ribbon|bow|hairband|headwear|hair_ornament|mask|bag|backpack|purse|handbag|umbrella|fan|brooch|tie|necktie|bowtie|scarf|belt|piercing)(?:$|\s)",
        ),
        (
            "07_ngoai_hinh",
            r"(?:^|\s)(?:hair|eyes?|eyebrows?|eyelashes|face|skin|fur|ears?|wings?|tails?|horns?|mouth|nose|lips?|teeth|tongue|breasts?|chest|cleavage|body|torso|shoulders?|arms?|hands?|fingers?|legs?|feet|foot|toes?|muscles?|belly|navel|nipples?|penis|vagina|anus|genitals?|anatomy|nails?|pupils?|sleeves?|ponytails?|twintails?|braids?|bangs|sidelocks|ahoge|tufts?|nude|naked)(?:$|\s)",
        ),
        (
            "08_tu_the_bieu_cam",
            r"(?:^|\s)(?:smil\w*|blush\w*|cry\w*|tear\w*|laugh\w*|angr\w*|surpris\w*|express\w*|look\w*|stand\w*|sitt\w*|ly\w*|walk\w*|runn\w*|jump\w*|pose\w*|gesture\w*|facin\w*|lean\w*|bend\w*|kneel\w*|danc\w*|wav\w*|hold\w*|rid\w*|fight\w*|hugg\w*|kiss\w*|dynamic\w*|arms_up|hand_on|crossed_arms)(?:$|\s)",
        ),
        (
            "09_boi_canh",
            r"(?:^|\s)(?:background|scenery|landscape|sky|clouds?|forest|trees?|flowers?|plants?|grass|ocean|sea|beach|water|river|lake|mountains?|hills?|city|street|road|indoors?|outdoors?|room|bedroom|classroom|school|building|house|castle|garden|park|shrine|temple|church|shop|cafe|restaurant|office|train_station|station|night_sky)(?:$|\s)",
        ),
        (
            "10_phong_cach",
            r"(?:^|\s)(?:anime|manga|illustration|art|artwork|artstyle|style|sketch|lineart|line_art|painting|painted|watercolor|watercolour|oil_paint|acrylic|cel_shading|cel_shaded|flat_color|flat_colour|monochrome|comic|chibi|realistic|realism|3d|digital|traditional|pixel_art|render|key_visual|screencap|screenshot|photorealistic|concept_art)(?:$|\s)",
        ),
        (
            "11_anh_sang_mau_sac",
            r"(?:^|\s)(?:light|lighting|shadow|shadows|glow|glowing|neon|sunlight|sunset|sunrise|moonlight|night|day|dark|bright|backlighting|backlight|rim_light|spotlight|firelight|candlelight|rainbow|gradient|palette|colored|colour|color|red|orange|yellow|green|blue|purple|pink|black|white|gray|grey)(?:$|\s)",
        ),
        (
            "12_bo_cuc_ky_thuat",
            r"(?:^|\s)(?:portrait|close.?up|full.?body|upper.?body|bust|cowboy_shot|wide_shot|view|angle|perspective|depth_of_field|focus|bokeh|composition|resolution|highres|absurdres|lowres|jpeg_artifacts|panorama|aspect_ratio|widescreen|text|watermark|signature|logo|camera|lens|fisheye|motion_blur)(?:$|\s)",
        ),
        (
            "13_vat_the",
            r"(?:^|\s)(?:weapon|sword|gun|rifle|pistol|knife|food|fruit|vegetable|furniture|chair|table|desk|bed|pillow|phone|smartphone|book|computer|laptop|car|vehicle|motorcycle|bicycle|instrument|piano|guitar|ball|toy|object|machine|robot)(?:$|\s)",
        ),
    ]
)

# These groups must win over broad words such as ``black``, ``water`` or
# ``colored`` that can appear in aliases of a clothing/object/technical tag.
_PRIORITY_GROUPS = (
    "05_trang_phuc_quan_ao",
    "06_phu_kien",
    "07_ngoai_hinh",
    "08_tu_the_bieu_cam",
    "12_bo_cuc_ky_thuat",
    "13_vat_the",
)


def _normalized(value: str) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"[_-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _pattern_text(row: list[str]) -> str:
    # Keep both source spelling (underscores) and normalized wording (spaces):
    # regexes can match canonical names and aliases without losing either form.
    values = tuple(str(value or "").casefold() for value in (row[0], row[3], row[4]))
    raw = " ".join(values)
    normalized = " ".join(_normalized(value) for value in values)
    return f"{raw} {normalized}"


def classify_row(row: list[str]) -> str:
    """Return one group id for a parsed five-column catalog row."""
    category = row[1]
    if category in {"1", "8"}:
        return "02_hoa_si"
    if category in {"3", "4", "10", "11"}:
        return "03_tac_pham_nhan_vat"
    if category in {"12", "15"}:
        return "04_loai_lore"
    text = _pattern_text(row)
    for group_id in _PRIORITY_GROUPS:
        if re.search(_PATTERNS[group_id], text, re.IGNORECASE):
            return group_id
    for group_id, expression in _PATTERNS.items():
        if group_id in _PRIORITY_GROUPS:
            continue
        if re.search(expression, text, re.IGNORECASE):
            return group_id
    return "99_khac"


def _write_row(writer: csv.writer, row: list[str]) -> None:
    if len(row) != EXPECTED_COLUMNS:
        raise ValueError(f"Catalog row phải có {EXPECTED_COLUMNS} cột, nhận {len(row)}")
    writer.writerow(row)


def split_catalog(input_path: Path, output_dir: Path) -> dict:
    """Split ``input_path`` into group CSVs and return a manifest dictionary."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    handles: dict[str, TextIO] = {}
    writers: dict[str, csv.writer] = {}
    counts = {group_id: 0 for group_id in GROUPS}
    try:
        for group_id in GROUPS:
            path = temporary / f"{group_id}.csv"
            handle = path.open("w", encoding="utf-8", newline="")
            handles[group_id] = handle
            writers[group_id] = csv.writer(handle, lineterminator="\n")
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.reader(source):
                if len(row) != EXPECTED_COLUMNS or not row[0].strip():
                    continue
                group_id = classify_row(row)
                _write_row(writers[group_id], row)
                counts[group_id] += 1
    finally:
        for handle in handles.values():
            handle.close()

    groups = []
    for group_id, label in GROUPS.items():
        path = temporary / f"{group_id}.csv"
        payload = path.read_bytes()
        groups.append(
            {
                "id": group_id,
                "label": label,
                "file": path.name,
                "rows": counts[group_id],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    total = sum(counts.values())
    with input_path.open("r", encoding="utf-8-sig") as source:
        source_rows = sum(1 for _ in source)
    manifest = {
        "source": input_path.name,
        "schema": "name,category,count,aliases,vietnamese_label",
        "exclusive": True,
        "classification": "category first, then canonical name/alias/label regex; fallback 99_khac",
        "source_rows": source_rows,
        "group_rows": total,
        "groups": groups,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = _readme(manifest)
    (temporary / "README.md").write_text(readme, encoding="utf-8")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.rename(output_dir)
    return manifest


def _readme(manifest: dict) -> str:
    lines = [
        "# Catalog prompt đã chia nhóm",
        "",
        f"Nguồn: `{manifest['source']}` · **{manifest['source_rows']:,} dòng**.",
        "Các file con dùng cùng schema 5 cột, không có header. Phân nhóm là **độc quyền**:",
        "mỗi tag xuất hiện đúng một lần; tag chưa khớp quy tắc nằm trong `99_khac.csv`.",
        "",
        "| File | Nhóm | Số dòng | Kích thước |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in manifest["groups"]:
        lines.append(
            f"| `{group['file']}` | {group['label']} | {group['rows']:,} | {group['bytes']:,} byte |"
        )
    lines += [
        "",
        "Tạo lại bằng:",
        "```bash",
        "python scripts/split_prompt_catalog.py",
        "```",
        "",
        "Quy tắc ưu tiên: danh mục họa sĩ/tác phẩm/nhân vật/loài trước; sau đó là",
        "trang phục → phụ kiện → ngoại hình → tư thế → bối cảnh → phong cách →",
        "ánh sáng/màu → bố cục/kỹ thuật → vật thể → khác.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = split_catalog(args.input, args.output_dir)
    print(
        f"Đã chia {manifest['group_rows']:,} dòng vào "
        f"{len(manifest['groups'])} nhóm trong `{args.output_dir}`."
    )
    for group in manifest["groups"]:
        print(f"  {group['file']}: {group['rows']:,} dòng")


if __name__ == "__main__":
    main()
