"""Tạo "file dịch" tiếng Việt cho autocomplete kiểu SAA.

Chạy từ thư mục gốc repo:
    python scripts/build_vietnamese_translate_file.py

Định bản sao chép đúng tệp tham chiếu tiếng Trung mà
mirabarukaso/character_select_stand_alone_app dùng cho Semi-Auto Tag Complete
(``data/danbooru_e621_merged_zh_cn.csv``):

* CSV không có dòng tiêu đề, một dòng cho mỗi thẻ;
* đúng ba trường ``tag,category,translation``;
* xuống dòng LF, UTF-8 không BOM (``--bom`` nếu muốn mở bằng Excel);
* trường dịch **không bao giờ chứa dấu phẩy**, vì bộ nạp JavaScript cắt mỗi dòng
  bằng ``line.split(',', 3)`` nên mọi thứ sau trường thứ ba sẽ bị bỏ.

Nhãn thật lấy từ cùng một bộ từ điển mà Studio dùng (``colab/studio.py``).
Họa sĩ, tác phẩm và nhân vật bị bỏ qua để không bịa tên riêng; mọi nhóm còn lại
được bổ sung nhãn dịch máy dự phòng khi thiếu bản dịch thật. Dấu gạch dưới trong
nhãn được đổi thành khoảng trắng.
"""

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colab.studio import (  # noqa: E402
    TAG_CATEGORIES,
    TAG_CSV_NAME,
    _is_translated_tag_label,
    parse_tag_csv,
)
from scripts.build_prompt_catalog_vi import (  # noqa: E402
    _machine_translate_tag,
    _machine_translation_tools,
)

TRANSLATE_CSV_NAME = "danbooru_e621_merged_vi_vn.csv"
# Không tự dịch tên riêng của họa sĩ, tác phẩm và nhân vật. Các nhóm còn lại
# (chung, loài, người đóng góp và lore) được bổ sung nhãn dịch máy khi thiếu.
TRANSLATE_SKIPPED_CATEGORIES = frozenset({"1", "3", "4", "8", "10", "11"})


def sanitize_translation(label):
    """Trả về nhãn tiếng Việt an toàn cho bộ nạp split theo dấu phẩy."""
    text = re.sub(r"[\r\n]+", " ", str(label or ""))
    text = text.replace(",", ";").replace('"', "'").replace("_", " ")
    words = re.sub(r"\s+", " ", text).strip().split()
    compact = []
    for word in words:
        if compact and word.casefold() == compact[-1].casefold():
            continue
        compact.append(word)
    return " ".join(compact)


def _has_curated_translation(name, category, label):
    if not _is_translated_tag_label(label, category):
        return False
    translation = sanitize_translation(label)
    # Match the historical builder's rule: a label is not useful only when it
    # is exactly the canonical spelling (including the tag's underscore form).
    return bool(translation) and translation.replace("_", " ").casefold() != name.casefold()


def translate_rows(rows, skip=TRANSLATE_SKIPPED_CATEGORIES):
    """Dựng các dòng dịch cho mọi nhóm không phải tên riêng bị bỏ qua.

    ``rows`` là kết quả của ``colab.studio.parse_tag_csv``: mỗi phần tử là
    ``(name, category, count, search_index, themes, label)`` theo thứ tự độ phổ biến
    giảm dần của CSV nguồn. Nhãn thật được giữ lại; tag còn thiếu nhận nhãn dịch máy
    dự phòng từ từ điển cục bộ, trong đó dấu gạch dưới được đổi thành khoảng trắng.
    """
    seen = set()
    result = []
    machine_seed = [[name, category, label] for name, category, _count, _index, _themes, label in rows]
    machine_words, machine_studio = _machine_translation_tools(machine_seed)
    for name, category, count, _search_index, _themes, label in rows:
        if category not in TAG_CATEGORIES or category in skip:
            continue
        if _has_curated_translation(name, category, label):
            translation = sanitize_translation(label)
        else:
            translation = _machine_translate_tag(name, category, machine_words, machine_studio)
            translation = sanitize_translation(translation)
        if translation:
            translation = translation[:1].upper() + translation[1:]
        if not translation:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append((name, category, translation, int(count)))
    return result


def render_translate_file(rows, bom=False):
    """Dựng nội dung tệp (bytes) từ các dòng đã lọc, LF, kết thúc bằng xuống dòng."""
    if not rows:
        return b""
    lines = [f"{name},{category},{translation}" for name, category, translation, _count in rows]
    text = "\n".join(lines) + "\n"
    return text.encode("utf-8-sig" if bom else "utf-8")


def write_bytes_atomic(destination, payload):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    return destination


def build_translate_file(
    input_path=ROOT / TAG_CSV_NAME,
    output_path=ROOT / TRANSLATE_CSV_NAME,
    bom=False,
    check=False,
):
    source = Path(input_path)
    if not source.is_file():
        raise SystemExit(f"Không thấy CSV nguồn: {source}")
    rows = parse_tag_csv(source.read_bytes().decode("utf-8-sig"))
    translated = translate_rows(rows)
    payload = render_translate_file(translated, bom=bom)
    destination = Path(output_path)

    machine_translated = sum(
        1
        for name, category, _count, _index, _themes, label in rows
        if category not in TRANSLATE_SKIPPED_CATEGORIES
        and not _has_curated_translation(name, category, label)
    )
    skipped = sum(1 for row in rows if row[1] in TRANSLATE_SKIPPED_CATEGORIES)
    stats = {
        "total": len(rows),
        "translated": len(translated),
        "machine_translated": machine_translated,
        "skipped": skipped,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "categories": _category_counts(translated),
        "output": destination,
    }

    if check:
        if not destination.is_file():
            raise SystemExit(f"--check: chưa có {destination}. Hãy chạy script không có --check.")
        current = destination.read_bytes()
        if current == payload:
            print(f"--check đạt: {destination} khớp nội dung dựng lại ({stats['translated']:,} dòng).")
            return stats
        print(
            f"--check lệch: {destination} có {len(current.splitlines()):,} dòng, "
            f"dựng lại được {stats['translated']:,} dòng. Hãy chạy lại script."
        )
        raise SystemExit(1)

    write_bytes_atomic(destination, payload)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    preview = ", ".join(f"{category}:{count:,}" for category, count in stats["categories"].items())
    print(
        f"Đã viết {stats['translated']:,} dòng dịch tiếng Việt "
        f"trên {stats['total']:,} thẻ của catalog "
        f"({stats['machine_translated']:,} nhãn dịch máy; "
        f"bỏ qua {stats['skipped']:,} thẻ tên riêng)."
    )
    print(f"Theo nhóm: {preview}")
    print(f"Tệp: {destination}\nKích thước: {stats['bytes']:,} byte\nSHA-256: {digest}")
    return stats


def _category_counts(rows):
    counts = {}
    for _name, category, _translation, _count in rows:
        counts[category] = counts.get(category, 0) + 1
    return {TAG_CATEGORIES[category]: count for category, count in sorted(counts.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tạo file dịch tiếng Việt định dạng SAA.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / TAG_CSV_NAME,
        help="CSV thẻ nguồn 4 hoặc 5 cột (mặc định: catalog trong thư mục gốc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / TRANSLATE_CSV_NAME,
        help=f"tệp đích (mặc định: {TRANSLATE_CSV_NAME} ở thư mục gốc)",
    )
    parser.add_argument("--bom", action="store_true", help="thêm BOM UTF-8 cho Excel/Windows")
    parser.add_argument(
        "--check",
        action="store_true",
        help="chỉ so sánh tệp hiện có với nội dung dựng lại, không ghi file",
    )
    args = parser.parse_args(argv)
    build_translate_file(
        input_path=args.input,
        output_path=args.output,
        bom=args.bom,
        check=args.check,
    )


if __name__ == "__main__":
    main()
