#!/usr/bin/env python3
"""Đo độ phủ và so hồi quy của bộ từ điển dịch thẻ tiếng Việt.

Hai lệnh (dùng mỗi đợt mở rộng từ điển trong `colab/studio.py`):

  python scripts/tag_vi_audit.py report
      Độ phủ theo nhóm chủ đề, chỉ tính các nhóm thẻ THẬT SỰ dịch được (loại danh mục
      tên họa sĩ/nhân vật/tác phẩm mà `_TAG_VI_NAME_CATEGORIES` đã loại khỏi tệp dịch),
      theo số thẻ và theo lượt dùng (heat). Kèm 10 thẻ nóng còn trống mỗi nhóm.

  python scripts/tag_vi_audit.py diff --baseline <đường-dẫn-studio-cũ> [--sample N] [--group TÊN]
      Chạy `vietnamese_tag_label` của bản cũ và bản mới trên TOÀN BỘ catalog rồi phân loại:
      MẤT (có bản dịch → mất), DỊCH MỚI, ĐỔI WORDING. Một đợt thêm từ phải có MẤT = 0 —
      đây là cách duy nhất phát hiện hồi quy âm thầm (thêm WORDS/HEADS làm quy tắc ghép
      chạy trước và đè nhãn đã curate).

Lệnh diff nhận baseline là TỆP .py (bản `colab/studio.py` trước khi sửa), không cần commit.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Danh mục có bản dịch thật (khớp bộ lọc của scripts/build_vietnamese_translate_file.py).
TRANSLATABLE = {"0", "5", "7", "12", "14"}


def load_studio(path: Path, name: str = "studio_under_audit"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def translated(module, name: str, category: str) -> bool:
    return module._is_translated_tag_label(module.vietnamese_tag_label(name, category), category)


def cmd_report(argv: argparse.Namespace) -> int:
    studio = load_studio(ROOT / "colab" / "studio.py")
    rows = studio.load_csv_tags()
    themes = [(name, re.compile(pattern)) for name, pattern in studio.TAG_THEMES.items()]
    stats: dict[str, list[int]] = {}
    seen: set[str] = set()
    for row in rows:
        name, category, heat = row[0], row[1], int(row[2] or 0)
        if category not in TRANSLATABLE:
            continue
        done = translated(studio, name, category)
        for theme, pattern in themes:
            if not pattern.search(name):
                continue
            key = (theme, name)
            if key in seen:
                continue
            seen.add(key)
            bucket = stats.setdefault(theme, [0, 0, 0, 0])
            bucket[0] += 1
            bucket[2] += heat
            if done:
                bucket[1] += 1
                bucket[3] += heat
    print("Độ phủ theo nhóm chủ đề (chỉ các danh mục dịch được):")
    print(f"  {'nhóm':26} {'thẻ đã dịch':>14} {'% thẻ':>8} {'% heat':>8} {'còn trống':>10}")
    for theme, (tot, done, heat, dheat) in sorted(stats.items(), key=lambda kv: -(kv[1][2] - kv[1][3])):
        print(f"  {theme:26} {done:7,}/{tot:<6,} {done/max(tot,1):7.1%} {dheat/max(heat,1):7.1%} {tot-done:10,}")
    gaps: list[tuple[int, str, str]] = []
    for row in rows:
        name, category = row[0], row[1]
        if category not in TRANSLATABLE or translated(studio, name, category):
            continue
        gaps.append((int(row[2] or 0), name, category))
    gaps.sort(reverse=True)
    print(f"\nToàn catalog (danh mục dịch được): còn {len(gaps):,} thẻ chưa dịch.\n")
    print("20 thẻ trống có lượt dùng cao nhất:")
    for heat, name, category in gaps[:20]:
        theme = next((t for t, p in themes if p.search(name)), "?")
        print(f"  {heat:9,}  [{theme}] {name}")
    return 0


def cmd_diff(argv: argparse.Namespace) -> int:
    old = load_studio(Path(argv.baseline), "studio_baseline")
    new = load_studio(ROOT / "colab" / "studio.py", "studio_current")
    rows = new.load_csv_tags()
    theme = re.compile(new.TAG_THEMES[argv.group]) if argv.group else None
    lost: list[tuple[str, str, str]] = []
    changed: list[tuple[str, str, str]] = []
    gained = 0
    for row in rows:
        name, category = row[0], row[1]
        if category not in TRANSLATABLE:
            continue
        if theme is not None and not theme.search(name):
            continue
        before = old.vietnamese_tag_label(name, category)
        after = new.vietnamese_tag_label(name, category)
        ok_old = old._is_translated_tag_label(before, category)
        ok_new = new._is_translated_tag_label(after, category)
        if ok_old and not ok_new:
            lost.append((name, before, after))
        elif not ok_old and ok_new:
            gained += 1
        elif ok_old and ok_new and before != after:
            changed.append((name, before, after))
    print(f"MẤT={len(lost)}  DỊCH MỚI={gained:,}  ĐỔI WORDING={len(changed):,}")
    if lost:
        print("\nHỒI QUY — phải sửa trước khi nộp:")
        for name, before, after in lost[:40]:
            print(f"  {name}\n    cũ:  {before}\n    mới: {after}")
    if changed:
        import random

        random.seed(argv.sample)
        print("\nĐỔI WORDING mẫu (chấp nhận được nếu nghĩa không tệ hơn):")
        for name, before, after in random.sample(changed, min(40, len(changed))):
            print(f"  {name:34} {before}   ||   {after}")
        tally = Counter(changed[i][0].rsplit("_", 1)[-1] for i in range(len(changed)))
        print("\nđuôi thẻ hay đổi nhất:", ", ".join(f"{k}×{v}" for k, v in tally.most_common(10)))
    return 1 if lost else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report").set_defaults(func=cmd_report)
    diff = sub.add_parser("diff")
    diff.add_argument("--baseline", required=True, help="đường dẫn colab/studio.py của bản trước")
    diff.add_argument("--sample", type=int, default=1, help="seed lấy mẫu đổi wording")
    diff.add_argument("--group", default=None, help="giới hạn vào một nhóm TAG_THEMES")
    diff.set_defaults(func=cmd_diff)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
