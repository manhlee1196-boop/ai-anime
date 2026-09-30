"""Personal WAI-illustrious Colab studio. Inlined into the standalone notebook.

The notebook's existing setup cells verify checkpoint and LoRA hashes before this
module is used. The model stays in Colab: Gradio provides a temporary public URL.
"""

import gc
import json
import math
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

SIZE_PRESETS = (
    "2688x2048",
    "768x768",
    "768x1024",
    "1024x768",
    "832x1216",
    "1216x832",
    "1024x1024",
    "1024x1344",
    "1344x1024",
)
DEFAULT_PROMPT = (
    "1girl, solo, cherry blossoms, spring, soft sunlight, "
    "detailed eyes, detailed clothing, masterpiece, best quality"
)
# Negative mặc định nhắm lỗi ngón tay/ngón chân; người dùng tự sửa theo ý mình.
DEFAULT_NEGATIVE = (
    "lowres, worst quality, low quality, blurry, bad anatomy, "
    "bad hands, deformed hands, extra fingers, missing fingers, fused fingers, "
    "malformed fingers, deformed feet, extra toes, missing toes, fused toes, "
    "malformed toes, extra limbs"
)
# Prompt do người dùng tự viết hoặc nạp từ thư viện; không lọc từ khóa nội dung.
CONTENT_ROOT = Path("/content")
# Thư viện prompt mẫu đóng kèm, dùng đúng định dạng file .txt mà UI đọc được.
# Người dùng có thể nạp file riêng của mình thay thế.
SAMPLE_PROMPT_LIBRARY = """=== 12 PROMPTS – NHÂN VẬT VĂN PHÒNG & THÀNH PHỐ (BẢN MẪU) ===

Tên tiếng Việt | Nội dung prompts tiếng Anh

Nhân vật công sở hiện đại, trang phục lịch sự kín đáo, ánh sáng trong trẻo, chi tiết cao

PROMPT 01 - Nữ nhân viên ngồi bàn làm việc, mỉm cười dịu
1girl, solo, office worker sitting at a tidy desk, crisp white shirt fully buttoned, navy blazer, knee-length pencil skirt, glasses, long black hair loosely tied, gentle warm smile, warm desk lamp light, modern office at night, anime illustration, detailed eyes, detailed clothing, masterpiece, best quality

PROMPT 02 - Nữ nhân viên cúi xem tài liệu rồi ngẩng lên
1girl, solo, office worker leaning over a desk to check documents, light blue shirt buttoned to the collar, long charcoal skirt, long brown hair in a neat bun, shy soft smile, soft window light, papers and coffee cup on the desk, anime illustration, detailed eyes, masterpiece, best quality

PROMPT 03 - Nam nhân viên đứng bên cửa sổ thành phố
1boy, solo, calm office worker standing by a large office window, white shirt with sleeves rolled up, dark tie, tailored trousers, short black hair, thoughtful expression, city night view, bokeh lights, soft mixed lighting, anime illustration, masterpiece, best quality
Negative: lowres, worst quality, blurry, bad anatomy, bad hands, extra fingers, deformed hands

PROMPT 04 - Họp nhóm trong phòng kính
multiple girls and boys, small team of office workers around a glass meeting table, whiteboard with charts, laptops and notebooks, daylight through tall windows, friendly expressions, clean modern interior, anime illustration, masterpiece, best quality
Steps: 28
Size: 1216x832

PROMPT 05 - Giờ nghỉ ở quán cà phê tầng trệt
1girl, solo, young office worker holding a paper coffee cup in a bright café, beige cardigan over a white shirt, pleated skirt, short wavy hair, relaxed smile, wooden interior with plants, soft morning light through glass, anime illustration, detailed eyes, masterpiece, best quality

PROMPT 06 - Đi bộ trong hành lang công ty
1girl, solo, office worker walking through a bright corporate corridor holding a folder, gray suit, short hair, determined expression, glossy floor reflections, sunbeams from side windows, dynamic composition, anime illustration, masterpiece, best quality

PROMPT 07 - Chờ thang máy lúc tan tầm
1girl, solo, office worker waiting for the elevator with a tote bag, long coat over a turtleneck, long straight hair, quiet tired smile, warm lobby lighting, marble walls, shallow depth of field, anime illustration, masterpiece, best quality

PROMPT 08 - Làm việc muộn bên đèn bàn
1girl, solo, office worker typing at a laptop late at night, glasses reflecting the screen, messy bun, oversized knit cardigan, focused expression, warm lamp glow, dark office, rain streaks on the window, cinematic lighting, anime illustration, masterpiece, best quality
Negative: lowres, worst quality, blurry, bad anatomy, bad hands, extra fingers, text, watermark
CFG: 7

PROMPT 09 - Ô trong suốt dưới mưa phố đêm
1girl, solo, office worker walking home with a transparent umbrella, trench coat and scarf, neon signs reflected in wet asphalt, light drizzle, calm expression, moody indigo palette, anime illustration, masterpiece, best quality

PROMPT 10 - Trên sân thượng giờ hoàng hôn
1girl, solo, office worker standing on a rooftop at sunset, blazer draped over one arm, hair moving in the wind, orange and violet sky above the skyline, expansive detailed background, hopeful atmosphere, anime illustration, masterpiece, best quality

PROMPT 11 - Thư viện và chồng sách cao
1girl, solo, young researcher between tall library shelves holding a stack of books, white blouse with a knitted vest, long braided hair, curious soft expression, dust particles in warm shafts of light, detailed interior, anime illustration, masterpiece, best quality

PROMPT 12 - Chuyến tàu cuối ngày
1girl, solo, office worker sitting by a train window at dusk, headphones on, soft sweater, long hair, gazing at passing city lights, reflections on the glass, quiet contemplative mood, cinematic anime film still, masterpiece, best quality
Seed: 20260928
"""
REPAIR_HINTS = {
    "hands": (
        "natural hands, anatomically correct fingers, detailed fingers",
        "extra fingers, missing fingers, fused fingers, malformed fingers, deformed hands",
    ),
    "legs": (
        "natural leg anatomy, well-formed feet, natural toes, balanced pose",
        "extra legs, broken legs, deformed feet, extra toes, missing toes, fused toes",
    ),
    "eyes": (
        "symmetrical eyes, detailed irises, perfect eyes",
        "misaligned eyes, deformed eyes, extra eyes",
    ),
    "custom": ("", ""),
}


def _add_prompt_tags(text, tags):
    """Append visible preset/repair tags once to an editable prompt."""
    seen = {part.strip().casefold() for part in text.split(",")}
    for tag in tags:
        tag = tag.strip()
        if tag and tag.casefold() not in seen:
            text = f"{text}, {tag}" if text else tag
            seen.add(tag.casefold())
    return text


def add_eyes_trigger(prompt, negative):
    """Explicit UI action: add the eye LoRA trigger once, visible and editable."""
    return _add_prompt_tags(prompt, ("perfect eyes",)), negative


def apply_repair_hints(positive, negative, target):
    """Explicit UI action: append suggestions in the editable fields before inpainting."""
    if target not in REPAIR_HINTS:
        raise ValueError("Chọn vùng sửa: tay, chân, mắt hoặc tùy chỉnh.")
    hint_pos, hint_neg = REPAIR_HINTS[target]
    return (
        _add_prompt_tags(positive, hint_pos.split(",")),
        _add_prompt_tags(negative, hint_neg.split(",")),
    )


# ---------------------------------------------------------------------------
# Thư viện prompt: nạp danh sách prompt từ file text rồi chọn một dòng để nạp.
#
#   === 50 PROMPTS – CHỦ ĐỀ ===
#   Tên tiếng Việt | Nội dung prompts tiếng Anh
#
#   PROMPT 01 - Tên tiếng Việt của prompt
#   Masterpiece, best quality, ultra-detailed anime style, ...
#   Negative: lowres, bad hands
#   Steps: 26
#
# Ngoài định dạng trên còn đọc được: bảng một dòng "Tên | Prompt", JSON
# [{"title": ..., "prompt": ...}], hoặc các đoạn prompt cách nhau dòng trống.
# Giới hạn thấp hơn mức runtime chấp nhận (2200/1700) để preset phong cách
# còn chỗ thêm thẻ mà không vượt ngưỡng khi gửi model.
# ---------------------------------------------------------------------------
PROMPT_LIBRARY_LIMIT = 2000
PROMPT_LIBRARY_NEGATIVE_LIMIT = 1500
PROMPT_LIBRARY_MAX_BYTES = 2_000_000

_LIBRARY_BANNER = re.compile(r"^\s*(?:={3,}|\*{3,})\s*(.+?)\s*(?:={3,}|\*{3,})\s*$")
_LIBRARY_MD_TITLE = re.compile(r"^\s*#\s+(.+?)\s*$")
# Bắt buộc có số thứ tự để dòng "Prompt: Masterpiece, ..." không bị coi là tiêu đề.
_LIBRARY_PROMPT_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?\**\s*prompt\s*\.?\s*(\d{1,3})\s*\**\s*[-–—:.)\]]*\s*(.*?)\s*$",
    re.IGNORECASE,
)
_LIBRARY_NUMBER_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?\[?(\d{1,3})\]?\s*[-–—.:/)]\s+(.+?)\s*$"
)
_LIBRARY_PIPE_ROW = re.compile(r"^\s*([^|\n]{1,90}?)\s*\|\s*([^|\n]{15,})\s*$")
_LIBRARY_TABLE_HEADER = re.compile(
    r"tên\s+tiếng\s+việt|nội\s+dung\s+prompt|prompt\s+tiếng\s+anh"
    r"|vietnamese\s+(?:name|title)|prompt\s*\(\s*en\w*\s*\)",
    re.IGNORECASE,
)
_LIBRARY_RULE = re.compile(r"^\s*(?:-{3,}|_{3,})\s*$")
_LIBRARY_BULLET = re.compile(r"^[-*•·]\s+")
_LIBRARY_PROMPT_PREFIX = re.compile(r"^\s*prompt\s*[:\-–—]\s*", re.IGNORECASE)
_LIBRARY_DIRECTIVES = (
    (
        "negative",
        re.compile(
            r"^\s*(?:negative(?:\s*prompt)?|loại\s+trừ|prompt\s+trừ)\s*[:\-–—]\s*(.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "steps",
        re.compile(
            r"^\s*(?:steps?|bước|số\s+bước)\s*[:\-–—]\s*(\d{1,3})\s*$", re.IGNORECASE
        ),
    ),
    (
        "cfg",
        re.compile(
            r"^\s*(?:cfg(?:\s*scale)?|guidance)\s*[:\-–—]\s*(\d{1,2}(?:[.,]\d)?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "size",
        re.compile(
            r"^\s*(?:size|kích\s+thước|resolution)\s*[:\-–—]\s*(\d{3,4})\s*[x×*]\s*(\d{3,4})\s*$",
            re.IGNORECASE,
        ),
    ),
    ("seed", re.compile(r"^\s*seed\s*[:\-–—]\s*(-?\d{1,12})\s*$", re.IGNORECASE)),
)


def _library_title(value):
    text = str(value or "").replace("*", "").replace("`", " ")
    text = _LIBRARY_BULLET.sub("", text)
    text = _LIBRARY_PROMPT_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()[:140]


def _library_body(lines):
    """Join one prompt body and extract the per-prompt parameters below it."""
    extras = {}
    parts = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        for key, pattern in _LIBRARY_DIRECTIVES:
            match = pattern.match(line)
            if not match:
                continue
            if key == "negative":
                extras["negative"] = match.group(1).strip()[
                    :PROMPT_LIBRARY_NEGATIVE_LIMIT
                ]
            elif key == "steps":
                extras["steps"] = min(45, max(10, int(match.group(1))))
            elif key == "cfg":
                extras["cfg"] = min(
                    12.0, max(1.0, float(match.group(1).replace(",", ".")))
                )
            elif key == "size":
                size = f"{int(match.group(1))}x{int(match.group(2))}"
                if size in SIZE_PRESETS:
                    extras["size"] = size
            elif key == "seed":
                value = int(match.group(1))
                if value == -1 or 0 <= value <= 2**32 - 1:
                    extras["seed"] = value
            break
        else:
            parts.append(_LIBRARY_BULLET.sub("", line).replace("`", "").strip("* "))
    prompt = _LIBRARY_PROMPT_PREFIX.sub("", re.sub(r"\s+", " ", " ".join(parts)))
    return prompt.strip(), extras


def _library_items(entries):
    items = []
    for title, body in entries:
        prompt, extras = _library_body(body)
        if not prompt:
            continue
        truncated = len(prompt) > PROMPT_LIBRARY_LIMIT
        if truncated:
            prompt = prompt[:PROMPT_LIBRARY_LIMIT]
        label = _library_title(title) or (
            f"{prompt[:58]}…" if len(prompt) > 58 else prompt
        )
        items.append(
            {
                "index": len(items) + 1,
                "label": f"{len(items) + 1:02d} · {label}",
                "title": label,
                "prompt": prompt,
                "negative": extras.get("negative", ""),
                "steps": extras.get("steps"),
                "cfg": extras.get("cfg"),
                "size": extras.get("size"),
                "seed": extras.get("seed"),
                "truncated": truncated,
            }
        )
    return tuple(items)


def _library_from_json(payload, name):
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    entries = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or row.get("name") or row.get("vi") or ""
        prompt = row.get("prompt") or row.get("en") or row.get("content") or ""
        entries.append((title, [str(prompt)]))
    items = _library_items(entries)
    return {"name": name, "description": "", "items": items} if items else None


def parse_prompt_library(text, name="Thư viện prompt"):
    """Parse a prompt-list file into {"name", "description", "items"}."""
    raw = str(text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        raise ValueError("File trống, không có prompt nào.")
    stripped = raw.strip()
    if stripped[:1] in "[{":
        try:
            parsed = _library_from_json(json.loads(stripped), name)
        except json.JSONDecodeError:
            parsed = None
        if parsed:
            return parsed

    lines = raw.split("\n")
    table_mode = (
        sum(
            1
            for line in lines
            if _LIBRARY_PIPE_ROW.match(line.strip())
            and not _LIBRARY_TABLE_HEADER.search(line)
        )
        >= 3
    )
    title = ""
    description = ""
    entries = []
    current = None
    expected = 1
    orphan = []

    def flush():
        nonlocal current
        if current is not None:
            entries.append(current)
            current = None

    for line in lines:
        line = line.rstrip()
        trimmed = line.strip()
        if not trimmed or _LIBRARY_RULE.match(trimmed):
            if current is not None:
                current[1].append("")
            continue
        banner = _LIBRARY_BANNER.match(trimmed)
        if banner and not entries and current is None and not title:
            title = _library_title(banner.group(1))
            continue
        if not title and not entries and current is None:
            heading = _LIBRARY_MD_TITLE.match(trimmed)
            if heading:
                title = _library_title(heading.group(1))
                continue
        if _LIBRARY_TABLE_HEADER.search(trimmed):
            continue
        numbered_prompt = _LIBRARY_PROMPT_HEADING.match(trimmed)
        if numbered_prompt:
            flush()
            expected = int(numbered_prompt.group(1)) + 1
            current = (numbered_prompt.group(2) or "", [])
            continue
        numbered = _LIBRARY_NUMBER_HEADING.match(trimmed)
        # Chỉ nhận "01 - Tên" khi số thứ tự khớp, tránh nhầm với prompt bắt đầu
        # bằng con số (ví dụ "3.5 mm lens").
        if numbered and int(numbered.group(1)) == expected:
            flush()
            expected += 1
            current = (numbered.group(2), [])
            continue
        row = _LIBRARY_PIPE_ROW.match(trimmed)
        if table_mode and row:
            flush()
            entries.append((row.group(1), [row.group(2)]))
            continue
        if current is not None:
            current[1].append(trimmed)
        elif not description and len(trimmed) > 12:
            orphan.append(trimmed)
    flush()
    description = " ".join(orphan)[:300]

    if not entries:
        # Không có tiêu đề nào: mỗi đoạn văn (cách nhau dòng trống) là một prompt.
        for block in re.split(r"\n\s*\n", raw):
            block = block.strip()
            if (
                not block
                or _LIBRARY_BANNER.match(block)
                or _LIBRARY_TABLE_HEADER.search(block)
            ):
                continue
            rows = [part.strip() for part in block.split("\n")]
            entries.append((_library_title(rows[0])[:70], rows))

    items = _library_items(entries)
    if not items:
        raise ValueError(
            "Không tìm thấy prompt nào. Mỗi prompt cần một dòng tiêu đề dạng "
            "“PROMPT 01 - Tên tiếng Việt” rồi đoạn prompt bên dưới, hoặc một "
            "dòng “Tên tiếng Việt | Nội dung prompts tiếng Anh”."
        )
    return {"name": title or name, "description": description, "items": items}


def read_prompt_library(path):
    """Parse an uploaded prompt list; Gradio hands over a temporary file path."""
    if not path:
        raise ValueError("Hãy chọn file .txt chứa danh sách prompt.")
    file = Path(path)
    if not file.is_file():
        raise ValueError("Không mở được file vừa tải lên.")
    if file.stat().st_size > PROMPT_LIBRARY_MAX_BYTES:
        raise ValueError("File prompt quá lớn. Giới hạn 2 MB.")
    return parse_prompt_library(
        file.read_text(encoding="utf-8", errors="replace"), name=file.name
    )


def load_prompt_library_text(text):
    """Parse a prompt list pasted into the textbox instead of uploaded."""
    if not str(text or "").strip():
        raise ValueError("Hãy dán nội dung file prompt vào ô trước khi đọc.")
    return parse_prompt_library(text, name="Danh sách đã dán")


def load_sample_prompt_library():
    return parse_prompt_library(SAMPLE_PROMPT_LIBRARY, name="Thư viện mẫu")


def prompt_library_dropdown(items):
    """Dropdown properties listing every prompt of the loaded library."""
    labels = tuple(item["label"] for item in items or ())
    return {"choices": labels, "value": None, "interactive": bool(labels)}


def prompt_library_status(library):
    items = library["items"]
    trimmed = sum(1 for item in items if item["truncated"])
    text = f"**{library['name']}** · {len(items)} prompt đã nạp. Chọn một dòng để nạp prompt."
    if library.get("description"):
        text += f"\n\n_{library['description']}_"
    if trimmed:
        text += f"\n\n{trimmed} prompt dài hơn {PROMPT_LIBRARY_LIMIT} ký tự nên đã bị cắt bớt."
    return text


def prompt_library_reset():
    return (
        (),
        {"choices": (), "value": None, "interactive": False},
        "Đã gỡ thư viện prompt.",
    )


def apply_prompt_choice(
    choice, items, prompt, negative, steps, cfg, seed, text_size, image_size
):
    """Fill the editable prompt fields (and listed parameters) from one entry."""
    items = tuple(items or ())
    selected = next((item for item in items if item["label"] == choice), None)
    if selected is None:
        return (
            prompt,
            negative,
            steps,
            cfg,
            seed,
            text_size,
            image_size,
            "Chưa nạp thư viện hoặc chưa chọn prompt nào trong danh sách.",
        )
    negative = selected["negative"] or negative
    steps = selected["steps"] if selected["steps"] is not None else steps
    cfg = selected["cfg"] if selected["cfg"] is not None else cfg
    seed = selected["seed"] if selected["seed"] is not None else seed
    sizes = (text_size, image_size)
    if selected["size"]:
        sizes = (selected["size"], selected["size"])
    applied = []
    if selected["negative"]:
        applied.append("Negative")
    if selected["steps"] is not None:
        applied.append(f"Steps {selected['steps']}")
    if selected["cfg"] is not None:
        applied.append(f"CFG {selected['cfg']:g}")
    if selected["size"]:
        applied.append(f"Kích thước {selected['size']}")
    if selected["seed"] is not None:
        applied.append(f"Seed {selected['seed']}")
    status = f"Đã nạp **{selected['label']}** vào ô *Prompt gửi model*."
    if applied:
        status += " Đã áp dụng: " + ", ".join(applied) + "."
    if selected["truncated"]:
        status += f" Prompt dài hơn {PROMPT_LIBRARY_LIMIT} ký tự nên đã bị cắt bớt."
    return (
        selected["prompt"],
        negative,
        steps,
        cfg,
        seed,
        sizes[0],
        sizes[1],
        status,
    )


def _number(value, name, low, high, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} phải là số từ {low} đến {high}.")
    if value < low or value > high or (integer and int(value) != value):
        raise ValueError(f"{name} phải là số từ {low} đến {high}.")
    return int(value) if integer else float(value)


def _preset_size(size):
    if size not in SIZE_PRESETS:
        raise ValueError("Chọn kích thước có sẵn trong danh sách.")
    return tuple(map(int, size.split("x")))


def _normalize_source(image, mask=None):
    """Resize source and painted mask together, keeping their pixel alignment."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise ValueError("Hãy tải ảnh nguồn PNG/JPG/WebP lên trước.")
    w, h = image.size
    if min(w, h) < 1 or w * h > 20_000_000 or max(w / h, h / w) > 1.75:
        raise ValueError(
            "Ảnh nguồn quá lớn hoặc quá dài (tối đa 20 MP và tỷ lệ 1,75:1). Hãy cắt ảnh trước."
        )
    if mask is not None and mask.size != image.size:
        raise ValueError(
            "Mask phải trùng chính xác kích thước ảnh nguồn trước khi thu nhỏ."
        )
    scale = min(1024 / max(w, h), max(1, 512 / min(w, h)))
    width = max(512, round(w * scale / 8) * 8)
    height = max(512, round(h * scale / 8) * 8)
    if width / height > 1.75:
        height = math.ceil(width / 1.75 / 8) * 8
    if height / width > 1.75:
        width = math.ceil(height / 1.75 / 8) * 8
    source = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    if mask is None:
        return source, None
    return source, mask.convert("L").resize((width, height), Image.Resampling.NEAREST)


def _editor_mask(editor, uploaded_mask):
    """Editor background is source; transparent painted layers mark the repair area."""
    from PIL import Image, ImageChops

    if not isinstance(editor, dict) or not isinstance(
        editor.get("background"), Image.Image
    ):
        raise ValueError("Tải ảnh cần sửa vào khung vẽ trước.")
    original = editor["background"]
    if uploaded_mask is not None:
        if (
            not isinstance(uploaded_mask, Image.Image)
            or uploaded_mask.size != original.size
        ):
            raise ValueError(
                "Mask PNG tải lên phải cùng kích thước ảnh nguồn. Trắng = sửa, đen = giữ."
            )
        mask = uploaded_mask.convert("L")
    else:
        mask = Image.new("L", original.size, 0)
        for layer in editor.get("layers") or []:
            if not isinstance(layer, Image.Image) or layer.size != original.size:
                raise ValueError(
                    "Lớp tô không trùng kích thước ảnh nguồn. Hãy xóa lớp rồi vẽ lại."
                )
            # The editor's unpainted canvas is transparent. An opaque RGB layer
            # would select the entire image, so require a real alpha channel.
            if "A" not in layer.getbands():
                raise ValueError(
                    "Lớp tô thiếu kênh trong suốt. Dùng cọ trực tiếp trên ảnh, hoặc tải mask PNG."
                )
            mask = ImageChops.lighter(mask, layer.getchannel("A"))
    binary = mask.point(lambda value: 255 if value >= 64 else 0)
    if binary.getbbox() is None or binary.getextrema() == (255, 255):
        raise ValueError(
            "Hãy tô một vùng nhỏ để sửa (không để mask rỗng hoặc trắng toàn bộ)."
        )
    return _normalize_source(original, binary)


class StudioRuntime:
    def __init__(
        self,
        *,
        torch,
        pipe,
        create_pipeline,
        checkpoint,
        lora_paths,
        lora_manifest,
        vram_mode,
        use_offload,
        output_dir,
        backup_dir="/content/wai_outputs",
    ):
        if pipe is None or not Path(checkpoint).is_file():
            raise RuntimeError(
                "Model chưa sẵn sàng; chạy lại các ô chuẩn bị checkpoint và nạp model."
            )
        root = CONTENT_ROOT.resolve()
        for label, directory in (
            ("OUTPUT_DIR", output_dir),
            ("backup_dir", backup_dir),
        ):
            path = Path(directory).absolute()
            resolved = path.resolve()
            if (
                not resolved.is_relative_to(root)
                or resolved == root
                or path.is_relative_to(CONTENT_ROOT / "drive")
                or resolved.is_relative_to(root / "drive")
            ):
                raise ValueError(f"{label} phải nằm dưới /content, không dùng Drive.")
        self.torch = torch
        self.pipe = pipe
        self.create_pipeline = create_pipeline
        self.checkpoint = Path(checkpoint)
        self.lora_paths = dict(lora_paths)
        self.lora_manifest = dict(lora_manifest)
        self.vram_mode = vram_mode
        self.use_offload = bool(use_offload)
        self.output_dir = Path(output_dir)
        self.backup_dir = Path(backup_dir)
        self.lock = threading.Lock()

    @property
    def execution_mode(self):
        return (
            "CPU offload (GPU tính toán từng phần)"
            if self.use_offload
            else "GPU trực tiếp"
        )

    def _parameters(
        self,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        count,
        anatomy_enabled,
        anatomy_weight,
        eyes_enabled,
        eyes_weight,
    ):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2200:
            raise ValueError("Prompt gửi model phải có từ 1 đến 2200 ký tự.")
        if not isinstance(negative, str) or len(negative) > 1700:
            raise ValueError("Negative gửi model tối đa 1700 ký tự.")
        steps = _number(steps, "Steps", 10, 45, integer=True)
        cfg = _number(cfg, "CFG", 1, 12)
        seed = _number(seed, "Seed", -1, 2**32 - 1, integer=True)
        if seed != -1 and seed < 0:
            raise ValueError("Seed phải là -1 hoặc số từ 0 đến 2^32 - 1.")
        count = _number(count, "Số ảnh", 1, 4, integer=True)
        if not isinstance(anatomy_enabled, bool) or not isinstance(eyes_enabled, bool):
            raise ValueError("Công tắc LoRA không hợp lệ.")
        loras = {
            "anatomy": (anatomy_enabled, _number(anatomy_weight, "Anatomy", 0.1, 1)),
            "eyes": (eyes_enabled, _number(eyes_weight, "Eyes", 0.1, 1)),
        }
        for name, (enabled, _) in loras.items():
            if enabled and name not in self.lora_paths:
                raise ValueError(
                    f"LoRA {name} chưa được nạp. Bật nó ở ô cấu hình và chạy lại các ô tải/nạp model."
                )
        # Prompt/negative là đúng hai ô người dùng đang thấy: không preset nào
        # và không vùng sửa nào được phép thêm thẻ ẩn lúc suy luận.
        return prompt, negative, steps, cfg, seed, count, loras

    def _apply_loras(self, choices):
        if self.lora_paths:
            names = list(self.lora_paths)
            # Weight zero disables a previously loaded adapter without deleting
            # the verified file or reloading the 6.94 GB checkpoint.
            weights = [choices[name][1] if choices[name][0] else 0.0 for name in names]
            # With CPU offload, the first inference may leave LoRA parameters as
            # inference tensors. PEFT's set_adapters toggles requires_grad; doing
            # that outside InferenceMode fails on the next image in PyTorch.
            with self.torch.inference_mode():
                self.pipe.set_adapters(names, adapter_weights=weights)

    def _infer_once(
        self,
        mode,
        source,
        mask,
        positive,
        negative,
        width,
        height,
        steps,
        cfg,
        seed,
        strength,
    ):
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        other_pipe = None
        if mode != "text":
            from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

            factory = (
                AutoPipelineForImage2Image
                if mode == "image"
                else AutoPipelineForInpainting
            )
            other_pipe = factory.from_pipe(self.pipe)  # reuse model weights
            if self.use_offload:
                self.pipe.remove_all_hooks()
                try:
                    other_pipe.enable_model_cpu_offload()
                except Exception:
                    self.pipe.enable_model_cpu_offload()
                    raise
        target = other_pipe or self.pipe
        options = dict(
            prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
        if mode != "text":
            options.update(image=source, strength=strength)
        if mode == "inpaint":
            options.update(mask_image=mask, padding_mask_crop=32)
        try:
            with self.torch.inference_mode():
                image = target(**options).images[0]
        finally:
            if other_pipe is not None and self.use_offload:
                other_pipe.remove_all_hooks()
                self.pipe.enable_model_cpu_offload()
        if image.size != (width, height):
            raise ValueError(
                "Pipeline trả về ảnh sai kích thước; không lưu để tránh lệch mask."
            )
        return image

    def _infer_with_retry(
        self,
        mode,
        source,
        mask,
        positive,
        negative,
        width,
        height,
        steps,
        cfg,
        seed,
        strength,
        choices,
    ):
        self._apply_loras(choices)
        oom = False
        try:
            return self._infer_once(
                mode,
                source,
                mask,
                positive,
                negative,
                width,
                height,
                steps,
                cfg,
                seed,
                strength,
            )
        except self.torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            self.torch.cuda.empty_cache()
            if self.vram_mode != "auto" or self.use_offload:
                raise RuntimeError(
                    "Hết VRAM. Giảm kích thước ảnh hoặc tắt LoRA và nạp lại model ở chế độ low_vram."
                )
            self.pipe = None
            gc.collect()
            self.torch.cuda.empty_cache()
            self.pipe = self.create_pipeline(True)  # same verified model + LoRAs
            self.use_offload = True
            self._apply_loras(choices)
            try:
                return self._infer_once(
                    mode,
                    source,
                    mask,
                    positive,
                    negative,
                    width,
                    height,
                    steps,
                    cfg,
                    seed,
                    strength,
                )
            except self.torch.cuda.OutOfMemoryError as exc:
                self.torch.cuda.empty_cache()
                raise RuntimeError(
                    "Vẫn thiếu VRAM sau khi thử CPU offload. Giảm kích thước ảnh."
                ) from exc

    def _save_png(self, image, mode, seed, metadata, embed):
        from PIL.PngImagePlugin import PngInfo

        filename = (
            f"wai_{mode}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{seed}.png"
        )
        info = PngInfo()
        if embed:
            info.add_text("parameters", json.dumps(metadata, ensure_ascii=False))

        def save(directory):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / filename
            partial = directory / (filename + ".partial")
            try:
                image.save(partial, format="PNG", pnginfo=info)
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
            return path

        try:
            return save(self.output_dir)
        except OSError:
            if self.output_dir == self.backup_dir:
                raise
            return save(self.backup_dir)

    def _generate(
        self,
        mode,
        source,
        mask,
        target,
        feather,
        strength,
        size,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        count,
        anatomy_enabled,
        anatomy_weight,
        eyes_enabled,
        eyes_weight,
        embed,
    ):
        from PIL import Image, ImageChops, ImageFilter

        positive, negative, steps, cfg, seed, count, choices = self._parameters(
            prompt,
            negative,
            steps,
            cfg,
            seed,
            count,
            anatomy_enabled,
            anatomy_weight,
            eyes_enabled,
            eyes_weight,
        )
        if mode == "text":
            width, height = _preset_size(size)
        elif mode == "image":
            width, height = _preset_size(size)
            if not isinstance(source, Image.Image):
                raise ValueError("Tải ảnh nguồn lên để dùng chế độ ảnh → ảnh.")
            if source.width * source.height > 20_000_000:
                raise ValueError("Ảnh nguồn tối đa 20 MP.")
            from PIL import ImageOps

            source = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (width, height),
                Image.Resampling.LANCZOS,
            )
            strength = _number(strength, "Strength", 0.2, 0.85)
        elif mode == "inpaint":
            if target not in REPAIR_HINTS:
                raise ValueError("Chọn vùng sửa: tay, chân, mắt hoặc tùy chỉnh.")
            strength = _number(strength, "Strength", 0.2, 0.85)
            feather = _number(feather, "Làm mềm viền", 0, 24, integer=True)
            if source is None or mask is None:
                raise ValueError("Tải ảnh và tô hoặc tải mask PNG trước khi sửa.")
            if mask.getbbox() is None or mask.getextrema() == (255, 255):
                raise ValueError(
                    "Mask phải tô một vùng nhỏ, không để rỗng hoặc trắng toàn bộ."
                )
            width, height = source.size
        else:
            raise ValueError("Chế độ tạo ảnh không được hỗ trợ.")

        paths = []
        gallery = []
        selected = []
        with self.lock:  # one GPU pipeline, even if separate UI actions are clicked
            for index in range(count):
                image_seed = (
                    secrets.randbelow(2**32) if seed == -1 else (seed + index) % 2**32
                )
                image = self._infer_with_retry(
                    mode,
                    source,
                    mask,
                    positive,
                    negative,
                    width,
                    height,
                    steps,
                    cfg,
                    image_seed,
                    strength,
                    choices,
                )
                if mode == "inpaint":
                    blend = (
                        ImageChops.multiply(
                            mask, mask.filter(ImageFilter.GaussianBlur(radius=feather))
                        )
                        if feather
                        else mask
                    )
                    image = Image.composite(image.convert("RGB"), source, blend)
                metadata = {
                    "model": self.checkpoint.name,
                    "operation": mode,
                    "prompt": positive,
                    "negative_prompt": negative,
                    "seed": image_seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "strength": strength if mode != "text" else None,
                    "loras": [
                        {
                            "name": name,
                            "weight": choices[name][1],
                            "version": self.lora_manifest[name]["version"],
                            "sha256": self.lora_manifest[name]["sha256"],
                        }
                        for name in self.lora_paths
                        if choices[name][0]
                    ],
                }
                path = self._save_png(image, mode, image_seed, metadata, bool(embed))
                paths.append(str(path))
                gallery.append((str(path), f"Seed {image_seed} · {width}×{height}"))
                selected.append(str(image_seed))
        status = (
            f"✅ Đã tạo {len(paths)} ảnh · seed: {', '.join(selected)}"
            f" · chế độ: {self.execution_mode} · đã lưu: {Path(paths[0]).parent}"
        )
        return gallery, paths, status, paths[-1]

    def text_to_image(
        self,
        size,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        count,
        anatomy_enabled,
        anatomy_weight,
        eyes_enabled,
        eyes_weight,
        embed,
    ):
        return self._generate(
            "text",
            None,
            None,
            None,
            0,
            1,
            size,
            prompt,
            negative,
            steps,
            cfg,
            seed,
            count,
            anatomy_enabled,
            anatomy_weight,
            eyes_enabled,
            eyes_weight,
            embed,
        )

    def image_to_image(
        self,
        source,
        size,
        strength,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        count,
        anatomy_enabled,
        anatomy_weight,
        eyes_enabled,
        eyes_weight,
        embed,
    ):
        return self._generate(
            "image",
            source,
            None,
            None,
            0,
            strength,
            size,
            prompt,
            negative,
            steps,
            cfg,
            seed,
            count,
            anatomy_enabled,
            anatomy_weight,
            eyes_enabled,
            eyes_weight,
            embed,
        )

    def inpaint(
        self,
        editor,
        mask_file,
        target,
        strength,
        feather,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        count,
        anatomy_enabled,
        anatomy_weight,
        eyes_enabled,
        eyes_weight,
        embed,
    ):
        source, mask = _editor_mask(editor, mask_file)
        return self._generate(
            "inpaint",
            source,
            mask,
            target,
            feather,
            strength,
            None,
            prompt,
            negative,
            steps,
            cfg,
            seed,
            count,
            anatomy_enabled,
            anatomy_weight,
            eyes_enabled,
            eyes_weight,
            embed,
        )


def build_app(runtime):
    """Build Gradio Blocks without opening a public tunnel until launch cell runs."""
    import gradio as gr
    from PIL import Image

    css = """
    #wai-studio {max-width: 1440px !important; margin: auto !important;}
    #wai-studio .form {gap: 0.5rem;}
    #wai-studio .block {border-radius: 12px;}
    #wai-studio span[data-testid="block-label"], #wai-studio .label-text {font-size: 0.82rem;}
    #studio-workspace {gap: 24px; align-items: flex-start;}
    #studio-controls, #studio-results {gap: 12px; min-width: 0 !important;}
    .studio-hero {display: flex; flex-wrap: wrap; align-items: center;
        justify-content: space-between; gap: 10px 20px; padding: 16px 20px;
        border: 1px solid var(--border-color-primary); border-radius: 16px;
        background: var(--block-background-fill);}
    .studio-brand {display: flex; align-items: center; gap: 12px;}
    .studio-mark {display: grid; place-items: center; width: 40px; height: 40px;
        border-radius: 12px; background: #ede9fe; color: #6d28d9; font-size: 22px;}
    .studio-hero h1 {font-size: 1.15rem; font-weight: 700; margin: 0;}
    .studio-hero p {font-size: 0.78rem; color: var(--body-text-color); opacity: 0.72; margin: 2px 0 0;}
    .studio-badge {font-size: 0.72rem; color: var(--body-text-color); opacity: 0.72;}
    .studio-section {display: flex; align-items: center; gap: 8px;
        font-size: 0.88rem; font-weight: 650; margin: 2px 0;}
    .studio-section span {display: inline-grid; place-items: center; width: 22px;
        height: 22px; border-radius: 7px; background: #ede9fe; color: #6d28d9;
        font-size: 0.7rem;}
    .studio-hint, .studio-hint p {font-size: 0.76rem !important; line-height: 1.5;
        color: var(--body-text-color); opacity: 0.8; margin: 0 !important;}
    #wai-studio .studio-primary {min-height: 44px; font-weight: 650;}
    #studio-modes .tab-nav {flex-wrap: wrap; gap: 2px;}
    #studio-modes .tab-nav button {font-size: 0.8rem; padding: 8px 10px;}
    #studio-status {padding: 10px 12px; border-radius: 10px;
        background: var(--block-background-fill); border: 1px solid var(--border-color-primary);}
    #studio-gallery {height: clamp(360px, calc(100dvh - 460px), 640px) !important;}
    #studio-security {border-top: 1px solid var(--border-color-primary); padding-top: 12px;}
    @media (min-width: 1000px) {
        #studio-controls {flex: 4 1 0% !important;}
        #studio-results {flex: 6 1 0% !important; position: sticky; top: 16px;}
    }
    @media (max-width: 999px) {
        #studio-workspace {flex-direction: column; gap: 20px;}
        #studio-controls, #studio-results {width: 100%; flex: none !important;}
        #studio-gallery {height: clamp(320px, 55vh, 560px) !important;}
    }
    @media (max-width: 480px) {
        #wai-studio {padding: 10px !important;}
        .studio-hero {padding: 12px;}
        #studio-result-actions {flex-direction: column;}
        #studio-result-actions button {width: 100%;}
    }
    """
    with gr.Blocks(
        title="WAI Studio · Colab GPU",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
        elem_id="wai-studio",
    ) as demo:
        gr.HTML(
            "<header class='studio-hero'>"
            "<div class='studio-brand'><span class='studio-mark' aria-hidden='true'>✦</span>"
            "<div><h1>WAI Studio</h1><p>Ý tưởng của bạn, thế giới anime của bạn.</p></div></div>"
            "<span class='studio-badge'>WAI-illustrious v17 · Google Colab GPU</span>"
            "</header>"
        )
        with gr.Row(equal_height=False, elem_id="studio-workspace"):
            with gr.Column(scale=4, min_width=0, elem_id="studio-controls"):
                gr.HTML("<h2 class='studio-section'><span>01</span> Thiết lập ảnh</h2>")
                prompt = gr.Textbox(
                    label="Prompt gửi model",
                    value=DEFAULT_PROMPT,
                    lines=4,
                    max_lines=7,
                    placeholder=(
                        "Mô tả nhân vật, trang phục, khung cảnh, ánh sáng và phong cách "
                        "vẽ bạn muốn (anime illustration, cel shading, watercolor...)"
                    ),
                    info=("Gửi nguyên văn · tự viết phong cách trong prompt."),
                )
                with gr.Accordion(
                    "Negative prompt · loại trừ chi tiết",
                    open=False,
                    elem_id="studio-negative",
                ):
                    negative = gr.Textbox(
                        label="Negative gửi model · ngón tay / ngón chân",
                        value=DEFAULT_NEGATIVE,
                        lines=3,
                        max_lines=6,
                        info="Nội dung này vẫn được gửi model khi thu gọn. Sửa hoặc xóa tùy ý.",
                    )
                eyes_trigger_button = gr.Button(
                    "+ Thêm perfect eyes",
                    size="sm",
                    elem_id="studio-eye-trigger",
                )
                with gr.Tabs(selected="text", elem_id="studio-modes") as modes:
                    with gr.Tab("Tạo ảnh", id="text"):
                        with gr.Row(equal_height=False):
                            text_size = gr.Dropdown(
                                choices=list(SIZE_PRESETS),
                                value="1024x1024",
                                label="Kích thước",
                                scale=1,
                            )
                            text_button = gr.Button(
                                "Tạo ảnh từ prompt",
                                variant="primary",
                                scale=2,
                                elem_classes="studio-primary",
                            )
                    with gr.Tab("Ảnh → ảnh", id="image"):
                        image_source = gr.Image(
                            label="Ảnh nguồn",
                            type="pil",
                            sources=["upload"],
                            image_mode="RGB",
                            height=230,
                        )
                        with gr.Row():
                            image_size = gr.Dropdown(
                                choices=list(SIZE_PRESETS),
                                value="1024x1024",
                                label="Kích thước đầu ra",
                            )
                            image_strength = gr.Slider(
                                0.2,
                                0.85,
                                value=0.45,
                                step=0.05,
                                label="Denoise strength",
                            )
                        image_button = gr.Button(
                            "Biến đổi ảnh",
                            variant="primary",
                            elem_classes="studio-primary",
                        )
                        gr.Markdown(
                            "Ảnh nguồn khác tỷ lệ sẽ được cắt giữa cho khớp kích thước "
                            "đầu ra, không làm méo.",
                            elem_classes="studio-hint",
                        )
                    with gr.Tab("Sửa vùng", id="inpaint"):
                        editor = gr.ImageEditor(
                            label="Ảnh & vùng tô",
                            type="pil",
                            image_mode="RGBA",
                            sources=["upload"],
                            height=380,
                            format="png",
                            transforms=(),
                            brush=gr.Brush(
                                default_size=40, colors=["#ffffff"], color_mode="fixed"
                            ),
                            eraser=gr.Eraser(default_size=40),
                            layers=False,
                        )
                        with gr.Accordion(
                            "Mask từ file · tùy chọn", open=False, elem_id="studio-mask"
                        ):
                            mask_file = gr.Image(
                                label="Hoặc tải mask trắng/đen PNG (ưu tiên hơn vùng tô)",
                                type="pil",
                                image_mode="L",
                                sources=["upload"],
                                height=150,
                            )
                        with gr.Row():
                            target = gr.Dropdown(
                                choices=["hands", "legs", "eyes", "custom"],
                                value="hands",
                                label="Chi tiết cần sửa",
                            )
                            inpaint_strength = gr.Slider(
                                0.2,
                                0.85,
                                value=0.45,
                                step=0.05,
                                label="Denoise strength",
                            )
                            feather = gr.Slider(
                                0,
                                24,
                                value=8,
                                step=2,
                                label="Làm mềm mép vùng sửa (px)",
                            )
                        with gr.Row():
                            repair_hints_button = gr.Button(
                                "Thêm gợi ý sửa vùng vào prompt đang hiển thị",
                                size="sm",
                                scale=2,
                            )
                            inpaint_button = gr.Button(
                                "Sửa vùng đã tô",
                                variant="primary",
                                scale=1,
                                elem_classes="studio-primary",
                            )
                        gr.Markdown(
                            "**Trắng / nét cọ = sửa; đen = giữ nguyên.** Ảnh tải lên "
                            "được thu về cạnh dài tối đa 1024 px cùng mask. Chọn "
                            "tay/chân/mắt không tự thêm từ vào prompt.",
                            elem_classes="studio-hint",
                        )
                with gr.Accordion(
                    "Thư viện prompt",
                    open=False,
                    elem_id="studio-library",
                ):
                    with gr.Row(equal_height=False):
                        prompt_file = gr.File(
                            label="File danh sách prompt (.txt/.md/.json · tối đa 2 MB)",
                            file_types=[".txt", ".md", ".json"],
                            file_count="single",
                            type="filepath",
                            scale=1,
                        )
                        prompt_paste = gr.Textbox(
                            label="Hoặc dán nội dung file vào đây",
                            lines=3,
                            max_lines=8,
                            placeholder=(
                                "=== 50 PROMPTS – CHỦ ĐỀ ===\n"
                                "PROMPT 01 - Tên tiếng Việt\n"
                                "1girl, solo, ..., masterpiece, best quality"
                            ),
                            scale=2,
                        )
                    with gr.Row():
                        paste_button = gr.Button("Đọc danh sách đã dán", size="sm")
                        sample_button = gr.Button(
                            "Nạp thư viện mẫu (12 prompt)", size="sm"
                        )
                    prompt_choice = gr.Dropdown(
                        choices=(),
                        value=None,
                        interactive=False,
                        label="Chọn prompt để nạp (gõ để lọc danh sách)",
                    )
                    prompt_library_state = gr.State(())
                    library_status = gr.Markdown(
                        "Chưa nạp thư viện. Nạp file của bạn hoặc bấm **Nạp thư viện "
                        "mẫu** để xem định dạng chuẩn.",
                        elem_classes="studio-hint",
                    )
                    gr.Markdown(
                        "Nhận: `PROMPT 01 - Tên` + đoạn prompt, bảng `Tên | Prompt`, "
                        'JSON `[{"title", "prompt"}]`, hoặc các đoạn cách nhau dòng '
                        "trống. Dòng `Negative:/Steps:/CFG:/Size:/Seed:` cũng được áp "
                        "dụng (kẹp về dải của giao diện). Chọn một dòng là nạp thẳng "
                        "vào ô prompt ở trên.",
                        elem_classes="studio-hint",
                    )
                with gr.Accordion(
                    "Thông số ảnh và LoRA", open=False, elem_id="studio-settings"
                ):
                    with gr.Row():
                        steps = gr.Slider(
                            10, 45, value=25, step=1, label="Số bước (steps)"
                        )
                        cfg = gr.Slider(
                            1, 12, value=6, step=0.5, label="CFG / độ bám prompt"
                        )
                    with gr.Row():
                        seed = gr.Number(
                            value=-1,
                            minimum=-1,
                            maximum=2**32 - 1,
                            precision=0,
                            label="Seed (-1 = ngẫu nhiên)",
                        )
                        count = gr.Slider(
                            1, 4, value=1, step=1, label="Số ảnh (tạo lần lượt)"
                        )
                    with gr.Row():
                        anatomy = gr.Checkbox(
                            label="Anatomy Helper · tay/chân",
                            value="anatomy" in runtime.lora_paths,
                            interactive="anatomy" in runtime.lora_paths,
                        )
                        anatomy_weight = gr.Slider(
                            0.1,
                            1,
                            value=runtime.lora_manifest.get("anatomy", {}).get(
                                "weight", 0.55
                            ),
                            step=0.05,
                            label="Cường độ Anatomy",
                        )
                    with gr.Row():
                        eyes = gr.Checkbox(
                            label="Perfect Eyes · mắt",
                            value="eyes" in runtime.lora_paths,
                            interactive="eyes" in runtime.lora_paths,
                        )
                        eyes_weight = gr.Slider(
                            0.1,
                            1,
                            value=runtime.lora_manifest.get("eyes", {}).get(
                                "weight", 0.45
                            ),
                            step=0.05,
                            label="Cường độ Eyes",
                        )
                    embed = gr.Checkbox(
                        label="Nhúng prompt vào metadata PNG (tắt nếu chia sẻ ảnh)",
                        value=False,
                    )
                    gr.Markdown(
                        "LoRA chỉ bật được nếu đã xác minh ở ô cấu hình; tắt/bật và đổi "
                        "cường độ ở đây không nạp lại checkpoint.",
                        elem_classes="studio-hint",
                    )
            with gr.Column(scale=6, min_width=0, elem_id="studio-results"):
                gr.HTML("<h2 class='studio-section'><span>02</span> Kết quả</h2>")
                gallery = gr.Gallery(
                    label="Kết quả · nhấn để xem lớn",
                    columns=2,
                    rows=1,
                    height=600,
                    preview=True,
                    elem_id="studio-gallery",
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                )
                status = gr.Markdown(
                    f"Sẵn sàng nhận ý tưởng của bạn. **Chế độ:** {runtime.execution_mode}",
                    elem_id="studio-status",
                    elem_classes="studio-hint",
                )
                with gr.Row(elem_id="studio-result-actions"):
                    to_image = gr.Button("↗ Biến đổi ảnh mới nhất", size="sm", scale=1)
                    to_inpaint = gr.Button(
                        "✎ Sửa vùng ảnh mới nhất", size="sm", scale=1
                    )
                with gr.Accordion(
                    "Tải ảnh PNG · danh sách file",
                    open=False,
                    elem_id="studio-downloads",
                ):
                    downloads = gr.File(
                        label="Tải ảnh PNG", file_count="multiple", interactive=False
                    )
                latest = gr.State(None)
                gr.Markdown(
                    "Tải từng ảnh bằng nút ⤓ trên ảnh, hoặc mở danh sách file. "
                    "**Tải xuống trước khi Colab ngắt** — ảnh trong `/content` sẽ mất.",
                    elem_classes="studio-hint",
                )

        gr.Markdown(
            "**Link tạm không có đăng nhập:** ai có link đều dùng được GPU của bạn. "
            "Đừng chia sẻ link; dừng runtime để thu hồi.",
            elem_id="studio-security",
            elem_classes="studio-hint",
        )
        shared = [
            prompt,
            negative,
            steps,
            cfg,
            seed,
            count,
            anatomy,
            anatomy_weight,
            eyes,
            eyes_weight,
            embed,
        ]
        outputs = [gallery, downloads, status, latest]
        events = (
            (text_button, runtime.text_to_image, [text_size, *shared]),
            (
                image_button,
                runtime.image_to_image,
                [image_source, image_size, image_strength, *shared],
            ),
            (
                inpaint_button,
                runtime.inpaint,
                [editor, mask_file, target, inpaint_strength, feather, *shared],
            ),
        )
        for button, fn, inputs in events:
            button.click(
                fn=fn,
                inputs=inputs,
                outputs=outputs,
                api_visibility="private",
                concurrency_id="wai_gpu",
                concurrency_limit=1,
            )

        def load_last(path):
            if not path or not Path(path).is_file():
                raise gr.Error("Hãy tạo ít nhất một ảnh trước.")
            return Image.open(path).convert("RGB")

        def transform_last(path):
            return load_last(path), gr.update(selected="image")

        def edit_last(path):
            image = load_last(path).convert("RGBA")
            return (
                {"background": image, "layers": [], "composite": image},
                gr.update(selected="inpaint"),
            )

        to_image.click(
            fn=transform_last,
            inputs=latest,
            outputs=[image_source, modes],
            api_visibility="private",
        )
        to_inpaint.click(
            fn=edit_last,
            inputs=latest,
            outputs=[editor, modes],
            api_visibility="private",
        )
        # Không còn preset: hai ô prompt/negative là đúng những gì gửi model.
        # Trigger LoRA mắt và gợi ý sửa vùng đều là nút bấm tường minh, người
        # dùng thấy và sửa/xóa được trước khi tạo ảnh.
        eyes_trigger_button.click(
            fn=add_eyes_trigger,
            inputs=[prompt, negative],
            outputs=[prompt, negative],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        repair_hints_button.click(
            fn=apply_repair_hints,
            inputs=[prompt, negative, target],
            outputs=[prompt, negative],
            api_visibility="private",
            queue=False,
        )
        # Thư viện prompt: đọc danh sách từ file hoặc đoạn văn bản đã dán, rồi
        # chọn một dòng để nạp thẳng vào ô prompt gửi model.
        library_outputs = [prompt_library_state, prompt_choice, library_status]

        def library_loaded(library):
            return (
                library["items"],
                gr.Dropdown(**prompt_library_dropdown(library["items"])),
                prompt_library_status(library),
            )

        def library_error(message):
            return gr.skip(), gr.skip(), f"⚠️ {message}"

        def load_library_from_file(path):
            try:
                library = read_prompt_library(path)
            except ValueError as error:
                return library_error(str(error))
            return library_loaded(library)

        def load_library_from_text(text):
            try:
                library = load_prompt_library_text(text)
            except ValueError as error:
                return library_error(str(error))
            return library_loaded(library)

        def load_sample_library():
            return library_loaded(load_sample_prompt_library())

        def clear_library():
            items, dropdown, message = prompt_library_reset()
            return items, gr.Dropdown(**dropdown), message

        prompt_file.upload(
            fn=load_library_from_file,
            inputs=prompt_file,
            outputs=library_outputs,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        prompt_file.clear(
            fn=clear_library,
            outputs=library_outputs,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        paste_button.click(
            fn=load_library_from_text,
            inputs=prompt_paste,
            outputs=library_outputs,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        sample_button.click(
            fn=load_sample_library,
            outputs=library_outputs,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        prompt_choice.select(
            fn=apply_prompt_choice,
            inputs=[
                prompt_choice,
                prompt_library_state,
                prompt,
                negative,
                steps,
                cfg,
                seed,
                text_size,
                image_size,
            ],
            outputs=[
                prompt,
                negative,
                steps,
                cfg,
                seed,
                text_size,
                image_size,
                library_status,
            ],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        demo.queue(max_size=4, default_concurrency_limit=1, api_open=False)
    # Gradio 6 applies CSS and themes at launch, not in the Blocks constructor.
    demo.studio_theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["ui-sans-serif", "system-ui", "sans-serif"],
    )
    demo.studio_css = css
    return demo
