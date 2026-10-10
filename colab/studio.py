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
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SIZE_PRESETS = (
    "512x512",
    "768x768",
    "768x1024",
    "1024x768",
    "832x1216",
    "1216x832",
    "1024x1024",
    "1024x1344",
    "1344x1024",
)
# Hires fix: SDXL tạo ảnh gốc quanh ~1 MP, rồi dùng Real-ESRGAN Anime6B x4
# để phóng to và tinh chỉnh tiếp bằng ảnh → ảnh ở denoise thấp.
HIRES_OFF = "Tắt"
HIRES_SCALES = {
    HIRES_OFF: 1.0,
    "1.25×": 1.25,
    "1.5×": 1.5,
    "1.75×": 1.75,
    "2×": 2.0,
}
HIRES_MAX_PIXELS = 4_200_000  # ≈ 2048×2048; giới hạn để không tràn VRAM/RAM Colab
HIRES_STRENGTH_RANGE = (0.2, 0.7)
HIRES_DEFAULT_STRENGTH = 0.4
REAL_ESRGAN_MODEL = {
    "name": "RealESRGAN_x4plus_anime_6B",
    "file": "RealESRGAN_x4plus_anime_6B.pth",
    "url": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    ),
    "size": 17_938_799,
    "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    "version": "xinntao/Real-ESRGAN@v0.2.2.4",
}
# Cập nhật mỗi lần sửa colab/studio.py: header in mã này để người dùng biết phiên
# Colab đang chạy bản nào (chạy lại riêng ô 9 KHÔNG cập nhật mã UI — nằm ở ô 7).
STUDIO_BUILD = "2026.10.10 · tách hàng đợi Kho thẻ"
REAL_ESRGAN_CACHE = "/content/wai_upscaler_cache"
REAL_ESRGAN_TILE_SIZE = 256
REAL_ESRGAN_TILE_PAD = 16
# Prompt mẫu viết theo THỨ TỰ CHUẨN mà Studio dùng: chủ thể đứng đầu để CLIP bám
# đối tượng, chi tiết nhân vật (tóc/mắt/mặt/biểu cảm) và trang phục đi ngay sau, rồi
# tư thế → bố cục → bối cảnh → ánh sáng → phong cách; thẻ lạ của bạn xếp sau phong
# cách, thẻ chất lượng đặt áp chót và thẻ độ nét chốt cuối. Bạn sửa/xóa tùy ý;
# Studio gửi đúng nội dung hai ô prompt/negative cho model.
DEFAULT_PROMPT = (
    "1girl, solo, adult woman, long dark hair, gentle smile, "
    "standing under cherry blossoms, petals falling, spring, soft sunlight, "
    "cel shading, anime illustration, masterpiece, best quality, amazing quality, absurdres"
)
# Negative mặc định nhắm lỗi ngón tay/ngón chân; người dùng tự sửa theo ý mình.
# Muốn ngắn hơn hãy nạp bộ "Chuẩn nhà phát hành WAI v17" trong UI — negative quá
# dài làm giảm chất lượng ảnh theo khuyến nghị của chính nhà phát hành.
DEFAULT_NEGATIVE = (
    "lowres, worst quality, low quality, blurry, bad anatomy, "
    "bad hands, deformed hands, extra fingers, missing fingers, fused fingers, "
    "malformed fingers, deformed feet, extra toes, missing toes, fused toes, "
    "malformed toes, extra limbs"
)
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
    "nails": (
        "natural nails, well-shaped fingernails, detailed nails",
        "deformed nails, missing nails, extra nails, chipped nails, broken nails",
    ),
    "legs": (
        "natural leg anatomy, well-formed feet, natural toes, balanced pose",
        "extra legs, broken legs, deformed feet, extra toes, missing toes, fused toes",
    ),
    "eyes": (
        "symmetrical eyes, detailed irises, perfect eyes",
        "misaligned eyes, deformed eyes, extra eyes",
    ),
    "face": (
        "symmetrical face, detailed face, natural skin tone",
        "bad face, poorly drawn face, asymmetrical eyes, deformed face, cross-eyed",
    ),
    "teeth": (
        "natural teeth, straight teeth",
        "bad teeth, crooked teeth, missing teeth, extra teeth",
    ),
    "hair": (
        "detailed hair, natural hair strands, consistent hairline",
        "messy hairline, fused hair strands, blurry hair, missing hair",
    ),
    "skin": (
        "smooth skin, even skin tone, consistent shading",
        "skin blemishes, acne, patchy skin, oily skin",
    ),
    "custom": ("", ""),
}
# Nhãn tiếng Việt cho dropdown; GIÁ TRỊ gửi vào runtime vẫn là khóa tiếng Anh ở trên
# (đúng khóa mà `_generate` kiểm tra khi sửa vùng), nên không đổi hành vi cũ.
REPAIR_LABELS = {
    "hands": "Bàn tay",
    "nails": "Móng tay / móng chân",
    "legs": "Chân / bàn chân",
    "eyes": "Mắt",
    "face": "Khuôn mặt",
    "teeth": "Răng",
    "hair": "Tóc",
    "skin": "Da",
    "custom": "Tùy chỉnh (tự viết)",
}
assert set(REPAIR_LABELS) == set(REPAIR_HINTS)

# ---------------------------------------------------------------------------
# CHI TIẾT MẮT & MÓNG: màu mắt, kiểu dáng móng tay, màu sơn móng tay/móng chân.
#
# Thẻ viết theo cú pháp Danbooru mà họ Illustrious bám tốt ("red nails",
# "almond-shaped nails", "heterochromia"…), mỗi lựa chọn là MỘT NHÓM thẻ để ghép
# được cả kiểu dáng + màu. Như mọi công cụ khác của Studio: chỉ ghi vào ô prompt
# ĐANG HIỂN THỊ để bạn sửa/xóa; không có thẻ nào được ghép ngầm lúc tạo ảnh.
# ---------------------------------------------------------------------------
LOOK_OFF = "Không thêm"
LOOK_FIELDS = (
    (
        "eye_color",
        "Màu mắt",
        {
            LOOK_OFF: (),
            "Xanh dương": ("blue eyes",),
            "Xanh dương nhạt": ("light blue eyes",),
            "Xanh ngọc": ("aqua eyes",),
            "Xanh lá": ("green eyes",),
            "Nâu": ("brown eyes",),
            "Hổ phách": ("amber eyes",),
            "Đỏ": ("red eyes",),
            "Hồng": ("pink eyes",),
            "Tím": ("purple eyes",),
            "Vàng": ("yellow eyes",),
            "Xám": ("grey eyes",),
            "Bạc": ("silver eyes",),
            "Đen": ("black eyes",),
            "Hai màu (heterochromia)": ("heterochromia", "multicolored eyes"),
            "Đổi màu (gradient)": ("gradient eyes",),
            "Mắt phát sáng": ("glowing eyes", "detailed pupils"),
        },
    ),
    (
        "nail_shape",
        "Kiểu dáng móng tay",
        {
            LOOK_OFF: (),
            "Tự nhiên, ngắn": ("natural nails", "short nails"),
            "Tròn ngắn": ("short round nails",),
            "Vuông": ("square nails",),
            "Bầu dục": ("oval nails",),
            "Hạnh nhân (almond)": ("almond-shaped nails",),
            "Dài nhọn (stiletto)": ("stiletto nails", "long fingernails"),
            "Dài đầu vuông (coffin)": ("coffin nails", "long fingernails"),
            "Móng dài": ("long fingernails",),
            "Móng sắc": ("sharp fingernails",),
        },
    ),
    (
        "nail_color",
        "Màu sơn móng tay",
        {
            LOOK_OFF: (),
            "Đỏ": ("red nails", "nail polish"),
            "Đen": ("black nails", "nail polish"),
            "Hồng": ("pink nails", "nail polish"),
            "Trắng": ("white nails", "nail polish"),
            "Xanh dương": ("blue nails", "nail polish"),
            "Tím": ("purple nails", "nail polish"),
            "Nude / da": ("nude nails", "nail polish"),
            "Gradient": ("gradient nails", "nail polish"),
            "Kim tuyến": ("glitter nails", "nail polish"),
            "French (đầu trắng)": ("french nails", "nail polish"),
            "Vẽ hoa văn": ("nail art", "nail polish"),
            "Không sơn": ("natural nails",),
        },
    ),
    (
        "toenail_color",
        "Màu sơn móng chân",
        {
            LOOK_OFF: (),
            "Đỏ": ("painted toenails", "red nail polish"),
            "Đen": ("painted toenails", "black nail polish"),
            "Hồng": ("painted toenails", "pink nail polish"),
            "Trắng": ("painted toenails", "white nail polish"),
            "Xanh dương": ("painted toenails", "blue nail polish"),
            "Tím": ("painted toenails", "purple nail polish"),
            "Nude / da": ("painted toenails", "nude nail polish"),
            "Kim tuyến": ("painted toenails", "glitter nail polish"),
            "Không sơn": ("natural toenails",),
        },
    ),
)
LOOK_OPTIONS = {field: choices for field, _, choices in LOOK_FIELDS}
LOOK_LABELS = {field: label for field, label, _ in LOOK_FIELDS}
for _field, _label, _choices in LOOK_FIELDS:
    assert LOOK_OFF in _choices, f"{_label} phải có lựa chọn '{LOOK_OFF}'."


def _add_prompt_tags(text, tags):
    """Append visible preset/repair tags once to an editable prompt.

    So khớp theo lõi thẻ, nên `(perfect eyes:1.1)` được coi là đã có `perfect eyes`
    và không bị chèn thêm một bản không trọng số.
    """
    base = text if isinstance(text, str) else ""
    seen = set()
    for part in base.split(","):
        core = tag_core(part)
        if core:
            seen.add(core.casefold())
    for tag in tags:
        tag = str(tag or "").strip()
        core = tag_core(tag)
        if tag and core and core.casefold() not in seen:
            base = f"{base}, {tag}" if base else tag
            seen.add(core.casefold())
    return base


def add_eyes_trigger(prompt, negative):
    """Explicit UI action: add the eye LoRA trigger once, visible and editable."""
    return _add_prompt_tags(prompt, ("perfect eyes",)), negative


def apply_repair_hints(positive, negative, target):
    """Explicit UI action: append suggestions in the editable fields before inpainting."""
    if target not in REPAIR_HINTS:
        raise ValueError(
            "Chọn vùng sửa: "
            + ", ".join(label.lower() for label in REPAIR_LABELS.values())
            + "."
        )
    hint_pos, hint_neg = REPAIR_HINTS[target]
    return (
        _add_prompt_tags(positive, hint_pos.split(",")),
        _add_prompt_tags(negative, hint_neg.split(",")),
    )


def look_tags(
    eye_color=LOOK_OFF,
    nail_shape=LOOK_OFF,
    nail_color=LOOK_OFF,
    toenail_color=LOOK_OFF,
):
    """Danh sách thẻ của các lựa chọn mắt/móng, theo thứ tự mắt → móng tay → móng chân."""
    chosen = {
        "eye_color": eye_color,
        "nail_shape": nail_shape,
        "nail_color": nail_color,
        "toenail_color": toenail_color,
    }
    tags = []
    for field, label in LOOK_LABELS.items():
        value = chosen.get(field, LOOK_OFF)
        options = LOOK_OPTIONS[field]
        if value not in options:
            raise ValueError(f"Chọn {label.lower()} trong danh sách hoặc '{LOOK_OFF}'.")
        tags.extend(options[value])
    return tuple(tags)


def apply_look_tags(
    positive,
    negative,
    eye_color=LOOK_OFF,
    nail_shape=LOOK_OFF,
    nail_color=LOOK_OFF,
    toenail_color=LOOK_OFF,
):
    """Hành động UI: thêm thẻ mắt/móng vào ô prompt đang hiển thị (sửa/xóa được)."""
    tags = look_tags(eye_color, nail_shape, nail_color, toenail_color)
    if not tags:
        raise ValueError(
            "Chưa chọn chi tiết nào: hãy chọn màu mắt, kiểu dáng móng hoặc màu sơn "
            f"móng (khác '{LOOK_OFF}') rồi bấm lại."
        )
    return _add_prompt_tags(positive, tags), negative


# ---------------------------------------------------------------------------
# QUY TRÌNH PROMPT CHUYÊN NGHIỆP (thứ tự thẻ chuẩn + negative theo mục đích +
# bộ kiểm tra prompt/thông số + kiểm tra thẻ trong prompt với kho thẻ CSV).
#
# Căn cứ đã đối chiếu (xem docs/QUY_TRINH_TAO_ANH.md, mục Nguồn):
#   • Nhà phát hành WAI-illustrious v17: quality head "masterpiece, best quality,
#     amazing quality"; negative ngắn "bad quality, worst quality, worst detail,
#     sketch, censor"; steps 15–30; CFG 5–7; Euler a; hires 1.5× với denoise
#     0.35–0.5; kèm cảnh báo KHÔNG thêm quá nhiều thẻ chất lượng/thẩm mỹ và
#     không viết negative quá dài vì sẽ làm giảm chất lượng, ảnh dễ mờ.
#   • Hướng dẫn cộng đồng Illustrious XL (SeaArt, tensor.art): họ Illustrious
#     bám negative rất tốt — "worst quality, low quality, very displeasing,
#     displeasing, oldest", "artistic error", "lowres, jpeg artifacts, censor,
#     watermark, bad hands, bad anatomy, traditional media"; thứ tự prompt:
#     chất lượng → nhãn phân loại → chủ thể → chi tiết → tư thế → bố cục/bối
#     cảnh → ánh sáng → phong cách → "absurdres, highres" ở cuối.
#   • SDXL/Illustrious mã hoá prompt theo khối 75 token: phần vượt quá bị bỏ
#     qua khối sau (mất ưu tiên) nên prompt phải ngắn, đúng thứ tự ưu tiên và
#     chỉ nhấn mạnh bằng (thẻ:1.1–1.2) thay vì lặp từ.
#
# Nguyên tắc của Studio vẫn giữ: mọi hàm dưới đây chỉ *trả về chuỗi hiển thị*
# trong hai ô prompt/negative để người dùng sửa hoặc xóa; không có thẻ nào được
# ghép ngầm lúc bấm tạo ảnh.
# ---------------------------------------------------------------------------
RECOMMENDED_STEPS = (15, 30)
RECOMMENDED_CFG = (5.0, 7.0)
RECOMMENDED_HIRES_STRENGTH = (0.35, 0.5)
SDXL_TOKEN_CHUNK = 75  # khối token CLIP của SDXL/Illustrious
QUALITY_TAGS = ("masterpiece", "best quality", "amazing quality")
QUALITY_TAIL = ("absurdres",)

# Negative theo mục đích: mỗi preset cố ý GIỮ NGẮN. Nhà phát hành cảnh báo
# negative quá dài làm giảm chất lượng; chuyên nghiệp là chọn đúng bộ cho đúng
# việc rồi cộng thêm vài thẻ riêng của ảnh đó, không dùng một danh sách khổng lồ.
NEGATIVE_PRESETS = (
    {
        "id": "publisher",
        "label": "Chuẩn nhà phát hành WAI v17 (ngắn nhất)",
        "tags": (
            "bad quality",
            "worst quality",
            "worst detail",
            "sketch",
            "censor",
        ),
        "when": "Mặc định an toàn cho mọi ảnh; ít rủi ro nghẹt chi tiết/làm mờ nhất.",
    },
    {
        "id": "core",
        "label": "Illustrious chuẩn · chất lượng + lỗi vẽ",
        "tags": (
            "worst quality",
            "low quality",
            "bad quality",
            "lowres",
            "jpeg artifacts",
            "bad anatomy",
            "bad hands",
            "extra digit",
            "fewer digits",
            "watermark",
            "signature",
            "text",
            "artistic error",
            "very displeasing",
            "oldest",
        ),
        "when": "Bộ dùng hằng ngày: vừa chặn lỗi chất lượng vừa chặn lỗi giải phẫu nhẹ.",
    },
    {
        "id": "anatomy",
        "label": "Sửa tay / chân / tỷ lệ cơ thể",
        "tags": (
            "bad anatomy",
            "bad hands",
            "deformed hands",
            "extra digit",
            "fewer digits",
            "fused fingers",
            "conjoined fingers",
            "extra limbs",
            "missing limbs",
            "extra arms",
            "extra legs",
            "bad proportions",
            "bad perspective",
            "deformed feet",
            "extra toes",
            "fused toes",
            "long neck",
        ),
        "when": "Ảnh có bàn tay/chân phức tạp; dùng kèm LoRA Anatomy hoặc inpaint vùng nhỏ.",
    },
    {
        "id": "anime2d",
        "label": "Giữ chất anime 2D (chống 3D/thực)",
        "tags": (
            "realistic",
            "photorealistic",
            "3d",
            "cgi",
            "render",
            "plastic skin",
            "realistic skin texture",
            "dull colors",
            "monochrome",
            "greyscale",
            "sketch",
            "traditional media",
            "jpeg artifacts",
            "bad anatomy",
        ),
        "when": "WAI v17 dễ ra chất nhựa/3D khi ít thẻ chất lượng; bộ này ép về nét vẽ 2D.",
    },
    {
        "id": "portrait",
        "label": "Chân dung · mặt, mắt, răng",
        "tags": (
            "bad face",
            "poorly drawn face",
            "deformed eyes",
            "asymmetrical eyes",
            "cross-eyed",
            "extra eyes",
            "dull eyes",
            "bad teeth",
            "crooked teeth",
            "skin blemishes",
            "acne",
            "bad anatomy",
            "bad hands",
            "extra digit",
            "jpeg artifacts",
            "watermark",
            "text",
        ),
        "when": "Ảnh cận mặt; nên kết hợp LoRA mắt (trigger `perfect eyes`) và inpaint vùng mắt.",
    },
    {
        "id": "scene",
        "label": "Phong cảnh · không có nhân vật",
        "tags": (
            "1girl",
            "1boy",
            "solo",
            "people",
            "crowd",
            "bad perspective",
            "bad proportions",
            "lowres",
            "blurry",
            "worst quality",
            "low quality",
            "jpeg artifacts",
            "text",
            "watermark",
            "signature",
            "logo",
        ),
        "when": "Background/key visual; chặn người lọt vào khung và lỗi phối cảnh.",
    },
    {
        "id": "sfw",
        "label": "An toàn nội dung (mọi nhân vật trưởng thành)",
        "tags": (
            "nsfw",
            "explicit",
            "nude",
            "loli",
            "shota",
            "child",
            "aged_down",
            "censored",
            "censor bars",
            "worst quality",
            "low quality",
            "jpeg artifacts",
        ),
        "when": "WAI có nhãn phân loại general/sensitive/nsfw/explicit: đưa nhãn unwanted vào negative.",
    },
    {
        "id": "inpaint",
        "label": "Inpaint / sửa vùng (rất ngắn)",
        "tags": (
            "bad quality",
            "worst quality",
            "lowres",
            "blurry",
            "jpeg artifacts",
            "watermark",
            "text",
            "bad anatomy",
        ),
        "when": "Khi sửa vùng nhỏ: negative ngắn để denoise thấp giữ được nét xung quanh.",
    },
)
NEGATIVE_REPLACE = "Ghi đè ô negative"
NEGATIVE_APPEND = "Nối thêm thẻ còn thiếu"
NEGATIVE_MODES = (NEGATIVE_REPLACE, NEGATIVE_APPEND)

# Thứ tự thẻ chuẩn: thẻ đứng trước được CLIP chú ý nhiều hơn. Chủ thể và chi tiết
# nhân vật (tóc/mắt/mặt/biểu cảm) dẫn đầu vì đó là thứ người dùng chỉnh nhiều nhất;
# trang phục — tư thế — bố cục — bối cảnh — ánh sáng — phong cách theo sau, thẻ lạ của
# người dùng đứng sau phong cách. Thẻ CHẤT LƯỢNG đặt áp chót: nhà phát hành WAI v17
# chỉ khuyến nghị *có* 2–3 thẻ masterpiece/best quality, còn để cuối thì khối chất
# lượng không chen mất vị trí mở đầu của thẻ tả nhân vật. absurdres luôn chốt cuối.
PROMPT_SECTIONS = (
    "subject",
    "rating",
    "appearance",
    "outfit",
    "pose",
    "composition",
    "background",
    "lighting",
    "style",
    "extra",
    "quality",
    "tail",
)
SECTION_LABELS = {
    "quality": "Chất lượng",
    "rating": "Nhãn phân loại",
    "subject": "Chủ thể",
    "appearance": "Ngoại hình",
    "outfit": "Trang phục",
    "pose": "Tư thế / hành động",
    "extra": "Thẻ khác của bạn",
    "composition": "Bố cục / góc máy",
    "background": "Bối cảnh",
    "lighting": "Ánh sáng / màu",
    "style": "Phong cách",
    "tail": "Độ nét (cuối prompt)",
}
# Thứ tự ưu tiên khi một thẻ khớp nhiều nhóm (khớp trước thắng).
_SECTION_RULES = (
    (
        "quality",
        r"^(?:masterpiece|(?:best|amazing|highest|great|top|ultra|high) quality|very aesthetic|newest)$",
    ),
    ("tail", r"^(?:absurdres|highres|high resolution|uhd|4k|8k)$"),
    ("rating", r"^(?:general|sensitive|questionable|explicit|safe|nsfw|rating_\w+)$"),
    (
        "subject",
        # `(?<![a-z])` để bắt được thẻ có số đứng trước như "1girl", "2boys":
        # giữa số và chữ không có ranh giới từ nên \b thường không khớp.
        r"(?:(?<![a-z])girls?\b|(?<![a-z])boys?\b|\bother\b|no humans|\bsolo\b|\bduo\b|\bcouple\b|"
        r"\bgroup\b|\b(?:female|male|vehicle|animal|food|object|profile) focus\b|\badult\b|\bmature\b|office worker|\bstudent\b|\bidol\b|"
        r"\bmaid\b|\bknight\b|\bwitch\b|\belf\b|\bandroid\b|\brobot\b|\bwaitress\b|"
        r"\bnurse\b|\bteacher\b|\bartists?\b|\bsamurai\b|\bninja\b|\bprincess\b|"
        r"\bqueen\b|\bking\b|\bsoldiers?\b|\bpilots?\b|\bchef\b|\bbarista\b|"
        r"\bresearchers?\b|\btravellers?\b|\btravelers?\b|\bangel\b|\bdemon\b|"
        r"\bvampire\b|\bcat girl\b|\bfox girl\b|\bmagical girl\b|\bcharacter\b)",
    ),
    (
        "pose",
        r"(\bstanding\b|\bsitting\b|\bwalking\b|\brunning\b|\bleaning\b|\blying\b|"
        r"\bkneeling\b|\bcrouching\b|\bjumping\b|\bdancing\b|\bflying\b|\bsleeping\b|"
        r"\bstretching\b|\bholding\b|\breaching\b|\bpointing\b|\bwaving\b|\breading\b|"
        r"\btyping\b|\bdrinking\b|\beating\b|\blooking\b|\bglancing\b|\bstaring\b|"
        r"\bsmiling\b|\bgrinning\b|\bfrowning\b|\bcrying\b|\bpose\b|\bposes\b|"
        r"\barms?\b|\bhands?\b|\blegs?\b|\bhead tilt\b|\bfrom behind\b|\bback\b|"
        r"\bhand on hip\b|\bhand in pocket\b|\bcrossed\b|\bturned\b|\bmid-air\b)",
    ),
    (
        "outfit",
        r"(\bshirt\b|\bblouse\b|\bdress\b|\bskirt\b|\bjacket\b|\bcoat\b|\bsuit\b|"
        r"\bblazer\b|\buniform\b|\bsweater\b|\bcardigan\b|\bhoodie\b|\bkimono\b|"
        r"\byukata\b|\barmor\b|\bgloves?\b|\bsocks?\b|\bstockings?\b|\bshoes?\b|"
        r"\bboots?\b|\bsneakers?\b|\bheels?\b|\btie\b|\bnecktie\b|\bscarf\b|\bhat\b|"
        r"\bcap\b|\bribbon\b|\bbow\b|\bapron\b|\bvest\b|\btrousers\b|\bpants\b|"
        r"\bjeans\b|\bshorts\b|\bbelt\b|\bbag\b|\bbackpack\b|\bjewelry\b|\bearing\b|"
        r"\bnecklace\b|\bring\b|\bcollar\b|\bhood\b|\bsleeves?\b|\bgown\b|\brobe\b|"
        r"\bcape\b|\bhelmet\b|\bmask\b|\boutfit\b|\bclothes\b|\bclothing\b|"
        r"\bwearing\b|\bdress shirt\b|\bpencil skirt\b|\bturtleneck\b|\bknit\b|thighhighs|pantyhose|\bgarter\b|\bsuspenders\b)",
    ),
    (
        # Nhóm này bắt cả thẻ móng/màu mắt của công cụ 'Chi tiết mắt & móng'.
        "appearance",
        r"(\bhair\b|\beyes?\b|\bskin\b|\bface\b|\bnose\b|\blips?\b|\bteeth\b|"
        r"\bears?\b|\btattoo\b|\bfreckles\b|\bmole\b|\bscar\b|\bglasses\b|"
        r"\bmakeup\b|\bexpression\b|\bsmile\b|\bblush\b|\btall\b|\bpetite\b|"
        r"\bslim\b|\bcurvy\b|\bmuscular\b|\bbraid\b|\bponytail\b|\btwintails\b|"
        r"\bbangs\b|\bbob cut\b|\bbun\b|\bwavy\b|\bstraight hair\b|\blong hair\b|"
        r"\bshort hair\b|\bblue eyes\b|\bbody\b|\bproportions\b|\bnails?\b|\bfingernails?\b|\btoenails?\b|\bnail polish\b|\bnail art\b|\bpupils?\b|\bheterochromia\b|\beyelashes\b|\beyebrows\b|\birides\b|\biris\b|\bbreasts?\b|\bchest\b|\bnavel\b|\bwaist\b|\bhips?\b|\bthighs?\b|\btails?\b|\bwings?\b|\bhorns?\b|\bclaws?\b|\bfangs?\b|\bscales\b|\bmuzzles?\b|\bhooves?\b|\btalons?\b|\bnipples?\b|\bareolas?\b|\bpiercings?\b|\btattoos?\b|\bgenitals?\b|\bpussy\b|\bpenis\b|\bvulva\b|\btesticles?\b|\bexoskeleton\b|\bantennae\b)"
        # Móng và chi tiết mắt được xếp vào nhóm Ngoại hình thay vì 'extra'. Cơ thể và bộ
        # phận nhân vật (vú, đuôi, cánh, sừng, bộ phận sinh dục…) cũng về nhóm này vì thẻ tả
        # nhân vật cần đứng đầu prompt. Tay/chân/đầu vẫn thuộc 'pose' do thứ tự khớp.

    ),
    (
        "composition",
        r"(close-up|\bcowboy shot\b|\bupper body\b|\bfull body\b|\bportrait\b|"
        r"wide shot|\bestablishing shot\b|dutch angle|from above|from below|"
        r"low angle|high angle|bird'?s?-eye|straight-on|depth of field|\bbokeh\b|"
        r"dynamic angle|rule of thirds|\bshot\b|\bangle\b|\bview\b|\bmacro\b|"
        r"\bfisheye\b|\bframing\b|\bsilhouette\b|\bsymmetrical\b|\bfocus\b)",
    ),
    (
        "background",
        r"(\bbackground\b|\bcity\b|cityscape|\bstreet\b|\balley\b|\boffice\b|"
        r"\bcaf[eé]\b|\brooftop\b|\bforest\b|\bsky\b|\bclouds?\b|cherry blossoms?|"
        r"\bsakura\b|\brain\b|\bsnow\b|\bnight\b|\bday\b|\bsunset\b|\bsunrise\b|"
        r"\bdawn\b|\bdusk\b|\bindoors\b|\boutdoors\b|\bsea\b|\bocean\b|\bbeach\b|"
        r"\bmountains?\b|\bhills?\b|\bgarden\b|\bpark\b|\blibrary\b|\btrains?\b|"
        r"\bspring\b|\bsummer\b|\bautumn\b|\bwinter\b|"
        r"\bstation\b|\bwindows?\b|\bdoors?\b|\bneon\b|\bvillage\b|\bcorridor\b|"
        r"\belevator\b|\blobby\b|\bdesks?\b|classroom|\bbedroom\b|\bkitchen\b|"
        r"\bbridge\b|\briver\b|\bfields?\b|\bflowers?\b|\bpetals?\b|\bstars\b|"
        r"\bmoon\b|\bsun\b|\bweather\b|\bskyline\b|\bfoliage\b|\bwalls?\b|\bfloor\b|\bceiling\b|\bpillars?\b|\bstairs?\b|\bcolumns?\b|\bplants?\b|\bshelves\b|\bfurniture\b|\bwind\b|\bwindy\b|\bfog\b|\bmisty\b|\bstorm\b|\blightning\b|\bhumid\b)",
    ),
    (
        "lighting",
        r"(\blight\b|\blights\b|\blighting\b|\bglow\b|\bglowing\b|sunlight|"
        r"moonlight|backlight|backlit|rim light|\bshadows?\b|cinematic lighting|"
        r"soft light|warm light|cool light|\bdramatic\b|\bmoody\b|lens flare|"
        r"god rays|sunbeam|\bcolou?rs?\b|\bpalette\b|\bpastel\b|\bvivid\b|"
        r"saturated|desaturated|\btone\b|\bcontrast\b|\bbrightness\b|\batmosphere\b|"
        r"\bambient\b|\bwarm\b|\bcool\b|\bdim\b|\bbright\b)",
    ),
    (
        "style",
        r"(\banime\b|illustration|cel shading|cel shaded|watercolou?r|"
        r"oil painting|acrylic|\bsketch\b|lineart|line art|flat colou?rs?|\bmanga\b|"
        r"\bcomic\b|\bcinematic\b|film still|key visual|art nouveau|ukiyo-e|"
        r"pixel art|\bchibi\b|semi-realistic|realistic|2\.5d|painterly|\bgouache\b|"
        r"\bink\b|digital painting|art by|style of|\bstudio\b|screencap|"
        r"retro artstyle|\bshading\b|\blineart\b|\bdetailed\b|\bdetail\b)",
    ),
)
_SECTION_RULE_MAP = dict(_SECTION_RULES)
# Thứ tự ưu tiên khi KHỚP, tách khỏi thứ tự hiển thị PROMPT_SECTIONS: nhóm bố cục
# phải được xét trước ngoại hình, nếu không "upper body" bị `\bbody\b` của nhóm
# ngoại hình bắt mất và prompt bị xếp sai chỗ.
_SECTION_MATCH_ORDER = (
    "quality",
    "tail",
    "rating",
    "subject",
    "composition",
    "pose",
    "outfit",
    "appearance",
    "background",
    "lighting",
    "style",
)
assert set(_SECTION_MATCH_ORDER) == set(
    _SECTION_RULE_MAP
), "Thứ tự khớp phải phủ đúng các nhóm có luật."
_SECTION_RES = tuple(
    (section, re.compile(_SECTION_RULE_MAP[section], re.IGNORECASE))
    for section in _SECTION_MATCH_ORDER
)
# Khung neo theo loại ảnh: chỉ được THÊM khi nhóm tương ứng đang trống, để không
# nhét thừa thẻ chất lượng (nhà phát hành cảnh báo thừa thẻ chất lượng làm mờ ảnh).
PROMPT_SCAFFOLDS = {
    "character": {
        "label": "Nhân vật · 1 nhân vật",
        "anchors": {
            "quality": QUALITY_TAGS,
            "subject": ("1girl", "solo"),
            "style": ("anime illustration", "cel shading"),
            "tail": QUALITY_TAIL,
        },
    },
    "portrait": {
        "label": "Chân dung cận mặt",
        "anchors": {
            "quality": QUALITY_TAGS,
            "subject": ("1girl", "solo"),
            "composition": ("close-up", "looking at viewer"),
            "lighting": ("soft lighting",),
            "style": ("anime illustration", "cel shading"),
            "tail": QUALITY_TAIL,
        },
    },
    "scene": {
        "label": "Phong cảnh · không nhân vật",
        "anchors": {
            "quality": QUALITY_TAGS,
            "subject": ("no humans",),
            "composition": ("wide shot",),
            "background": ("detailed background", "scenery"),
            "style": ("anime background",),
            "tail": QUALITY_TAIL,
        },
    },
    "action": {
        "label": "Hành động / key visual",
        "anchors": {
            "quality": QUALITY_TAGS,
            "subject": ("1girl", "solo"),
            "pose": ("dynamic pose",),
            "composition": ("dynamic angle", "depth of field"),
            "lighting": ("dramatic lighting",),
            "style": ("anime key visual", "cel shading"),
            "tail": QUALITY_TAIL,
        },
    },
}
SCAFFOLD_CHOICES = tuple((item["label"], key) for key, item in PROMPT_SCAFFOLDS.items())

# Thẻ/cú pháp của hệ Pony không có tác dụng trên Illustrious/WAI.
PONY_ONLY_TAGS = re.compile(
    r"^(?:score_\w+|source_\w+|rating_(?:safe|explicit|questionable))$", re.IGNORECASE
)
WEIGHT_RE = re.compile(r"^\((?P<body>.+?):(?P<weight>-?\d+(?:\.\d+)?)\)$")
QUALITY_TAG_SET = {
    "masterpiece",
    "best quality",
    "amazing quality",
    "highest quality",
    "great quality",
    "ultra quality",
    "high quality",
    "very aesthetic",
    "newest",
}
STYLE_TAG_HINTS = (
    "anime",
    "illustration",
    "cel shading",
    "watercolor",
    "manga",
    "lineart",
    "painting",
    "flat color",
    "key visual",
    "film still",
    "screencap",
    "artstyle",
)
SUBJECT_TAG_RE = re.compile(
    r"(?:\d+\s*)?(?:girls?|boys?|other)\b|no humans|\bsolo\b|\b(?:female|male|vehicle|animal|food|object|profile) focus\b|"
    r"\bcouple\b|\bduo\b|\bgroup\b",
    re.IGNORECASE,
)


def negative_preset_choices():
    """Dropdown choices: nhãn mô tả → id preset."""
    return tuple((preset["label"], preset["id"]) for preset in NEGATIVE_PRESETS)


def find_negative_preset(preset_id):
    for preset in NEGATIVE_PRESETS:
        if preset["id"] == preset_id:
            return preset
    raise ValueError("Chọn một bộ negative trong danh sách trước khi nạp.")


def split_tags(text):
    """Tách chuỗi prompt thành danh sách thẻ, giữ nguyên cú pháp nhấn mạnh."""
    if not isinstance(text, str):
        raise ValueError("Prompt phải là chuỗi văn bản.")
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def tag_core(tag):
    """Bỏ cú pháp nhấn mạnh ((thẻ), [thẻ], (thẻ:1.2)) để so khớp nội dung thẻ.

    Gỡ từng lớp ngoặc và trọng số, nên ``((perfect eyes:1.1))`` khớp ``perfect eyes``
    chứ không còn sót ``:1.1``.
    """
    core = str(tag or "").strip()
    for _ in range(8):
        match = WEIGHT_RE.fullmatch(core)
        if match:
            body = match.group("body").strip()
            if body and body.count("(") == body.count(")"):
                core = body
                continue
        if len(core) >= 2 and {"(": ")", "[": "]"}.get(core[0]) == core[-1]:
            inner = core[1:-1].strip()
            if inner:
                core = inner
                continue
        break
    return core.strip("()[] ").strip()


def estimate_tokens(text):
    """Ước lượng token CLIP (BPE) — heuristic, không dùng tokenizer thật.

    Mỗi từ ≈ 1 token; dấu gạch dưới, số dài và dấu câu thường bị tách thành token
    riêng nên được cộng thêm. Con số dùng để cảnh báo vượt khối 75 token, không
    phải số token chính xác của CLIP.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    total = 1  # token mở đầu chuỗi
    for word in re.findall(r"[^\s,]+", text):
        total += 1
        total += word.count("_")
        total += max(0, len(re.findall(r"\d+", word)) - 1)
        total += len(re.findall(r"[^\w]", word))
        if len(word) > 12:
            total += 1
    return total


def classify_tag(tag):
    """Xếp một thẻ vào nhóm trong PROMPT_SECTIONS (không khớp → 'extra').

    Luật khớp được viết theo dạng người đọc (`blue eyes`, `from behind`) trong khi prompt
    Danbooru viết `blue_eyes`: giữa `_` và chữ không có ranh giới từ nên `\\beyes\\b`
    không khớp. Vì vậy thử trên bản đã đổi `_`/`-` thành dấu cách TRƯỚC, rồi mới thử trên
    bản gốc để các luật có dấu gạch dưới trong regex (vd. `rating_\\w+`) vẫn chạy.
    """
    core = tag_core(tag)
    spaced = re.sub(r"[_\-]+", " ", core)
    for section, pattern in _SECTION_RES:
        if pattern.search(spaced) or pattern.search(core):
            return section
    return "extra"


def sort_prompt_tags(tags):
    """Sắp xếp thẻ theo thứ tự chuẩn, khử trùng lặp, giữ thứ tự gốc trong nhóm."""
    buckets = {section: [] for section in PROMPT_SECTIONS}
    seen = set()
    duplicates = []
    for tag in tags:
        key = tag_core(tag).casefold()
        if not key:
            continue
        if key in seen:
            duplicates.append(tag)
            continue
        seen.add(key)
        buckets[classify_tag(tag)].append(tag)
    ordered = [tag for section in PROMPT_SECTIONS for tag in buckets[section]]
    return (
        ordered,
        duplicates,
        {section: list(buckets[section]) for section in PROMPT_SECTIONS},
    )


def structure_prompt(prompt, kind="character"):
    """Sắp lại prompt theo thứ tự chuẩn và thêm thẻ neo còn thiếu của khung đã chọn.

    Trả về (prompt, ghi chú) — chuỗi prompt chỉ để hiển thị trong ô prompt, người
    dùng sửa/xóa được trước khi tạo ảnh.
    """
    if kind not in PROMPT_SCAFFOLDS:
        raise ValueError("Chọn một khung prompt trong danh sách.")
    tags = split_tags(prompt)
    ordered, duplicates, buckets = sort_prompt_tags(tags)
    scaffold = PROMPT_SCAFFOLDS[kind]["anchors"]
    added = []
    for section, anchors in scaffold.items():
        if buckets.get(section):
            continue
        for anchor in anchors:
            if all(tag_core(tag).casefold() != anchor.casefold() for tag in ordered):
                ordered.append(anchor)
                added.append(anchor)
        ordered, _, _ = sort_prompt_tags(ordered)
    result = ", ".join(ordered)
    if len(result) > 2200:
        raise ValueError(
            "Prompt sau khi sắp xếp dài hơn 2200 ký tự. Hãy bỏ bớt thẻ ít quan trọng "
            "rồi sắp xếp lại."
        )
    used = [
        f"{SECTION_LABELS[section]} ({len(buckets[section])})"
        for section in PROMPT_SECTIONS
        if buckets[section]
    ]
    note = (
        f"**{PROMPT_SCAFFOLDS[kind]['label']}** · đã sắp xếp {len(ordered)} thẻ theo "
        "thứ tự: chủ thể → nhãn phân loại → ngoại hình → trang phục → tư thế → bố cục "
        "→ bối cảnh → ánh sáng → phong cách → thẻ khác → chất lượng → độ nét."
    )
    if used:
        note += "\n\n- Phân nhóm: " + ", ".join(used) + "."
    if added:
        note += f"\n- Thêm thẻ neo còn thiếu: `{'`, `'.join(added)}`."
    if duplicates:
        note += f"\n- Bỏ {len(duplicates)} thẻ trùng: `{'`, `'.join(duplicates)}`."
    note += "\n\nChuỗi này chỉ được ghi vào ô prompt để bạn xem và sửa; không có thẻ nào được thêm ngầm khi tạo ảnh."
    return result, note


def apply_negative_preset(preset_id, negative, mode=NEGATIVE_REPLACE):
    """Nạp một bộ negative tối ưu vào ô negative đang hiển thị."""
    preset = find_negative_preset(preset_id)
    tags = list(preset["tags"])
    if mode == NEGATIVE_APPEND:
        result = _add_prompt_tags(negative or "", tags)
    elif mode == NEGATIVE_REPLACE:
        result = ", ".join(tags)
    else:
        raise ValueError("Chọn cách áp dụng: ghi đè hoặc nối thêm.")
    if len(result) > 1700:
        raise ValueError(
            "Negative sau khi ghép dài hơn 1700 ký tự. Hãy dùng chế độ ghi đè."
        )
    action = "Nối thêm" if mode == NEGATIVE_APPEND else "Đã nạp"
    note = (
        f"{action} **{preset['label']}** ({len(tags)} thẻ) vào ô *Negative gửi model*. "
        f"{preset['when']}"
    )
    return result, note


def analyze_prompt(
    prompt,
    negative,
    steps=None,
    cfg=None,
    size=None,
    hires_scale=HIRES_OFF,
    hires_strength=HIRES_DEFAULT_STRENGTH,
):
    """Kiểm tra prompt/thông số theo khuyến nghị WAI v17 + thực hành Illustrious.

    Trả về (findings, stats); mỗi finding là (level, message) với level thuộc
    {"ok", "warn", "info"}. Chỉ đọc và báo cáo, không sửa prompt.
    """
    findings = []
    prompt = prompt if isinstance(prompt, str) else ""
    negative = negative if isinstance(negative, str) else ""
    tags = split_tags(prompt)
    cores = [tag_core(tag) for tag in tags]
    lowered = [core.casefold() for core in cores]
    tokens = estimate_tokens(prompt)
    neg_tags = split_tags(negative)
    quality_tags = [core for core, low in zip(cores, lowered) if low in QUALITY_TAG_SET]
    stats = {
        "tags": len(tags),
        "tokens": tokens,
        "negative_tags": len(neg_tags),
        "negative_tokens": estimate_tokens(negative),
    }
    if not tags:
        findings.append(("warn", "Prompt trống: model sẽ tạo ảnh gần như ngẫu nhiên."))
        return findings, stats

    if tokens <= SDXL_TOKEN_CHUNK:
        findings.append(
            ("ok", f"Prompt ≈ {tokens} token, vừa trong một khối 75 token của SDXL.")
        )
    elif tokens <= SDXL_TOKEN_CHUNK * 2:
        findings.append(
            (
                "warn",
                f"Prompt ≈ {tokens} token: vượt khối 75 token nên phần cuối bị đẩy "
                "sang khối sau và mất ưu tiên. Hãy bỏ thẻ ít quan trọng hoặc chèn "
                "`BREAK` (viết hoa) để tách khối có chủ đích.",
            )
        )
    else:
        findings.append(
            (
                "warn",
                f"Prompt ≈ {tokens} token — quá dài cho SDXL/Illustrious. Prompt dài "
                "làm mỗi thẻ yếu đi; giữ khoảng 20–40 thẻ quan trọng nhất.",
            )
        )

    if len(quality_tags) > 3:
        findings.append(
            (
                "warn",
                f"Có {len(quality_tags)} thẻ chất lượng ({', '.join(quality_tags[:5])}…). "
                "Nhà phát hành WAI v17 cảnh báo thừa thẻ chất lượng/thẩm mỹ làm giảm "
                "chất lượng và làm mờ ảnh; giữ 2–3 thẻ là đủ.",
            )
        )
    elif not quality_tags:
        findings.append(
            (
                "info",
                "Chưa có thẻ chất lượng. WAI v17 khuyến nghị thêm "
                "`masterpiece, best quality, amazing quality` (không cần nhiều hơn); "
                "Studio xếp khối này áp chót, ngay trước `absurdres`, để phần chi tiết "
                "nhân vật được CLIP đọc trước.",
            )
        )

    _, duplicates, _ = sort_prompt_tags(tags)
    if duplicates:
        findings.append(
            (
                "warn",
                f"Thẻ trùng lặp: `{'`, `'.join(duplicates)}`. Lặp thẻ tốn token mà "
                "không tăng trọng số; muốn nhấn mạnh hãy dùng `(thẻ:1.1)`.",
            )
        )

    negative_cores = {tag_core(tag).casefold() for tag in neg_tags}
    conflicts = sorted(
        {core for core, low in zip(cores, lowered) if low in negative_cores}
    )
    if conflicts:
        findings.append(
            (
                "warn",
                f"Vừa có trong prompt vừa có trong negative: `{'`, `'.join(conflicts)}`. "
                "Hai lệnh ngược nhau làm model dao động; giữ ở một phía.",
            )
        )

    pony = sorted({core for core in cores if PONY_ONLY_TAGS.match(core)})
    if pony:
        findings.append(
            (
                "warn",
                f"`{'`, `'.join(pony)}` là cú pháp của họ Pony (score_/source_). "
                "WAI-illustrious không dùng hệ này; thay bằng thẻ chất lượng chuẩn.",
            )
        )

    watermarkish = [
        core
        for core, low in zip(cores, lowered)
        if low in {"text", "watermark", "signature", "logo", "username", "artist name"}
    ]
    if watermarkish:
        findings.append(
            (
                "info",
                f"`{'`, `'.join(watermarkish)}` đang nằm trong prompt dương: trừ khi "
                "bạn thật sự muốn chữ/ký hiệu trong ảnh, hãy bỏ chúng và đưa sang "
                "negative.",
            )
        )

    if any(low == "detailed eyes" for low in lowered):
        findings.append(
            (
                "info",
                "`detailed eyes` được cộng đồng báo là gần như không tác dụng trên họ "
                "Illustrious. Muốn mắt đẹp hãy bật LoRA mắt và thêm trigger "
                "`perfect eyes`, hoặc mô tả cụ thể (`sharp focus on eyes`, màu mắt).",
            )
        )

    has_style = any(hint in low for low in lowered for hint in STYLE_TAG_HINTS)
    if not has_style:
        findings.append(
            (
                "info",
                "Chưa có thẻ phong cách. WAI v17 dễ ra chất 3D/nhựa khi thiếu thẻ "
                "chất lượng và phong cách: thêm `cel shading, anime illustration` "
                "(hoặc `realistic, 3d` vào negative nếu muốn giữ nét 2D).",
            )
        )
    else:
        findings.append(("ok", "Đã có thẻ phong cách/chất liệu vẽ trong prompt."))

    subject_tags = []
    seen_subjects = set()
    for core in cores:
        if SUBJECT_TAG_RE.search(core) and core.casefold() not in seen_subjects:
            seen_subjects.add(core.casefold())
            subject_tags.append(core)
    if not subject_tags:
        findings.append(
            (
                "info",
                "Chưa khai báo số lượng chủ thể. Illustrious bám rất tốt các thẻ "
                "`1girl, solo` / `1boy, solo` / `no humans`; thiếu chúng dễ thừa nhân vật.",
            )
        )
    else:
        findings.append(
            ("ok", f"Đã khai báo chủ thể: `{'`, `'.join(subject_tags[:3])}`.")
        )

    weights = []
    for tag in tags:
        match = WEIGHT_RE.match(tag.strip())
        if match:
            weights.append((match.group("body"), float(match.group("weight"))))
    heavy = [body for body, weight in weights if weight > 1.2 or weight < 0.5]
    if heavy:
        findings.append(
            (
                "warn",
                f"Trọng số ngoài khoảng an toàn: `{'`, `'.join(heavy)}`. Quá 1.2 dễ "
                "cháy nét/mất bố cục; ưu tiên mô tả bằng từ ngữ và chỉ nhấn 1.05–1.2.",
            )
        )
    elif weights:
        findings.append(
            ("ok", f"Dùng nhấn mạnh `(thẻ:trọng số)` hợp lý cho {len(weights)} thẻ.")
        )

    underscores = [core for core in cores if "_" in core]
    if underscores:
        findings.append(
            (
                "info",
                f"Có {len(underscores)} thẻ dùng dấu gạch dưới (`{'`, `'.join(underscores[:3])}`…). "
                "Illustrious đọc cả dạng cách (`long hair`) và tốn ít token hơn.",
            )
        )

    if not neg_tags:
        findings.append(
            (
                "info",
                "Negative trống. Họ Illustrious bám negative tốt; tối thiểu nên có "
                "`bad quality, worst quality, worst detail, sketch, censor`.",
            )
        )
    elif len(neg_tags) > 40 or stats["negative_tokens"] > 2 * SDXL_TOKEN_CHUNK:
        findings.append(
            (
                "warn",
                f"Negative có {len(neg_tags)} thẻ (≈{stats['negative_tokens']} token). "
                "Nhà phát hành WAI v17 cảnh báo negative quá dài làm giảm chất lượng "
                "và làm mờ ảnh; chọn một bộ theo mục đích rồi thêm vài thẻ riêng.",
            )
        )
    else:
        findings.append(("ok", f"Negative {len(neg_tags)} thẻ — độ dài hợp lý."))

    if steps is not None:
        try:
            value = float(steps)
        except (TypeError, ValueError):
            value = None
        if (
            value is not None
            and not RECOMMENDED_STEPS[0] <= value <= RECOMMENDED_STEPS[1]
        ):
            findings.append(
                (
                    "info",
                    f"Steps = {int(value)}: WAI v17 khuyến nghị "
                    f"{RECOMMENDED_STEPS[0]}–{RECOMMENDED_STEPS[1]} bước (Euler a). "
                    "Nhiều bước hơn chủ yếu tốn thời gian.",
                )
            )
    if cfg is not None:
        try:
            value = float(cfg)
        except (TypeError, ValueError):
            value = None
        if value is not None and not RECOMMENDED_CFG[0] <= value <= RECOMMENDED_CFG[1]:
            findings.append(
                (
                    "info",
                    f"CFG = {value:g}: WAI v17 khuyến nghị "
                    f"{RECOMMENDED_CFG[0]:g}–{RECOMMENDED_CFG[1]:g}. Trên 7 ảnh dễ "
                    "cháy màu, dưới 5 dễ lệch prompt.",
                )
            )
    if size:
        match = re.match(r"^(\d+)x(\d+)$", str(size))
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            if width * height < 900_000:
                findings.append(
                    (
                        "info",
                        f"Kích thước {width}×{height} nhỏ hơn ~1 MP mà SDXL được huấn "
                        "luyện. Hãy tạo ở 1024×1024/832×1216 rồi dùng hires 1.5–2×.",
                    )
                )
    if hires_scale != HIRES_OFF and not (
        RECOMMENDED_HIRES_STRENGTH[0]
        <= float(hires_strength)
        <= RECOMMENDED_HIRES_STRENGTH[1]
    ):
        findings.append(
            (
                "info",
                f"Hires strength = {float(hires_strength):g}: nhà phát hành gợi ý "
                f"{RECOMMENDED_HIRES_STRENGTH[0]}–{RECOMMENDED_HIRES_STRENGTH[1]} cho "
                "bước hires (thấp giữ bố cục, cao thêm chi tiết nhưng dễ đổi nét).",
            )
        )
    return findings, stats


def format_prompt_report(findings, stats):
    """Kết quả kiểm tra thành Markdown cho giao diện Studio."""
    icons = {"warn": "⚠️", "info": "ℹ️", "ok": "✅"}
    order = {"warn": 0, "info": 1, "ok": 2}
    lines = [
        "**🩺 Kết quả kiểm tra** · "
        f"{stats['tags']} thẻ · ≈{stats['tokens']} token (ước lượng) · "
        f"negative {stats['negative_tags']} thẻ"
    ]
    if not findings:
        lines.append("\n✅ Không phát hiện vấn đề.")
        return "\n".join(lines)
    for level, message in sorted(findings, key=lambda item: order.get(item[0], 3)):
        lines.append(f"\n{icons.get(level, 'ℹ️')} {message}")
    lines.append(
        "\n---\n**Thông số tham chiếu WAI v17:** Euler a · steps "
        f"{RECOMMENDED_STEPS[0]}–{RECOMMENDED_STEPS[1]} · CFG "
        f"{RECOMMENDED_CFG[0]:g}–{RECOMMENDED_CFG[1]:g} · ~1 MP (1024×1024, "
        "832×1216) · hires 1.5× với strength "
        f"{RECOMMENDED_HIRES_STRENGTH[0]}–{RECOMMENDED_HIRES_STRENGTH[1]}. "
        "Báo cáo này chỉ đọc prompt/thông số, không sửa gì cả."
    )
    return "\n".join(lines)


def run_prompt_check(
    prompt,
    negative,
    steps,
    cfg,
    size,
    hires_scale=HIRES_OFF,
    hires_strength=HIRES_DEFAULT_STRENGTH,
):
    """Sự kiện UI: kiểm tra prompt/thông số hiện tại, không đổi giá trị nào."""
    findings, stats = analyze_prompt(
        prompt,
        negative,
        steps=steps,
        cfg=cfg,
        size=size,
        hires_scale=hires_scale,
        hires_strength=hires_strength,
    )
    return format_prompt_report(findings, stats)


# ---------------------------------------------------------------------------
# AUTO-DETAILER · tự phát hiện mặt / bàn tay rồi inpaint lại đúng vùng đó.
#
# Tương đương ADetailer của A1111/ComfyUI, viết theo kiến trúc của Studio:
# detector YOLOv8 (cùng họ weight mà ADetailer dùng, tập huấn có cả ảnh anime)
# trả bounding box → cắt vùng → inpaint vùng cắt bằng chính pipeline inpaint
# đang có → dán lại với viền mềm. Hai ô prompt/negative đang hiển thị được dùng
# NGUYÊN VĂN: không có thẻ nào được thêm ngầm lúc chạy.
#
# Weight detector được ghim SHA-256 + số byte như checkpoint/LoRA: file tải về
# sai hash bị xóa và từ chối nạp.
# ---------------------------------------------------------------------------
DETAILER_OFF = "Tắt"
DETAILER_TARGETS = {
    DETAILER_OFF: (),
    "Mặt": ("face",),
    "Tay": ("hands",),
    "Mặt + tay": ("face", "hands"),
}
DETAILER_LABELS = {"face": "mặt", "hands": "tay"}
DETAILER_STRENGTH_RANGE = (0.2, 0.7)
DETAILER_DEFAULT_STRENGTH = 0.4
DETAILER_CONF_RANGE = (0.1, 0.9)
DETAILER_DEFAULT_CONF = 0.3
DETAILER_MAX_RANGE = (1, 4)
DETAILER_DEFAULT_MAX = 2
DETAILER_PADDING = 0.25  # nới bounding box thêm 25% mỗi chiều
DETAILER_FEATHER = 12  # px làm mềm mép khi dán vùng đã sửa trở lại
DETAILER_MIN_CROP = 512  # vùng nhỏ hơn sẽ được phóng lên trước khi inpaint
DETAILER_REPO = "Bingsu/adetailer"
DETAILER_REVISION = "c310c2160bcd1b249e93d1a1e4984949e5cc2d96"
DETAILER_CACHE = "/content/wai_detailer_cache"
DETAILER_MODELS = {
    "face": {
        "file": "face_yolov8n.pt",
        "sha256": "70b640f8f60b1cf0dcc72f30caf3da9495eb2fb6509da48c53374ad6806e6a9c",
        "size": 6_230_011,
        "version": "Bingsu/adetailer@c310c21",
    },
    "hands": {
        "file": "hand_yolov8n.pt",
        "sha256": "3991202eb69e9ddcb3b9ba80cdeb41e734ffaf844403d6c9f47d515cd88c6f29",
        "size": 6_237_883,
        "version": "Bingsu/adetailer@c310c21",
    },
}


def detailer_targets(label):
    """Nhãn trong dropdown → các vùng cần phát hiện."""
    if label not in DETAILER_TARGETS:
        raise ValueError("Chọn auto-detailer: Tắt, Mặt, Tay hoặc Mặt + tay.")
    return DETAILER_TARGETS[label]


def validate_detailer(label, strength, conf, max_regions):
    """Cấu hình auto-detailer hợp lệ, hoặc None khi người dùng để Tắt."""
    targets = detailer_targets(label)
    if not targets:
        return None
    return {
        "label": label,
        "targets": targets,
        "strength": _number(strength, "Detailer strength", *DETAILER_STRENGTH_RANGE),
        "conf": _number(conf, "Ngưỡng phát hiện", *DETAILER_CONF_RANGE),
        "max_regions": _number(
            max_regions, "Số vùng tối đa", *DETAILER_MAX_RANGE, integer=True
        ),
    }


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_realesrgan_weight(cache_dir=REAL_ESRGAN_CACHE, spec=REAL_ESRGAN_MODEL):
    """Tải weight Anime6B vào /content, kiểm tra kích thước và SHA-256."""
    import urllib.request

    folder = Path(cache_dir)
    cached = folder / spec["file"]
    if cached.is_file():
        if cached.stat().st_size != spec["size"] or sha256_file(cached) != spec["sha256"]:
            raise RuntimeError(
                f"Weight {spec['file']} trong {folder} sai kích thước/SHA-256. "
                "Hãy tự xóa file đó rồi chạy lại; hệ thống không ghi đè file hỏng."
            )
        return cached

    folder.mkdir(parents=True, exist_ok=True)
    partial = cached.with_name(cached.name + ".partial")
    partial.unlink(missing_ok=True)
    received = 0
    try:
        with urllib.request.urlopen(spec["url"], timeout=90) as response:
            with partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > spec["size"]:
                        raise RuntimeError("File tải về vượt kích thước đã ghim.")
                    output.write(chunk)
        if received != spec["size"] or sha256_file(partial) != spec["sha256"]:
            raise RuntimeError(
                f"Weight {spec['file']} tải về không khớp kích thước/SHA-256 "
                f"đã ghim ({spec['version']})."
            )
        os.replace(partial, cached)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError) and "không khớp" in str(exc):
            raise
        raise RuntimeError(
            f"Không tải/xác minh được Real-ESRGAN Anime6B: {exc}. "
            "Kiểm tra mạng Colab; các chế độ tạo ảnh thường vẫn dùng được."
        ) from exc
    return cached


def ensure_detailer_weight(kind, cache_dir=DETAILER_CACHE):
    """Tải weight YOLOv8 một lần rồi dùng lại; từ chối file sai hash."""
    if kind not in DETAILER_MODELS:
        raise ValueError("Detector không tồn tại: " + str(kind))
    spec = DETAILER_MODELS[kind]
    folder = Path(cache_dir)
    cached = folder / spec["file"]
    if cached.is_file():
        if (
            cached.stat().st_size != spec["size"]
            or sha256_file(cached) != spec["sha256"]
        ):
            raise RuntimeError(
                f"File detector {spec['file']} trong {folder} sai kích thước/SHA-256. "
                "Hãy tự xóa file đó rồi chạy lại; hệ thống không ghi đè file hỏng."
            )
        return cached
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - ô 2 đã ghim huggingface_hub
        raise RuntimeError(
            "Thiếu huggingface_hub để tải weight auto-detailer. Chạy lại ô 2."
        ) from exc
    folder.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=DETAILER_REPO,
            filename=spec["file"],
            revision=DETAILER_REVISION,
            local_dir=str(folder),
            token=False,
        )
    )
    if (
        downloaded.stat().st_size != spec["size"]
        or sha256_file(downloaded) != spec["sha256"]
    ):
        downloaded.unlink(missing_ok=True)
        raise RuntimeError(
            f"Weight {spec['file']} tải về không khớp SHA-256 đã ghim "
            f"({spec['version']}); đã xóa file và không nạp. Kiểm tra mạng rồi thử lại."
        )
    return downloaded


def load_yolo_model(path):
    """Nạp weight YOLOv8; báo lỗi rõ ràng nếu ô 2 chưa cài ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài `ultralytics` nên auto-detailer không chạy được. Chọn "
            "Runtime → Restart runtime → Run all để ô 2 cài lại thư viện, hoặc "
            "để auto-detailer ở mức Tắt."
        ) from exc
    return YOLO(str(path))


def pad_box(box, size, padding=DETAILER_PADDING):
    """Nới bounding box, kẹp trong ảnh, cạnh chia hết cho 8 và ≥ 64 px."""
    width, height = size
    left, top, right, bottom = box
    x1, x2 = sorted((min(max(0.0, left), width), min(max(0.0, right), width)))
    y1, y2 = sorted((min(max(0.0, top), height), min(max(0.0, bottom), height)))
    grow_x = (x2 - x1) * padding
    grow_y = (y2 - y1) * padding
    x1 = max(0, int(x1 - grow_x) // 8 * 8)
    y1 = max(0, int(y1 - grow_y) // 8 * 8)
    x2 = min(width, math.ceil((x2 + grow_x) / 8) * 8)
    y2 = min(height, math.ceil((y2 + grow_y) / 8) * 8)
    if x2 - x1 < 64:
        x2 = min(width, x1 + 64)
        x1 = max(0, x2 - 64)
    if y2 - y1 < 64:
        y2 = min(height, y1 + 64)
        y1 = max(0, y2 - 64)
    return (x1, y1, x2, y2)


def build_detailer_detector(kind, cache_dir=DETAILER_CACHE):
    """Weight đã xác minh hash → detector YOLOv8 sẵn sàng predict."""
    return load_yolo_model(ensure_detailer_weight(kind, cache_dir))


def detect_detail_regions(
    image,
    targets,
    conf,
    max_regions,
    detector=build_detailer_detector,
    cache_dir=DETAILER_CACHE,
):
    """Trả về [(box, loại vùng, độ tin cậy)] theo độ tin cậy giảm dần."""
    found = []
    for kind in targets:
        model = detector(kind, cache_dir)
        for result in model.predict(source=image, conf=conf, verbose=False):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                coords = getattr(box, "xyxy", None)
                scores = getattr(box, "conf", None)
                if coords is None or len(coords) == 0:
                    continue
                x1, y1, x2, y2 = (float(value) for value in list(coords[0])[:4])
                confidence = (
                    float(scores[0]) if scores is not None and len(scores) else conf
                )
                found.append((pad_box((x1, y1, x2, y2), image.size), kind, confidence))
    found.sort(key=lambda item: item[2], reverse=True)
    return found[:max_regions]


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
        lines = [str(prompt)]
        negative = row.get("negative") or row.get("negative_prompt")
        if negative:
            lines.append("Negative: " + " ".join(str(negative).split()))
        for key, label in (
            ("steps", "Steps"),
            ("cfg", "CFG"),
            ("size", "Size"),
            ("seed", "Seed"),
        ):
            if row.get(key) not in (None, ""):
                lines.append(f"{label}: {row[key]}")
        entries.append((title, lines))
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


def prompt_library_choices(items):
    """Properties for the library picker (a Radio list, dễ chạm trên điện thoại)."""
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


def _hires_size(width, height, scale_label):
    """Return (width, height, clamped) for the hires pass, or None when it is off.

    Kích thước là bội số của 8 (yêu cầu của SDXL/VAE). Nếu bản phóng to vượt
    HIRES_MAX_PIXELS thì hệ số được giảm cho vừa thay vì báo lỗi.
    """
    if scale_label not in HIRES_SCALES:
        raise ValueError("Chọn độ phân giải cao có sẵn trong danh sách.")
    scale = HIRES_SCALES[scale_label]
    if scale <= 1:
        return None
    clamped = False
    if width * height * scale * scale > HIRES_MAX_PIXELS:
        scale = math.sqrt(HIRES_MAX_PIXELS / (width * height))
        clamped = True
    if scale < 1.05:
        return None
    return (
        max(width, int(width * scale) // 8 * 8),
        max(height, int(height * scale) // 8 * 8),
        clamped,
    )


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


def _build_realesrgan_model(torch):
    """Build the x4 Anime 6B RRDBNet without installing legacy ESRGAN packages.

    Layer names and dimensions match the official BasicSR RRDBNet checkpoint.
    The weight file is pinned separately and never loaded until its SHA-256 passes.
    """
    nn = torch.nn
    functional = torch.nn.functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(96, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 32, 3, 1, 1)
            self.conv4 = nn.Conv2d(160, 32, 3, 1, 1)
            self.conv5 = nn.Conv2d(192, 64, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value):
            first = self.lrelu(self.conv1(value))
            second = self.lrelu(self.conv2(torch.cat((value, first), dim=1)))
            third = self.lrelu(
                self.conv3(torch.cat((value, first, second), dim=1))
            )
            fourth = self.lrelu(
                self.conv4(torch.cat((value, first, second, third), dim=1))
            )
            fifth = self.conv5(
                torch.cat((value, first, second, third, fourth), dim=1)
            )
            return fifth * 0.2 + value

    class ResidualInResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, value):
            output = self.rdb1(value)
            output = self.rdb2(output)
            output = self.rdb3(output)
            return output * 0.2 + value

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(
                *(ResidualInResidualDenseBlock() for _ in range(6))
            )
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value):
            features = self.conv_first(value)
            body = self.conv_body(self.body(features))
            features = features + body
            features = self.lrelu(
                self.conv_up1(
                    functional.interpolate(features, scale_factor=2, mode="nearest")
                )
            )
            features = self.lrelu(
                self.conv_up2(
                    functional.interpolate(features, scale_factor=2, mode="nearest")
                )
            )
            return self.conv_last(self.lrelu(self.conv_hr(features)))

    return RRDBNet()


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
        detailer_cache=DETAILER_CACHE,
        detailer_detector=build_detailer_detector,
        upscaler_cache=REAL_ESRGAN_CACHE,
        realesrgan_upscaler=None,
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
        self.detailer_cache = str(detailer_cache)
        self.detailer_detector = detailer_detector
        self.upscaler_cache = str(upscaler_cache)
        # A test hook avoids downloads; production lazily builds the pinned x4 model.
        self.realesrgan_upscaler = realesrgan_upscaler
        self._realesrgan_model = None
        self._realesrgan_device = None
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

    def _detail_pass(self, image, positive, negative, steps, cfg, seed, spec, choices):
        """Auto-detailer: inpaint lại từng vùng mặt/tay vừa phát hiện được."""
        from PIL import Image, ImageDraw, ImageFilter

        regions = detect_detail_regions(
            image,
            spec["targets"],
            spec["conf"],
            spec["max_regions"],
            detector=self.detailer_detector,
            cache_dir=self.detailer_cache,
        )
        wanted = "/".join(DETAILER_LABELS[kind] for kind in spec["targets"])
        if not regions:
            return image, (
                f"auto-detailer không thấy {wanted} nào ở ngưỡng "
                f"{spec['conf']:g} — hãy hạ ngưỡng phát hiện rồi thử lại"
            )
        base = image.convert("RGB")
        notes = []
        for index, (box, kind, confidence) in enumerate(regions):
            x1, y1, x2, y2 = box
            crop = base.crop(box)
            scale = 1.0
            if max(crop.size) < DETAILER_MIN_CROP:
                scale = DETAILER_MIN_CROP / max(crop.size)
                crop = crop.resize(
                    (
                        max(64, round(crop.width * scale / 8) * 8),
                        max(64, round(crop.height * scale / 8) * 8),
                    ),
                    Image.Resampling.LANCZOS,
                )
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).rectangle(
                (0, 0, crop.width - 1, crop.height - 1), fill=255
            )
            refined = self._infer_with_retry(
                "inpaint",
                crop,
                mask,
                positive,
                negative,
                crop.width,
                crop.height,
                steps,
                cfg,
                seed + index,
                spec["strength"],
                choices,
            )
            if scale != 1.0:
                box_size = (x2 - x1, y2 - y1)
                refined = refined.resize(box_size, Image.Resampling.LANCZOS)
                mask = mask.resize(box_size, Image.Resampling.LANCZOS)
            blend = mask.filter(ImageFilter.GaussianBlur(radius=DETAILER_FEATHER))
            base.paste(
                Image.composite(refined.convert("RGB"), base.crop(box), blend),
                (x1, y1),
            )
            notes.append(
                f"{DETAILER_LABELS[kind]} {x2 - x1}×{y2 - y1}px "
                f"@({x1},{y1}) · tin cậy {confidence:.2f}"
            )
        return base, "auto-detailer đã sửa " + "; ".join(notes)

    def _load_realesrgan_model(self):
        """Lazily load the pinned anime upscaler, preferring CUDA with CPU fallback."""
        if self._realesrgan_model is not None:
            return self._realesrgan_model, self._realesrgan_device

        weight = ensure_realesrgan_weight(self.upscaler_cache)
        try:
            try:
                checkpoint = self.torch.load(
                    str(weight), map_location="cpu", weights_only=True
                )
            except TypeError:  # PyTorch before the weights_only argument existed.
                checkpoint = self.torch.load(str(weight), map_location="cpu")
            model = _build_realesrgan_model(self.torch)
            state = None
            if isinstance(checkpoint, dict):
                state = checkpoint.get("params_ema")
                if state is None:
                    state = checkpoint.get("params")
                if state is None and "conv_first.weight" in checkpoint:
                    state = checkpoint
            if not isinstance(state, dict):
                raise ValueError("Checkpoint không có state dict params_ema/params.")
            model.load_state_dict(state, strict=True)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Không nạp được kiến trúc/weight Real-ESRGAN Anime6B đã xác minh."
            ) from exc

        device = "cuda" if self.torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                model = model.to(device).half()
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                model = model.to("cpu").float()
                device = "cpu"
        else:
            model = model.to("cpu").float()
        self._realesrgan_model = model
        self._realesrgan_device = device
        return model, device

    def _realesrgan_x4_on_device(self, image, model, device):
        """Run the official x4 RRDBNet in overlapping tiles; return an RGB PIL image."""
        import numpy as np
        from PIL import Image

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        # Real-ESRGAN's reference inference path feeds OpenCV BGR tensors.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        dtype = self.torch.float16 if device == "cuda" else self.torch.float32
        source = self.torch.from_numpy(bgr.transpose(2, 0, 1).copy())
        source = source.unsqueeze(0).to(device=device, dtype=dtype).div_(255.0)
        pad = 10
        padded = self.torch.nn.functional.pad(
            source, (pad, pad, pad, pad), mode="reflect"
        )
        del source

        result_bgr = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
        tile_size = REAL_ESRGAN_TILE_SIZE
        tile_pad = REAL_ESRGAN_TILE_PAD
        with self.torch.inference_mode():
            for top in range(0, height, tile_size):
                bottom = min(top + tile_size, height)
                input_top = max(0, pad + top - tile_pad)
                input_bottom = min(padded.shape[-2], pad + bottom + tile_pad)
                for left in range(0, width, tile_size):
                    right = min(left + tile_size, width)
                    input_left = max(0, pad + left - tile_pad)
                    input_right = min(padded.shape[-1], pad + right + tile_pad)
                    tile = padded[..., input_top:input_bottom, input_left:input_right]
                    tile_output = model(tile)
                    expected_shape = (
                        (input_bottom - input_top) * 4,
                        (input_right - input_left) * 4,
                    )
                    if tuple(tile_output.shape[-2:]) != expected_shape:
                        raise RuntimeError(
                            "Real-ESRGAN trả về kích thước tile không phải x4."
                        )
                    crop_top = (pad + top - input_top) * 4
                    crop_left = (pad + left - input_left) * 4
                    crop_bottom = crop_top + (bottom - top) * 4
                    crop_right = crop_left + (right - left) * 4
                    core = tile_output[
                        0, :, crop_top:crop_bottom, crop_left:crop_right
                    ].detach()
                    patch = (
                        core.float()
                        .clamp_(0, 1)
                        .mul_(255)
                        .round_()
                        .to(device="cpu", dtype=self.torch.uint8)
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    result_bgr[top * 4 : bottom * 4, left * 4 : right * 4] = patch

        result_rgb = np.ascontiguousarray(result_bgr[:, :, ::-1])
        return Image.fromarray(result_rgb, mode="RGB")

    def _realesrgan_x4(self, image):
        model, device = self._load_realesrgan_model()
        try:
            return self._realesrgan_x4_on_device(image, model, device)
        except self.torch.cuda.OutOfMemoryError as exc:
            if device != "cuda":
                raise RuntimeError(
                    "Không đủ bộ nhớ để chạy Real-ESRGAN Anime6B trên CPU."
                ) from exc
            self.torch.cuda.empty_cache()
            self._realesrgan_model = model.to("cpu").float()
            self._realesrgan_device = "cpu"
            try:
                return self._realesrgan_x4_on_device(
                    image, self._realesrgan_model, "cpu"
                )
            except self.torch.cuda.OutOfMemoryError as cpu_exc:
                raise RuntimeError(
                    "Real-ESRGAN Anime6B hết bộ nhớ cả trên CPU; hãy dùng ảnh nhỏ hơn."
                ) from cpu_exc

    def _upscale_with_realesrgan(self, image, size):
        """Run Anime6B x4, then downsample to the selected exact output dimensions."""
        from PIL import Image

        if self.realesrgan_upscaler is not None:
            result = self.realesrgan_upscaler(image.convert("RGB"), size)
        else:
            result = self._realesrgan_x4(image).resize(
                size, Image.Resampling.LANCZOS
            )
        if not isinstance(result, Image.Image) or result.size != tuple(size):
            raise RuntimeError(
                "Real-ESRGAN không trả về đúng kích thước đã chọn; ảnh không được lưu."
            )
        return result.convert("RGB")

    def _hires_pass(
        self,
        image,
        size,
        positive,
        negative,
        steps,
        cfg,
        seed,
        strength,
        choices,
    ):
        """Anime6B upscale followed by img2img refinement at the chosen scale."""
        # VAE tiling giữ đỉnh VRAM thấp khi giải mã ảnh > 1 MP (pipe SDXL thật có
        # hàm này; dùng chung VAE với pipeline ảnh → ảnh nên bật một lần là đủ).
        enable_tiling = getattr(self.pipe, "enable_vae_tiling", None)
        if callable(enable_tiling):
            enable_tiling()
        upscaled = self._upscale_with_realesrgan(image, size)
        return self._infer_with_retry(
            "image",
            upscaled,
            None,
            positive,
            negative,
            size[0],
            size[1],
            steps,
            cfg,
            seed,
            strength,
            choices,
        )

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
        hires_scale=HIRES_OFF,
        hires_strength=HIRES_DEFAULT_STRENGTH,
        detailer=DETAILER_OFF,
        detailer_strength=DETAILER_DEFAULT_STRENGTH,
        detailer_conf=DETAILER_DEFAULT_CONF,
        detailer_max=DETAILER_DEFAULT_MAX,
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
        # Kiểm tra cấu hình detailer trước cả phần kiểm tra ảnh nguồn: lỗi cấu hình
        # phải báo ngay, không để người dùng tải ảnh lên rồi mới bị từ chối.
        detailer_spec = validate_detailer(
            detailer, detailer_strength, detailer_conf, detailer_max
        )
        if detailer_spec and mode == "inpaint":
            raise ValueError(
                "Chế độ sửa vùng đã vẽ lại đúng vùng bạn tô rồi; hãy để auto-detailer "
                "ở mức Tắt, hoặc dùng Ảnh → ảnh nếu muốn tự sửa mặt/tay."
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
        elif mode == "upscale":
            from PIL import ImageOps

            if not isinstance(source, Image.Image):
                raise ValueError("Tải ảnh lên để phóng to.")
            if source.width * source.height > 20_000_000:
                raise ValueError("Ảnh nguồn tối đa 20 MP.")
            source = ImageOps.exif_transpose(source).convert("RGB")
            if (
                min(source.size) < 256
                or max(source.width / source.height, source.height / source.width)
                > 1.75
            ):
                raise ValueError(
                    "Ảnh cần có cạnh ngắn ≥256 px và tỷ lệ không quá 1,75:1 để phóng to."
                )
            width = round(source.width / 8) * 8
            height = round(source.height / 8) * 8
            if (width, height) != source.size:  # SDXL/VAE cần bội số của 8
                source = source.resize((width, height), Image.Resampling.LANCZOS)
            strength = None
            if hires_scale == HIRES_OFF:
                raise ValueError("Chọn hệ số phóng trong danh sách từ 1.25× đến 2×.")
        else:
            raise ValueError("Chế độ tạo ảnh không được hỗ trợ.")

        hires = None
        if mode in ("text", "image", "upscale"):
            hires = _hires_size(width, height, hires_scale)
            if hires is None and mode == "upscale":
                raise ValueError(
                    f"Ảnh đã ở sát giới hạn ≈{HIRES_MAX_PIXELS / 1e6:.1f} MP, "
                    "không thể phóng to thêm."
                )
            if hires:
                hires_strength = _number(
                    hires_strength, "Hires strength", *HIRES_STRENGTH_RANGE
                )
        elif hires_scale != HIRES_OFF:
            raise ValueError("Độ phân giải cao không áp dụng cho chế độ sửa vùng.")
        base_size = (width, height)
        final_size = hires[:2] if hires else base_size

        paths = []
        gallery = []
        selected = []
        detail_notes = []
        with self.lock:  # one GPU pipeline, even if separate UI actions are clicked
            for index in range(count):
                image_seed = (
                    secrets.randbelow(2**32) if seed == -1 else (seed + index) % 2**32
                )
                if mode == "upscale":
                    image = source  # không tạo lại: chỉ phóng to rồi tinh chỉnh
                else:
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
                if detailer_spec:
                    # Sửa mặt/tay ở độ phân giải gốc, trước hires fix: vùng cắt nhỏ
                    # nên rẻ hơn và ảnh đã sạch lỗi mới được phóng to.
                    image, detail_note = self._detail_pass(
                        image,
                        positive,
                        negative,
                        steps,
                        cfg,
                        image_seed,
                        detailer_spec,
                        choices,
                    )
                    detail_notes.append(detail_note)
                if hires:
                    image = self._hires_pass(
                        image,
                        final_size,
                        positive,
                        negative,
                        steps,
                        cfg,
                        image_seed,
                        hires_strength,
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
                    "width": final_size[0],
                    "height": final_size[1],
                    "steps": steps,
                    "cfg": cfg,
                    "strength": strength if mode in ("image", "inpaint") else None,
                    "hires": (
                        {
                            "scale": hires_scale,
                            "upscaler": {
                                "name": REAL_ESRGAN_MODEL["name"],
                                "version": REAL_ESRGAN_MODEL["version"],
                                "sha256": REAL_ESRGAN_MODEL["sha256"],
                            },
                            "base_width": base_size[0],
                            "base_height": base_size[1],
                            "strength": hires_strength,
                        }
                        if hires
                        else None
                    ),
                    "detailer": (
                        {
                            "targets": list(detailer_spec["targets"]),
                            "strength": detailer_spec["strength"],
                            "conf": detailer_spec["conf"],
                            "max_regions": detailer_spec["max_regions"],
                            "weights": [
                                DETAILER_MODELS[kind]["version"]
                                for kind in detailer_spec["targets"]
                            ],
                        }
                        if detailer_spec
                        else None
                    ),
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
                gallery.append(
                    (str(path), f"Seed {image_seed} · {final_size[0]}×{final_size[1]}")
                )
                selected.append(str(image_seed))
        status = (
            f"✅ Đã tạo {len(paths)} ảnh · seed: {', '.join(selected)}"
            f" · chế độ: {self.execution_mode} · đã lưu: {Path(paths[0]).parent}"
        )
        if hires:
            status += (
                f" · {'phóng to' if mode == 'upscale' else 'hires fix'} "
                f"{base_size[0]}×{base_size[1]} → "
                f"{final_size[0]}×{final_size[1]}"
                f" · upscaler: {REAL_ESRGAN_MODEL['name']}"
            )
            if hires[2]:
                status += (
                    f" (đã giảm hệ số để không vượt {HIRES_MAX_PIXELS / 1e6:.1f} MP)"
                )
        if detailer_spec:
            status += f" · {detailer_spec['label'].lower()} → {detail_notes[-1]}"
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
        hires_scale=HIRES_OFF,
        hires_strength=HIRES_DEFAULT_STRENGTH,
        detailer=DETAILER_OFF,
        detailer_strength=DETAILER_DEFAULT_STRENGTH,
        detailer_conf=DETAILER_DEFAULT_CONF,
        detailer_max=DETAILER_DEFAULT_MAX,
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
            hires_scale,
            hires_strength,
            detailer,
            detailer_strength,
            detailer_conf,
            detailer_max,
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
        hires_scale=HIRES_OFF,
        hires_strength=HIRES_DEFAULT_STRENGTH,
        detailer=DETAILER_OFF,
        detailer_strength=DETAILER_DEFAULT_STRENGTH,
        detailer_conf=DETAILER_DEFAULT_CONF,
        detailer_max=DETAILER_DEFAULT_MAX,
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
            hires_scale,
            hires_strength,
            detailer,
            detailer_strength,
            detailer_conf,
            detailer_max,
        )

    def upscale(
        self,
        source,
        hires_scale,
        hires_strength,
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
        """Anime6B upscale an existing image, then refine with img2img."""
        return self._generate(
            "upscale",
            source,
            None,
            None,
            0,
            None,
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
            hires_scale,
            hires_strength,
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


# CSV is data only: lazily downloaded, pinned and verified; never executed.
TAG_CSV_NAME = "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"
TAG_CSV_SHA256 = "9e51c0bde90e0d9535ce8d4ce52c9e422ac347926bb493546ebe42f6f3b786df"
# Keep accepting the older four-column catalog when a user uploads it manually.
TAG_CSV_LEGACY_SHA256 = "287bb5ad86fcc56f535b9ebae8d3e696e5ff3fa5c057d6fa6884282fb083e738"
TAG_CSV_MAX_BYTES = 32_000_000
TAG_CSV_URL = (
    "https://raw.githubusercontent.com/manhlee1196-boop/ai-anime/"
    "96cc543afa7664345129e5bbef3bd57167365091/" + TAG_CSV_NAME
)
TAG_CATEGORIES = {
    "0": "Danbooru · Chung", "1": "Danbooru · Họa sĩ",
    "3": "Danbooru · Tác phẩm", "4": "Danbooru · Nhân vật",
    "5": "Danbooru · Metadata", "7": "e621 · Chung",
    "8": "e621 · Họa sĩ", "9": "e621 · Người đóng góp",
    "10": "e621 · Tác phẩm", "11": "e621 · Nhân vật",
    "12": "e621 · Loài", "14": "e621 · Metadata", "15": "e621 · Lore",
}
TAG_THEMES = {
    "Ngoại hình": r"hair|eyes?|ears?|face|skin|fur|wings?|tail|horns?",
    "Trang phục & phụ kiện": r"dress|shirt|skirt|uniform|clothes|clothing|hat|ribbon|armor|jewelry|boots?|gloves?",
    "Biểu cảm & tư thế": r"smile|blush|crying|laughing|looking|standing|sitting|lying|pose|running|wink",
    "Bối cảnh & thiên nhiên": r"background|sky|cloud|forest|tree|flower|ocean|beach|city|street|indoors|outdoors|room|mountain|snow|rain",
    "Ánh sáng & màu sắc": r"light|shadow|sunset|sunrise|night|glow|neon|monochrome|color|colour",
    "Bố cục & kỹ thuật": r"view|portrait|close.up|full.body|perspective|focus|sketch|painting|watercolor|resolution|highres|absurdres|digital",
}
# Vietnamese captions are display/search metadata only, never generation tokens.
TAG_VI_LABELS = {
    "solo": "Một nhân vật", "1girl": "Một nhân vật nữ", "1boy": "Một nhân vật nam",
    "2girls": "Hai nhân vật nữ", "2boys": "Hai nhân vật nam",
    "multiple_girls": "Nhiều nhân vật nữ", "multiple_boys": "Nhiều nhân vật nam",
    "male": "Giới tính nam", "female": "Giới tính nữ", "duo": "Hai nhân vật",
    "anthro": "Động vật nhân hóa", "mammal": "Động vật có vú",
    "canid": "Họ chó", "canine": "Loài chó", "felid": "Họ mèo", "feline": "Loài mèo",
    "long_hair": "Tóc dài", "short_hair": "Tóc ngắn", "medium_hair": "Tóc dài vừa",
    "very_long_hair": "Tóc rất dài", "curly_hair": "Tóc xoăn", "wavy_hair": "Tóc gợn sóng",
    "straight_hair": "Tóc thẳng", "messy_hair": "Tóc rối", "ponytail": "Tóc đuôi ngựa",
    "twintails": "Tóc buộc hai bên", "braid": "Tóc tết", "bangs": "Tóc mái",
    "hair": "Tóc", "fur": "Lông thú", "tail": "Đuôi", "wings": "Đôi cánh",
    "horns": "Sừng", "animal_ears": "Tai động vật", "cat_ears": "Tai mèo",
    "looking_at_viewer": "Nhìn người xem", "looking_away": "Nhìn sang chỗ khác",
    "smile": "Mỉm cười", "blush": "Đỏ mặt", "open_mouth": "Miệng mở",
    "closed_mouth": "Miệng khép", "closed_eyes": "Nhắm mắt", "wink": "Nháy mắt",
    "crying": "Đang khóc", "tears": "Nước mắt", "laughing": "Đang cười",
    "angry": "Tức giận", "surprised": "Ngạc nhiên", "sleeping": "Đang ngủ",
    "standing": "Đang đứng", "sitting": "Đang ngồi", "lying": "Đang nằm",
    "walking": "Đang đi bộ", "running": "Đang chạy", "jumping": "Đang nhảy",
    "arms_up": "Giơ tay lên", "crossed_arms": "Khoanh tay", "hand_on_hip": "Tay chống hông",
    "dress": "Váy liền", "shirt": "Áo sơ mi", "skirt": "Chân váy",
    "clothing": "Trang phục", "school_uniform": "Đồng phục học sinh",
    "jacket": "Áo khoác", "kimono": "Áo kimono", "armor": "Áo giáp",
    "hat": "Mũ", "ribbon": "Ruy băng", "gloves": "Găng tay", "boots": "Ủng",
    "glasses": "Kính mắt", "earrings": "Khuyên tai", "necklace": "Vòng cổ",
    "simple_background": "Nền đơn giản", "transparent_background": "Nền trong suốt",
    "indoors": "Trong nhà", "outdoors": "Ngoài trời", "sky": "Bầu trời",
    "cloud": "Mây", "clouds": "Những đám mây", "forest": "Rừng", "tree": "Cây",
    "flower": "Hoa", "flowers": "Những bông hoa", "cherry_blossoms": "Hoa anh đào",
    "ocean": "Đại dương", "beach": "Bãi biển", "city": "Thành phố",
    "street": "Đường phố", "room": "Căn phòng", "mountain": "Núi",
    "snow": "Tuyết", "rain": "Mưa", "water": "Nước", "night": "Ban đêm",
    "sunset": "Hoàng hôn", "sunrise": "Bình minh", "sunlight": "Ánh nắng",
    "moonlight": "Ánh trăng", "backlighting": "Ánh sáng ngược", "shadow": "Bóng đổ",
    "monochrome": "Đơn sắc", "greyscale": "Thang xám", "blurry": "Mờ nhòe",
    "portrait": "Chân dung", "close-up": "Cận cảnh", "full_body": "Toàn thân",
    "upper_body": "Nửa thân trên", "from_above": "Góc nhìn từ trên",
    "from_below": "Góc nhìn từ dưới", "from_side": "Góc nhìn bên cạnh",
    "from_behind": "Góc nhìn từ phía sau", "depth_of_field": "Độ sâu trường ảnh",
    "sketch": "Phác thảo", "watercolor": "Màu nước", "lineart": "Nét vẽ",
    "highres": "Độ phân giải cao", "hi_res": "Độ phân giải cao",
    "absurdres": "Độ phân giải cực cao", "absurd_res": "Độ phân giải cực cao",
    "lowres": "Độ phân giải thấp", "animated": "Ảnh động", "comic": "Truyện tranh",
    "text": "Chữ trong ảnh", "signature": "Chữ ký", "watermark": "Dấu bản quyền",
    "translation_request": "Yêu cầu dịch", "translated": "Đã dịch",
    "official_art": "Tranh chính thức", "original": "Sáng tác gốc",
    "breasts": "Ngực", "large_breasts": "Ngực lớn", "small_breasts": "Ngực nhỏ",
    "cleavage": "Khe ngực", "nude": "Khỏa thân", "nipples": "Núm vú",
    "genitals": "Bộ phận sinh dục", "nsfw": "Nội dung nhạy cảm",
}
TAG_VI_LABELS.update({
    # Đợt 11 — thẻ nóng ghép máy không tự nhiên nên viết tay.
    "hand_on_another's_shoulder": "Đặt tay lên vai người khác",
    "hands_on_another's_shoulders": "Đặt hai tay lên vai người khác",
    "hand_on_another's_chest": "Đặt tay lên ngực người khác",
    "hand_on_another's_stomach": "Đặt tay lên bụng người khác",
    "hand_on_another's_waist": "Đặt tay lên eo người khác",
    "hand_on_another's_thigh": "Đặt tay lên đùi người khác",
    "hand_on_another's_cheek": "Đặt tay lên má người khác",
    "hands_on_another's_cheeks": "Đặt hai tay lên má người khác",
    "hand_on_another's_chin": "Đặt tay lên cằm người khác",
    "hand_on_another's_neck": "Đặt tay lên cổ người khác",
    "hand_on_another's_back": "Đặt tay lên lưng người khác",
    "hand_on_another's_ass": "Đặt tay lên mông người khác",
    "hand_on_another's_hip": "Đặt tay lên hông người khác",
    "hand_on_another's_leg": "Đặt tay lên chân người khác",
    "hand_on_another's_arm": "Đặt tay lên cánh tay người khác",
    "head_on_another's_shoulder": "Tựa đầu lên vai người khác",
    "grabbing_another's_ass": "Bóp mông người khác",
    "grabbing_another's_arm": "Nắm cánh tay người khác",
    "grabbing_another's_thighs": "Nắm đùi người khác",
    "grabbing_another's_penis": "Nắm dương vật người khác",
    "holding_another's_wrist": "Nắm cổ tay người khác",
    "holding_another's_arm": "Nắm cánh tay người khác",
    "holding_another's_leg": "Giữ chân người khác",
    "holding_another's_face": "Ôm mặt người khác",
    "spreading_another's_legs": "Kéo rộng chân người khác",
    "spreading_another's_pussy": "Mở âm hộ người khác",
    "finger_in_another's_mouth": "Đưa ngón tay vào miệng người khác",
    "arm_around_another's_waist": "Vòng tay ôm eo người khác",
    "arm_around_another's_shoulder": "Vòng tay qua vai người khác",
    "licking_another's_face": "Liếm mặt người khác",
    "kissing_on_the_cheek": "Hôn lên má",
    "missionary_position": "Tư thế truyền giáo",
    "reverse_missionary_position": "Tư thế truyền giáo ngược",
    "reverse_cowgirl_position": "Tư thế cưỡi ngựa ngược",
    "squatting_cowgirl_position": "Tư thế cưỡi ngựa ngồi xổm",
    "doggystyle_position": "Tư thế doggy",
    "spooning_position": "Tư thế thuyền chung thủy",
    "sixty_nine_position": "Tư thế 69",
    "kneeling_oral_position": "Tư thế quỳ bằng miệng",
    "stand_and_carry_position": "Tư thế bế đứng",
    "prone_bone_position": "Tư thế nằm sấp",
    "fully_clothed_female": "Nữ mặc kín đồ",
    "fully_clothed_male": "Nam mặc kín đồ",
    "fully_clothed_anthro": "Nhân thú mặc kín đồ",
    "fully_clothed_person": "Nhân vật mặc kín đồ",
    "mostly_nude_female": "Nữ gần khỏa thân",
    "mostly_nude_male": "Nam gần khỏa thân",
    "mostly_nude_anthro": "Nhân thú gần khỏa thân",
    "partially_clothed_female": "Nữ mặc hở một phần",
    "partially_clothed_male": "Nam mặc hở một phần",
    "partially_clothed_anthro": "Nhân thú mặc hở một phần",
    "topless_anthro": "Nhân thú hở trên",
    "topless_animal": "Động vật hở trên",
    "bottomless_female": "Nữ hở dưới",
    "bottomless_male": "Nam hở dưới",
    "bottomless_anthro": "Nhân thú hở dưới",
    "morbidly_obese_female": "Nữ béo bệnh lý",
    "morbidly_obese_male": "Nam béo bệnh lý",
    "morbidly_obese_anthro": "Nhân thú béo bệnh lý",
    "anatomically_correct_genitalia": "Bộ phận sinh dục đúng giải phẫu",
    "anatomically_correct_penis": "Dương vật đúng giải phẫu",
    "anatomically_correct_vulva": "Âm hộ đúng giải phẫu",
    "anatomically_correct_breasts": "Ngực đúng giải phẫu",
    "anatomically_correct_anus": "Hậu môn đúng giải phẫu",
    "nine_ball_maid_uniform": "Đồng phục hầu gái Nine Ball",
    "riding_crop": "Roi cưỡi ngựa",
    "cloaked": "Có áo choàng",
    "gynomorph_penetrating_female": "Dạng cái thâm nhập nữ",
    "gynomorph_penetrating_male": "Dạng cái thâm nhập nam",
    "andromorph_penetrating_female": "Dạng đực thâm nhập nữ",
    "andromorph_penetrating_male": "Dạng đực thâm nhập nam",
    "intersex_penetrating_female": "Liên giới tính thâm nhập nữ",
    "intersex_penetrating_male": "Liên giới tính thâm nhập nam",
    "shirt_hold": "Nắm áo sơ mi",
    "female_penetrating_male": "Nữ thâm nhập nam",
    "female_penetrating_female": "Nữ thâm nhập nữ",
    "male_penetrating_male": "Nam thâm nhập nam",
    "feral_penetrating_anthro": "Thú thâm nhập nhân thú",
    "feral_penetrating_female": "Thú thâm nhập nữ",
    "feral_penetrating_male": "Thú thâm nhập nam",
    "holding_riding_crop": "Cầm roi cưỡi ngựa",
    "crop_top_lift": "Nhấc áo croptop",
    "crop_top_hoodie": "Áo hoodie croptop",
    "crop_top_only": "Chỉ mặc áo croptop",
    "split_crop": "Áo croptop tách đôi",
    "crop_top_overhang": "Vạt thừa lộ trên áo croptop",
    "out_of_frame": "Lọt ra ngoài khung hình",
    "in_frame": "Trong khung hình",
    "foot_out_of_frame": "Chân lọt ra ngoài khung",
    "head_out_of_frame": "Đầu lọt ra ngoài khung",
    "knees_out_of_frame": "Đầu gối lọt ra ngoài khung",
    "hand_out_of_frame": "Bàn tay lọt ra ngoài khung",
    "leg_out_of_frame": "Chân lọt ra ngoài khung",
    "upper_body_out_of_frame": "Thân trên lọt ra ngoài khung",
    "frame_by_frame": "Từng khung hình một",
    "spoken_question_mark": "Dấu hỏi trong bóng thoại",
    "spoken_exclamation_mark": "Dấu chấm than trong bóng thoại",
    "spoken_musical_note": "Nốt nhạc trong bóng thoại",
    "spoken_note": "Nốt nhạc trong bóng thoại",
    "speech_bubble_start": "Đầu bong bóng thoại",
    "speech_bubble_tail": "Đuôi bong bóng thoại",
    # Thẻ phổ biến về số lượng, biểu cảm và tư thế.
    "couple": "Cặp đôi", "multiple_people": "Nhiều người",
    "female_focus": "Tập trung vào nhân vật nữ", "male_focus": "Tập trung vào nhân vật nam",
    "solo_focus": "Tập trung vào nhân vật chính", "looking_at_another": "Nhìn người khác",
    "looking_at_another_person": "Nhìn người khác", "looking_back": "Ngoái nhìn",
    "looking_down": "Nhìn xuống", "looking_up": "Nhìn lên",
    "looking_to_the_side": "Nhìn sang bên", "eyes_visible_through_hair": "Mắt lộ qua tóc",
    "one_eye_closed": "Nhắm một mắt", "half-closed_eyes": "Mắt lim dim",
    "narrowed_eyes": "Nheo mắt", "closed_eyelids": "Mí mắt khép",
    "smug": "Vẻ mặt tự mãn", "frown": "Nhíu mày", "pout": "Chu môi",
    "happy": "Vui vẻ", "sad": "Buồn", "serious": "Nghiêm túc",
    "embarrassed": "Ngượng ngùng", "expressionless": "Không biểu cảm",
    "nervous": "Lo lắng", "tired": "Mệt mỏi", "confused": "Bối rối",
    "tongue_out": "Thè lưỡi", "fangs": "Răng nanh", "teeth": "Răng",
    "kneeling": "Đang quỳ", "crouching": "Đang cúi người", "squatting": "Đang ngồi xổm",
    "bending_over": "Cúi người về phía trước", "leaning_forward": "Nghiêng người về phía trước",
    "on_back": "Nằm ngửa", "on_side": "Nằm nghiêng", "on_stomach": "Nằm sấp",
    "arms_behind_back": "Hai tay ra sau lưng", "hands_on_hips": "Hai tay chống hông",
    "hand_on_own_hip": "Tay chống hông", "waving": "Đang vẫy tay",
    "holding": "Đang cầm", "holding_weapon": "Cầm vũ khí", "holding_sword": "Cầm kiếm",
    "holding_gun": "Cầm súng", "holding_phone": "Cầm điện thoại",
    "holding_flower": "Cầm hoa", "eating": "Đang ăn", "drinking": "Đang uống",
    "smoking": "Đang hút thuốc", "dancing": "Đang nhảy múa",
    # Trang phục, phụ kiện và ngoại hình.
    "bikini": "Đồ bơi bikini", "swimsuit": "Đồ bơi", "one_piece_swimsuit": "Đồ bơi một mảnh",
    "thighhighs": "Tất cao đùi", "pantyhose": "Quần tất", "stockings": "Bít tất dài",
    "knee_highs": "Tất cao đến gối", "barefoot": "Chân trần", "high_heels": "Giày cao gót",
    "sneakers": "Giày thể thao", "short_sleeves": "Tay áo ngắn", "long_sleeves": "Tay áo dài",
    "sleeveless": "Không tay", "collared_shirt": "Áo sơ mi có cổ",
    "t-shirt": "Áo phông", "hoodie": "Áo hoodie", "sweater": "Áo len",
    "coat": "Áo khoác dài", "shorts": "Quần short", "pants": "Quần dài",
    "hair_ornament": "Đồ trang trí tóc", "hair_ribbon": "Ruy băng cài tóc",
    "hair_bow": "Nơ cài tóc", "hairband": "Băng đô", "hairclip": "Kẹp tóc",
    "thick_thighs": "Đùi đầy đặn", "bare_shoulders": "Vai trần", "bare_legs": "Chân trần",
    "bare_arms": "Cánh tay trần", "navel": "Rốn", "collarbone": "Xương quai xanh",
    "thighs": "Đùi", "feet": "Bàn chân", "fingers": "Ngón tay", "toes": "Ngón chân",
    "hand": "Bàn tay", "hands": "Bàn tay", "eyelashes": "Lông mi",
    "pointy_ears": "Tai nhọn", "animal_tail": "Đuôi động vật", "dragon": "Rồng",
    "fox": "Cáo", "wolf": "Sói", "cat": "Mèo", "dog": "Chó", "rabbit": "Thỏ",
    "bird": "Chim", "horse": "Ngựa", "angel": "Thiên thần", "demon": "Quỷ",
    # Bối cảnh, loại hình ảnh và metadata.
    "black_background": "Nền đen", "blue_background": "Nền xanh dương",
    "red_background": "Nền đỏ", "outdoors": "Ngoài trời", "indoors": "Trong nhà",
    "bedroom": "Phòng ngủ", "classroom": "Lớp học", "kitchen": "Nhà bếp",
    "cityscape": "Cảnh thành phố", "scenery": "Phong cảnh", "landscape": "Phong cảnh",
    "rainy": "Trời mưa", "snowing": "Trời đang có tuyết", "sunny": "Trời nắng",
    "day": "Ban ngày", "evening": "Buổi tối", "dusk": "Chạng vạng",
    "dawn": "Rạng đông", "night_sky": "Bầu trời đêm", "starry_sky": "Bầu trời đầy sao",
    "wind": "Gió", "windy": "Trời có gió", "water": "Nước", "ocean": "Đại dương",
    "digital_media_(artwork)": "Tranh kỹ thuật số", "traditional_media": "Chất liệu truyền thống",
    "pixel_art": "Tranh pixel", "3d": "Hình ảnh 3D", "photorealistic": "Phong cách ảnh chân thực",
    "anime_coloring": "Tô màu anime", "speech_bubble": "Bong bóng thoại",
    "english_text": "Chữ tiếng Anh", "japanese_text": "Chữ tiếng Nhật",
    "no_humans": "Không có người", "no_hats": "Không có mũ",
    "male/female": "Nam và nữ", "male/male": "Nam và nam", "female/female": "Nữ và nữ",
    "1person": "Một người", "2people": "Hai người", "3girls": "Ba nhân vật nữ",
    "3boys": "Ba nhân vật nam", "4girls": "Bốn nhân vật nữ", "4boys": "Bốn nhân vật nam",
})
TAG_VI_LABELS.update({
    # Từ điển bổ sung cho các thẻ chung thường gặp trong toàn bộ CSV.
    "jewelry": "Trang sức", "bodily_fluids": "Dịch cơ thể", "hair_between_eyes": "Tóc giữa hai mắt",
    "bow": "Nơ", "sex": "Quan hệ tình dục", "tongue": "Lưỡi", "clothed": "Mặc quần áo",
    "underwear": "Đồ lót", "balls": "Tinh hoàn", "butt": "Mông", "genital_fluids": "Dịch sinh dục",
    "erection": "Cương cứng", "sweat": "Mồ hôi", "cum": "Tinh dịch", "vulva": "Âm hộ",
    "shoes": "Giày", "penetration": "Thâm nhập", "panties": "Quần lót", "ahoge": "Tóc chỉa",
    "weapon": "Vũ khí", "white_body": "Cơ thể trắng", "sidelocks": "Tóc mai", "heart": "Trái tim",
    "anus": "Hậu môn", "horn": "Sừng", ":d": "Biểu cảm :D", "cowboy_shot": "Khung hình đầu đến đùi",
    "penile": "Thuộc dương vật", "scalie": "Nhân vật bò sát nhân hóa", "food": "Thức ăn",
    "hetero": "Dị tính", "claws": "Móng vuốt", "muscular": "Cơ bắp",
    "mythological_creature": "Sinh vật thần thoại", "vaginal": "Thuộc âm đạo", "ass": "Mông",
    "feral": "Động vật không nhân hóa", "frills": "Viền bèo", "topwear": "Trang phục thân trên",
    "artist_name": "Tên họa sĩ", "open_clothes": "Quần áo mở", "spread_legs": "Dạng chân",
    "parted_lips": "Môi hé", "piercing": "Khuyên xỏ", "censored": "Đã kiểm duyệt",
    "dialogue": "Lời thoại", "necktie": "Cà vạt", "pleated_skirt": "Chân váy xếp ly",
    "choker": "Vòng cổ choker", "humanoid_genitalia": "Bộ phận sinh dục dạng người",
    "biped": "Đi bằng hai chân", "alternate_costume": "Trang phục thay thế",
    "male_penetrating": "Nhân vật nam chủ động thâm nhập", "animal_genitalia": "Bộ phận sinh dục động vật",
    "collar": "Vòng cổ", "anal": "Quan hệ qua đường hậu môn", "humanoid_penis": "Dương vật dạng người",
    "areola": "Quầng vú", "belt": "Thắt lưng", "detached_sleeves": "Tay áo rời",
    "markings": "Dấu vết / hoa văn", "group": "Nhóm", "bottomwear": "Trang phục thân dưới",
    "hand_up": "Giơ tay", "muscular_male": "Nam cơ bắp", "big_butt": "Mông lớn",
    "plant": "Thực vật", "japanese_clothes": "Trang phục Nhật Bản", "halo": "Vầng hào quang",
    "puffy_sleeves": "Tay áo phồng", "wide_hips": "Hông rộng", "pussy": "Âm hộ",
    "size_difference": "Chênh lệch kích thước", "cum_inside": "Xuất tinh bên trong",
    "oral": "Quan hệ bằng miệng", "headgear": "Đồ đội đầu", "bowtie": "Nơ cổ",
    "equid": "Họ ngựa", "midriff": "Vùng eo và bụng", "spreading": "Dạng chân",
    "multicolored_body": "Cơ thể nhiều màu", "grin": "Cười toe toét", "heart_symbol": "Biểu tượng trái tim",
    "hood": "Mũ trùm", "elbow_gloves": "Găng tay dài quá khuỷu", "ear_piercing": "Khuyên tai",
    "penile_penetration": "Dương vật thâm nhập", "domestic_dog": "Chó nhà", "fang": "Răng nanh",
    "animal_penis": "Dương vật động vật", "saliva": "Nước bọt", "eyes_closed": "Nhắm mắt",
    "anal_penetration": "Thâm nhập hậu môn", "streaked_hair": "Tóc có lọn màu khác",
    "twitter_username": "Tên người dùng Twitter", "ambiguous_gender": "Giới tính không rõ",
    "sweatdrop": "Giọt mồ hôi", "blunt_bangs": "Tóc mái ngang", "fingerless_gloves": "Găng tay hở ngón",
    "furniture": "Đồ nội thất", "stomach": "Bụng", "sword": "Kiếm", "interspecies": "Khác loài",
    "eyewear": "Kính mắt", "sailor_collar": "Cổ áo thủy thủ", "abs": "Cơ bụng",
    "female_penetrated": "Nhân vật nữ bị thâm nhập", "chibi": "Chibi", "belly": "Bụng",
    "felis": "Họ mèo", "two-tone_hair": "Tóc hai màu", "mole": "Nốt ruồi",
    "legwear": "Trang phục chân", "hair_flower": "Hoa cài tóc", "miniskirt": "Váy ngắn",
    "serafuku": "Đồng phục thủy thủ", "border": "Đường viền", "hair_bun": "Tóc búi",
    "outside": "Bên ngoài", "bed": "Giường", "feathers": "Lông vũ", "hair_over_one_eye": "Tóc che một mắt",
    "ejaculation": "Xuất tinh", "bra": "Áo ngực", "paws": "Bàn chân thú", "5_fingers": "Năm ngón tay",
    "animal_ear_fluff": "Lông viền tai động vật", "lagomorph": "Họ thỏ", "eyebrows": "Lông mày",
    "makeup": "Trang điểm", "pawpads": "Đệm chân thú", "footwear": "Giày dép",
    "star_(symbol)": "Ngôi sao", "male_penetrating_female": "Nam thâm nhập nữ",
    "striped_clothes": "Quần áo sọc", "vest": "Áo gile", "bracelet": "Vòng tay",
    "off_shoulder": "Trễ vai", "nail_polish": "Sơn móng", "digital_drawing_(artwork)": "Tranh vẽ kỹ thuật số",
    "black_nose": "Mũi đen", "black_fur": "Lông đen", "leotard": "Đồ liền thân",
    "grey_fur": "Lông xám", "leporid": "Họ thỏ", "multicolored_fur": "Lông nhiều màu",
    "dated": "Có ghi ngày tháng", "cup": "Cốc", "anthro_on_anthro": "Nhân vật nhân hóa với nhau",
    "pillow": "Gối", "looking_pleasured": "Vẻ mặt thỏa mãn", "parted_bangs": "Tóc mái rẽ",
    "overweight": "Thừa cân", "headwear": "Đồ đội đầu", "arm_up": "Giơ tay",
    "mosaic_censoring": "Che bằng khảm", "young": "Trẻ", "big_balls": "Tinh hoàn lớn",
    "muscular_anthro": "Nhân vật nhân hóa cơ bắp", "male_penetrated": "Nhân vật nam bị thâm nhập",
    "tattoo": "Hình xăm", "fellatio": "Quan hệ bằng miệng", "sharp_teeth": "Răng nhọn",
    "completely_nude": "Hoàn toàn khỏa thân", "fingernails": "Móng tay", "facial_hair": "Râu",
    "knot": "Nút thắt", "vein": "Mạch máu", "tan_body": "Cơ thể rám nắng", "scar": "Sẹo",
    "clothes_lift": "Vén quần áo", "wet": "Ướt", "on_bed": "Nằm trên giường",
    "handwear": "Găng tay", "hands_up": "Giơ hai tay", "black_choker": "Vòng cổ choker đen",
    "glowing": "Phát sáng", "multiple_views": "Nhiều góc nhìn", "neckerchief": "Khăn quàng cổ",
    "precum": "Dịch tiền xuất tinh", "mask": "Mặt nạ", "lips": "Môi", "pecs": "Cơ ngực",
    "soles": "Lòng bàn chân", "bdsm": "BDSM", "holding_object": "Cầm vật",
    "orgasm": "Cực khoái", "v-shaped_eyebrows": "Lông mày chữ V",
    "veiny_penis": "Dương vật nổi gân", "detailed_background": "Nền chi tiết",
    "feathered_wings": "Cánh lông vũ", "puffy_short_sleeves": "Tay áo ngắn phồng",
    "upper_teeth_only": "Chỉ thấy răng trên", "pantherine": "Phân họ báo",
    "aqua_eyes": "Mắt xanh ngọc", "symbol-shaped_pupils": "Đồng tử hình biểu tượng",
    "blood": "Máu", "white_panties": "Quần lót trắng", "dark-skinned_female": "Nhân vật nữ da sẫm",
    "thigh_strap": "Đai đùi", "mole_under_eye": "Nốt ruồi dưới mắt",
    "vaginal_fluids": "Dịch âm đạo", "bulge": "Phần phồng lên", "two_tone_body": "Cơ thể hai màu",
    "masturbation": "Thủ dâm", "twin_braids": "Hai bím tóc", "two_side_up": "Hai lọn tóc buộc lên hai bên",
    "scales": "Vảy", "loli": "Nhân vật nhỏ tuổi", "accessory": "Phụ kiện", "hug": "Ôm",
    "sex_toy": "Đồ chơi tình dục", "rear_view": "Góc nhìn từ phía sau", "gun": "Súng",
    "licking": "Đang liếm", "bar_censor": "Thanh che", "gradient_background": "Nền chuyển sắc",
    "strapless": "Không dây", "feet_out_of_frame": "Bàn chân ngoài khung hình",
    "sleeves_past_wrists": "Tay áo che qua cổ tay", "fake_animal_ears": "Tai động vật giả",
    "flat_chest": "Ngực phẳng", "stripes": "Sọc", "fur_trim": "Viền lông",
    "sleeveless_shirt": "Áo sơ mi không tay", "sleeveless_dress": "Váy không tay",
    "foreskin": "Bao quy đầu", "book": "Sách", "window": "Cửa sổ", "variant_set": "Bộ biến thể",
    "gradient_hair": "Tóc chuyển sắc", "sparkle": "Lấp lánh", "horse_girl": "Cô gái ngựa",
    "widescreen": "Màn hình rộng", "white_border": "Viền trắng", "grass": "Cỏ",
    "motion_lines": "Vệt chuyển động", "penis_in_vagina": "Dương vật trong âm đạo",
    "countershading": "Màu lưng và bụng tương phản", "wrist_cuffs": "Vòng cổ tay",
    "front_view": "Góc nhìn chính diện", "see-through_clothes": "Quần áo xuyên thấu",
    "toe_claws": "Móng vuốt ở ngón chân", "phone": "Điện thoại", "bondage": "Trói buộc",
    "ring_piercing": "Khuyên vòng", "buttons": "Cúc áo", "bell": "Chuông", "cosplay": "Cosplay",
    "colored_skin": "Da có màu", "neck_ribbon": "Ruy băng cổ", "heterochromia": "Mắt hai màu",
    "maid_headdress": "Mũ hầu gái", "maid": "Hầu gái", "chair": "Ghế", "spikes": "Gai",
    "sound_effects": "Hiệu ứng âm thanh", "group_sex": "Quan hệ tình dục nhóm",
    "all_fours": "Tư thế chống bằng bốn chi", "femboy": "Nhân vật nam nữ tính",
    "from_behind_position": "Tư thế từ phía sau", "ascot": "Khăn ascot", "huge_butt": "Mông rất lớn",
    "fruit": "Trái cây", "covered_nipples": "Núm vú được che", "submissive": "Phục tùng",
    "dominant": "Thống trị", "canine_penis": "Dương vật chó", "seductive": "Quyến rũ",
})
TAG_VI_LABELS.update({
    "penis": "Dương vật", "socks": "Tất", "virtual_youtuber": "YouTuber ảo",
    "official_alternate_costume": "Trang phục thay thế chính thức", "big_penis": "Dương vật lớn",
    "huge_penis": "Dương vật rất lớn", "canis": "Họ chó", "mythological_scalie": "Nhân vật bò sát thần thoại",
    "equine": "Thuộc loài ngựa", "domestic_cat": "Mèo nhà", "vaginal_penetration": "Thâm nhập qua âm đạo",
    "bag": "Túi", "scarf": "Khăn quàng", "cape": "Áo choàng", "apron": "Tạp dề",
    "armpits": "Nách", "crop_top": "Áo croptop", "yuri": "Tình cảm nữ-nữ",
    "intersex": "Liên giới tính", "presenting": "Đưa ra phía trước", "bottomless": "Không mặc quần",
    "reptile": "Loài bò sát", "bound": "Bị trói", "character_name": "Tên nhân vật",
    "chinese_commentary": "Chú thích tiếng Trung", "anthro_penetrated": "Nhân vật nhân hóa bị thâm nhập",
    "shaded": "Có đổ bóng", "hair_intakes": "Lọn tóc cuộn vào trong", ":o": "Biểu cảm :O",
    "gynomorph": "Nhân vật nữ có cơ quan sinh dục nam", "3d_(artwork)": "Tranh 3D",
    "bovid": "Họ bò", "inside": "Bên trong", "avian": "Loài chim", "male_penetrating_male": "Nam thâm nhập nam",
    "side_ponytail": "Tóc đuôi ngựa lệch bên", "canine_genitalia": "Bộ phận sinh dục chó",
    "fluffy": "Bồng bềnh", "anthro_penetrating": "Nhân vật nhân hóa chủ động thâm nhập",
    "hybrid": "Sinh vật lai", "one-piece_swimsuit": "Đồ bơi một mảnh", "torn_clothes": "Quần áo rách",
    "drooling": "Chảy nước miếng", "bodysuit": "Đồ liền thân", "sash": "Đai vải",
    "pov": "Góc nhìn thứ nhất", "plaid_clothes": "Quần áo kẻ caro", "4_toes": "Bốn ngón chân",
    "rodent": "Loài gặm nhấm", "covered_navel": "Rốn được che", "uncensored": "Không kiểm duyệt",
    "dripping": "Nhỏ giọt", "young_anthro": "Nhân vật nhân hóa trẻ", "cum_in_ass": "Xuất tinh vào hậu môn",
    "tan_fur": "Lông rám nắng", "glistening": "Lấp lánh", "trio": "Bộ ba",
    "cum_in_vagina": "Xuất tinh vào âm đạo", "equine_genitalia": "Bộ phận sinh dục ngựa",
    "sunglasses": "Kính râm", "floating_hair": "Tóc bay", "tank_top": "Áo ba lỗ",
    "not_furry": "Không có lông thú", "siblings": "Anh chị em ruột", "pose": "Tư thế",
    "sheath": "Bao kiếm", "slightly_chubby": "Hơi mũm mĩm", "robot": "Người máy",
    "pubic_hair": "Lông mu", "turtleneck": "Áo cổ lọ", "chain": "Dây chuyền", "groin": "Háng",
    "animal_humanoid": "Động vật dạng người", "detached_collar": "Cổ áo rời", "aquatic": "Sinh vật thủy sinh",
    "blurry_background": "Nền mờ", "thigh_highs": "Tất cao đùi", "single_braid": "Một bím tóc",
    "generation_1_pokemon": "Pokémon thế hệ đầu", "copyright_name": "Tên tác phẩm/bản quyền",
    "tail_markings": "Hoa văn trên đuôi", "clitoris": "Âm vật", "table": "Bàn", "profile": "Góc nghiêng",
    "dress_shirt": "Áo sơ mi", "open_smile": "Cười miệng mở", "raised_tail": "Đuôi dựng lên",
    "non-web_source": "Nguồn ngoài web", "petals": "Cánh hoa", "double_bun": "Hai búi tóc",
    "hooves": "Móng guốc", "capelet": "Áo choàng ngắn", "anthrofied": "Được nhân hóa",
    "big_belly": "Bụng lớn", "aqua_hair": "Tóc xanh ngọc", "cellphone": "Điện thoại di động",
    "black_clothing": "Trang phục đen", "pupils": "Đồng tử", "head_tilt": "Nghiêng đầu",
    "bent_over": "Cúi gập người", "chest_tuft": "Chùm lông ngực", "blush_stickers": "Hình dán má hồng",
    "female_anthro": "Nhân vật nữ nhân hóa", "clenched_teeth": "Nghiến răng", "fire": "Lửa",
    "dutch_angle": "Góc nghiêng kiểu Hà Lan", "partially_clothed": "Mặc quần áo một phần",
    "3_toes": "Ba ngón chân", "ring": "Nhẫn", "leaf": "Lá cây", "sandals": "Dép xăng đan",
    "holding_food": "Cầm thức ăn", "game_asset": "Tài nguyên trò chơi", "cat_girl": "Cô gái mèo",
    "topless": "Để ngực trần", "equine_penis": "Dương vật ngựa", "black_bikini": "Bikini đen",
    "zettai_ryouiki": "Khoảng hở giữa tất cao đùi và váy", "skindentation": "Vết hằn trên da",
    "highleg": "Cắt cao ở hông", "inner_ear_fluff": "Lông tơ trong tai", "eyeshadow": "Phấn mắt",
    "pubes": "Lông mu", "colored": "Có màu", "nipple_piercing": "Khuyên núm vú",
    "kneehighs": "Tất cao đến gối", "thick_eyebrows": "Lông mày rậm", "cum_on_body": "Tinh dịch trên cơ thể",
    "swimwear": "Đồ bơi", "two_tone_fur": "Lông hai màu", "biceps": "Cơ nhị đầu",
    "presenting_hindquarters": "Đưa phần thân sau ra phía trước",
})
# Chỉ ghép nhãn từ các từ thông dụng, rõ nghĩa và cấu trúc tag quen thuộc.
_TAG_VI_COLORS = {
    "black": "đen", "white": "trắng", "blue": "xanh dương", "green": "xanh lá",
    "red": "đỏ", "yellow": "vàng", "orange": "cam", "pink": "hồng",
    "purple": "tím", "brown": "nâu", "grey": "xám", "gray": "xám",
    "silver": "bạc", "gold": "vàng kim",
    "aqua": "xanh ngọc", "teal": "xanh cổ vịt", "cyan": "xanh lơ", "turquoise": "xanh ngọc lam",
    "navy": "xanh navy", "olive": "xanh ô liu", "lavender": "tím oải hương", "violet": "tím",
    "magenta": "hồng cánh sen", "maroon": "nâu đỏ", "beige": "màu be", "ivory": "màu ngà",
    "cream": "màu kem", "bronze": "đồng thau", "copper": "đồng đỏ", "tan": "nâu rám nắng",
    "rainbow": "cầu vồng", "pastel": "màu pastel", "neon": "neon", "metallic": "ánh kim",
    "sepia": "nâu vàng cổ điển", "multitone": "nhiều tông",
}
_TAG_VI_WORDS = {
    **_TAG_VI_COLORS,
    "long": "dài", "short": "ngắn", "medium": "vừa", "very": "rất",
    "small": "nhỏ", "big": "to", "large": "lớn", "huge": "khổng lồ",
    "thick": "dày", "thin": "mỏng", "curly": "xoăn", "wavy": "gợn sóng",
    "straight": "thẳng", "messy": "rối", "braided": "tết", "fluffy": "lông xù",
    "spiky": "dựng", "pointy": "nhọn", "round": "tròn", "wide": "rộng",
    "narrow": "hẹp", "open": "mở", "closed": "khép", "light": "nhạt",
    "dark": "sẫm", "bright": "sáng", "pale": "nhạt", "multicolored": "nhiều màu",
    "animal": "động vật", "human": "người", "cat": "mèo", "dog": "chó",
    "fox": "cáo", "wolf": "sói", "rabbit": "thỏ", "bird": "chim",
    "horse": "ngựa", "dragon": "rồng", "mouse": "chuột", "deer": "nai",
    "tiger": "hổ", "lion": "sư tử", "bear": "gấu", "raccoon": "gấu mèo",
    # Họa tiết, chất liệu và trạng thái — đủ an toàn để ghép phía sau danh từ tiếng Việt.
    "striped": "kẻ sọc", "plaid": "kẻ caro", "checkered": "kẻ ca-rô", "dotted": "chấm tròn",
    "torn": "rách", "ripped": "xé rách", "wet": "ướt", "dry": "khô", "tight": "ôm sát",
    "matte": "nhám", "glowing": "phát sáng", "lit": "được chiếu sáng", "shadowed": "có bóng",
    "lace": "ren", "denim": "jean", "jeans": "jean", "leather": "da", "latex": "latex",
    "rubber": "cao su", "wool": "len", "silk": "lụa", "satin": "sa tanh", "velvet": "nhung",
    "cotton": "cotton", "metal": "kim loại", "golden": "màu vàng", "glass": "thủy tinh",
    "paper": "giấy", "wooden": "bằng gỗ", "stone": "đá", "snow": "tuyết", "water": "nước",
    "frilled": "nhún xếp", "ruffled": "xếp tầng", "pleated": "xếp ly", "buttons": "cúc",
    "patterned": "có hoa văn", "print": "họa tiết",
    "textured": "có kết cấu", "smooth": "trơn", "furry": "nhiều lông", "bald": "trọc",
    "clean": "sạch", "dirty": "bẩn", "bloody": "vấy máu", "sweaty": "đầm mồ hôi",
    "dusty": "bụi bặm", "old": "cũ", "new": "mới", "mini": "siêu ngắn", "micro": "rất nhỏ",
    "half": "một nửa", "partial": "một phần",
    "partially": "một phần", "mostly": "chủ yếu", "barely": "hầu như không",
    "muscular": "cơ bắp", "chubby": "mũm mĩm", "overweight": "thừa cân", "obese": "béo phì",
    "feminine": "nữ tính", "masculine": "nam tính", "cute": "dễ thương", "sexy": "gợi cảm",
    "formal": "trang trọng", "casual": "thường ngày", "sporty": "thể thao", "military": "quân đội",
    "school": "học đường", "christmas": "Giáng sinh", "halloween": "Halloween",
    "summer": "mùa hè", "winter": "mùa đông", "spring": "mùa xuân", "autumn": "mùa thu",
    # Bộ phận cơ thể đứng ở vị trí bổ ngữ: "Đồ xăm ở cánh tay", "Hoa văn chân"…
    "head": "đầu", "face": "mặt", "hair": "tóc", "fur": "lông", "skin": "da",
    "eye": "mắt", "eyes": "mắt", "ear": "tai", "ears": "tai", "nose": "mũi",
    "mouth": "miệng", "lip": "môi", "lips": "môi", "chin": "cằm", "cheek": "gò má",
    "neck": "cổ", "shoulder": "vai", "shoulders": "vai", "chest": "ngực",
    "back": "lưng", "stomach": "bụng", "belly": "bụng", "navel": "rốn",
    "armpit": "nách", "armpits": "nách", "butt": "mông", "ass": "mông",
    "arm": "cánh tay", "arms": "cánh tay", "elbow": "khuỷu tay", "wrist": "cổ tay",
    "hand": "bàn tay", "hands": "bàn tay", "finger": "ngón tay", "fingers": "ngón tay",
    "leg": "chân", "legs": "chân", "thigh": "đùi", "thighs": "đùi", "knee": "đầu gối",
    "ankle": "cổ chân", "foot": "bàn chân", "feet": "bàn chân", "toe": "ngón chân",
    "toes": "ngón chân", "tail": "đuôi", "wing": "cánh", "wings": "cánh",
    "horn": "sừng", "horns": "sừng", "paw": "chân thú", "paws": "chân thú",
    "penis": "dương vật", "vulva": "âm hộ", "anus": "hậu môn", "breast": "ngực",
    "breasts": "ngực", "nipple": "núm vú", "nipples": "núm vú", "genitals": "bộ phận sinh dục",
    "body": "cơ thể", "figure": "dáng người", "silhouette": "bóng", "profile": "góc nghiêng",
    "fire": "lửa", "smoke": "khói", "steam": "hơi nước",
    "spiked": "có gai", "puffy": "phồng", "cropped": "lửng", "sailor": "thủy thủ",
    "bat": "dơi", "bunny": "thỏ", "demon": "quỷ", "devil": "quỷ", "angel": "thiên thần",
    "ghost": "ma", "skeleton": "bộ xương", "witch": "phù thủy", "maid": "hầu gái",
    "ninja": "ninja", "pirate": "cướp biển", "robot": "người máy", "mecha": "mecha",
    "frog": "ếch", "snake": "rắn", "spider": "nhện", "insect": "côn trùng",
    "butterfly": "bươm bướm", "bee": "ong", "fish": "cá", "shark": "cá mập",
    "whale": "cá voi", "octopus": "bạch tuộc", "squid": "mực", "crab": "cua",
    "goat": "dê", "sheep": "cừu", "cow": "bò", "pig": "lợn", "duck": "vịt",
    "swan": "thiên nga", "owl": "cú mèo", "eagle": "đại bàng", "crow": "quạ",
    "panda": "gấu trúc", "koala": "koala", "otter": "rái cá", "ferret": "chồn bạc",
    "squirrel": "sóc", "hedgehog": "nhím gai", "hyena": "linh cẩu", "jackal": "chồn sói",
    "mythological": "thần thoại", "prehistoric": "tiền sử", "extinct": "tuyệt chủng",
    "inanimate": "vô tri", "object": "vật thể", "food": "thức ăn", "drink": "đồ uống",
    "dessert": "món tráng miệng", "meat": "thịt", "fruit": "trái cây", "vegetable": "rau",
    "cake": "bánh ngọt", "cookie": "bánh quy", "candy": "kẹo", "chocolate": "sô-cô-la",
    "bread": "bánh mì", "rice": "cơm", "egg": "trứng", "milk": "sữa",
    "coffee": "cà phê", "tea": "trà", "beer": "bia", "wine": "rượu vang", "sake": "rượu sake",
    "book": "sách", "letter": "bức thư", "envelope": "phong bì", "card": "thẻ",
    "money": "tiền", "coin": "đồng xu", "key": "chìa khóa",
    "sword": "kiếm", "shield": "khiên", "bomb": "bom", "arrow": "mũi tên",
    "spotlight": "đèn rọi", "candle": "nến", "lamp": "đèn", "lantern": "đèn lồng",
    "fireworks": "pháo hoa", "explosion": "vụ nổ", "magic": "phép thuật",
    "sparkle": "ánh lấp lánh", "sparkles": "ánh lấp lánh", "bubble": "bọt khí",
    "bubbles": "bọt khí", "blood": "máu", "pus": "mủ", "saliva": "nước bọt",
    "tears": "nước mắt",
    # Đợt 2: số đếm, trạng thái, chất liệu — đủ an toàn để ghép sau danh từ chính.
    "own": "của mình", "one": "một", "two": "hai", "three": "ba", "four": "bốn",
    "five": "năm", "six": "sáu", "seven": "bảy", "eight": "tám", "nine": "chín",
    "ten": "mười", "multi": "nhiều", "piece": "mảnh", "tone": "tông", "shaped": "hình",
    "framed": "khung", "trimmed": "viền", "covered": "được che", "covering": "che",
    "missing": "thiếu", "sleeveless": "không tay", "strapless": "không quai",
    "collared": "có cổ", "laced": "buộc dây", "bandaged": "băng bó",
    "freckled": "có tàn nhang", "raised": "dựng lên", "spread": "dạng ra", "flat": "phẳng",
    "pointed": "nhọn", "curvy": "cong", "mismatched": "lệch đôi", "mechanical": "cơ khí",
    "tinted": "nhuộm màu", "layered": "nhiều lớp", "impossible": "bất khả thi",
    "visible": "lộ rõ", "invisible": "tàng hình", "glistening": "bóng nhẫy",
    "unworn": "không mặc", "worn": "đang mặc", "hyper": "phóng đại",
    "countershade": "bóng ngược", "inner": "mặt trong", "outer": "mặt ngoài",
    "fake": "giả", "fishnet": "lưới", "sports": "thể thao", "polka": "bi", "dot": "chấm",
    # Tính từ trạng thái/bề mặt (thiếu ở các đợt chèn trước, nay khai báo đủ bộ).
    "glossy": "bóng loáng", "sheer": "xuyên thấu", "shiny": "bóng", "slim": "thon",
    "fit": "dáng thể hình", "tall": "cao", "young": "trẻ", "marbled": "vân đá",
    "gradient": "chuyển sắc", "spotted": "có đốm", "translucent": "bán trong suốt",
    "transparent": "trong suốt",
    "single": "một chiếc", "double": "kép", "extra": "phụ thêm",
    "matching": "đồng bộ", "asymmetrical": "bất đối xứng",
    "naked": "trần", "scar": "sẹo", "feather": "lông vũ", "heart": "trái tim",
    "star": "ngôi sao", "moon": "mặt trăng", "sun": "mặt trời",
    "bell": "chuông", "chain": "dây xích", "high": "cao", "low": "thấp",
    "longer": "dài hơn", "shorter": "ngắn hơn", "older": "lớn tuổi hơn",
    "younger": "nhỏ tuổi hơn", "larger": "to hơn", "smaller": "nhỏ hơn",
    "taller": "cao hơn", "shortstack": "thấp và mũm mĩm",
    # Danh từ vừa là danh từ chính vừa hay đứng làm bổ ngữ: khai báo ở cả hai phía.
    "anthro": "nhân hóa", "feral": "bán thú", "kemono": "kemono", "humanoid": "dạng người",
    "girl": "nữ", "boy": "nam", "male": "nam", "female": "nữ", "intersex": "liên giới tính",
    "ponytail": "đuôi ngựa", "twintails": "hai búi",
    "braid": "bím tóc", "bun": "búi tóc", "bangs": "tóc mái", "glasses": "kính",
    "ribbon": "ruy băng", "bow": "nơ", "collar": "cổ áo", "strap": "dây",
    "zipper": "khóa kéo", "button": "cúc",
    "loose": "rộng",
}
_TAG_VI_WORDS.update({
    # Đợt 11 — danh từ họ hàng cho họ thẻ <A>_and_<B>.
    "parent": "cha mẹ",
    "son": "con trai",
    "daughter": "con gái",
    "mother": "mẹ",
    "father": "cha",
    "brothers": "anh em trai",
    "grandmother": "bà",
    "grandfather": "ông",
    "twin_sister": "chị em gái song sinh",
    # Đợt 11 — bổ ngữ cho họ giới tính/loài/trang phục.
    "pregnant": "mang thai",
    "athletic": "cường tráng",
    "toned": "thon săn",
    "voluptuous": "nóng bỏng",
    "dominant": "chiếm ưu thế",
    "submissive": "nhượng bộ",
    "bottomless": "hở phần dưới",
    "topless": "hở phần trên",
    "clothed": "mặc đồ",
    "nude": "khỏa thân",
    "bound": "bị trói",
    "gaping": "há rộng",
    "inflated": "phồng to",
    "censored": "đã kiểm duyệt",
    "presenting": "chĩa ra",
    "spreading": "mở rộng",
    "felid": "họ mèo",
    "canid": "họ chó",
    "equid": "họ ngựa",
    "leporid": "họ thỏ",
    "lagomorph": "bộ thỏ",
    "rodent": "bộ gặm nhấm",
    "arthropod": "chân khớp",
    "scalie": "bò sát",
    "mustelid": "họ chồn",
    "procyonid": "họ gấu trúc",
    "pantherine": "họ mèo lớn",
    "sciurid": "sóc",
    "aquatic": "dưới nước",
    "alien": "ngoài hành tinh",
    "anal": "hậu môn",
    "vaginal": "âm đạo",
    "oral": "bằng miệng",
    "penile": "thuộc dương vật",
    "cock": "dương vật",
    "pussy": "âm hộ",
    "testes": "tinh hoàn",
    "balls": "tinh hoàn",
    "gynomorph": "dạng cái",
    "andromorph": "dạng đực",
    "dildo": "dương vật giả",
    "kissing": "hôn",
    "bikini": "bikini",
    "thong": "chữ T",
    "highleg": "cắt cao",
    "sweater": "áo len",
    "rope": "dây thừng",
    "suit": "vest",
    "jacket": "áo khoác",
    "garter": "dây giữ bít tất",
    "armwear": "đồ tay",
    "neckwear": "khăn cổ",
    "forehead": "trán",
    "nape": "gáy",
    "sole": "lòng bàn chân",
    "heel": "gót",
    "microphone": "mic",
    "size": "kích thước",
    "height": "chiều cao",
    "width": "bề ngang",
    "weight": "cân nặng",
    "missionary": "truyền giáo",
    "doggystyle": "doggy",
    "watermark": "hình mờ",
    "logo": "biểu tượng",
    "macro": "phóng to",
    # Đợt 11 — bổ ngữ cho họ giới tính/loài/trang phục.
    "assertive": "tự tin",
    "elderly": "lớn tuổi",
    "featureless": "không chi tiết",
    "hipped": "hông",
    "outstretched": "duỗi thẳng",
    "arched": "cong",
    "inverted": "đảo ngược",
    "throbbing": "giật thình thịch",
    "bouncing": "nảy",
    "constricted": "co hẹp",
    "limited": "hạn chế",
    "rectangular": "hình chữ nhật",
    "circular": "hình tròn",
    "triangular": "hình tam giác",
    "anatomical": "giải phẫu",
    "symbolic": "tượng trưng",
    "imminent": "sắp xảy ra",
    "delayed": "trễ",
    "centered": "canh giữa",
    "spaced": "giãn cách",
    "mollusk": "thân mềm",
    "cephalopod": "đầu chân vòng",
    "arachnid": "họ nhện",
    "lagomorphs": "bộ thỏ",
    "monstrous": "quái dị",
    "elemental": "nguyên tố",
    "goo": "nhầy",
    "robotic": "người máy",
    "mechanized": "cơ giới hóa",
    "fennec": "cáo sa mạc",
    "hunting": "săn mồi",
    "domestic": "nhà",
    "wild": "dã ngoại",
    "toothed": "có răng",
    "baleen": "tấm sừng",
    "genital": "sinh dục",
    "boobs": "ngực",
    "futa": "futanari",
    "buttplug": "chậu hậu môn",
    "deep": "sâu",
    "virgin": "trinh nữ",
    "competition": "thi đấu",
    "visor": "lưỡi trai",
    "bridal": "cô dâu",
    "wedding": "cưới",
    "thighband": "dải đùi",
    "spaghetti": "dây mảnh",
    "slingshot": "dây chéo",
    "potted": "trồng chậu",
    "stuffed": "nhồi bông",
    "shin": "cẳng chân",
    "knuckle": "khớp ngón",
    "bouquet": "bó hoa",
    "fork": "dĩa",
    "spoon": "thìa",
    "chopsticks": "đũa",
    "wand": "đũa phép",
    "pen": "bút máy",
    "axe": "rìu",
    "popsicle": "kem que",
    "scythe": "lưỡi hái",
    "dagger": "dao găm",
    "hammer": "búa",
    "paintbrush": "cọ vẽ",
    "guitar": "đàn guitar",
    "syringe": "ống tiêm",
    "controller": "tay cầm",
    "can": "lon",
    "vase": "lọ hoa",
    "clipboard": "bảng kẹp",
    "pompoms": "bông cổ vũ",
    "mop": "cây lau nhà",
    "age": "tuổi",
    "species": "loài",
    "gender": "giới tính",
    "musical": "âm nhạc",
    "barbell": "thanh thẳng",
    "septum": "vách mũi",
    "industrial": "industrial",
    "medial": "phía trong",
    "lateral": "ngoài",
    "proximal": "gần gốc",
    "twin": "đôi",
    "triple_": "ba",
    "cowgirl": "cưỡi ngựa",
    "reverse": "ngược",
    "spooning": "thuyền chung thủy",
    "milling": "quay",
    "sample": "mẫu",
    "company": "công ty",
    "serial": "sê-ri",
    "production": "sản xuất",
    "newsboy": "newsboy",
    "pilot": "phi công",
    "abyssal": "vực thẳm",
    "weibo": "Weibo",
    "pixiv": "Pixiv",
    "twitter": "Twitter",
    "instagram": "Instagram",
    "spoken": "trong bóng thoại",
    "thought": "trong bong bóng nghĩ",
    "question": "hỏi",
    "exclamation": "cảm thán",
    "speech": "nói",
    "tile": "gạch",
    "brick": "gạch nung",
    "concrete": "bê tông",
    "ceramic": "gốm",
    "disposable": "dùng một lần",
    "giant": "khổng lồ",
    "tiny": "rất nhỏ",
    "tiny_": "nhỏ xíu",
    "germ": "vi trùng",
    # "space_uniform": có mặt "space" trong WORDS để quy tắc ghép thắng khung tên riêng.
    "space": "vũ trụ",
    # Đợt 3 – chi tiết nhân vật: nghề/nhóm người, loài, hình khối, chất liệu mũ áo.
    "monotone": "một tông", "solid": "màu đặc", "amber": "hổ phách", "spiral": "xoắn ốc",
    "creepy": "rợn người", "dull": "mờ", "sharp": "sắc", "bushy": "rậm", "prick": "dựng đứng",
    "lop": "cụp", "straw": "rơm", "cowboy": "cao bồi", "police": "cảnh sát",
    "bucket": "xô", "cone": "nón nhọn", "asian": "châu Á", "japanese": "Nhật Bản",
    "idol": "thần tượng", "cheerleader": "cổ động viên", "nurse": "y tá",
    "doctor": "bác sĩ", "farmer": "nông dân", "firefighter": "cứu hỏa", "soldier": "lính",
    "monk": "tu sĩ", "knight": "hiệp sĩ", "samurai": "samurai", "vampire": "ma cà rồng",
    "mermaid": "nàng tiên cá", "zombie": "thây ma", "wizard": "phù thủy nam",
    "cube": "khối lập phương", "sphere": "hình cầu", "crystal": "pha lê", "anchor": "cái neo",
    "snowflake": "bông tuyết", "bead": "hạt", "pearl": "ngọc trai", "gem": "đá quý",
    "monkey": "khỉ", "dolphin": "cá heo", "seal": "hải cẩu", "walrus": "hải mã",
    "penguin": "cánh cụt", "rat": "chuột cống", "camel": "lạc đà", "gym": "thể dục",
    "sport": "thể thao", "naval": "hải quân", "army": "lục quân", "circle": "hình tròn",
    "alternate": "khác",
    "boot": "bốt",
    "boxing": "quyền Anh",
    "broken": "bị gãy",
    "camo": "rằn ri",
    "camouflage": "rằn ri",
    "detailed": "có chi tiết",
    "diamond": "hình thoi",
    "disembodied": "tách rời",
    "eyelids": "mí mắt",
    "fiery": "bằng lửa",
    "flaming": "bốc cháy",
    "floral": "thêu hoa",
    "frilly": "nhún diềm",
    "front": "phía trước",
    "furred": "có lông",
    "gingham": "kẻ ô vuông",
    "glove": "găng tay",
    "hairless": "không lông",
    "heeled": "có gót",
    "horizontal": "ngang",
    "horned": "có sừng",
    "iridescent": "ánh kim",
    "keyhole": "lỗ khóa",
    "latex": "cao su",
    "linked": "được nối",
    "multiple": "nhiều",
    "nearly": "gần như",
    "office": "văn phòng",
    "overskirt": "chân váy phủ ngoài",
    "patch": "miếng vá",
    "patches": "miếng vá",
    "pattern": "hoa văn",
    "pinstripe": "sọc chỉ",
    "pseudo": "giả",
    "pubic": "vùng kín",
    "seam": "đường may",
    "seams": "đường may",
    "skinsuit": "bộ đồ ôm sát",
    "studded": "đính đinh",
    "tag": "nhãn",
    "tentacle": "xúc tu",
    "tentacles": "xúc tu",
    "tip": "đầu chóp",
    "tips": "đầu chóp",
    "underskirt": "chân váy lót",
    "unusual": "bất thường",
    "vertical": "dọc",
    "ahoge": "tóc ahoge",
    "argyle": "kẻ quả thoi",
    "curled": "uốn xoăn",
    "detached": "tách rời",
    "diagonal": "chéo",
    "earclip": "kẹp tai",
    "facepaint": "vẽ mặt",
    "furrow": "nếp hằn",
    "hairy": "nhiều lông",
    "intake": "lọn tóc",
    "lidded": "hở mí",
    "lolita": "lolita",
    "median": "giữa",
    "nun": "nữ tu",
    "oversized": "quá khổ",
    "showgirl": "gợi cảm",
    "skinny": "gầy gò",
    "sparkling": "lấp lánh",
    "stirrup": "đai giữ",
    "sundress": "váy dạo nắng",
    "tailcoat": "áo đuôi cá",
    "taut": "căng chặt",
    "turtleneck": "cổ lọ",
    "uneven": "không đều",
    "wagging": "vẫy",
    "leaf": "chiếc lá", "skull": "đầu lâu", "bone": "xương", "tassel": "chuỗi tua",
    "crescent": "hình lưỡi liềm", "hourglass": "đồng hồ cát", "moon": "mặt trăng",
    "star": "ngôi sao", "cloud": "đám mây", "flame": "ngọn lửa", "bell": "chuông",
    "ribbon": "ruy băng", "bow": "nơ", "paw": "chân thú",
    "claw": "móng vuốt", "fang": "nanh", "whisker": "râu mép", "scales": "vảy",
    "shell": "vỏ", "hoof": "móng guốc", "beak": "mỏ",
    "antler": "gạc hươu", "fangs": "nanh", "gemstone": "đá quý", "chain": "dây xích",
    "whiskers": "râu mép",
    "padlock": "ổ khóa", "key": "chìa khóa", "locket": "mặt dây chuyền ảnh",
    "brooch": "trâm cài", "tiara": "vương miện nhỏ", "circlet": "vòng đội đầu",
    "armored": "bọc giáp",
    "backless": "hở lưng",
    "flipped": "lật sang bên",
    "halter": "dạng yếm",
    "notched": "chẻ",
    "quilted": "bông chần",
    "ribbed": "sống dọc",
    "toeless": "hở ngón chân",
    "triangle": "hình tam giác",
    "winged": "có cánh",
    "aisle": "lối đi",
    "bench": "ghế dài",
    "bicycle": "xe đạp",
    "blanket": "chăn",
    "branch": "cành cây",
    "broccoli": "bông cải xanh",
    "cage": "lồng",
    "camera": "máy ảnh",
    "car": "ô tô",
    "carpet": "thảm trải sàn",
    "carrot": "cà rốt",
    "cave": "hang động",
    "cliff": "vách đá",
    "cockpit": "buồng lái",
    "counter": "quầy",
    "creature": "sinh vật",
    "curtain": "rèm",
    "diaper": "tã lót",
    "floor": "sàn nhà",
    "genitalia": "bộ phận sinh dục",
    "ground": "mặt đất",
    "hammock": "võng",
    "ladder": "cái thang",
    "lake": "hồ",
    "lap": "đùi",
    "mattress": "đệm nằm",
    "motorcycle": "xe máy",
    "note": "nốt nhạc",
    "people": "những người",
    "person": "người",
    "pillow": "gối ôm",
    "plant": "cây cảnh",
    "podium": "bục",
    "pond": "ao",
    "railing": "lan can",
    "river": "sông",
    "sea": "biển",
    "slide": "cầu trượt",
    "stage": "sân khấu",
    "stairs": "cầu thang",
    "stool": "ghế đẩu",
    "surface": "bề mặt",
    "swing": "xích đu",
    "tent": "lều",
    "throne": "ngai vàng",
    "toilet": "bồn cầu",
    "trampoline": "tấm bạt nhún",
    "truck": "xe tải",
    "tv": "ti vi",
    "wall": "bức tường",
    "bomber": "bomber",
    "cargo": "túi hộp",
    "cross": "chữ thập",
    "downcast": "cụp xuống",
    "fingerless": "hở ngón",
    "fur": "lông thú",
    "hanger": "móc treo",
    "opera": "ô-pê-ra",
    "sleepy": "buồn ngủ",
    "spade": "hình cái xẻng",
    "teary": "ngấn lệ",
    "tired": "mệt mỏi",
    "upturned": "hướng lên",
    "varsity": "bóng rổ trường",
    "avian": "chim",
    "bovine": "bò",
    "canine": "chó",
    "caprine": "dê",
    "cervine": "hươu",
    "equine": "ngựa",
    "feline": "mèo",
    "furniture": "đồ nội thất",
    "instrument": "nhạc cụ",
    "leporine": "thỏ",
    "lupine": "sói",
    "master": "chủ nhân",
    "murine": "chuột",
    "musteline": "chồn",
    "other": "người khác",
    "others": "những người khác",
    "ovine": "cừu",
    "owner": "chủ nhân",
    "piscine": "cá",
    "pokemon": "Pokémon",
    "porcine": "lợn",
    "predator": "vật săn mồi",
    "prey": "con mồi",
    "reader": "người đọc",
    "reptilian": "bò sát",
    "rider": "người cưỡi",
    "tool": "dụng cụ",
    "ursine": "gấu",
    "vehicle": "xe cộ",
    "viewer": "người xem",
    "vulpine": "cáo",
    "weapon": "vũ khí",
    # Đợt 6 — chi tiết nhân vật:
    "facial": "trên mặt",
    "furgonomic": "nửa người nửa thú",
    "colored": "nhuộm màu",
    "skinned": "da",
    "blonde": "vàng hoe",
    "flag": "hình cờ",
    "living": "có sự sống",
    "sparse": "thưa",
    "side": "lệch bên",
    "shared": "dùng chung",
    "themed": "chủ đề",
    "drawn": "vẽ",
    "sided": "mặt",
    "flying": "bay",
    "suggestive": "gợi cảm",
    "holographic": "toàn ảnh",
    "energy": "năng lượng",
    "chinese": "Trung Quốc",
    "shirtless": "không mặc áo",
    "sleeve": "tay áo",
    "strawberry": "dâu tây",
    "liquid": "dạng lỏng",
    "cosmic": "vũ trụ",
    "prehensile": "có khả năng quấn",
    "stitched": "khâu",
    "polo": "polo",
    "undersized": "quá nhỏ",
    "excessive": "quá nhiều",
    "convenient": "tiện ích",
    "feathered": "có lông vũ",
    "burnt": "cháy",
    "starry": "đầy sao",
    "skeletal": "hình bộ xương",
    "medical": "y tế",
    "party": "tiệc",
    "chef": "đầu bếp",
    "flower": "hoa",
    "platform": "đế dày",
    "cracked": "nứt",
    "borrowed": "mượn",
    "cocktail": "cocktail",
    "bovid": "họ trâu bò",
    "soccer": "bóng đá",
    "kindergarten": "mẫu giáo",
    "training": "tập luyện",
    "pride": "tự hào",
    "underwear": "đồ lót",
    "clothing": "quần áo",
    "muzzle": "mõm",
    "snout": "mõm",
    "antlers": "gạc hươu",
    "hooves": "móng guốc",
    "tuft": "túm lông",
    "tufts": "túm lông",
    "fluff": "lông xù",
    "fin": "vây",
    "fins": "vây",
    "scale": "vảy",
    "hip": "hông",
    "hips": "hông",
    "cheeks": "má",
    "jaw": "hàm",
    "tongue": "lưỡi",
    "gums": "nướu",
    "eyelid": "mí mắt",
    "eyelash": "lông mi",
    "eyelashes": "lông mi",
    "eyebrow": "lông mày",
    "eyebrows": "lông mày",
    "nostril": "lỗ mũi",
    "nostrils": "lỗ mũi",
    "pore": "lỗ chân lông",
    "pores": "lỗ chân lông",
    "wrinkle": "nếp nhăn",
    "wrinkles": "nếp nhăn",
    "makeup": "trang điểm",
    "nail": "móng",
    "nails": "móng",
    "freckles": "tàn nhang",
    "tattoo": "hình xăm",
    "piercing": "khuyên",
    "bruise": "vết bầm",
    "dirt": "bụi bẩn",
    "drool": "dãi",
    "tear": "giọt nước mắt",
    "spit": "nước bọt",
    "soap": "bọt xà phòng",
    "glitter": "kim tuyến",
    "dust": "bụi",
    "rain": "mưa",
    "wind": "gió",
    "stars": "sao",
    "hearts": "trái tim",
    "bones": "xương",
    "muscle": "cơ",
    "muscles": "cơ",
    "spine": "cột sống",
    "rib": "xương sườn",
    "ribs": "xương sườn",
    "employee": "nhân viên",
    "bandage": "băng quấn",
    "santa": "ông già Noel",
    "crazy": "điên loạn",
    "chastity": "trinh tiết",
    "coil": "cuộn tròn",
    "forked": "chẻ đôi",
    "chewing": "nhai",
    "leopard": "báo",
    "zebra": "ngựa vằn",
    "lemur": "culi",
    "boar": "lợn rừng",
    "goose": "ngỗng",
    "chicken": "gà",
    "lizard": "thằn lằn",
    "gecko": "tắc kè",
    "turtle": "rùa",
    "crocodile": "cá sấu",
    "wasp": "ong bắp cày",
    "moth": "bướm đêm",
    "beetle": "bọ",
    "ladybug": "bọ rùa",
    "dragonfly": "chuồn chuồn",
    "ant": "kiến",
    "worm": "giun",
    "snail": "ốc sên",
    "lobster": "tôm hùm",
    "shrimp": "tôm",
    "jellyfish": "sứa",
    "starfish": "sao biển",
    "cheetah": "báo gấm",
    "panther": "báo đen",
    "elephant": "voi",
    "rhinoceros": "tê giác",
    "hippo": "hà mã",
    "giraffe": "hươu cao cổ",
    "kangaroo": "chuột túi",
    "gorilla": "khỉ đột",
    "weasel": "chồn",
    "skunk": "chồn hôi",
    "badger": "lửng",
    "mole": "chuột chũi",
    "beaver": "hải ly",
    "bison": "bò rừng",
    "buffalo": "trâu",
    "moose": "nai sừng tấm",
    "reindeer": "tuần lộc",
    "antelope": "linh dương",
    "pony": "ngựa lùn",
    "mare": "ngựa cái",
    "cub": "con non",
    "fawn": "nai con",
    "pup": "con non",
    "lamb": "cừu con",
    "sloth": "lười",
    "wyvern": "wyvern",
    "unicorn": "kỳ lân",
    "pegasus": "ngựa có cánh",
    "serpent": "xà",
    "wyrm": "rồng không cánh",
    # Đợt 6 — vòng 2: chất liệu, phụ kiện, kiểu tóc.
    "hat": "mũ",
    "dress": "váy",
    "tie": "cà vạt",
    "shirt": "áo sơ mi",
    "color": "màu",
    "bondage": "trói buộc",
    "american": "Mỹ",
    "combat": "chiến đấu",
    "teardrop": "giọt nước",
    "slit": "xẻ tà",
    "seamed": "có đường chỉ",
    "faceless": "không mặt",
    "cybernetic": "người máy cấy ghép",
    "veiny": "lộ gân",
    "eyeball": "mắt cầu",
    "eared": "có tai",
    "tailed": "có đuôi",
    "pom": "quả bông",
    "tree": "cây",
    "grill": "nướng",
    "foil": "giấy bạc",
    "chainmail": "áo giáp lưới",
    "plate": "giáp tấm",
    "vinyl": "nhựa vinyl",
    "mesh": "lưới",
    "corduroy": "nhẵn nhung",
    "linen": "vải lanh",
    "fur_trim": "viền lông",
    "plastic": "nhựa",
    "iron": "sắt",
    "steel": "thép",
    "wood": "gỗ",
    "marble": "đá cẩm thạch",
    "porcelain": "sứ",
    "clay": "đất sét",
    "wax": "sáp",
    "honey": "mật ong",
    "soda": "nước ngọt",
    "juice": "nước ép",
    # Đợt 6 — vòng 3: hình khối, chất liệu, chủ đề trái cây/bánh.
    "curved": "cong",
    "split": "chia hai",
    "official": "chính thức",
    "pumpkin": "bí ngô",
    "clover": "cỏ ba lá",
    "flowing": "tung bay",
    "darkened": "tối đi",
    "severed": "bị cắt lìa",
    "apple": "táo",
    "painted": "vẽ họa tiết",
    "skimpy": "ít vải",
    "crooked": "vẹo",
    "embroidered": "thêu",
    "shaved": "cạo ngắn",
    "lined": "có lớp lót",
    "prosthetic": "nhân tạo",
    "dipstick": "hình que",
    "ring": "vòng",
    "gear": "bánh răng",
    "forearm": "cẳng tay",
    "foreskin": "bao quy đầu",
    "stained": "vấy bẩn",
    "cherry": "hoa anh đào",
    "peach": "đào",
    "melon": "dưa",
    "grape": "nho",
    "lemon": "chanh",
    "watermelon": "dưa hấu",
    "banana": "chuối",
    "mushroom": "nấm",
    "leaves": "lá",
    "vine": "dây leo",
    "cactus": "xương rồng",
    "bamboo": "tre",
    "pine": "thông",
    "maple": "phong",
    "oak": "sồi",
    "willow": "liễu",
    "lotus": "sen",
    "tulip": "tulip",
    "daisy": "cúc dại",
    "rose": "hoa hồng",
    "lily": "huệ",
    "sakura": "hoa anh đào",
    "sunflower": "hướng dương",
    "holly": "cây nhựa xanh",
    "ivy": "trường xuân",
    "moss": "rêu",
    "wheat": "lúa mì",
    "corn": "ngô",
    "bean": "đậu",
    "chestnut": "hạt dẻ",
    "acorn": "quả sồi",
    "walnut": "óc chó",
    "peanut": "đậu phộng",
    "almond": "hạnh nhân",
    "ginger": "gừng",
    "garlic": "tỏi",
    "onion": "hành tây",
    "pepper": "tiêu",
    "salt": "muối",
    "sugar": "đường",
    "caramel": "karamel",
    "vanilla": "vani",
    "matcha": "trà matcha",
    "cocoa": "ca cao",
    "butter": "bơ",
    "cheese": "phô mai",
    "toast": "bánh mì nướng",
    "pancake": "bánh kếp",
    "waffle": "bánh quế",
    "donut": "bánh donut",
    "pie": "bánh nhân",
    "pudding": "bánh pudding",
    "jelly": "thạch",
    # Đợt 7 — trang phục: môn thể thao, vùng miền, kiểu cắt.
    "fool": "kẻ ngốc",
    "national": "quốc gia",
    "team": "đội",
    "baseball": "bóng chày",
    "basketball": "bóng rổ",
    "volleyball": "bóng chuyền",
    "tennis": "quần vợt",
    "track": "điền kinh",
    "swim": "bơi",
    "butler": "quản gia",
    "prison": "tù",
    "tactical": "chiến thuật",
    "work": "lao động",
    "workout": "tập thể dục",
    "biker": "mô tô",
    "tribal": "văn hoá bộ lạc",
    "ornate": "cầu kỳ",
    "patchwork": "chắp vá",
    "padded": "có đệm",
    "hooded": "có mũ trùm",
    "sideless": "hở bên",
    "shoulderless": "không vai",
    "breastless": "không ngực",
    "tube": "dạng ống",
    "wringing": "vắt",
    "korean": "Hàn Quốc",
    "german": "Đức",
    "italian": "Ý",
    "french": "Pháp",
    "spanish": "Tây Ban Nha",
    "english": "Anh",
    "indian": "Ấn Độ",
    "russian": "Nga",
    "hawaiian": "Hawaii",
    "aloha": "aloha",
    "ainu": "Ainu",
    "meiji": "Minh Trị",
    "roman": "La Mã",
    "greek": "Hy Lạp",
    "egyptian": "Ai Cập",
    "arabian": "Ả Rập",
    "persian": "Ba Tư",
    "viking": "kỵ sĩ Bắc Âu",
    "native": "bản địa",
    "victorian": "thời Victoria",
    "edwardian": "thời Edward",
    "medieval": "trung cổ",
    "renaissance": "phục hưng",
    "baroque": "baroque",
    "gothic": "gothic",
    "cyberpunk": "cyberpunk",
    "steampunk": "steampunk",
    "religious": "tôn giáo",
    "ceremonial": "nghi lễ",
    "traditional": "truyền thống",
    "modern": "hiện đại",
    "futuristic": "tương lai",
    "antique": "cổ",
    "exotic": "độc lạ",
    "regal": "hoàng gia",
    "royal": "hoàng gia",
    "noble": "quý tộc",
    "peasant": "nông dân",
    "servant": "người hầu",
    "schoolgirl": "nữ sinh",
    "schoolboy": "nam sinh",
    "uniformed": "mặc đồng phục",
    "crossdressing": "mặc đồ khác giới",
    "boob": "ngực",
    "ofuda": "bùa ofuda",
    "kimono": "kimono",
    "overall": "yếm",
    "overalls": "quần yếm",
    "skirt": "chân váy",
    "belt": "thắt lưng",
    "sash": "dải thắt",
    "poncho": "áo poncho",
    "tunic": "áo tuynic",
    "doublet": "áo doublet",
    "bodice": "áo corset trên",
    "gown": "đầm dạ hội",
    "waist": "eo",
    "crotch": "đũng quần",
    "openchest": "hở ngực",
    "midriff": "vòng eo",
    "crop": "cắt ngắn",
    "highwaist": "cạp cao",
    "lowleg": "cạp trễ",
    "zippered": "có khóa kéo",
    "buttoned": "có cúc",
    "cutout": "mổ xẻ",
    "gartered": "có dây treo",
    "unarmored": "không giáp",
    "store": "cửa hàng",
    "basket": "cái rổ",
    "band": "dải băng",
    "baton": "cái gậy",
    "ruff": "diềm xếp",
    "snap": "cúc bấm",
    "worship": "sùng bái",
    "bloomers": "quần phồng",
    "pillbox": "hộp nhỏ",
    "porkpie": "porkpie",
    "hard": "bảo hộ",
    "fold": "gập",
    "pencil": "bút chì",
    "dixie": "dixie",
    "cup": "cái cốc",
    "bowl": "cái bát",
    "boat": "thuyền",
    # Đợt 9 — bố cục, bối cảnh, ánh sáng, thời tiết.
    "blurred": "mờ",
    "photo": "ảnh chụp",
    "grid": "lưới",
    "nature": "thiên nhiên",
    "halftone": "chấm nửa tông",
    "geometric": "hình học",
    "bedding": "ga trải giường",
    "tiled": "lát gạch",
    "mountainous": "núi non",
    "blank": "trống",
    "palm": "cọ",
    "bare": "trụi lá",
    "foliage": "lá cây",
    "horizon": "đường chân trời",
    "stump": "gốc cây",
    "shade": "bóng râm",
    "blossom": "hoa nở",
    "blossoms": "hoa nở",
    "raining": "trời mưa",
    "snowing": "tuyết rơi",
    "windy": "gió mạnh",
    "humid": "ẩm ướt",
    "lightning": "sấm sét",
    "twilight": "chạng vạng",
    "dusk": "hoàng hôn",
    "dawn": "bình minh",
    "restroom": "nhà vệ sinh",
    "locker": "tủ để đồ",
    "kitchen": "bếp",
    "attic": "gác xép",
    "basement": "tầng hầm",
    "corridor": "hành lang",
    "balcony": "ban công",
    "terrace": "sân thượng",
    "nightclub": "hộp đêm",
    "aquarium": "bể cá",
    "greenhouse": "nhà kính",
    "warehouse": "nhà kho",
    "garage": "gara",
    "laundry": "phòng giặt đồ",
    "infirmary": "phòng y vụ",
    "clubroom": "phòng câu lạc bộ",
    "interior": "bên trong",
    "exterior": "bên ngoài",
    "alleyway": "ngõ nhỏ",
    "lights": "đèn",
    "bulb": "bóng đèn",
    "flash": "nháy",
    "glow": "phát sáng",
    "illuminated": "được chiếu sáng",
    "lamplight": "ánh đèn",
    "sunlight": "ánh nắng",
    "moonlight": "ánh trăng",
    "luminescence": "sự phát quang",
    "broom": "cây chổi",
    "beachball": "quả bóng bãi biển",
    "umbrella": "cái ô",
    "towel": "khăn tắm",
    "wreath": "vòng hoa",
    "pot": "chậu",
    "grain": "hạt",
    "film": "phim",
    "electricity": "dòng điện",
    "skyscraper": "nhà chọc trời",
    "snowman": "người tuyết",
    "sandcastle": "lâu đài cát",
    "lighthouse": "ngọn hải đăng",
    "pavement": "vỉa hè",
    "crosswalk": "vạch qua đường",
    "fireplace": "lò sưởi",
    "chandelier": "đèn chùm",
    "bookshelf": "giá sách",
    "bookcase": "tủ sách",
    "sidewalk": "vỉa hè",
    "fence": "hàng rào",
    "gate": "cổng",
    "waterfall": "thác nước",
    "glacier": "sông băng",
    "canyon": "hẻm núi",
    "volcano": "núi lửa",
    "tornado": "lốc xoáy",
    "balloon": "bóng bay",
    "confetti": "giấy vụn trang trí",
    "bonfire": "lửa trại",
    "torch": "đèn đuốc",
    "bust": "bán thân",
    "headshot": "đầu và vai",
    "fullbody": "toàn thân",
    "sketch": "bản phác thảo",
    "sketchy": "vẽ phác",
    "page": "trang",
    "mismatch": "không khớp",
    "resolution": "độ phân giải",
    "perspective": "phối cảnh",
    "viewpoint": "góc nhìn",
    "zoom": "phóng to",
    "uncropped": "không bị cắt",
    "colorized": "được tô màu",
    "uncolored": "chưa tô màu",
    "monochrome": "đơn sắc",
    "grayscale": "xám đơn sắc",
    "muted": "trầm",
    "vivid": "rực rỡ",
    "desaturated": "giảm bão hòa",
    "extremities": "ngón tay và ngón chân",
    "swatch": "mẫu màu",
    "guide": "hướng dẫn",
    "coded": "mã hóa",
    "connection": "nối",
    "halftones": "chấm nửa tông",
    "linework": "nét vẽ",
    "rough": "thô",
    "polished": "bóng bẩy",
    "vintage": "cổ",
    "retro": "hoài cổ",
    "ancient": "cổ đại",
    "ruined": "hoang tàn",
    "abandoned": "bỏ hoang",
    "crowded": "đông đúc",
    "empty": "trống",
    "cluttered": "bừa bộn",
    "tidy": "gọn gàng",
    "cozy": "ấm cúng",
    "spooky": "đáng sợ",
    "dreamy": "mơ màng",
    "documentary": "tài liệu",
    "nostalgic": "hoài niệm",
    "ominous": "tăm tối",
    "serene": "yên bình",
    "peaceful": "yên tĩnh",
    "chaotic": "hỗn loạn",
    "surreal": "siêu thực",
    "whimsical": "kỳ ảo",
    "eerie": "rợn ngợp",
    "magical": "kỳ diệu",
    "mysterious": "bí ẩn",
    "romantic": "lãng mạn",
    "gloomy": "ảm đạm",
    "cheerful": "vui tươi",
    "melancholic": "u buồn",
    "ethereal": "thoát tục",
    "divine": "thiêng liêng",
    "infernal": "địa ngục",
    "heavenly": "thiên đường",
    "celestial": "thiên thể",
    "paradise": "thiên đường",
    "nightmare": "ác mộng",
    "dream": "giấc mơ",
    "restraint": "dây trói",
    "musky": "có mùi xạ hương",
    "neonate": "sơ sinh",
    "husky": "chó husky",
    "malamute": "chó malămut",
    "oceanic": "đại dương",
    "mountain": "núi",
    "swamp": "đầm lầy",
    "desert": "sa mạc",
    "jungle": "rừng rậm",
    "meadow": "đồng cỏ",
    "valley": "thung lũng",
    "island": "đảo",
    "peninsula": "bán đảo",
    "archipelago": "quần đảo",
    "reef": "rạn san hô",
    "tide": "thủy triều",
    "wave": "sóng",
    # Đợt 10 — mắt, đuôi, tóc, phụ kiện tóc, giới từ trên cơ thể.
    "hypnotic": "bị thôi miên",
    "compound": "kép",
    "bloodshot": "vằn máu",
    "hollow": "trũng",
    "derp": "ngơ ngác",
    "obscured": "bị che",
    "crossed": "chéo",
    "unusually": "bất thường",
    "sparkly": "lấp lánh",
    "glittery": "lấp lánh",
    "droning": "trố",
    "bulging": "lồi",
    "contracted": "co lại",
    "dilated": "giãn",
    "pupillary": "thu đồng tử",
    "tapering": "vót nhọn",
    "entwined": "quấn vào nhau",
    "cleft": "chẻ đôi",
    "nub": "cộc",
    "cable": "cáp",
    "cetacean": "họ cá voi",
    "tapir": "heo vòi",
    "crocodilian": "họ cá sấu",
    "axolotl": "kỳ giông axolotl",
    "raptor": "khủng long săn mồi",
    "corvid": "họ quạ",
    "sylvan": "rừng",
    "eastern": "phương Đông",
    "western": "phương Tây",
    "nubbin": "mẫu",
    "mane": "bờm",
    "stray": "lơ thơ",
    "sideburns": "tóc mai",
    "sideburn": "tóc mai",
    "extensions": "tóc nối",
    "swoop": "chổm",
    "splayed": "xòe",
    "rounded": "tròn",
    "quiff": "ngửa ra sau",
    "drill": "xoắn ốc",
    "ringlets": "cuộn",
    "tucked": "giắt",
    "unkempt": "bờ phơ",
    "slicked": "vuốt",
    "greasy": "bết dầu",
    "bandaid": "băng cá nhân",
    "character": "nhân vật",
    "dice": "xúc xắc",
    "lifebuoy": "phao cứu sinh",
    "smokestack": "ống khói",
    "hexagon": "lục giác",
    "pentagon": "ngũ giác",
    "screw": "ốc vít",
    "nut": "đai ốc",
    "bolt": "bu lông",
    "fan": "quạt",
    "planet": "hành tinh",
    "crown": "vương miện",
    "antenna": "ăng-ten",
    "antennae": "ăng-ten",
    "gill": "mang",
    "gauze": "gạc y tế",
    "urine": "nước tiểu",
    "feces": "phân",
    "vomit": "nôn",
    "writing": "chữ viết",
    "paint": "sơn",
    "splatter": "văng",
    "lipstick": "son môi",
    "mark": "vệt",
    "sticker": "nhãn dán",
    "stamp": "dấu",
    "mud": "bùn",
    "syrup": "xi-rô",
    "case": "bao đựng",
    "diving": "lặn",
    "snorkel": "ống thở",
    "welding": "hàn",
    "sleep": "ngủ",
    "triple": "ba",
    "quad": "bốn",
    "quintuple": "năm",
    "single_pair": "một đôi",
    "shading": "tô bóng",
    "internal": "bên trong",
    "organs": "nội tạng",
    "organ": "nội tạng",
    "parking": "đậu xe",
    "entry": "lối vào",
    "sale": "bán",
    "present": "xuất hiện",
    "lube": "chất bôi trơn",
    "dickey": "cổ áo giả",
    "dickie": "cổ áo giả",
    "smiley": "mỉm cười",
    "legwear": "quần tất",
    "rudder": "bánh lái",
    "lounge": "thư giãn",
    "drawing": "vẽ",
    "tablet": "bảng vẽ",
    "sketching": "phác",
    "mammal": "động vật có vú",
    "nonmammal": "loài không có vú",
    "position": "vị thế",
    "positions": "vị thế",
})
# Tính từ mô tả (hình thái, trạng thái, kích thước) — khác với danh từ bổ nghĩa ở
# vị trí trong cụm tiếng Việt, nên tách riêng.
_TAG_VI_ADJECTIVE_MODIFIERS = frozenset({
    "long", "short", "medium", "very", "small", "big", "large", "huge", "thick", "thin",
    "curly", "wavy", "straight", "messy", "braided", "fluffy", "spiky", "pointy", "round",
    "wide", "narrow", "open", "closed", "multicolored", "striped", "plaid", "checkered",
    "dotted", "spotted", "gradient", "marbled", "torn", "ripped", "wet", "dry", "tight",
    "loose", "sheer", "shiny", "glossy", "matte", "glowing", "lit", "shadowed", "frilled",
    "ruffled", "pleated", "textured", "smooth", "furry", "bald", "clean", "dirty", "bloody",
    "sweaty", "dusty", "old", "new", "mini", "micro", "half", "partial", "partially",
    "mostly", "barely", "muscular", "chubby", "overweight", "obese", "slim", "fit", "tall",
    "young", "feminine", "masculine", "cute", "sexy", "formal", "casual", "sporty",
    "patterned", "translucent", "transparent", "puffy", "cropped", "spiked", "golden",
    "single", "double", "extra", "matching", "asymmetrical",
    "light", "dark", "bright", "pale",
    "own", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "multi", "piece", "tone", "shaped", "framed", "trimmed", "covered", "covering",
    "missing", "sleeveless", "strapless", "collared", "laced", "bandaged", "freckled",
    "raised", "spread", "flat", "pointed", "curvy", "mismatched", "mechanical", "tinted",
    "layered", "impossible", "visible", "invisible", "glistening", "unworn", "worn",
    "hyper", "inner", "outer", "high", "low", "sports", "longer", "shorter", "older",
    "younger", "larger", "smaller", "taller",
    "monotone", "solid", "spiral", "creepy", "dull", "sharp", "bushy", "prick", "lop",
    "circle", "sleepy", "tired", "teary", "upturned", "downcast", "spade", "cross",
    "armored", "winged", "notched", "toeless", "flipped", "backless", "halter",
    "triangle", "ribbed", "quilted",
    "skinny", "hairy", "sparkling", "taut", "curled", "detached", "oversized", "uneven", "lidded", "diagonal", "argyle", "turtleneck", "wagging",
    "detailed", "broken", "frilly", "floral", "alternate", "horned", "furred", "hairless", "studded", "iridescent", "flaming", "fiery", "vertical", "horizontal", "nearly", "unusual", "multiple", "linked", "disembodied", "heeled", "pseudo",
})
# Danh từ bổ nghĩa (vật liệu, loài, bộ phận, đồ vật) đứng ngay sau danh từ chính và
# ngược thứ tự so với tiếng Anh: rabbit_ear_hat → "Mũ tai thỏ".
_TAG_VI_NOUN_MODIFIERS = frozenset(
    part for part in _TAG_VI_WORDS
    if part not in _TAG_VI_ADJECTIVE_MODIFIERS and part not in _TAG_VI_COLORS
)


_TAG_VI_COMPOSITE_HEADS = {
    "hair": "Tóc", "eyes": "Mắt", "background": "Nền", "skin": "Da",
    "dress": "Váy", "shirt": "Áo sơ mi", "skirt": "Chân váy", "sleeves": "Tay áo",
    "gloves": "Găng tay", "boots": "Bốt", "shoes": "Giày", "socks": "Tất",
    "stockings": "Bít tất", "ears": "Tai", "tail": "Đuôi", "wings": "Cánh",
    "horns": "Sừng", "breasts": "Ngực", "hands": "Bàn tay", "feet": "Bàn chân",
    "fingers": "Ngón tay", "toes": "Ngón chân", "arms": "Cánh tay", "legs": "Chân",
    "thighs": "Đùi", "mouth": "Miệng", "face": "Khuôn mặt", "teeth": "Răng",
    "tongue": "Lưỡi", "glasses": "Kính", "hat": "Mũ", "jacket": "Áo khoác",
    "coat": "Áo khoác", "pants": "Quần", "shorts": "Quần short", "body": "Cơ thể",
    "fur": "Lông", "panties": "Quần lót", "pantyhose": "Quần tất",
    "thighhighs": "Tất cao đùi", "nose": "Mũi", "bow": "Nơ",
    "ribbon": "Ruy băng", "necklace": "Vòng cổ", "flower": "Hoa", "flowers": "Hoa",
    "sky": "Bầu trời", "cloud": "Mây", "clouds": "Mây",
}
# Danh từ ghép được mở rộng theo các nhóm hay gặp nhất trong catalog. Mỗi mục ở đây
# Mỗi mục mở ra một họ thẻ "<modifier>_<danh từ>" mà không phải liệt kê từng thẻ một.
_TAG_VI_COMPOSITE_HEADS.update({
    # Đợt 11 — danh từ chính mới (đồ vật cầm tay, họ loài, khung hình).
    "bouquet": "Bó hoa",
    "fork": "Dĩa",
    "spoon": "Thìa",
    "chopsticks": "Đôi đũa",
    "wand": "Cây đũa phép",
    "pen": "Bút máy",
    "axe": "Rìu",
    "popsicle": "Cây kem que",
    "scythe": "Lưỡi hái",
    "dagger": "Dao găm",
    "hammer": "Búa",
    "paintbrush": "Cọ vẽ",
    "guitar": "Đàn guitar",
    "syringe": "Ống tiêm",
    "controller": "Tay cầm chơi game",
    "can": "Lon",
    "vase": "Lọ hoa",
    "clipboard": "Bảng kẹp giấy",
    "microphone": "Micrô",
    "mop": "Cây lau nhà",
    "bucket": "Xô",
    "ladder": "Thang",
    "pompoms": "Bông cổ vũ",
    "genitalia": "Bộ phận sinh dục",
    "cumshot": "Tinh dịch bắn ra",
    "sweater_vest": "Áo len gilê",
    "stuffed_toy": "Thú nhồi bông",
    # Đợt 11 — danh từ chính mới (đồ vật cầm tay, họ loài, khung hình).
    "smoking_pipe": "Tẩu thuốc",
    "game_controller": "Tay cầm chơi game",
    "handheld_console": "Máy chơi game cầm tay",
    "ranged_weapon": "Vũ khí tầm xa",
    "poke_ball": "Poké Ball",
    "gohei": "Gậy gohei",
    "difference": "Chênh lệch",
    "frame": "Khung hình",
    "playtime": "Giờ chơi",
    "visor_cap": "Mũ lưỡi trai có lưới chắn",
    "thong_bikini": "Bikini dây",
    "potted_plant": "Chậu cây cảnh",
    "rope_belt": "Thắt lưng dây",
    # Trang phục
    "uniform": "Đồng phục", "clothes": "Quần áo", "clothing": "Trang phục",
    "underwear": "Đồ lót", "swimwear": "Đồ bơi", "swimsuit": "Đồ bơi",
    "bikini": "Bikini", "bra": "Áo ngực", "thong": "Quần lót dây",
    "top": "Áo", "tank": "Áo ba lỗ", "costume": "Trang phục", "suit": "Bộ đồ",
    "bodysuit": "Đồ liền thân", "leotard": "Leotard", "jumpsuit": "Áo liền quần",
    "sweater": "Áo len", "cardigan": "Áo cardigan", "vest": "Áo ghi-lê",
    "hoodie": "Áo hoodie", "hood": "Mũ trùm", "kimono": "Kimono",
    "cloak": "Áo choàng", "cape": "Áo choàng", "apron": "Tạp dề",
    "armor": "Giáp", "helmet": "Mũ bảo hiểm", "cap": "Mũ",
    "headband": "Băng đô", "headwear": "Phụ kiện đầu", "legwear": "Đồ che chân",
    "armwear": "Đồ che tay", "lingerie": "Đồ lót nữ", "sleepwear": "Đồ ngủ",
    "sneakers": "Giày thể thao", "heels": "Giày cao gót", "loafers": "Giày lười",
    # Phụ kiện
    "belt": "Thắt lưng", "buckle": "Khóa", "collar": "Cổ áo", "tie": "Cà vạt",
    "necktie": "Cà vạt", "bowtie": "Nơ cổ", "scarf": "Khăn quàng",
    "veil": "Màn che mặt", "goggles": "Kính bảo hộ", "eyewear": "Kính",
    "jewelry": "Trang sức", "bracelet": "Vòng tay", "anklet": "Vòng chân",
    "earring": "Khuyên tai", "earrings": "Khuyên tai", "pendant": "Mặt dây chuyền",
    "tiara": "Vương miện nhỏ", "crown": "Vương miện", "brooch": "Trâm cài",
    "tattoo": "Hình xăm", "piercing": "Khuyên", "makeup": "Trang điểm",
    "lipstick": "Son môi", "perfume": "Nước hoa", "handbag": "Túi xách",
    "bag": "Túi", "backpack": "Ba lô", "purse": "Túi cầm tay",
    # Cơ thể
    "eye": "Mắt", "eyelid": "Mí mắt", "eyelash": "Lông mi", "eyelashes": "Lông mi",
    "eyebrow": "Lông mày", "eyebrows": "Lông mày", "pupil": "Đồng tử",
    "pupils": "Đồng tử", "sclera": "Củng mạc", "iris": "Mống mắt",
    "ear": "Tai", "horn": "Sừng", "wing": "Cánh", "nose": "Mũi",
    "nostril": "Lỗ mũi", "snout": "Mõm", "muzzle": "Mõm", "beak": "Mỏ",
    "lip": "Môi", "lips": "Môi", "cheek": "Má", "cheeks": "Má",
    "chin": "Cằm", "jaw": "Hàm", "forehead": "Trán", "temple": "Thái dương",
    "neck": "Cổ", "throat": "Cổ họng", "shoulder": "Vai", "shoulders": "Vai",
    "chest": "Ngực", "pecs": "Cơ ngực", "pectorals": "Cơ ngực",
    "back": "Lưng", "spine": "Cột sống", "waist": "Eo", "hip": "Hông",
    "hips": "Hông", "belly": "Bụng", "stomach": "Bụng", "navel": "Rốn",
    "armpit": "Nách", "armpits": "Nách", "butt": "Mông", "ass": "Mông",
    "arm": "Cánh tay", "elbow": "Khuỷu tay", "wrist": "Cổ tay",
    "hand": "Bàn tay", "palm": "Lòng bàn tay", "finger": "Ngón tay",
    "thumb": "Ngón cái", "nail": "Móng tay", "nails": "Móng tay",
    "fingernails": "Móng tay", "toenails": "Móng chân", "leg": "Chân",
    "thigh": "Đùi", "knee": "Đầu gối", "knees": "Đầu gối", "calf": "Bắp chân",
    "heel": "Gót chân", "sole": "Lòng bàn chân", "paw": "Bàn chân thú",
    "paws": "Bàn chân thú", "hindpaw": "Bàn chân sau", "forepaw": "Bàn chân trước",
    # Sinh dục (nhãn mô tả, khớp cách Studio đang hiển thị)
    "penis": "Dương vật", "dick": "Dương vật", "cock": "Dương vật",
    "balls": "Tinh hoàn", "testicles": "Tinh hoàn", "scrotum": "Bìu",
    "vulva": "Âm hộ", "pussy": "Âm hộ", "vagina": "Âm đạo",
    "anus": "Hậu môn", "asshole": "Hậu môn", "clitoris": "Âm vật",
    "areola": "Quầng vú", "areolas": "Quầng vú", "nipple": "Núm vú",
    "sheath": "Bao dương vật", "ovipositor": "Cơ quan đẻ trứng",
})
# Lớp danh từ chính thứ hai: thêm một mục ở đây mở khóa cả họ "<bổ ngữ>_<danh từ>"
# mà không phải liệt kê từng thẻ; quy tắc ghép vẫn chỉ chạy khi MỌI token bổ ngữ đã biết.
_TAG_VI_COMPOSITE_HEADS.update({
    "anthro": "Nhân vật nhân hóa", "feral": "Nhân vật bán thú", "kemono": "Kemono",
    "humanoid": "Dạng người", "human": "Người", "girl": "Cô gái", "boy": "Cậu trai",
    "female": "Nhân vật nữ", "male": "Nhân vật nam", "intersex": "Người liên giới tính",
    "gynomorph": "Nhân vật gynomorph", "andromorph": "Nhân vật andromorph",
    "print": "Họa tiết", "pattern": "Hoa văn", "markings": "Vằn", "mark": "Dấu",
    "tuft": "Túm lông", "fluff": "Lông tơ", "nape": "Gáy",
    "ponytail": "Tóc đuôi ngựa", "twintails": "Tóc hai búi", "braid": "Bím tóc",
    "braids": "Bím tóc", "drill": "Tóc xoắn", "drills": "Tóc xoắn",
    "curls": "Lọn tóc xoăn", "curl": "Lọn tóc xoăn", "bun": "Búi tóc",
    "bangs": "Tóc mái", "hairband": "Băng buộc tóc", "hairpin": "Kẹp tóc",
    "ringlets": "Tóc xoăn lọn", "pigtails": "Tóc hai bím",
    "areolas": "Quầng vú", "prepuce": "Da quy đầu", "knot": "Nút thắt",
    "testes": "Tinh hoàn", "cloaca": "Lỗ huyệt", "maw": "Hàm",
    "whiskers": "Râu mép", "chestplate": "Giáp ngực", "pauldron": "Giáp vai",
    "focus": "Tập trung vào", "position": "Tư thế", "view": "Góc nhìn",
    "angle": "Góc máy", "portrait": "Ảnh chân dung", "shot": "Cắt cảnh",
    "penetration": "Sự thâm nhập", "masturbation": "Thủ dâm", "orgasm": "Lên đỉnh",
    "ejaculation": "Xuất tinh", "insertion": "Đưa vào", "grab": "Bóp",
    "pull": "Kéo", "lift": "Nâng lên", "slip": "Trượt hở", "cutout": "Lỗ cắt",
    "blush": "Đỏ mặt", "smile": "Nụ cười", "frown": "Nhăn mặt",
    "expression": "Biểu cảm", "fluids": "Dịch", "wear": "Đồ mặc",
    "topwear": "Áo", "bottomwear": "Đồ mặc dưới", "footwear": "Giày dép",
    "headwear": "Phụ kiện đầu", "neckwear": "Phụ kiện cổ", "wristwear": "Phụ kiện cổ tay",
    "halo": "Vầng hào quang", "choker": "Vòng cổ", "neckerchief": "Khăn quàng cổ",
    "gauntlet": "Găng tay giáp", "capelet": "Áo choàng ngắn", "scrunchie": "Dây buộc tóc",
    "hair_ornament": "Phụ kiện tóc", "ornament": "Đồ trang trí", "eyeliner": "Kẻ mắt",
    "border": "Viền ảnh", "motif": "Họa tiết", "watermark": "Dấu bản quyền",
    "strap": "Dây đeo", "straps": "Dây đeo", "pouch": "Túi con",
    "ball": "Quả bóng", "bowl": "Bát", "mug": "Cốc có quai", "page": "Trang",
    "poster": "Áp phích", "number": "Chữ số", "name": "Tên", "prey": "Con mồi",
    "predator": "Vật săn mồi", "partner": "Bạn tình", "sibling": "Anh chị em",
    "brother": "Anh em trai", "sister": "Chị em gái", "idol": "Idol",
})


# Với các danh từ này, tiếng Việt tự nhiên hơn khi nối màu bằng "màu".
_TAG_VI_COLOR_MARKER_HEADS = frozenset({
    "hair", "background", "skin", "dress", "shirt", "skirt", "gloves", "boots", "shoes",
    "clothes", "clothing", "uniform", "underwear", "swimsuit", "swimwear", "bikini", "bra",
    "top", "suit", "jacket", "coat", "sweater", "vest", "hood", "cape", "cloak", "apron",
    "socks", "stockings", "thighhighs", "kneehighs", "panties", "pantyhose", "shorts",
    "pants", "legwear", "armwear", "jewelry", "collar", "hat", "cap", "helmet",
})


_TAG_VI_COMPOSITE_HEADS.update({
    "less": "Không có", "symbol": "Biểu tượng", "logo": "Logo", "text": "Chữ",
    "theme": "Chủ đề", "outline": "Đường viền", "jewel": "Đá quý",

    "blush": "Má hồng",
    "dimples": "Lúm đồng tiền",
    "eyeliner": "Kẻ mắt",
    "eyeshadow": "Phấn mắt",
    "freckles": "Tàn nhang",
    "handwear": "Đồ che tay",
    "highlighter": "Phấn bắt sáng",
    "makeup": "Trang điểm",
    "mascara": "Mascara",
    "moles": "Nốt ruồi",
    "note": "Nốt nhạc",
})

_TAG_VI_COMPOSITE_HEADS.update({
    # Đợt 4: danh từ hình khối hay đứng cuối cụm chi tiết.
    "heart": "Trái tim", "star": "Ngôi sao", "moon": "Mặt trăng", "cloud": "Đám mây",
    "flame": "Ngọn lửa", "drop": "Giọt", "leaf": "Lá", "bone": "Xương", "feather": "Lông vũ",
    "bell": "Chuông", "ribbon": "Ruy băng", "bow": "Nơ", "skull": "Đầu lâu", "anchor": "Cái neo", "snowflake": "Bông tuyết",
    "spiral": "Đường xoáy", "hourglass": "Đồng hồ cát", "clock": "Đồng hồ",
    "accessories": "Phụ kiện",
    "accessory": "Phụ kiện",
    "ahoge": "Tóc ahoge",
    "armpits": "Nách",
    "bands": "Dải",
    "bellies": "Bụng",
    "belts": "Thắt lưng",
    "bows": "Nơ",
    "buttocks": "Mông",
    "calves": "Bắp chân",
    "caps": "Mũ",
    "censor": "Kiểm duyệt",
    "claws": "Móng vuốt",
    "curls": "Lọn tóc xoăn",
    "dresses": "Váy liền",
    "elbows": "Khuỷu tay",
    "eyeball": "Nhãn cầu",
    "eyeballs": "Nhãn cầu",
    "fangs": "Nanh",
    "feathers": "Lông vũ",
    "fingers": "Ngón tay",
    "hair_tubes": "Lọn tóc ống",
    "hairband": "Băng đô",
    "hairbands": "Băng đô",
    "hats": "Mũ",
    "headdress": "Đồ đội đầu",
    "headdresses": "Đồ đội đầu",
    "hooves": "Vó",
    "horns": "Sừng",
    "knees": "Đầu gối",
    "navels": "Rốn",
    "nostrils": "Lỗ mũi",
    "pasties": "Miếng che ngực",
    "ribbons": "Ruy băng",
    "shirts": "Áo sơ mi",
    "skirts": "Chân váy",
    "soles": "Bàn chân",
    "spear": "Ngọn giáo",
    "spears": "Ngọn giáo",
    "straps": "Dây đeo",
    "tails": "Đuôi",
    "ties": "Cà vạt",
    "toes": "Ngón chân",
    "tresses": "Lọn tóc",
    "tubes": "Ống",
    "whiskers": "Râu mèo",
    "boot": "Bốt",
    "eyelids": "Mí mắt",
    "focus": "Trọng tâm",
    "glove": "Găng tay",
    "growth": "Sự mọc",
    "latex": "Cao su",
    "office_chair": "Ghế văn phòng",
    "overskirt": "Chân váy phủ ngoài",
    "patch": "Miếng vá",
    "pattern": "Hoa văn",
    "seam": "Đường may",
    "seams": "Đường may",
    "skinsuit": "Bộ đồ ôm sát",
    "strand": "Lọn tóc",
    "tentacle": "Xúc tu",
    "tentacles": "Xúc tu",
    "tip": "Đầu chóp",
    "tips": "Đầu chóp",
    "underskirt": "Chân váy lót",
    "ape": "Vượn",
    "bed": "Giường",
    "bee": "Ong",
    "beetle": "Bọ cánh cứng",
    "bench": "Ghế dài",
    "bicycle": "Xe đạp",
    "bird": "Chim",
    "blanket": "Chăn",
    "boat": "Thuyền",
    "book": "Quyển sách",
    "bottle": "Cái chai",
    "bunny": "Thỏ",
    "bus": "Xe buýt",
    "butterfly": "Bướm",
    "camera": "Máy ảnh",
    "candle": "Nến",
    "car": "Ô tô",
    "carpet": "Thảm",
    "cat": "Mèo",
    "chair": "Ghế",
    "chairs": "Ghế",
    "chick": "Gà con",
    "chicken": "Gà",
    "couch": "Ghế sofa",
    "cow": "Bò",
    "crab": "Cua",
    "cup": "Cốc",
    "curtain": "Rèm",
    "deer": "Hươu",
    "desk": "Bàn làm việc",
    "dog": "Chó",
    "dolphin": "Cá heo",
    "door": "Cánh cửa",
    "drink": "Đồ uống",
    "duck": "Vịt",
    "elephant": "Voi",
    "ferret": "Chồn",
    "fish": "Cá",
    "floor": "Sàn nhà",
    "flower": "Hoa",
    "food": "Đồ ăn",
    "frog": "Ếch",
    "fur": "Lông",
    "giraffe": "Hươu cao cổ",
    "goat": "Dê",
    "gorilla": "Khỉ đột",
    "ground": "Mặt đất",
    "hamster": "Chuột hamster",
    "hedgehog": "Nhím",
    "hippo": "Hà mã",
    "kitten": "Mèo con",
    "koala": "Gấu túi",
    "ladybug": "Bọ rùa",
    "lamp": "Đèn",
    "lizard": "Thằn lằn",
    "lobster": "Tôm hùm",
    "mirror": "Cái gương",
    "monkey": "Khỉ",
    "moth": "Ngài",
    "motorcycle": "Xe máy",
    "mouse": "Chuột nhắt",
    "octopus": "Bạch tuộc",
    "otter": "Rái cá",
    "owl": "Cú",
    "panda": "Gấu trúc",
    "phone": "Điện thoại",
    "pig": "Lợn",
    "pillow": "Gối",
    "plane": "Máy bay",
    "plant": "Cây cảnh",
    "plate": "Cái đĩa",
    "pony": "Ngựa Pony",
    "pose": "Tư thế",
    "puppy": "Chó con",
    "rabbit": "Thỏ",
    "rhino": "Tê giác",
    "rock": "Tảng đá",
    "scorpion": "Bọ cạp",
    "seal": "Hải cẩu",
    "shark": "Cá mập",
    "sheep": "Cừu",
    "ship": "Con tàu",
    "sink": "Bồn rửa",
    "snake": "Rắn",
    "sofa": "Ghế sofa",
    "spider": "Nhện",
    "squid": "Mực ống",
    "squirrel": "Sóc",
    "stairs": "Cầu thang",
    "stool": "Ghế đẩu",
    "table": "Bàn",
    "toilet": "Bồn cầu",
    "tree": "Cây",
    "truck": "Xe tải",
    "turtle": "Rùa",
    "wall": "Bức tường",
    "wasp": "Ong bắp cày",
    "whale": "Cá voi",
    "window": "Cửa sổ",
    "zebra": "Ngựa vằn",
    # Đợt 6 — chi tiết nhân vật:
    "ring": "Nhẫn",
    "beard": "Râu",
    "headgear": "Phụ kiện đội đầu",
    "foreskin": "Bao quy đầu",
    "eyepatch": "Miếng bịt mắt",
    "tears": "Nước mắt",
    "furniture": "Nội thất",
    "hairclip": "Kẹp tóc",
    "miniskirt": "Chân váy ngắn",
    "cuffs": "Ống bọc cổ tay",
    "spots": "Đốm",
    "outerwear": "Áo khoác ngoài",
    "sweatshirt": "Áo nỉ",
    "bondage": "Trói buộc",
    "paint": "Màu vẽ",
    "nipples": "Núm vú",
    "wraps": "Băng quấn",
    "sitting": "Tư thế ngồi",
    "expansion": "Độ giãn",
    "headphones": "Tai nghe",
    "tailband": "Dải buộc đuôi",
    "facepaint": "Vẽ mặt",
    "weapon": "Vũ khí",
    "chain": "Dây chuyền",
    "mask": "Mặt nạ",
    "gem": "Đá quý",
    "lineart": "Nét vẽ",
    "gear": "Bánh răng",
    "tag": "Nhãn",
    "scales": "Vảy",
    "scar": "Sẹo",
    "garter": "Dây treo",
    "tailcoat": "Áo đuôi tôm",
    "torture": "Tra tấn",
    "fan": "Quạt",
    "earbuds": "Tai nghe nhét tai",
    "fins": "Vây",
    "frill": "Diềm xếp",
    "mane": "Bờm",
    "birthmark": "Nốt bớt",
    "forearms": "Cẳng tay",
    "forearm": "Cẳng tay",
    "head": "Đầu",
    "spikes": "Gai nhọn",
    "undershirt": "Áo lót trong",
    "bulge": "Chỗ lùm lùm",
    "chocolate": "Sô-cô-la",
    "gum": "Kẹo cao su",
    "balloon": "Bóng bay",
    "screen": "Màn hình",
    "t_shirt": "Áo thun",
    "top_hat": "Mũ chóp cao",
    "antenna": "Ăng-ten",
    "antennae": "Ăng-ten",
    "drawing": "Hình vẽ",
    "hair_ring": "Vòng tóc",
    "hair_ties": "Cột tóc",
    "hair_tie": "Cột tóc",
    "collar_tag": "Nhãn vòng cổ",
    "foundation": "Kem nền",
    "nail_polish": "Sơn móng tay",
    "manicure": "Sửa móng tay",
    "freckle": "Tàn nhang",
    "mole": "Chuột chũi",
    "dimple": "Lúm đồng tiền",
    "unibrow": "Lông mày rậm liền",
    "stubble": "Râu lởm chởm",
    "moustache": "Ria mép",
    "sideburns": "Tóc mai",
    "sideburn": "Tóc mai",
    "side_ponytail": "Tóc đuôi ngựa lệch bên",
    "fringe": "Mái",
    "bear": "Gấu",
    "fox": "Cáo",
    "wolf": "Sói",
    "lemur": "Culi",
    "raccoon": "Gấu mèo",
    "horse": "Ngựa",
    "boar": "Lợn rừng",
    "rat": "Chuột cống",
    "eagle": "Đại bàng",
    "crow": "Quạ",
    "swan": "Thiên nga",
    "goose": "Ngỗng",
    "bat": "Dơi",
    "gecko": "Tắc kè",
    "crocodile": "Cá sấu",
    "dragonfly": "Chuồn chuồn",
    "ant": "Kiến",
    "worm": "Giun",
    "snail": "Ốc sên",
    "shrimp": "Tôm",
    "jellyfish": "Sứa",
    "starfish": "Sao biển",
    "tiger": "Hổ",
    "lion": "Sư tử",
    "leopard": "Báo",
    "cheetah": "Báo gấm",
    "panther": "Báo đen",
    "hyena": "Linh cẩu",
    "rhinoceros": "Tê giác",
    "camel": "Lạc đà",
    "kangaroo": "Chuột túi",
    "penguin": "Cánh cụt",
    "weasel": "Chồn",
    "skunk": "Chồn hôi",
    "badger": "Lửng",
    "beaver": "Hải ly",
    "bison": "Bò rừng",
    "buffalo": "Trâu",
    "moose": "Nai sừng tấm",
    "reindeer": "Tuần lộc",
    "antelope": "Linh dương",
    "mare": "Ngựa cái",
    "cub": "Con non",
    "fawn": "Nai con",
    "pup": "Con non",
    "lamb": "Cừu con",
    "sloth": "Con lười",
    "wyvern": "Wyvern",
    "dragon": "Rồng",
    "unicorn": "Kỳ lân",
    "pegasus": "Ngựa có cánh",
    "mermaid": "Nàng tiên cá",
    "serpent": "Xà",
    "wyrm": "Rồng không cánh",
    # Đợt 6 — vòng 2: chất liệu, phụ kiện, kiểu tóc.
    "color": "Màu",
    "transformation": "Sự biến đổi",
    "warmers": "Băng giữ ấm",
    "animal": "Động vật",
    "stud": "Khuyên nụ",
    "lock": "Lọn tóc",
    "day": "Ngày",
    "hairstyle": "Kiểu tóc",
    "creature": "Sinh vật",
    "cookie": "Bánh quy",
    "corset": "Áo corset",
    "armlet": "Vòng đeo tay",
    "stocking": "Tất dài",
    "removal": "Sự tháo bỏ",
    "beads": "Hạt chuỗi",
    "booties": "Giày mềm",
    "stripe": "Sọc",
    "ridge": "Gờ",
    "hold": "Cái giữ",
    "fetish": "Sở thích",
    "swing": "Cái đu",
    "loss": "Sự mất",
    "glue": "Keo",
    "sticker": "Nhãn dán",
    "stamp": "Con dấu",
    "seal_stamp": "Con dấu",
    "decoration": "Đồ trang trí",
    "emblem": "Huy hiệu",
    "button": "Cúc áo",
    "zipper": "Khóa kéo",
    "pocket": "Túi áo",
    "hem": "Gấu áo",
    "sleeve": "Tay áo",
    "cuff": "Ống tay",
    "cloaks": "Áo choàng",
    "tee": "Áo thun",
    "tank_top": "Áo hai dây",
    "garter_belt": "Dây treo tất",
    "garters": "Dây treo tất",
    "kneehighs": "Tất đầu gối",
    "leggings": "Quần legging",
    "onesie": "Bộ liền thân",
    "pajamas": "Đồ ngủ",
    "nightgown": "Áo ngủ",
    "robe": "Áo choàng",
    "bathrobe": "Áo choàng tắm",
    "yukata": "Áo yukata",
    "hakama": "Quần hakama",
    "sari": "Váy sari",
    "cheongsam": "Xường xám",
    "hanbok": "Áo hanbok",
    # Đợt 7 — trang phục: bộ phận món đồ.
    "bloomers": "Quần phồng",
    "ruff": "Diềm cổ",
    "tassel": "Chùm tua rua",
    "band": "Dải băng",
    "basket": "Cái rổ",
    "baton": "Cái gậy",
    "snap": "Cúc bấm",
    "store": "Cửa hàng",
    "worship": "Sự sùng bái",
    "overhang": "Vạt thừa",
    "upskirt": "Ảnh hớ váy",
    "flip": "Cái lật",
    "poncho": "Áo poncho",
    "tunic": "Áo tuynic",
    "gown": "Đầm dạ hội",
    "doublet": "Áo doublet",
    "bodice": "Áo corset",
    "dirndl": "Váy dirndl",
    # Đợt 9 — danh từ chính cho bố cục và bối cảnh.
    "lights": "Đèn",
    "room": "Căn phòng",
    "interior": "Bên trong",
    "exterior": "Bên ngoài",
    "sketch": "Bản phác thảo",
    "pages": "Trang",
    "shadow": "Bóng",
    "shadows": "Bóng",
    "highlights": "Điểm nhấn",
    "colors": "Màu",
    "palette": "Bảng màu",
    "horizon": "Đường chân trời",
    "skyline": "Đường chân trời thành phố",
    "stump": "Gốc cây",
    "wreath": "Vòng hoa",
    "pot": "Chậu",
    "grain": "Hạt phim",
    "bulb": "Bóng đèn",
    "trail": "Vệt",
    "beam": "Chùm",
    "ray": "Tia",
    "rays": "Tia",
    "field": "Cánh đồng",
    "meadow": "Đồng cỏ",
    "tower": "Tháp",
    "skyscraper": "Nhà chọc trời",
    "snowman": "Người tuyết",
    "umbrella": "Cái ô",
    "towel": "Khăn tắm",
    "broom": "Cây chổi",
    "lantern": "Đèn lồng",
    "fence": "Hàng rào",
    "gate": "Cổng",
    "pond": "Ao nước",
    "waterfall": "Thác nước",
    "cliff": "Vách đá",
    "cave": "Hang động",
    "island": "Đảo",
    "reef": "Rạn san hô",
    "swamp": "Đầm lầy",
    "valley": "Thung lũng",
    "glacier": "Sông băng",
    "canyon": "Hẻm núi",
    "volcano": "Núi lửa",
    "mural": "Tranh tường",
    "banner": "Biểu ngữ",
    "signboard": "Biển hiệu",
    "billboard": "Biển quảng cáo",
    "painting": "Bức tranh",
    "drawings": "Bản vẽ",
    "sketchbook": "Sổ phác thảo",
    "viewfinder": "Ống ngắm",
    # Đợt 10 — danh từ chính cho phụ kiện và hoa văn cơ thể.
    "hair_ornaments": "Phụ kiện tóc",
    "ornaments": "Đồ trang trí",
    "headset": "Tai nghe",
    "hair_tube": "Ống tóc",
    "hair_extensions": "Tóc nối",
    "hair_ribbon": "Ruy băng tóc",
    "hair_bow": "Nơ tóc",
    "hair_flower": "Hoa cài tóc",
    "hair_scrunchie": "Lạt tóc scrunchie",
    "hair_feather": "Lông chim cài tóc",
    "hair_horns": "Sừng cài tóc",
    "tail_ornament": "Trang trí đuôi",
    "ear_ornament": "Trang trí tai",
    "head_ornament": "Đồ đội đầu",
    "leg_ornament": "Trang trí chân",
    "arm_ornament": "Trang trí tay",
    "waist_ornament": "Đồ trang trí eo",
    "back_ornament": "Trang trí lưng",
    "face_marking": "Vằn mặt",
    "marking": "Vằn",
})


_TAG_VI_NAME_CATEGORIES = frozenset({"1", "3", "4", "8", "9", "10", "11", "15"})
_TAG_VI_CATEGORY_FALLBACKS = {
    "1": "Họa sĩ", "3": "Tác phẩm", "4": "Nhân vật", "5": "Metadata",
    "7": "Chưa có bản dịch", "8": "Họa sĩ", "9": "Người đóng góp",
    "10": "Tác phẩm", "11": "Nhân vật", "12": "Loài", "14": "Metadata",
    "15": "Lore", "0": "Chưa có bản dịch",
}
TAG_VI_TRANSLATION_FALLBACK = "Chưa có bản dịch"

# Sinh các tổ hợp màu + danh từ quen thuộc mà không cần dịch máy từng dòng.
for _color, _vi_color in _TAG_VI_COLORS.items():
    for _part, _vi_part in {
        "hair": "Tóc", "eyes": "Mắt", "background": "Nền", "dress": "Váy",
        "shirt": "Áo", "skirt": "Chân váy", "body": "Cơ thể", "fur": "Lông",
        "tail": "Đuôi", "gloves": "Găng tay", "panties": "Quần lót",
        "pantyhose": "Quần tất", "thighhighs": "Tất cao đùi", "nose": "Mũi",
    }.items():
        TAG_VI_LABELS.setdefault(f"{_color}_{_part}", f"{_vi_part} {_vi_color}")
TAG_VI_LABELS["blonde_hair"] = "Tóc vàng"
TAG_VI_LABELS.update({
    # Đợt 2: thẻ một từ và cụm cố định mà quy tắc ghép vẫn không với tới.
    "anthro_penetrating_anthro": "Nhân vật nhân hóa thâm nhập nhân vật nhân hóa",     "headband": "Băng đô",
    "animal": "Động vật",     "pectorals": "Cơ ngực",
    "steam": "Hơi nước",     "large_variant_set": "Bộ biến thể lớn",
    "skin_fang": "Răng nanh xuyên da",     "internal": "Nội tạng",
    "squish": "Bị ép dẹt",     "buckle": "Khóa",
    "hair_over_shoulder": "Tóc vắt qua một bên vai",     "bat": "Dơi",
    "disembodied_hand": "Bàn tay rời",     "pantyshot": "Ảnh lộ quần lót",
    "tan": "Da rám nắng",     "o-ring": "Vòng chữ O",
    "human_penetrating": "Người thâm nhập",     "jeans": "Quần jean",
    "qipao": "Sườn xám",     "colored_sclera": "Củng mạc có màu",
    "trap": "Trap",     "mythological_canine": "Chó thần thoại",
    "ball": "Quả bóng",     "string_bikini": "Bikini dây",
    "couch": "Ghế sofa",     "nipple_outline": "Đường viền núm vú",
    "plate": "Chiếc đĩa",     "curvy": "Dáng cong",
    "pelvic_curtain": "Rèm xương chậu",     "buckteeth": "Răng vẩu",
    "penile_masturbation": "Thủ dâm dương vật",     "mature_male": "Nam trưởng thành",
    "no_pants": "Không mặc quần",     "gauntlets": "Găng tay giáp",
    "looking_back_at_viewer": "Nhìn lại người xem",     "floating": "Lơ lửng",
    "foot_fetish": "Ái bàn chân",     "crescent": "Trăng lưỡi liềm",
    "cover_page": "Trang bìa",     "low-angle_view": "Góc nhìn từ dưới lên",
    "hand_gesture": "Cử chỉ tay",     "5_toes": "Năm ngón chân",
    "clothed_sex": "Quan hệ khi còn mặc đồ",     "foreshortening": "Co ngắn phối cảnh",
    "ribbons": "Dải ruy băng",     "cropped_torso": "Thân trên cắt lửng",
    "index_finger_raised": "Giơ ngón trỏ",     "furgonomics": "Furgonomics",
    "public": "Nơi công cộng",     "knotted_penis": "Dương vật có nút",
    "cigarette": "Thuốc lá",     "hand_on_own_face": "Tay chạm mặt mình",
    "earth_pony": "Ngựa đất",     "breast_play": "Kích thích ngực",
    "uniform": "Đồng phục",     "one_eye_covered": "Che một mắt",
    "desk": "Bàn làm việc",     "digimon_(species)": "Loài Digimon",
    "inflation": "Phình to",     "leaking_cum": "Tinh dịch rỉ ra",
    "patreon_username": "Tên người dùng Patreon",     "skimpy": "Hở hang",
    "gold_trim": "Viền vàng",     "licking_lips": "Liếm môi",
    "tearing_up": "Rơm rớm nước mắt",     "zipper": "Khóa kéo",
    "mouth_closed": "Miệng ngậm",     "ass_up": "Mông hướng lên",
    "waist_apron": "Tạp dề ngang eo",     "abstract_background": "Nền trừu tượng",
    "three-quarter_portrait": "Ảnh chân dung ba phần tư",     "dinosaur": "Khủng long",
    "armlet": "Vòng tay",     "lizard": "Thằn lằn",
    "pom_pom_(clothes)": "Bông tua rua",     "dominant_male": "Nam thống trị",
    "magic": "Phép thuật",     "sand": "Cát",
    "between_breasts": "Giữa hai ngực",     "facial_markings": "Vằn trên mặt",
    "twin_drills": "Tóc xoắn đôi",     "shota": "Shota",
    "faceless": "Không mặt",     "spot_color": "Màu đốm",
    "legendary_pokemon": "Pokémon huyền thoại",     "skull": "Đầu lâu",
    "empty_eyes": "Mắt trống rỗng",     "interlocked_fingers": "Các ngón tay đan vào nhau",
    "pregnant": "Mang thai",     "white_outline": "Đường viền trắng",
    "drinking_glass": "Ly uống",     "lace_trim": "Viền ren",
    "revealing_clothes": "Trang phục hở hang",     "hands-free": "Không dùng tay",
    "hair_rings": "Nhẫn tóc",     "furrowed_brow": "Nhíu mày",
    "web_address": "Địa chỉ web",     "moan": "Rên rỉ",
    "reflection": "Ảnh phản chiếu",     "red_fox": "Cáo đỏ",
    "moobs": "Ngực nam",     "goggles_on_head": "Kính bảo hộ trên đầu",
    "rock": "Tảng đá",     "letterboxed": "Viền đen trên dưới ảnh",
    "male_pubic_hair": "Lông mu nam",     "science_fiction": "Khoa học viễn tưởng",
    "short_hair_with_long_locks": "Tóc ngắn chừa lại lọn dài",     "camisole": "Áo hai dây",
    "sweater_vest": "Áo ghi-lê len",     "mane": "Bờm",
    "vehicle": "Xe cộ",     "happy_birthday": "Chúc mừng sinh nhật",
    "breast_press": "Ép ngực",     "flying": "Đang bay",
    "2d_animation": "Hoạt hình 2D",     "hakama": "Quần hakama",
    "smaller_penetrated": "Bên bị thâm nhập nhỏ hơn",     "hair_bobbles": "Tóc buộc lọn",
    "gift": "Quà tặng",     "eulipotyphlan": "Bộ ăn sâu bọ",
    "handgun": "Súng lục",     "multiple_tails": "Nhiều đuôi",
    "neck_bell": "Chuông cổ",     "cum_while_penetrated": "Xuất tinh khi đang bị thâm nhập",
    "presenting_anus": "Hướng hậu môn ra trước",     "skirt_set": "Bộ váy đồng bộ",
    "beads": "Chuỗi hạt",     "gagged": "Bị nhét miệng",
    "adapted_costume": "Trang phục chuyển thể",     "mature_anthro": "Nhân vật nhân hóa trưởng thành",
    "male_on_bottom": "Nam ở phía dưới",     "full_moon": "Trăng tròn",
    "artist_logo": "Logo họa sĩ",     "arm_at_side": "Để tay dọc thân",
    "rifle": "Súng trường",     "standing_sex": "Quan hệ ở tư thế đứng",
    "feral_penetrating": "Nhân vật bán thú thâm nhập",     "tsurime": "Mắt xếch lên",
    "carrying": "Đang mang",     "bone": "Xương",
    "sex_toy_insertion": "Đưa đồ chơi tình dục vào",     "vertical-striped_clothes": "Quần áo kẻ sọc dọc",
    "blood_on_face": "Máu trên mặt",     "sofa": "Ghế bành",
    "semi-rimless_eyewear": "Kính gọng nửa",     "puffy_nipples": "Núm vú căng",
    "breasts_apart": "Hai ngực tách rời",     "veins": "Gân",
    "female_on_human": "Nữ với người",     "generation_7_pokemon": "Pokémon thế hệ 7",
    "lactating": "Sữa rỉ ra",     "motor_vehicle": "Xe cơ giới",
    "santa_hat": "Mũ ông già Noel",     "doll": "Búp bê",
    "harness": "Đai buộc",     "western_dragon": "Rồng phương Tây",
    "bike_shorts": "Quần đạp xe",     "jingle_bell": "Chuông leng keng",
    "erect_nipples": "Núm vú cương",     "dual_persona": "Hai nhân cách",
    "socks_(marking)": "Hoa văn trên tất",     "panty_pull": "Kéo quần lót",
    "genderswap": "Đổi giới tính",     "three-quarter_view": "Góc nhìn ba phần tư",
    ":p": "Biểu cảm :p",     "clitoral_hood": "Quy đầu âm vật",
    "pattern_clothing": "Quần áo có hoa văn",     "hand_fan": "Quạt cầm tay",
    "multiple_scenes": "Nhiều cảnh",     "mustache": "Ria mép",
    "forehead": "Trán",     "procyonid": "Họ gấu mèo",
    "peaked_cap": "Mũ có lưỡi trai",     "straight-on": "Nhìn chính diện",
    "clenched_hands": "Nắm hai tay",     "feral_penetrated": "Nhân vật bán thú bị thâm nhập",
    "belt_buckle": "Mặt khóa thắt lưng",     "facial": "Thuộc khuôn mặt",
    "androgynous": "Hòa hợp nam nữ",     "pocket": "Túi áo",
    "frilled_shirt_collar": "Cổ áo sơ mi nhún xếp",     "big_nipples": "Núm vú lớn",
    "cunnilingus": "Kích thích âm hộ bằng miệng",     "toned": "Cơ săn chắc",
    "hand_in_pocket": "Tay trong túi",     "breast_squish": "Ngực bị ép",
    "marsupial": "Thú có túi",     "vibrator": "Đồ chơi rung",
    "spikes_(anatomy)": "Gai (giải phẫu)",     "side_slit": "Đường xẻ bên",
    "mole_on_breast": "Nốt ruồi trên ngực",     "intersex/male": "Liên giới tính và nam",
    "pastoral_dog": "Chó chăn cừu",     "clothed_female_nude_male": "Nữ mặc đồ còn nam khỏa thân",
    "no_sound": "Không có âm thanh",     "half_updo": "Búi tóc một nửa",
    "bad_link": "Đường link hỏng",     "looking_ahead": "Nhìn về phía trước",
    "toony": "Phong cách hoạt hình",     "pokephilia": "Yêu Pokémon",
    "clothes_writing": "Chữ viết trên quần áo",     "feral_on_feral": "Bán thú với bán thú",
    "paizuri": "Ép ngực vào dương vật",     "knee_up": "Co một gối",
    "top_hat": "Mũ chóp cao",     "seaside": "Bên biển",
    "colored_eyelashes": "Lông mi có màu",     "herding_dog": "Chó chăn gia súc",
    "excessive_genital_fluids": "Rất nhiều dịch sinh dục",     "musk": "Xạ hương",
    "sleeves_rolled_up": "Cuộn tay áo",     "featureless_crotch": "Không vẽ bộ phận sinh dục",
    "saliva_string": "Sợi nước bọt",     "low-tied_long_hair": "Tóc dài buộc thấp",
    "exclamation_point": "Dấu chấm than",     "intersex/female": "Liên giới tính và nữ",
    "turtleneck_sweater": "Áo len cổ lọ",     "pencil_skirt": "Váy bút chì",
    "single_glove": "Một chiếc găng tay",     "covering_privates": "Che chỗ kín",
    "high_heel_boots": "Bốt cao gót",     "mirror": "Cái gương",
    "upskirt": "Chụp từ dưới váy",     "tareme": "Mắt cụp xuống",
    "copyright_notice": "Thông báo bản quyền",     "mind_control": "Kiểm soát tâm trí",
    "shirt_tucked_in": "Sơ-vin áo",     "male_pov": "Góc nhìn của nam",
    "mary_janes": "Giày Mary Jane",     "mecha": "Mecha",
    "dipstick_ears": "Tai cụt",     "female_on_top": "Nữ ở phía trên",
    "barazoku": "Barazoku",     "source_filmmaker_(artwork)": "Nhà làm phim nguồn",
    "flat_chested": "Ngực phẳng",     "cattle": "Gia súc",
    "track_jacket": "Áo khoác thể thao",     "cute_fangs": "Răng nanh dễ thương",
    "excessive_cum": "Rất nhiều tinh dịch",     "hair_flaps": "Tóc bay hai bên",
    "crying_with_eyes_open": "Khóc mở mắt",     "hand_on_another's_head": "Tay đặt lên đầu người khác",
    "spoken_ellipsis": "Dấu ba chấm trong bong bóng",     "wading": "Lội nước",
    "on_chair": "Trên ghế",     "high-waist_skirt": "Chân váy cạp cao",
    "arm_behind_back": "Để tay sau lưng",     "ringed_eyes": "Mắt có vòng tròn",
    "humor": "Hài hước",     "pinup": "Tranh pinup",
    "holding_staff": "Cầm gậy",     "side_boob": "Nghiêng lộ ngực",
    "tray": "Cái khay",     "lens_flare": "Lóa ống kính",
    "notice_lines": "Nét nhấn mạnh",     "wide-eyed": "Mở to mắt",
    "door": "Cánh cửa",     "kemonomimi_mode": "Chế độ tai thú",
    "between_legs": "Giữa hai chân",     "holding_umbrella": "Cầm ô",
    "tentacle_hair": "Tóc xúc tu",     "larger_dom": "Bên thống trị to hơn",
    "nose_ring": "Nhẫn mũi",     "finger_to_mouth": "Đặt ngón tay lên môi",
    "werecreature": "Người hóa thú",     "front-tie_top": "Áo buộc dây phía trước",
    "pauldrons": "Giáp vai",     "male_on_anthro": "Nam với nhân vật nhân hóa",
    "submissive_female": "Nữ phục tùng",     "skin_tight": "Bó sát da",
    "3_fingers": "Ba ngón tay",     "blunt_ends": "Đầu cắt bằng",
    "painting_(artwork)": "Tranh vẽ",     "camel_toe": "Lộ khe quần lót",
    "ear_covers": "Miếng che tai",     "arm_warmers": "Ống giữ ấm cánh tay",
    "pasties": "Miếng che núm vú",     "black_claws": "Móng vuốt đen",
    "bite": "Cắn",     "precum_drip": "Dịch trước xuất tinh nhỏ giọt",
    "aged_up": "Già hóa",     "patreon_logo": "Logo Patreon",
    "winged_unicorn": "Kỳ lân có cánh",     "pink_pawpads": "Đệm gan bàn chân hồng",
    "legs_together": "Chụm hai chân",     "pouch": "Túi con",
    "swim_ring": "Vòng bơi",     "blouse": "Áo kiểu nữ",
    "futanari": "Futanari",     "imminent_sex": "Sắp quan hệ",
    "reaching": "Đang với tay",     "arm_behind_head": "Để tay sau đầu",
    "oni": "Oni",     "heavy_breathing": "Thở gấp",
    "clothed_male": "Nam mặc đồ",     "2koma": "Truyện tranh 2 khung",
    "seiza": "Ngồi seiza",     "hair_behind_ear": "Giắt tóc sau tai",
    "human_on_feral": "Người với nhân vật bán thú",     "hair_over_eye": "Tóc che một mắt",
    "fins": "Vây",     "pov_hands": "Bàn tay góc nhìn thứ nhất",
    "colored_nails": "Móng tay màu",     "curtained_hair": "Tóc rẽ hai bên như rèm",
    "drinking_straw": "Ống hút",     "question_mark": "Dấu hỏi",
    "naughty_face": "Mặt tinh nghịch",     "blindfold": "Băng bịt mắt",
    "lab_coat": "Áo choàng phòng thí nghiệm",     "sciurid": "Họ sóc",
    "selfie": "Ảnh tự chụp",     "convenient_censoring": "Che kiểm duyệt đúng chỗ",
    "cum_on_self": "Tinh dịch lên chính mình",     "macro": "Cận cảnh",
    "car": "Ô tô",     "tail_motion": "Vệt chuyển động đuôi",
    "leaning": "Tựa người",     "bedding": "Đồ trải giường",
    "jitome": "Nhìn xiên",     "sign": "Bảng hiệu",
    "side-tie_panties": "Quần lót buộc bên",     "model_sheet": "Bảng tạo hình nhân vật",
    ":q": "Biểu cảm :q",     "clothing_lift": "Tốc quần áo",
    "shrug_(clothing)": "Áo choàng vai",     "2_horns": "Hai sừng",
    "musclegut": "Cơ bụng một múi",     "missionary": "Tư thế truyền giáo",
    "penis_in_mouth": "Dương vật trong miệng",     "nordic_sled_dog": "Chó kéo xe Bắc Âu",
    "goatee": "Râu quai nón nhọn",     "pivoted_ears": "Tai xoay ngang",
    "gaping": "Hở hoác",     "one-hour_drawing_challenge": "Thử thách vẽ một giờ",
    "ribbed_sweater": "Áo len gân",     "garter_belt": "Thắt lưng giữ tất",
    "scared": "Sợ hãi",     "annoyed": "Bực dọc",
    "cleft_of_venus": "Khe giữa hai mông",     "underwater": "Dưới nước",
    "pokemon_focus": "Tập trung vào Pokémon",     "genderswap_(mtf)": "Đổi giới tính (nam thành nữ)",
    "stubble": "Râu lởm chởm",     "looking_at_partner": "Nhìn bạn tình",
    "hair_spread_out": "Tóc xòe ra",     "alternate_color": "Màu khác",
    "tapering_penis": "Dương vật thuôn nhọn",     "overalls": "Áo yếm",
    "leather": "Da thuộc",     "tokin_hat": "Mũ tokin",
    "hands_behind_back": "Hai tay sau lưng",     "object_in_ass": "Dị vật trong hậu môn",
    "werecanid": "Người hóa chó sói",     "ambiguous_penetration": "Thâm nhập không rõ ràng",
    "personification": "Nhân cách hóa",     "android": "Người máy",
    "furred_scalie": "Bò sát nhân hóa có lông",     "mouth_mask": "Khẩu trang",
    "panties_aside": "Kéo quần lót sang bên",     "lollipop": "Kẹo mút",
    "hat_ornament": "Trang sức trên mũ",     "bottomwear_down": "Hạ đồ mặc dưới",
    "breastplate": "Giáp ngực",     "4k": "Độ phân giải 4K",
    "falling_petals": "Cánh hoa rơi",     "off-shoulder_dress": "Váy trễ vai",
    "musteline": "Họ chồn",     "scut_tail": "Đuôi dẹt dựng đứng",
    "partially_retracted_foreskin": "Da quy đầu tụt một phần",     "skinsuit": "Bộ đồ mô phỏng",
    "epaulettes": "Bản vai",     "hoop_earrings": "Khuyên tai vòng",
    "hanging_breasts": "Ngực chảy xệ",     "head_rest": "Gối tựa đầu",
    "tailwag": "Đuôi vẫy",     "white_feathers": "Lông vũ trắng",
    "hair_down": "Tóc xõa",
})

TAG_VI_LABELS.update({
    # Đợt 3 – chi tiết nhân vật: mặt, tóc, mắt, tai/đuôi/sừng, trang phục.
    "expressionless": "Mặt không biểu cảm",
    "orgasm_face": "Mặt khi lên đỉnh",
    "smug_face": "Mặt tự mãn",
    "cupped_face": "Hai tay ôm mặt",
    "wide-eyed": "Mở to mắt",
    "multiple_eyes": "Nhiều mắt",
    "blank_eyes": "Mắt trống rỗng",
    "rolling_eyes": "Đang đảo mắt",
    "eye_roll": "Đảo mắt",
    "solid_oval_eyes": "Mắt hình bầu dục đặc",
    "solid_circle_eyes": "Mắt hình tròn đặc",
    "sleepy_eyes": "Mắt buồn ngủ",
    "teary_eyes": "Mắt ngấn lệ",
    "upturned_eyes": "Mắt hướng lên",
    "downcast_eyes": "Mắt cụp xuống",
    "amber_eyes": "Mắt màu hổ phách",
    "spiral_eyes": "Mắt xoắn ốc",
    "cross_eyed": "Mắt lé",
    "wall_eyes": "Nhiều mắt trên tường",
    "eyeball_only": "Chỉ có nhãn cầu",
    "ringed_eyes": "Mắt có viền tròn",
    "jaw_down": "Hàm trễ xuống",
    "seductive_smile": "Nụ cười gợi cảm",
    "awkward_smile": "Nụ cười gượng",
    "drooling": "Đang chảy dãi",
    "tongue_out": "Thè lưỡi",
    "hand_over_own_mouth": "Tay che miệng mình",
    "hair_up": "Tóc buộc lên",
    "loose_hair": "Tóc xõa",
    "tied_hair": "Tóc buộc",
    "hair_strand": "Lọn tóc",
    "hair_bead": "Hạt cài tóc",
    "hair_beads": "Chuỗi hạt cài tóc",
    "hair_bell": "Chuông cài tóc",
    "hair_stick": "Trâm cài tóc",
    "hair_slicked_back": "Tóc vuốt ngược",
    "hair_front_braid": "Bím tóc trước",
    "hair_feathers": "Lông vũ cài tóc",
    "hair_gel": "Keo vuốt tóc",
    "hair_ribbon": "Ruy băng buộc tóc",
    "hair_bow": "Nơ buộc tóc",
    "hair_ornament": "Phụ kiện tóc",
    "hairclip": "Kẹp tóc",
    "hairpin": "Kẹp tóc",
    "no_bangs": "Không có tóc mái",
    "side_ponytail": "Đuôi ngựa lệch một bên",
    "short_ponytail": "Đuôi ngựa ngắn",
    "cone_head": "Nón chóp nhọn",
    "santa_hat": "Mũ ông già Noel",
    "beanie": "Mũ len",
    "beret": "Mũ nồi",
    "headdress": "Đồ đội đầu",
    "hat_ornament": "Trang trí mũ",
    "crownlette": "Vương miện nhỏ",
    "headwear": "Phụ kiện đội đầu",
    "earmuffs": "Băng chụp tai",
    "ear_tag": "Kẹp tai",
    "ear_ornament": "Trang trí tai",
    "ear_covering": "Tai được che",
    "ear_piercing": "Xuyên tai",
    "pointed_ears": "Tai nhọn",
    "animal_ears": "Tai động vật",
    "fluffy_ears": "Tai lông xù",
    "horse_ears": "Tai ngựa",
    "tail_ribbon": "Ruy băng buộc đuôi",
    "tail_bell": "Chuông đeo đuôi",
    "tail_bow": "Nơ buộc đuôi",
    "tail_fluff": "Búi lông đuôi",
    "tail_stripe": "Vằn đuôi",
    "tail_ring": "Vòng đuôi",
    "tail_ornament": "Trang trí đuôi",
    "tail_feather": "Lông vũ ở đuôi",
    "heart_tail": "Đuôi hình trái tim",
    "spade_tail": "Đuôi hình cái xẻng",
    "ringed_tail": "Đuôi có khoanh",
    "flaming_tail": "Đuôi lửa",
    "beaver_tail": "Đuôi hải ly",
    "lion_tail": "Đuôi sư tử",
    "monkey_tail": "Đuôi khỉ",
    "long_tail_ribbon": "Ruy băng dài buộc đuôi",
    "feathered_wings": "Cánh lông vũ",
    "wing_ornament": "Trang trí cánh",
    "horn_ring": "Vòng sừng",
    "horn_ornament": "Trang trí sừng",
    "horse_horns": "Sừng ngựa",
    "paw_pose": "Tư thế vuốt",
    "detailed_bulge": "Hạ bộ có chi tiết",
    "bags_under_eyes": "Quầng thâm dưới mắt",
    "sportswear": "Quần áo thể thao",
    "microskirt": "Váy siêu ngắn",
    "miniskirt": "Váy ngắn",
    "pinafore_dress": "Váy yếm",
    "suspender_skirt": "Chân váy có dây đeo",
    "undershirt": "Áo lót trong",
    "overalls": "Quần yếm",
    "clothes_down": "Hạ quần áo xuống",
    "dress_aside": "Kéo váy sang bên",
    "dress_lift": "Nâng váy",
    "thong": "Quần lót dây",
    "chastity_device": "Đai cấm dục",
    "chastity_belt": "Đai cấm dục",
    "elbow_gloves": "Găng tay dài quá khuỷu",
    "opera_gloves": "Găng tay dài",
    "fingerless_gloves": "Găng tay hở ngón",
    "lace-up_boots": "Bốt buộc dây",
    "rain_boots": "Bốt đi mưa",
    "snow_boots": "Bốt đi tuyết",
    "knee_boots": "Bốt cao đến gối",
    "ankle_boots": "Bốt cổ chân",
    "fur_boots": "Bốt lông",
    "legwear_above_shoes": "Đồ chân đi trên giày",
    "sheer_legwear": "Tất xuyên thấu",
    "patterned_panties": "Quần lót có hoa văn",
    "lace_panties": "Quần lót ren",
    "cotton_panties": "Quần lót cotton",
    "leather_jacket": "Áo khoác da",
    "sleeveless_jacket": "Áo khoác không tay",
    "hooded_jacket": "Áo khoác có mũ",
    "varsity_jacket": "Áo khoác bóng rổ trường",
    "bomber_jacket": "Áo khoác bomber",
    "denim_shorts": "Quần short bò",
    "cargo_shorts": "Quần short túi hộp",
    "denim_skirt": "Chân váy bò",
    "pleated_skirt": "Chân váy ly",
    "denim_jacket": "Áo khoác bò",
    "clothes_hanger": "Móc treo quần áo",
    "hanger": "Móc treo",
    "eyeliner": "Kẻ mắt",
    "eye_shadow": "Phấn mắt",
    "eyeshadow": "Phấn mắt",
    "makeup_on_face": "Trang điểm trên mặt",
    "greasepaint_makeup": "Trang điểm hí họa",
    "war_paint": "Vẽ mặt trận",
    "face_mask": "Mặt nạ che mặt",
    "pulled_down_by_self": "Kéo xuống (tự làm)",
    "clothes_around_ankles": "Quần áo trễ xuống mắt cá",
    "clothes_lift": "Nâng quần áo",
    "skirt_basket": "Váy dạng giỏ",
    "front_slit_bikini": "Bikini chẻ trước",
    "high_waist_panties": "Quần lót cạp cao",
    "boy_shorts": "Quần short boxer",
    "briefs": "Quần lót tam giác",
    "swim_trunks": "Quần bơi nam",
})

TAG_VI_LABELS.update({
    # Đợt 4 – thẻ chi tiết hot còn sót (danh từ ghép, số nhiều, trạng thái).
    "hakama_skirt": "Váy hakama",
    "one_eye_obstructed": "Một mắt bị che",
    "stud_earrings": "Khuyên tai dạng nụ",
    "furred_dragon": "Rồng có lông",
    "tracen_school_uniform": "Đồng phục trường Tracen",
    "spear": "Ngọn giáo",
    "pseudo_hair": "Tóc kết từ vật khác",
    "teddy_bear": "Gấu bông",
    "under-rim_eyewear": "Kính gọng dưới",
    "hat_flower": "Hoa cài mũ",
    "sitting_on_another": "Ngồi lên người khác",
    "east_asian_clothing": "Trang phục Đông Á",
    "off-shoulder_shirt": "Áo trễ vai",
    "against_surface": "Tựa vào bề mặt",
    "eyebrows_hidden_by_hair": "Lông mày bị tóc che",
    "retracted_foreskin": "Da quy đầu kéo ngược",
    "unretracted_foreskin": "Da quy đầu chưa kéo",
    "wide_eyed": "Mở to mắt",
    "wide-eyed": "Mở to mắt",
    "eyebrow_piercing": "Xuyên lông mày",
    "looking_aside": "Nhìn sang bên",
    "looking_afar": "Nhìn xa xăm",
    "wedding_dress": "Váy cưới",
    "horned_humanoid": "Nhân vật có sừng",
    "hand_on_another's_face": "Tay chạm mặt người khác",
    "split-color_hair": "Tóc chia màu",
    "full-face_blush": "Đỏ cả mặt",
    "absurdly_long_hair": "Tóc dài phi lý",
    "food-themed_hair_ornament": "Phụ kiện tóc hình đồ ăn",
    "tail_feathers": "Lông vũ ở đuôi",
    "claw_pose": "Tư thế vuốt",
    "heart_censor": "Kiểm duyệt hình trái tim",
    "eye_patch": "Miếng che mắt",
    "curled_horns": "Sừng xoắn",
    "facesitting": "Ngồi lên mặt",
    "tress_ribbon": "Ruy băng buộc lọn tóc",
    "tail_accessory": "Phụ kiện đuôi",
    "eyeball": "Nhãn cầu",
    "parallel_hairclips": "Kẹp tóc song song",
    "fairy_wings": "Cánh tiên",
    "invisible_chair": "Ghế vô hình",
    "bikini_armor": "Giáp bikini",
    "sundress": "Váy dạo nắng",
    "hairy": "Có nhiều lông",
    "skinny": "Gầy gò",
    "eye_mask": "Mặt nạ che mắt",
    "frilled_hair_tubes": "Ống tóc nhún",
    "taut_clothes": "Quần áo căng chặt",
    "evil_smile": "Nụ cười gian ác",
    "faceless_female": "Nữ không mặt",
    "polar_bear": "Gấu Bắc Cực",
    "sparkling_eyes": "Mắt lấp lánh",
    "face-to-face": "Đối mặt nhau",
    "hat_feather": "Lông vũ cài mũ",
    "averting_eyes": "Nhìn lảng đi",
    "see-through_shirt": "Áo sơ mi xuyên thấu",
    "alternate_eye_color": "Màu mắt khác",
    "clothing_pull": "Kéo quần áo",
    "stirrup_legwear": "Tất có đai bàn chân",
    "bikini_skirt": "Chân váy bikini",
    "teeth_showing": "Để lộ răng",
    "deep_skin": "Da sẫm màu",
    "two_tails": "Hai cái đuôi",
    "2_tails": "Hai cái đuôi",
    "nervous_smile": "Nụ cười lo lắng",
    "uneven_legwear": "Đồ chân không đều",
    "backwards_hat": "Đội mũ ngược",
    "heart_ahoge": "Tóc ahoge hình trái tim",
    "oversized_clothes": "Quần áo quá khổ",
    "alternate_hair": "Kiểu tóc khác",
    "alternate_hair_color": "Màu tóc khác",
    "ice_wings": "Cánh băng",
    "diagonal-striped_clothes": "Quần áo kẻ sọc chéo",
    "simple_eyes": "Mắt vẽ đơn giản",
    "multiple_hair_bows": "Nhiều nơ buộc tóc",
    "lolita_hairband": "Băng đô lolita",
    "eye_scar": "Sẹo ở mắt",
    "one-eyed": "Một mắt",
    "swallowing": "Đang nuốt",
    "nun_headdress": "Khăn trùm nữ tu",
    "holding_another's_hair": "Nắm tóc người khác",
    "dress_bow": "Nơ trên váy",
    "detached_wings": "Cánh tách rời",
    "argyle_clothes": "Quần áo kẻ quả thoi",
    "clothes_grab": "Nắm quần áo",
    "lidded_eyes": "Mở hé mí",
    "animal_ear_headphones": "Tai nghe thú",
    "braided_hair_rings": "Vòng buộc tóc tết",
    "showgirl_skirt": "Chân váy showgirl",
    "median_furrow": "Đường rãnh giữa bụng",
    "furrification": "Hoá thú",
    "earphones": "Tai nghe nhét tai",
    "turtleneck_shirt": "Áo sơ mi cổ lọ",
    "curved_horn": "Sừng cong",
    "unusual_tail": "Đuôi khác thường",
    "mini_top_hat": "Mũ chóp cao nhỏ",
    "crescent_hat_ornament": "Trang trí mũ hình lưỡi liềm",
    "heart_pasties": "Miếng che ngực hình tim",
    "tail_wagging": "Đuôi vẫy",
    "hakama_short_skirt": "Chân váy hakama ngắn",
    "multiple_poses": "Nhiều tư thế",
    "wingless_dragon": "Rồng không cánh",
    "hasu_no_sora_school_uniform": "Đồng phục trường Hasu no Sora",
    "crossdressing_(mtf)": "Giả gái (MTF)",
    "dildo_sitting": "Ngồi trên dương vật giả",
    "glowing_genitalia": "Bộ phận phát sáng",
    "worm's-eye_view": "Góc nhìn từ dưới lên",
})

TAG_VI_LABELS.update({
    # Đợt 3 – chi tiết nhân vật: mặt, tóc, mắt, tai/đuôi/sừng, trang phục.
    "tied_up": "Bị trói",
    "tied_down": "Bị trói giữ",
    "tied_up_(nonsexual)": "Bị trói (không khiêu dâm)",
    "tied_to_chair": "Trói vào ghế",
    "untied_bikini_top": "Áo bikini cởi dây",
    "untied_bikini_bottom": "Quần bikini cởi dây",
    "untied_panties": "Quần lót cởi dây",
    "untied_shoelaces": "Dây giày cởi",
    "holding_with_tail": "Giữ bằng đuôi",
    "holding_with_feet": "Giữ bằng bàn chân",
    "holding_with_foot": "Giữ bằng bàn chân",
    "holding_with_gesture": "Giữ bằng cử chỉ tay",
    "holding_with_chopsticks": "Giữ bằng đôi đũa",
    "holding_with_tongue": "Giữ bằng lưỡi",
    "holding_with_two_hands": "Giữ bằng hai tay",
    "holding_with_tentacle": "Giữ bằng xúc tu",
    "holding_with_penis": "Giữ bằng dương vật",
    "holding_with_stirrups": "Giữ bằng bàn đạp",
    "clothes_lift": "Vén quần áo",
    "side_ponytail": "Tóc đuôi ngựa lệch bên",
    "short_ponytail": "Tóc đuôi ngựa ngắn",
    "flat_cap": "Mũ lưỡi phẳng",
    "winter_cap": "Mũ len mùa đông",
    "nurse_cap": "Mũ y tá",
    "police_cap": "Mũ cảnh sát",
})

TAG_VI_LABELS.update({
    # Bổ sung theo độ phổ biến: thẻ một từ và tổ hợp mà quy tắc ghép không với tới.
    "photoshop_(medium)": "Ảnh chỉnh bằng Photoshop",     "pokemon_(species)": "Loài Pokémon",
    "tuft": "Túm lông",     "humanoid": "Dạng người",
    "human": "Người",     "commission": "Tranh đặt hàng",
    "untranslatable_commentary": "Chú thích không dịch được",     "clothing_cutout": "Lỗ cắt trên quần áo",
    "hyper": "Phóng đại",     "v": "Tư thế chữ V",
    "pokemon_(creature)": "Pokémon (sinh vật)",     "mythological_equine": "Ngựa thần thoại",
    ":3": "Biểu cảm :3",     "korean_commentary": "Chú thích tiếng Hàn",
    "video": "Video",     "legs": "Đôi chân",
    "bear": "Gấu",     "non-mammal_breasts": "Ngực của loài không phải thú",
    "low_twintails": "Tóc hai búi thấp",     "mostly_nude": "Khỏa thân phần lớn",
    "meme": "Meme",     "faceless_male": "Nam không có mặt",
    "membrane_(anatomy)": "Màng (giải phẫu)",     "4_fingers": "Bốn ngón tay",
    "chinese_clothes": "Quần áo kiểu Trung Quốc",     "swept_bangs": "Tóc mái rẽ một bên",
    "colored_inner_hair": "Tóc lớp trong tô màu",     "curvy_figure": "Dáng người cong",
    "gem": "Đá quý",     "freckles": "Tàn nhang",
    "fish": "Cá",     "machine": "Máy móc",
    "caprine": "Họ dê",     "partial_commentary": "Chú thích một phần",
    "overweight_male": "Nam thừa cân",     "headphones": "Tai nghe chụp đầu",
    "^_^": "Biểu cảm ^_^",     "cum_in_mouth": "Tinh dịch trong miệng",
    "sideboob": "Nghiêng lộ ngực bên",     "shiny_skin": "Da bóng",
    "digitigrade": "Đi bằng ngón chân",     "denim": "Vải bò",
    "slit_pupils": "Đồng tử khe dọc",     "beret": "Mũ beret",
    "pussy_juice": "Dịch âm hộ",     "helmet": "Mũ bảo hiểm",
    "plantigrade": "Bàn chân áp đất",     "bright_pupils": "Đồng tử sáng",
    "from_front_position": "Tư thế nhìn từ phía trước",     "backsack": "Ba lô",
    "black_hairband": "Băng buộc tóc đen",     "facial_mark": "Điểm trên mặt",
    "witch_hat": "Mũ phù thủy",     "on_top": "Ở phía trên",
    "pink_nipples": "Núm vú hồng",     "tentacles": "Xúc tu",
    "playboy_bunny": "Đồ thỏ Playboy",     "moon": "Mặt trăng",
    "rose": "Hoa hồng",     "crown": "Vương miện",
    "url": "Đường link",     "nose_blush": "Đỏ sống mũi",
    "beak": "Mỏ",     "breath": "Hơi thở",
    "cumshot": "Xuất tinh lên ảnh",     "fox_girl": "Cô gái cáo",
    "alternate_hairstyle": "Kiểu tóc khác",     "ear_ring": "Khuyên vành tai",
    "nude_anthro": "Nhân vật nhân hóa khỏa thân",     "facial_tuft": "Túm lông mặt",
    "holding_hands": "Nắm tay nhau",     "halterneck": "Áo quàng cổ",
    "hood_down": "Hạ mũ trùm",     "kemono": "Kemono",
    "furry": "Furry",     "bed_sheet": "Ga trải giường",
    "bob_cut": "Tóc bob",     "age_difference": "Chênh lệch tuổi tác",
    "membranous_wings": "Cánh màng",     "cum_in_pussy": "Tinh dịch trong âm hộ",
    "arm_support": "Đỡ người bằng tay",     "smaller_male": "Nam nhỏ hơn",
    "demon_horns": "Sừng quỷ",     "beard": "Râu quai nón",
    "smartphone": "Điện thoại thông minh",     "larger_male": "Nam to hơn",
    "mature_female": "Nữ trưởng thành",     "pony": "Ngựa pony",
    "testicles": "Tinh hoàn",     "overweight_anthro": "Nhân vật nhân hóa thừa cân",
    "penis_in_ass": "Dương vật trong hậu môn",     "elf": "Tiên",
    "eye_contact": "Giao tiếp bằng mắt",     "humanoid_hands": "Bàn tay dạng người",
    "crossover": "Giao thoa tác phẩm",     "suit": "Bộ vest",
    "bedroom_eyes": "Ánh mắt gợi tình",     "after_sex": "Sau khi quan hệ",
    "generation_4_pokemon": "Pokémon thế hệ 4",     "garter_straps": "Dây giữ bít tất",
    "shirt_lift": "Tốc áo",     "trembling": "Run rẩy",
    "glans": "Quy đầu",     "flying_sweatdrops": "Giọt mồ hôi văng",
    "male_anthro": "Nhân vật nam nhân hóa",     "dark-skinned_male": "Nam da sẫm",
    "cheek_tuft": "Túm lông má",     "demon_girl": "Cô gái quỷ",
    "lipstick": "Son môi",     "unicorn": "Kỳ lân",
    "groping": "Sờ soạng",     "electronics": "Đồ điện tử",
    "cleavage_cutout": "Lỗ cắt khoe khe ngực",     "paid_reward_available": "Có phần thưởng trả phí",
    "dildo": "Dương vật giả",     "spots": "Đốm",
    "drill_hair": "Tóc xoắn",     "bottle": "Cái chai",
    "bandages": "Băng gạc",     "pectoral": "Cơ ngực",
    "floral_print": "Họa tiết hoa",     "goggles": "Kính bảo hộ",
    "blush_lines": "Vệt đỏ mặt",     "eyewear_on_head": "Kính đội trên đầu",
    "on_bottom": "Ở phía dưới",     "antenna_hair": "Tóc anten",
    "1other": "Một nhân vật khác",     "cardigan": "Áo cardigan",
    "container": "Vật chứa",     "logo": "Logo",
    "umbrella": "Cái ô",     "high_ponytail": "Tóc đuôi ngựa cao",
    "faceless_character": "Nhân vật không mặt",     "crossed_legs": "Bắt chéo chân",
    "oral_penetration": "Thâm nhập bằng miệng",     "underboob": "Ngực lộ phía dưới",
    "standing_on_one_leg": "Đứng một chân",     "no_bra": "Không mặc áo ngực",
    "handjob": "Thủ dâm bằng tay",     "female_pubic_hair": "Lông mu nữ",
    "whiskers": "Râu mép",     "cameltoe": "Lộ khe quần lót",
    "no_shoes": "Không đi giày",     "side-tie_bikini_bottom": "Quần bikini buộc bên",
    "sisters": "Chị em gái",     "rope": "Dây thừng",
    "child": "Trẻ em",     "bestiality": "Quan hệ với động vật",
    "copyright_request": "Yêu cầu bản quyền",     "holding_cup": "Cầm cốc",
    "mammal_humanoid": "Động vật có vú dạng người",     "outline": "Đường viền",
    "forced": "Bị ép buộc",     "restrained": "Bị giữ chặt",
    "hand_on_butt": "Tay đặt lên mông",     "heart-shaped_pupils": "Đồng tử hình tim",
    "back": "Lưng",     "stuffed_toy": "Thú nhồi bông",
    "parody": "Tranh chế",     "dipstick_tail": "Đuôi cụt",
    "building": "Tòa nhà",     "magical_girl": "Thiếu nữ phép thuật",
    "x_hair_ornament": "Phụ kiện tóc chữ X",     "finger_claws": "Móng vuốt ở ngón tay",
    "cowgirl_position": "Tư thế cưỡi ngựa",     "lingerie": "Đồ lót nữ gợi cảm",
    "4koma": "Truyện tranh 4 khung",     "ass_visible_through_thighs": "Lộ mông qua kẽ đùi",
    "yaoi": "Yaoi",     "facial_piercing": "Khuyên mặt",
    "true_fox": "Cáo thật",     "animated_gif": "GIF động",
    "first_person_view": "Góc nhìn thứ nhất",     "own_hands_together": "Chắp hai tay trước mặt",
    "undressing": "Đang cởi quần áo",     "hyper_genitalia": "Bộ phận sinh dục phóng đại",
    "armwear": "Đồ che cánh tay",     "human_on_anthro": "Người với nhân vật nhân hóa",
    "full-length_portrait": "Ảnh toàn thân",     "no_panties": "Không mặc quần lót",
    "threesome": "Quan hệ ba người",     "md5_mismatch": "MD5 không khớp",
    "cross": "Thập tự",     "tiger": "Hổ",
    "snout": "Mõm",     "deer": "Nai",
    "backpack": "Ba lô",     "underwear_only": "Chỉ mặc đồ lót",
    "pointing": "Chỉ tay",     "military": "Quân đội",
    "alien": "Người ngoài hành tinh",     "hair_accessory": "Phụ kiện tóc",
    "straddling": "Ngồi dạng chân qua",     "arthropod": "Động vật chân khớp",
    "skeb_commission": "Tranh đặt trên Skeb",     "transformation": "Biến hình",
    "translucent": "Bán trong suốt",     "towel": "Khăn tắm",
    "leg_up": "Giơ chân",     "cutie_mark": "Cutie mark",
    "gesture": "Cử chỉ tay",     "two-piece_swimsuit": "Đồ bơi hai mảnh",
    "vore": "Nuốt chửng",     "murid": "Họ chuột",
    "thigh_gap": "Kẽ đùi",     "wariza": "Ngồi kiểu wariza",
    "animal_print": "Họa tiết da thú",     "quadruped": "Bốn chân",
    "leash": "Dây dắt",     "murine": "Thuộc họ chuột",
    "doggystyle": "Tư thế doggy",     "double-parted_bangs": "Tóc mái rẽ ngôi giữa",
    "light_smile": "Mỉm cười nhẹ",     "curtains": "Rèm cửa",
    "scrunchie": "Dây buộc tóc vải",     "mouth_hold": "Cầm bằng miệng",
    "crossed_bangs": "Tóc mái chéo",     "fingering": "Kích thích bằng ngón tay",
    "clothes_pull": "Kéo quần áo",     "non-mammal_nipples": "Núm vú của loài không phải thú",
    "scar_on_face": "Sẹo trên mặt",     "rape": "Cưỡng dâm",
    "knife": "Con dao",     "legs_up": "Giơ hai chân",
    "obi": "Đai obi",     "spiked_hair": "Tóc có gai",
    "mob_cap": "Mũ trùm đầu",     "cum_on_face": "Tinh dịch trên mặt",
    "sex_from_behind": "Quan hệ từ phía sau",     "6+girls": "Sáu nhân vật nữ trở lên",
    "cloudy_sky": "Bầu trời nhiều mây",     "nude_female": "Nữ khỏa thân",
    "candy": "Kẹo",     "third-party_source": "Nguồn bên thứ ba",
    "outside_border": "Ngoài viền ảnh",     "antlers": "Gạc",
    "generation_3_pokemon": "Pokémon thế hệ 3",     "christmas": "Giáng sinh",
    "holidays": "Ngày lễ",     "cover": "Ảnh bìa",
    "muscular_female": "Nữ cơ bắp",     "shoulder_armor": "Giáp vai",
    "thong": "Quần lót dây",     "hair_tubes": "Ống tóc",
    "eyepatch": "Miếng che mắt",     "armband": "Băng tay",
    "facing_viewer": "Đối diện người xem",     "abdominal_bulge": "Bụng phình",
    "second-party_source": "Nguồn bên thứ hai",     "skirt_lift": "Tốc váy",
    "latex": "Latex",     "crossdressing": "Trang phục chéo giới",
    "suspenders": "Dây treo quần",     "veil": "Màn che mặt",
    "strapless_leotard": "Leotard không quai",     "submissive_male": "Nam phục tùng",
    "juliet_sleeves": "Tay áo phồng kiểu Juliet",     "alpha_channel": "Kênh alpha",
    "nude_male": "Nam khỏa thân",     "clothed_anthro": "Nhân vật nhân hóa mặc đồ",
    "areola_slip": "Trượt hở quầng vú",     "floppy_ears": "Tai cụp",
    "voluptuous": "Nóng bỏng",     "tight_clothing": "Quần áo bó",
    "topless_female": "Nữ ngực trần",     "big_muscles": "Cơ bắp lớn",
    "dot_nose": "Mũi chấm",     "bug": "Bọ",
    "cloak": "Áo choàng",     "foot_focus": "Tập trung vào bàn chân",
    "head_wings": "Cánh trên đầu",     "nature": "Thiên nhiên",
    "sound": "Âm thanh",     "monster": "Quái vật",
    "blazer": "Áo blazer",     "snake": "Rắn",
    "puffy_long_sleeves": "Tay dài phồng",     "hyper_penis": "Dương vật phóng đại",
    "box": "Cái hộp",     "tassel": "Tua rua",
    "bandaid": "Băng cá nhân",     ">_<": "Biểu cảm >_<",
    "tiara": "Vương miện nhỏ",     "hand_on_breast": "Tay đặt lên ngực",
    "generation_2_pokemon": "Pokémon thế hệ 2",     "staff": "Gậy trường",
    "kiss": "Hôn",     "generation_6_pokemon": "Pokémon thế hệ 6",
    "clothing_aside": "Kéo quần áo sang bên",     "eeveelution": "Eeveelution",
    "robe": "Áo choàng dài",     "cum_on_penis": "Tinh dịch trên dương vật",
    "grabbing_another's_breast": "Bóp ngực người khác",     "outstretched_arm": "Duỗi cánh tay",
    "stuffed_animal": "Thú nhồi bông",     ";d": "Biểu cảm ;d",
    "mustelid": "Họ chồn",     "side_view": "Nhìn từ bên",
    "restraints": "Dụng cụ trói",     "brooch": "Trâm cài",
    "polearm": "Vũ khí cán dài",     "gag": "Nhét miệng",
    "scan": "Ảnh scan",     "young_male": "Nam trẻ",
    "kissing": "Đang hôn",     "larger_female": "Nữ to hơn",
    "outstretched_arms": "Duỗi hai cánh tay",     "fishnets": "Đồ lót lưới",
    "bat_wings": "Cánh dơi",     "bara": "Bara",
    "aged_down": "Trẻ hóa",     "perineum": "Tầng sinh môn",
    "single_hair_bun": "Một búi tóc",     "official_alternate_hairstyle": "Kiểu tóc thay thế chính thức",
    "white_sailor_collar": "Cổ thủy thủ trắng",     "girl_on_top": "Nữ ở phía trên",
    "cum_on_breasts": "Tinh dịch trên ngực",     "smirk": "Cười mỉa",
    "torn_clothing": "Quần áo rách",     "ahegao": "Ahegao",
    "fully_clothed": "Mặc kín",     "blue_sailor_collar": "Cổ thủy thủ xanh dương",
    "neck_tuft": "Túm lông cổ",     "melee_weapon": "Vũ khí cận chiến",
    "extra_ears": "Tai phụ",     "hand_on_own_chest": "Tay đặt lên ngực mình",
    "pokemorph": "Nhân vật dạng Pokémon",     "baseball_cap": "Mũ lưỡi trai",
    "demon_tail": "Đuôi quỷ",     "hooded_jacket": "Áo khoác có mũ",
    "bovine": "Họ bò",     "onomatopoeia": "Từ tượng thanh",
    "crossgender": "Xuyên giới tính",     "prehistoric_species": "Loài tiền sử",
    "thought_bubble": "Bong bóng suy nghĩ",     "athletic": "Dáng thể thao",
    "lifting_own_clothes": "Tự tốc quần áo",     "breasts_out": "Ngực lộ ra",
    "musical_note": "Nốt nhạc",     "clothed_female": "Nữ mặc đồ",
    "clenched_hand": "Nắm tay",     "hat_bow": "Nơ trên mũ",
    "alcohol": "Đồ uống có cồn",     "overweight_female": "Nữ thừa cân",
    "katana": "Kiếm katana",     "mixed-language_commentary": "Chú thích nhiều ngôn ngữ",
    "short_twintails": "Tóc hai búi ngắn",     "hat_ribbon": "Ruy băng trên mũ",
    "beverage": "Đồ uống",     "topless_male": "Nam ngực trần",
    "side_braid": "Bím tóc bên",     "goat": "Dê",
    "cum_drip": "Tinh dịch nhỏ giọt",     "star_(sky)": "Ngôi sao",
    "traditional_media_(artwork)": "Chất liệu truyền thống",     "on_front": "Ở phía trước",
    "flaccid": "Không cương",     "lion": "Sư tử",
    "profanity": "Lời thô tục",     "smoke": "Khói",
    "wing_collar": "Còng cổ có cánh",     "loafers": "Giày lười",
    "polka_dot": "Họa tiết chấm bi",     "hindpaw": "Bàn chân sau",
    "humanoid_pointy_ears": "Tai nhọn dạng người",     "smaller_female": "Nữ nhỏ hơn",
    "two_tone_hair": "Tóc hai tông màu",     "butterfly": "Bươm bướm",
    "character_request": "Yêu cầu nhân vật",     "single_thighhigh": "Một tất cao đùi",
    "pegasus": "Ngựa Pegasus",     "chibi_only": "Chỉ có chibi",
    "low_ponytail": "Tóc đuôi ngựa thấp",     "maid_apron": "Tạp dề hầu gái",
    "revision": "Bản chỉnh sửa",     "bouncing_breasts": "Ngực nảy",
    "knees_up": "Co gối",     "generation_5_pokemon": "Pokémon thế hệ 5",
    "shaded_face": "Mặt có đổ bóng",     "hood_up": "Kéo mũ trùm",
    "microphone": "Micro",     "countershade_torso": "Đổ bóng ngược thân trên",
    "disembodied_penis": "Dương vật rời",     "ribbon_trim": "Viền ruy băng",
    "heart_eyes": "Mắt hình trái tim",     "anger_vein": "Gân tức giận",
    "black_and_white": "Đen trắng",     "toenails": "Móng chân",
    "instrument": "Nhạc cụ",     "cropped_jacket": "Áo khoác lửng",
    "wristband": "Băng cổ tay",     "condom": "Bao cao su",
    "tachi-e": "Tachi-e",     "light_particles": "Hạt sáng",
    "on_ground": "Trên mặt đất",     "shark": "Cá mập",
    "monster_girl": "Quái vật nữ",     "3d_animation": "Hoạt hình 3D",
    "bridal_gauntlets": "Găng tay cô dâu",     "nails": "Móng tay",
    "spoken_heart": "Trái tim trong bong bóng",     "gloves_(marking)": "Hoa văn găng tay",
    "corset": "Áo corset",     "red_neckerchief": "Khăn quàng cổ đỏ",
    "partially_submerged": "Ngâm một phần",     "arm_hair": "Lông tay",
    "armpit_hair": "Lông nách",     "body_hair": "Lông cơ thể",
    "leg_hair": "Lông chân",     "back_hair": "Lông lưng",
    "stomach_hair": "Lông bụng",     "hand_hair": "Lông tay",
    "foot_hair": "Lông chân",     "hair_over_eyes": "Tóc che mắt",
    "smiling_at_viewer": "Cười nhìn người xem",     "holding_book": "Cầm sách",
    "mole_under_mouth": "Nốt ruồi dưới môi",     "sleeves_past_fingers": "Tay áo quá ngón tay",
    "highleg_leotard": "Leotard cắt cao",     "glistening_body": "Cơ thể bóng nhẫy",
    "presenting_vulva": "Hướng âm hộ ra trước",
    # Đợt 5 – thẻ hot còn sót: tên series + trang phục, trạng thái cơ thể, sọc dọc/ngang.
    "new_year": "Năm mới",
    "faceless_human": "Người không mặt",
    "faceless_anthro": "Thú nhân hóa không mặt",
    "happy_new_year": "Chúc mừng năm mới",
    "not_furry_focus": "Không chú trọng furry",
    "heart_of_string": "Trái tim bằng dây",
    "alternate_hair_length_(longer)": "Độ dài tóc khác (dài hơn)",
    "alternate_hair_length_(shorter)": "Độ dài tóc khác (ngắn hơn)",
    "furgonomic_piercing": "Khuyên furgonomic",
    "single_hair_intake": "Lọn tóc đơn",
    "eyes_out_of_frame": "Mắt nằm ngoài khung",
    "hatching_(texture)": "Vẽ gạch sọc (kết cấu)",
    "star_hat_ornament": "Trang trí mũ hình ngôi sao",
    "mitakihara_school_uniform": "Đồng phục trường Mitakihara",
    "kita_high_school_uniform": "Đồng phục trường Kita",
    "sakuragaoka_high_school_uniform": "Đồng phục trường Sakuragaoka",
    "ooarai_school_uniform": "Đồng phục trường Ooarai",
    "sailor_senshi_uniform": "Đồng phục Sailor Senshi",
    "the_pose": "Tư thế The Pose",
    "standing_split": "Chẻ chân khi đứng",
    "third_eye": "Mắt thứ ba",
    "beard_stubble": "Râu lởm chởm",
    "streaming_tears": "Nước mắt chảy ròng ròng",
    "chair_position": "Tư thế ngồi ghế",
    "tail_fetish": "Ái đuôi",
    "tail_play": "Trò chơi với đuôi",
    "footwear_bow": "Nơ trên giày",
    "sweater_dress": "Váy len",
    "swivel_chair": "Ghế xoay",
    "wool_(fur)": "Lông cừu",
    "tail_fin": "Vây đuôi",
    "grabbing_another's_hair": "Nắm tóc người khác",
    "ineffective_clothing": "Quần áo chẳng che được gì",
    "cat_ear_headphones": "Tai nghe tai mèo",
    "bondage_gear": "Đồ trói buộc",
    "pom_pom_hair_ornament": "Phụ kiện tóc quả bông",
    "convenient_hair": "Tóc che đúng chỗ cần che",
    "single_ear_cover": "Che một bên tai",
    "dressing": "Đang mặc đồ",
    "unicorn_horn": "Sừng kỳ lân",
    "rimless_eyewear": "Kính không gọng",
    "winking_at_viewer": "Nháy mắt với người xem",
    "skirt_pull": "Kéo chân váy",
    "shirt_pull": "Kéo áo sơ mi",
    "full_armor": "Giáp toàn thân",
    "power_armor": "Giáp cường lực",
    "bandage_on_face": "Băng quấn trên mặt",
    "parted_hair": "Tóc rẽ ngôi",
    "spoken_blush": "Đỏ mặt (thoại)",
    "multi-tied_hair": "Tóc buộc nhiều mối",
    "heart_(marking)": "Dấu hình trái tim",
    "teardrop": "Giọt nước mắt",
    "teardrop_earring": "Bông tai giọt nước",
    "face_fucking": "Đút vào mặt",
    "rectangular_eyewear": "Kính gọng chữ nhật",
    "gauged_ear": "Tai xỏ lỗ to",
    "ring_(jewelry)": "Nhẫn",
    "drawing_(object)": "Bản vẽ",
    "4_ears": "Bốn cái tai",
    "1_eye": "Một mắt",
    "armchair": "Ghế bành",
    "thorns": "Gai",
    "pear-shaped_figure": "Dáng quả lê",
    "tailed_humanoid": "Nhân vật có đuôi",
    "heart-shaped_box": "Hộp hình trái tim",
    "earpiece": "Tai nghe một bên",
    "colored_inner_animal_ears": "Lòng tai thú màu khác",
    "furry_with_non-furry": "Furry với không furry",
    "musical_note_hair_ornament": "Phụ kiện tóc hình nốt nhạc",
    "athletic_wear": "Quần áo thể thao",
    "furisode": "Kimono furisode",
    "vertical_striped_clothes": "Quần áo kẻ sọc dọc",
    "horizontal_striped_clothes": "Quần áo kẻ sọc ngang",
    "vertical_striped_dress": "Váy kẻ sọc dọc",
    "vertical_striped_panties": "Quần lót kẻ sọc dọc",
    "disembodied_linked_eye": "Mắt rời được nối lại",
    "high_heeled_shoes": "Giày cao gót",
    "high_heeled_boots": "Bốt cao gót",
    "high_heeled_sandals": "Dép quai cao gót",
    "pattern_dress": "Váy có hoa văn",
    "pattern_skirt": "Chân váy có hoa văn",
    "pattern_shirt": "Áo sơ mi có hoa văn",
    "earbuds": "Tai nghe nhét tai",
    "tail_stripe": "Vằn trên đuôi",
    "tail_stripes": "Vằn trên đuôi",
    "choker_(jewelry)": "Vòng cổ",
    "anklet_(jewelry)": "Vòng chân",
    "toe_ring": "Nhẫn chân",
    "toeless_boots": "Bốt kín ngón",
    "toeless_socks": "Tất kín ngón",
    "bare_legs": "Chân trần",
    "bare_shoulders": "Trần vai",
    "bare_back": "Lưng trần",
    "bare_neck": "Cổ trần",
    "bare_chest": "Ngực trần",
    "bare_belly": "Bụng trần",
    "bare_thighs": "Đùi trần",
    "bare_arms": "Cánh tay trần",
    "bare_feet": "Bàn chân trần",
    "bare_hands": "Bàn tay trần",
    "bare_hips": "Hông trần",
    "bare_waist": "Eo trần",
    "bare_skin": "Da trần",
    "bare_foot": "Bàn chân trần",
    # Đợt 6 — chi tiết nhân vật:
    "looking_at_self": "Nhìn chính mình",
    "boxers_(clothing)": 'Quần boxer',
    "partially_undressed": "Cởi đồ một phần",
    "wearing_chastity_cage": "Đeo lồng trinh tiết",
    "chastity_cage": "Lồng trinh tiết",
    "dress_flower": "Hoa đính trên váy",
    "tail_coil": "Đuôi cuộn tròn",
    "blowing_bubble_gum": "Thổi bóng kẹo cao su",
    "chewing_gum": "Nhai kẹo cao su",
    "asking": "Đang hỏi",
    "action_pose": "Tư thế hành động",
    "coattails": "Đuôi áo",
    "bandage_over_one_eye": "Băng quấn che một mắt",
    "looking_back_at_another": "Quay lại nhìn người khác",
    "bird's-eye_view": 'Góc nhìn từ trên cao',
    "greco-roman_clothes": 'Quần áo Hy-La cổ',
    "ancient_greek_clothes": "Quần áo Hy Lạp cổ đại",
    "throwing": "Đang ném",
    "blushing_profusely": "Đỏ mặt dữ dội",
    "anal_tail": "Đuôi hậu môn",
    "shibari_over_clothes": "Trói shibari đè quần áo",
    "screen_face": "Mặt màn hình",
    "running_makeup": "Trang điểm lem nhem",
    "hair_flowing_over": "Tóc bay lả lơi",
    "unconvincing_armor": "Giáp phi thực tế",
    "meme_clothing": "Quần áo in meme",
    "microdress": "Váy siêu ngắn",
    "user_interface": "Giao diện người dùng",
    "gloved_handjob": "Thủ dâm bằng găng tay",
    "flame-tipped_tail": 'Đuôi đầu lửa',
    "under-elbow_gloves": 'Găng tay quá khuỷu tay',
    "heart_o-ring": 'Nhẫn O hình trái tim',
    "eyepatch_bikini": "Bikini kèm bịt mắt",
    "4_eyes": "Bốn mắt (đeo kính)",
    "unusual_wing_placement": "Vị trí cánh khác thường",
    "tears_of_pleasure": "Nước mắt khoái cảm",
    "hairpods": "Kẹp tóc hình tai nghe",
    "heart_hands_duo": "Hai người tạo hình trái tim bằng tay",
    "heart_arms_duo": "Hai người ôm nhau tạo hình trái tim",
    "heart_tail_duo": "Hai người tạo hình trái tim bằng đuôi",
    "heart_ahoge_duo": "Hai người tạo hình trái tim bằng cọng tóc ngược",
    "tail_sex": "Quan hệ bằng đuôi",
    "horn_sex": "Quan hệ bằng sừng",
    "ear_sex": "Quan hệ bằng tai",
    "hair_sex": "Quan hệ bằng tóc",
    "wing_sex": "Quan hệ bằng cánh",
    "pubic_hair_peek": "Lông mu lấp ló",
    "armpit_hair_peek": "Lông nách lấp ló",
    "chest_hair_peek": "Lông ngực lấp ló",
    "navel_hair_peek": "Lông rốn lấp ló",
    "male_underwear_peek": "Đồ lót nam lấp ló",
    "heart_facial_mark": "Hình trái tim trên mặt",
    "gears": "Bánh răng",
    "santa_dress": "Váy ông già Noel",
    "demon_slayer_uniform": "Đồng phục diệt quỷ",
    "employee_uniform": "Đồng phục nhân viên",
    "root_(hair)": 'Gốc tóc',
    "roots_(hair)": 'Gốc tóc',
    "single_hair_streak": "Một lọn tóc màu khác",
    "hair_streak": "Lọn tóc màu khác",
    "hat_belt": "Dây lưng giữ mũ",
    "dress_pull": "Kéo váy",
    "dress_shoes": "Giày diện váy",
    "hat_tip": "Vành mũ",
    "clothing_bow": "Nơ trên quần áo",
    "shirt_collar": "Cổ áo sơ mi",
    "shirt_bow": "Nơ trên áo sơ mi",
    "footwear_ribbon": "Ruy băng trên giày",
    "footwear_flower": "Hoa đính trên giày",
    "footwear_focus": "Nhấn mạnh giày",
    "underwear_outline": "Vết hằn đồ lót",
    "underwear_grab": "Nắm đồ lót",
    "underwear_sniffing": "Đang ngửi đồ lót",
    "clothes_tug": "Kéo quần áo",
    "clothes_pin": "Ghim quần áo",
    "clothes_theft": "Trộm quần áo",
    "clothing_swap": "Đổi quần áo",
    "shirt_tug": "Kéo áo sơ mi",
    "glasses_pull": "Kéo kính",
    "hat_grab": "Nắm mũ",
    "tail_tag": "Nhãn đuôi",
    "collar_tag_(object)": 'Nhãn vòng cổ',
    "ear_chain": "Dây chuyền tai",
    "tail_chain": "Dây chuyền đuôi",
    "body_chain": "Dây chuyền thân",
    "ear_garter": "Dây treo tai",
    "tail_garter": "Dây treo đuôi",
    "ear_lick": "Liếm tai",
    "tail_lick": "Liếm đuôi",
    "ear_bite": "Cắn tai",
    "tail_bite": "Cắn đuôi",
    "ear_hug": "Ôm bằng tai",
    "tail_hug": "Ôm bằng đuôi",
    "wing_hug": "Ôm bằng cánh",
    "ear_torture": "Tra tấn tai",
    "tail_torture": "Tra tấn đuôi",
    # Đợt 6 — vòng 2: chất liệu, phụ kiện, kiểu tóc.
    "wall-eyed": "Mắt lé ra ngoài",
    "glossy_eyed": "Mắt bóng",
    "starry_eyed": "Mắt đầy sao",
    "one_eye_half-closed": "Một mắt nửa nhắm",
    "eyes_mostly_closed": "Mắt gần khép",
    "eye_half_closed": "Mắt nửa khép",
    "half_closed_eye": "Mắt nửa khép",
    "ear_stud": "Khuyên tai dạng nụ",
    "eyebrow_stud": "Khuyên mày dạng nụ",
    "underwear_sex": "Quan hệ qua lớp đồ lót",
    "clothing_sex": "Quan hệ qua lớp quần áo",
    "eye_sex": "Quan hệ bằng mắt",
    "flying_sex": "Quan hệ trên không",
    "tongue_showing": "Lưỡi thè ra",
    "cracked_skin": "Da nẻ",
    "cocktail_glass": "Ly cocktail",
    "wine_glass": "Ly rượu vang",
    "heart-shaped_lock": "Lọn tóc hình trái tim",
    "tail_beads": "Hạt chuỗi trên đuôi",
    "ear_beads": "Hạt chuỗi trên tai",
    "clothing_transformation": "Biến đổi quần áo",
    "underwear_transformation": "Biến đổi đồ lót",
    "ear_transformation": "Biến đổi tai",
    "tail_transformation": "Biến đổi đuôi",
    "hair_transformation": "Biến đổi tóc",
    "fur_transformation": "Biến đổi lông",
    "clothing_loss": "Mất quần áo",
    "hair_loss": "Rụng tóc",
    "fur_loss": "Rụng lông",
    "horn_removal": "Tháo sừng",
    "clothing_removal": "Tháo quần áo",
    "eye_removal": "Tháo mắt",
    "borrowed_hairstyle": "Kiểu tóc mượn",
    "matching_hairstyle": "Kiểu tóc đôi",
    "fur_armlet": "Vòng lông đeo tay",
    "pearl_armlet": "Vòng ngọc trai đeo tay",
    "winged_armlet": "Vòng tay có cánh",
    "tail_stocking": "Tất đuôi",
    "ribbon-trimmed_corset": "Áo corset viền ruy băng",
    "armored_corset": "Áo corset giáp",
    "glove_corset": "Áo corset kèm găng",
    "heart-shaped_cookie": "Bánh quy hình trái tim",
    "eye_creature": "Sinh vật con mắt",
    "heart_creature": "Sinh vật trái tim",
    "hair_creature": "Sinh vật tóc",
    "flower_swing": "Đu quay hoa",
    "sword_swing": "Cái vung kiếm",
    "weapon_swing": "Cái vung vũ khí",
    "object_swing": "Cái vung vật",
    "pride_color_clothing": "Quần áo màu cờ tự hào",
    "pride_color_topwear": "Áo trên màu cờ tự hào",
    "pride_color_legwear": "Quần màu cờ tự hào",
    "side-tie_clothing": "Quần áo buộc dây bên hông",
    "back-tie_clothing": "Quần áo buộc sau lưng",
    "front-tie_clothing": "Quần áo buộc phía trước",
    "neck-tie_clothing": "Quần áo thắt cổ",
    "dress_swimsuit": "Đồ bơi kiểu váy",
    "military_dress_uniform": "Đồng phục lễ quân đội",
    "american_flag_legwear": "Quần cờ Mỹ",
    "american_flag_dress": "Váy cờ Mỹ",
    "american_flag_shirt": "Áo cờ Mỹ",
    "combat_boots": "Giày chiến đấu",
    "military_combat_uniform": "Đồng phục tác chiến quân đội",
    "combat_gear": "Đồ tác chiến",
    "applying_makeup": "Đang trang điểm",
    "applying_lipstick": "Đang tô son",
    "applying_manicure": "Đang sơn móng tay",
    "applying_own_makeup": "Đang tự trang điểm",
    "white-tailed_deer": "Hươu đuôi trắng",
    "long-tailed_weasel": "Chồn đuôi dài",
    "flat-tailed_gecko": "Tắc kè đuôi bằng",
    "seamed_legwear": "Quần tất có đường chỉ",
    "back-seamed_legwear": "Quần tất có đường chỉ sau",
    "front-seamed_legwear": "Quần tất có đường chỉ trước",
    "side-seamed_legwear": "Quần tất có đường chỉ bên",
    "x_eyes": "Mắt chữ X",
    "x_eye": "Mắt chữ X",
    "slit_dress": "Váy xẻ tà",
    "side_slit_dress": "Váy xẻ tà bên",
    "side_slit_skirt": "Chân váy xẻ tà bên",
    "faceless_feral": "Thú không mặt",
    "faceless_humanoid": "Người không mặt",
    "bat-eared_fox": "Cáo tai dơi",
    "eared_owl": "Cú có tai",
    "long-eared_owl": "Cú tai dài",
    "short-eared_owl": "Cú tai ngắn",
    "tree_horns": "Sừng cây",
    "christmas_tree_hat": "Mũ cây thông Noel",
    "pom_pom_earrings": "Khuyên tai quả bông",
    "pom_hat": "Mũ quả bông",
    "teardrop_facial_mark": "Hình giọt nước trên mặt",
    "teardrop_tattoo": "Xăm hình giọt nước",
    "human_and_animal_ears": "Tai người và tai thú",
    "fur_and_scales": "Lông và vảy",
    "ears_as_hair": "Tai thay tóc",
    "ribbon_as_bra": "Ruy băng thay áo ngực",
    "food_as_clothes": "Thức ăn thay quần áo",
    "flower_as_hat": "Hoa thay mũ",
    # Đợt 6 — vòng 3: thẻ nóng cần dịch cả cụm.
    "hair_color_connection": "Nối màu tóc",
    "eye_color_connection": "Nối màu mắt",
    "skin_color_connection": "Nối màu da",
    "hair_color_change": "Đổi màu tóc",
    "eye_color_change": "Đổi màu mắt",
    "skin_change": "Đổi màu da",
    "official_alternate_hair_color": "Màu tóc phụ chính thức",
    "official_alternate_eye_color": "Màu mắt phụ chính thức",
    "official_alternate_skin_color": "Màu da phụ chính thức",
    "official_alternate_outfit": "Trang phục phụ chính thức",
    "official_alternate_hair_ornament": "Trang sức tóc phụ chính thức",
    "split_ponytail": "Tóc đuôi ngựa chia hai",
    "split_dress": "Váy xẻ đôi",
    "split-color_clothes": "Quần áo hai màu tách biệt",
    "split-color_skin": "Da hai màu tách biệt",
    "ring-tailed_lemur": "Culi đuôi vòng",
    "ring-tailed_cat": "Mèo đuôi vòng",
    "heart_ring_choker": "Vòng cổ trái tim hình nhẫn",
    "four-leaf_clover_earrings": "Khuyên tai cỏ bốn lá",
    "flowing_hair": "Tóc bay trong gió",
    "flowing_clothing": "Quần áo bay trong gió",
    "crooked_smile": "Nụ cười lệch",
    "severed_head": "Đầu bị cắt lìa",
    "dipstick_horn": "Sừng hình que",
    "dipstick_hair": "Tóc hình que",
    "button_up_skirt": "Chân váy cài cúc",
    "lace-up_legwear": "Quần tất buộc dây",
    "lace-up_gloves": "Găng tay buộc dây",
    "fur_lined_clothing": "Quần áo lót lông",
    "fur-lined_gloves": "Găng tay lót lông",
    "half-shaved_hair": "Tóc cạo một nửa",
    "x-shaped_eyewear": "Kính hình chữ X",
    "x-uniform": "Đồng phục chữ X",
    "head_up_shirt": "Áo sơ mi trễ cổ",
    "under_dress": "Đồ mặc trong váy",
    "under_clothes": "Đồ mặc trong quần áo",
    "on_swing": "Trên đu quay",
    "on_furniture": "Trên nội thất",
    "foreskin_day": "Ngày bao quy đầu",
    # Đợt 7 — trang phục: thẻ nóng.
    "national_soccer_team_uniform": "Đồng phục đội tuyển bóng đá quốc gia",
    "soccer_uniform": "Đồng phục bóng đá",
    "baseball_uniform": "Đồng phục bóng chày",
    "basketball_uniform": "Đồng phục bóng rổ",
    "volleyball_uniform": "Đồng phục bóng chuyền",
    "tennis_uniform": "Đồng phục quần vợt",
    "track_uniform": "Đồng phục điền kinh",
    "swim_cap": "Mũ bơi",
    "skirt_suit": "Bộ vest chân váy",
    "pencil_dress": "Váy bút chì",
    "tube_dress": "Váy quây",
    "sideless_dress": "Váy chẽ bên",
    "pillbox_hat": "Mũ hộp nhỏ",
    "porkpie_hat": "Mũ porkpie",
    "dixie_cup_hat": "Mũ giấy Dixie Cup",
    "hard_hat": "Mũ bảo hộ",
    "fold-over_boots": "Bốt gập cổ",
    "bowl_hat": "Mũ hình bát",
    "bowler_hat": "Mũ bowler",
    "ancient_egyptian_clothes": "Quần áo Ai Cập cổ đại",
    "ancient_roman_clothes": "Quần áo La Mã cổ đại",
    "roman_clothes": "Quần áo La Mã",
    "greek_clothes": "Quần áo Hy Lạp",
    "egyptian_clothing": "Trang phục Ai Cập",
    "arabian_clothes": "Quần áo Ả Rập",
    "korean_clothes": "Trang phục Hàn Quốc",
    "german_clothes": "Trang phục Đức",
    "ainu_clothes": "Trang phục Ainu",
    "hawaiian_shirt": "Áo Hawaii",
    "aloha_shirt": "Áo aloha",
    "meiji_schoolgirl_uniform": "Đồng phục nữ sinh thời Minh Trị",
    "native_american_clothes": "Trang phục người bản địa châu Mỹ",
    "victorian_clothes": "Quần áo thời Victoria",
    "edwardian_clothes": "Quần áo thời Edward",
    "medieval_clothes": "Quần áo trung cổ",
    "renaissance_clothes": "Quần áo thời phục hưng",
    "tactical_clothes": "Quần áo chiến thuật",
    "biker_clothes": "Quần áo dân phượt",
    "prison_clothes": "Quần áo tù nhân",
    "religious_clothing": "Trang phục tôn giáo",
    "work_uniform": "Đồng phục lao động",
    "workout_clothing": "Quần áo tập thể dục",
    "through_clothes": "Xuyên qua quần áo",
    "ofuda_on_clothes": "Bùa ofuda dán trên quần áo",
    "fool's_hat": "Mũ của chú hề",
    "lifting_another's_clothes": "Đang nhấc quần áo người khác lên",
    "pulling_another's_clothes": "Đang kéo quần áo người khác",
    "grabbing_another's_shirt": "Đang nắm áo sơ mi người khác",
    "undressing_another": "Đang cởi đồ người khác",
    "undressing_self": "Đang tự cởi đồ",
    "undressing_partner": "Đang cởi đồ cho bạn tình",
    "dressing_another": "Đang mặc đồ cho người khác",
    "putting_on_gloves": "Đang đeo găng tay",
    "putting_on_jewelry": "Đang đeo trang sức",
    "wringing_clothes": "Đang vắt quần áo",
    "padded_gloves": "Găng tay có đệm",
    "hooded_dress": "Đầm có mũ trùm",
    "patchwork_clothes": "Quần áo chắp vá",
    "ornate_clothes": "Quần áo cầu kỳ",
    "boob_hat": "Mũ hình ngực",
    "kimono_skirt": "Chân váy kimono",
    "overall_skirt": "Chân váy yếm",
    "belt_boots": "Bốt có quai",
    "waist_ribbon": "Ruy băng eo",
    "skirt_bow": "Nơ trên chân váy",
    "hat_band": "Dải băng mũ",
    "hat_tassel": "Tua rua trên mũ",
    "hat_bobbles": "Quả bông trên mũ",
    "glove_snap": "Cúc bấm găng tay",
    "ribbons_(anatomy)": "Dải ruy băng",
    "armored_vehicle": "Xe cơ giới bọc thép",
    "clothing_store": "Cửa hàng quần áo",
    "shoulderless_shirt": "Áo không vai",
    "breastless_clothes": "Quần áo không ngực",
    "crotchless_clothing": "Quần áo hở đũng",
    "undone_neck_ribbon": "Caravat nơ cổ chưa cài",
    "crossdressing_male": "Nam mặc đồ nữ",
    "crossdressing_female": "Nữ mặc đồ nam",
    "beaded_jewelry": "Trang sức hạt cườm",
    "team_skull_uniform": "Đồng phục Team Skull",
    # Đợt 9 — thẻ nóng bố cục và bối cảnh.
    "palm_tree": "Cây cọ",
    "bare_tree": "Cây trụi lá",
    "flower_field": "Cánh đồng hoa",
    "forest_background": "Nền rừng",
    "nature_background": "Nền thiên nhiên",
    "blurred_background": "Nền mờ",
    "photo_background": "Nền ảnh chụp",
    "grid_background": "Nền lưới",
    "halftone_background": "Nền chấm nửa tông",
    "geometric_background": "Nền hình học",
    "bedding_background": "Nền ga trải giường",
    "text_background": "Nền chữ",
    "starry_sky_background": "Nền bầu trời đầy sao",
    "star_symbol_background": "Nền biểu tượng ngôi sao",
    "city_background": "Nền thành phố",
    "bokeh_background": "Nền xóa phông",
    "blur_background": "Nền mờ",
    "bathroom": "Phòng tắm",
    "restroom": "Nhà vệ sinh",
    "public_restroom": "Nhà vệ sinh công cộng",
    "locker_room": "Phòng thay đồ",
    "living_room": "Phòng khách",
    "train_interior": "Bên trong tàu hỏa",
    "bamboo_forest": "Rừng tre",
    "nightclub": "Hộp đêm",
    "restroom_stall": "Buồng vệ sinh",
    "flight_deck": "Boong đáp",
    "mushroom_forest": "Rừng nấm",
    "coral_reef": "Rạn san hô",
    "broom": "Cây chổi",
    "broom_riding": "Cưỡi chổi",
    "holding_broom": "Cầm cây chổi",
    "electricity": "Dòng điện",
    "husky": "Chó husky",
    "beachball": "Quả bóng bãi biển",
    "beach_ball": "Quả bóng bãi biển",
    "skyscraper": "Nhà chọc trời",
    "snowman": "Người tuyết",
    "beach_umbrella": "Ô che nắng bãi biển",
    "beach_towel": "Khăn tắm bãi biển",
    "raincoat": "Áo mưa",
    "raining": "Trời đang mưa",
    "light_rays": "Tia sáng",
    "light_beam": "Chùm sáng",
    "light_trail": "Vệt sáng",
    "light_bulb": "Bóng đèn",
    "city_lights": "Đèn thành phố",
    "neon_lights": "Đèn neon",
    "christmas_lights": "Đèn Giáng sinh",
    "stage_lights": "Đèn sân khấu",
    "ceiling_light": "Đèn trần",
    "flashlight": "Đèn pin",
    "lighter": "Bật lửa",
    "glowstick": "Thanh phát sáng",
    "afterglow": "Ánh chiều tàn",
    "sidelighting": "Ánh sáng bên",
    "dappled_sunlight": "Nắng đốm",
    "drop_shadow": "Bóng đổ",
    "colored_shadow": "Bóng có màu",
    "window_shadow": "Bóng cửa sổ",
    "tree_shade": "Bóng cây",
    "shadow_face": "Mặt trong bóng tối",
    "lightning_bolt_symbol": "Biểu tượng tia sét",
    "flower_wreath": "Vòng hoa đội đầu",
    "flower_pot": "Chậu hoa",
    "film_grain": "Hạt phim",
    "mountainous_horizon": "Đường chân trời núi non",
    "oceanic_dolphin": "Cá heo đại dương",
    "cumulonimbus_cloud": "Mây vũ tích",
    "colorful": "Sặc sỡ",
    "color_swatch": "Mẫu màu",
    "color_guide": "Bảng hướng dẫn màu",
    "color_coded": "Mã hóa theo màu",
    "color_coded_text": "Chữ mã hóa theo màu",
    "color_connection": "Nối màu",
    "muted_colors": "Màu trầm",
    "warm_colors": "Màu ấm",
    "cool_colors": "Màu lạnh",
    "partially_colored": "Được tô màu một phần",
    "colored_extremities": "Ngón tay và ngón chân có màu",
    "colored_speech_bubble": "Bong bóng thoại có màu",
    "colored_blood": "Máu có màu",
    "colored_fire": "Lửa có màu",
    "colored_shoe_soles": "Đế giày có màu",
    "color_edit": "Chỉnh sửa màu",
    "incredibly_absurdres": "Độ nét cực cao",
    "nude_beach": "Bãi biển khỏa thân",
    "fart_cloud": "Đám mây trung tiện",
    "brainwashing": "Tẩy não",
    "neon_trim": "Viền phát quang",
    "half-length_portrait": "Ảnh chân dung nửa người",
    "bust_portrait": "Ảnh chân dung bán thân",
    "headshot_portrait": "Ảnh chân dung đầu và vai",
    "duo_portrait": "Ảnh chân dung hai người",
    "group_portrait": "Ảnh chân dung nhóm",
    "high-angle_view": "Góc nhìn từ trên cao",
    "top-down_view": "Góc nhìn từ trên xuống",
    "x-ray_view": "Cảnh chụp X-quang",
    "cross-section_view": "Mặt cắt",
    "profile_view": "Góc nhìn nghiêng",
    "three_quarter_view": "Góc nhìn ba phần tư",
    "resolution_mismatch": "Độ phân giải không khớp",
    "perspective": "Phối cảnh",
    "viewfinder": "Ống ngắm",
    "sketch_page": "Trang phác thảo",
    "sketchbook": "Sổ phác thảo",
    "colored_sketch": "Bản phác thảo có màu",
    "duo_focus": "Trọng tâm hai nhân vật",
    "group_focus": "Trọng tâm nhóm",
    "text_focus": "Trọng tâm chữ",
    "genital_focus": "Trọng tâm bộ phận sinh dục",
    "gore_focus": "Trọng tâm máu me",
    "reflection_focus": "Trọng tâm hình phản chiếu",
    "genital_close-up": "Cận cảnh bộ phận sinh dục",
    "full-body_tattoo": "Hình xăm toàn thân",
    "lighting": "Ánh sáng",
    "highlights_(coloring)": "Điểm sáng khi tô màu",
    "digital_painting": "Tranh kỹ thuật số",
    "traditional_painting": "Tranh truyền thống",
    "traditional_watercolor": "Màu nước truyền thống",
    "colored_pencil": "Bút chì màu",
    "slightly_chubby_female": "Nữ hơi mũm mĩm",
    "slightly_chubby_anthro": "Nhân vật hơi mũm mĩm",
    "slightly_chubby_male": "Nam hơi mũm mĩm",
    "restrained_arms": "Tay bị trói",
    "stationary_restraints": "Dụng cụ cố định",
    "musky_penis": "Dương vật có mùi xạ hương",
    "musky_butt": "Mông có mùi xạ hương",
    "musky_balls": "Tinh hoàn có mùi xạ hương",
    "neonate": "Sơ sinh",
    "bedroom_sex": "Quan hệ trong phòng ngủ",
    "sex_on_the_beach": "Quan hệ trên bãi biển",
    "talking_to_viewer": "Nói chuyện với người xem",
    "aiming_at_viewer": "Chĩa về phía người xem",
    "smirking_at_viewer": "Nhếch mép nhìn người xem",
    "grinning_at_viewer": "Cười toe nhìn người xem",
    "waving_at_viewer": "Vẫy tay với người xem",
    "blushing_at_viewer": "Đỏ mặt nhìn người xem",
    "frowning_at_viewer": "Nhăn mặt với người xem",
    "attacking_viewer": "Tấn công người xem",
    "feeding_viewer": "Đút cho người xem",
    "teasing_viewer": "Trêu người xem",
    "asking_viewer": "Hỏi người xem",
    "inviting_viewer": "Mời người xem",
    "flirting_with_viewer": "Tán tỉnh người xem",
    "seducing_viewer": "Cuốn hút người xem",
    "hypnotizing_viewer": "Thôi miên người xem",
    "punching_viewer": "Đấm người xem",
    "insulting_viewer": "Chửi người xem",
    "offering_to_viewer": "Đưa cho người xem",
    "gesturing_at_viewer": "Ra hiệu với người xem",
    "leashed_viewer": "Người xem bị xích",
    "viewer_holding_phone": "Người xem cầm điện thoại",
    "viewer_holding_leash": "Người xem cầm dây dắt",
    "background_character": "Nhân vật nền",
    "farting_at_viewer": "Đánh rắm vào người xem",
    # Đợt 10 — đuôi nhóm ngoại hình.
    "beady_eyes": "Mắt tròn nhỏ",
    "v-shaped_eyes": "Mắt hình chữ V",
    "unusually_open_eyes": "Mở mắt to bất thường",
    "wide-eyed_open": "Mở to mắt",
    "dotted_eyes": "Mắt chấm",
    "quad_tails": "Bốn cái đuôi",
    "tri_tails": "Ba cái đuôi",
    "9_tails": "Chín cái đuôi",
    "eastern_dragon_tail": "Đuôi rồng phương Đông",
    "eastern_dragon_horns": "Sừng rồng phương Đông",
    "mane_hair": "Tóc bờm",
    "non-mammal_hair": "Tóc của loài không có vú",
    "single_hair_tube": "Một ống tóc",
    "tucking_hair": "Đang giắt tóc",
    "twirling_hair": "Đang xoắn tóc",
    "playing_with_own_hair": "Đang chơi với tóc của mình",
    "stray_pubic_hair": "Sợi lông mu xổ ra",
    "anal_hair": "Lông hậu môn",
    "hand_in_another's_hair": "Tay đặt trong tóc người khác",
    "hands_on_another's_face": "Hai tay đặt lên mặt người khác",
    "hand_on_another's_tail": "Tay đặt lên đuôi người khác",
    "centered_hair_bow": "Nơ tóc chính giữa",
    "character_hair_ornament": "Phụ kiện tóc hình nhân vật",
    "d-pad_hair_ornament": "Phụ kiện tóc hình D-pad",
    "bandaid_hair_ornament": "Phụ kiện tóc băng cá nhân",
    "lightning_bolt_hair_ornament": "Phụ kiện tóc hình tia sét",
    "farting_on_face": "Đánh rắm lên mặt",
    "gauze_on_face": "Gạc y tế trên mặt",
    "urine_on_face": "Nước tiểu trên mặt",
    "feces_on_face": "Phân trên mặt",
    "paint_splatter_on_face": "Sơn văng trên mặt",
    "lipstick_mark_on_face": "Vệt son môi trên mặt",
    "instrument_case_on_back": "Bao nhạc cụ trên lưng",
    "diving_mask_on_head": "Mặt nạ lặn trên đầu",
    "writing_on_thigh": "Chữ viết trên đùi",
    "writing_on_chest": "Chữ viết trên ngực",
    "urine_on_chest": "Nước tiểu trên ngực",
    "smiley_face": "Khuôn mặt cười",
    "fisheye": "Ống kính mắt cá",
    "lounge_chair": "Ghế dài thư giãn",
    "drawing_tablet": "Bảng vẽ",
    "rudder_footwear": "Giày bánh lái",
    "legwear_garter": "Dây giữ quần tất",
    "69_position": "Vị trí 69",
    "no_dickey": "Không có cổ áo giả",
    "no_irises": "Không có mống mắt",
    "no_shading": "Không tô bóng",
    "no_internal_organs": "Không có nội tạng",
    "no_entry_sign": "Biển cấm vào",
    "no_parking_sign": "Biển cấm đậu xe",
    "not_for_sale": "Không bán",
    "not_present": "Không xuất hiện",
    "no_lube": "Không có chất bôi trơn",
    "partially_shaded_face": "Mặt tô bóng một phần",
    "furrowed_eyebrows": "Lông mày cau",
    "hypnotic_eyes": "Mắt bị thôi miên",
    "compound_eyes": "Mắt kép",
    "bloodshot_eyes": "Mắt vằn máu",
    "hollow_eyes": "Mắt trũng",
    "obscured_eyes": "Mắt bị che",
    "derp_eyes": "Mắt ngơ ngác",
    "cable_tail": "Đuôi dạng cáp",
    "tapering_tail": "Đuôi vót nhọn",
    "entwined_tails": "Đuôi quấn vào nhau",
    "nub_tail": "Đuôi cộc",
    "cleft_tail": "Đuôi chẻ đôi",
    "cetacean_tail": "Đuôi cá voi",
    "tapir_tail": "Đuôi heo vòi",
    "crocodilian_tail": "Đuôi họ cá sấu",
    "hair_extensions": "Tóc nối",
    "ovum_with_heart": "Trứng có tim",
    # Đợt 10 — tên đội/trường ghép nhầm thành tính từ + danh từ thường.
    "orange_planet_uniform": "Đồng phục Orange Planet",
    "angry_sweatdrop": "Giọt mồ hôi bực bội",
    "embarrassed_sweatdrop": "Giọt mồ hôi ngượng ngùng",
    "band-aid_on_face": "Băng cá nhân trên mặt",
    "band-aid_on_arm": "Băng cá nhân trên tay",
    "band-aid_on_leg": "Băng cá nhân trên chân",
    "0_0": "Mắt tròn xoe",
})


def _vi_singular(token):
    """Số ít của một token (chỉ dùng khi bản gốc không có trong từ điển)."""
    fixed = _TAG_VI_PLURAL_IRREGULAR.get(token)
    if fixed:
        return fixed
    if len(token) < 5 or not token.endswith("s") or token in _TAG_VI_PLURAL_TRAPS:
        return None
    candidates = []
    if token.endswith("ies"):
        candidates.append(token[:-3] + "y")
    if token.endswith(("hes", "xes", "zes", "ches", "shes", "ses")):
        candidates.append(token[:-2])
    candidates.append(token[:-1])
    for cand in candidates:
        if len(cand) >= 3 and cand.isalpha() and cand != token:
            if (cand in _TAG_VI_WORDS or cand in _TAG_VI_COMPOSITE_HEADS
                    or cand in _TAG_VI_COLORS):
                return cand
    return None


def _vi_word(token):
    """Nhãn tiếng Việt của một từ bổ nghĩa, chấp nhận số nhiều và sở hữu cách."""
    label = _TAG_VI_WORDS.get(token) or _TAG_VI_COLORS.get(token)
    if label:
        return label
    possessive = _TAG_VI_POSSESSIVES.get(token)
    if possessive:
        return possessive
    if token.endswith("'s"):
        stem = token[:-2]
        stem_label = (_TAG_VI_WORDS.get(stem) or _TAG_VI_COMPOSITE_HEADS.get(stem)
                      or TAG_VI_LABELS.get(stem))
        if stem_label:
            return "của " + _lower_first(stem_label)
    singular = _vi_singular(token)
    if singular:
        return _TAG_VI_WORDS.get(singular) or _TAG_VI_COLORS.get(singular)
    return None


def _vi_head(key):
    """Danh từ chính của cụm; thử cả số ít của token cuối nếu cần."""
    head = _TAG_VI_COMPOSITE_HEADS.get(key)
    if head:
        return head
    parts = key.split("_")
    singular = _vi_singular(parts[-1])
    if singular:
        return _TAG_VI_COMPOSITE_HEADS.get("_".join([*parts[:-1], singular]))
    return None


# Động từ tác động lên món đồ: dạng <món đồ>_<động từ> do quy tắc cụm đảm nhiệm.
_TAG_VI_GARMENT_ACTION_VERBS = {
    "lift", "pull", "grab", "adjust", "lower", "unzip", "tug", "remove", "wear",
}

# Động từ hai ngôi trong thẻ e621: "<A>_penetrating_<B>" → "<A> thâm nhập <B>".
_TAG_VI_TRANSITIVE_RELATIONS = {
    "penetrating": "thâm nhập", "rimming": "liếm hậu môn", "fingering": "đút ngón tay vào",
    "molesting": "sàm sỡ", "undressing": "cởi đồ của", "dickriding": "cưỡi lên",
    "facesitting": "ngồi lên mặt", "riding": "cưỡi lên", "kissing": "hôn",
}


def _vi_modifier(part):
    """Bản dịch một từ bổ nghĩa trong cụm: WORDS, màu, số lượng hoặc danh từ chính.

    Danh từ chính cũng dùng làm bổ ngữ được (`penis` trong `penis_size_difference`,
    `species` trong `species_transformation`) — chỉ mở khi token đó có trong HEADS nên
    không thể tạo nhãn bừa.
    

    "4_arms" → "4" không có trong WORDS nhưng tiếng Việt đọc là lượng từ ("Bốn cánh tay").
    Chỉ nhận chữ số nào có sẵn trong bảng (1..20), nếu không "69_position" sẽ thành
    "Sáu mươi chín cái vị trí".
    """
    # WORDS thắng (giữ "single" → "một chiếc"); số lượng chỉ là phương án dự phòng
    # cho chữ số 1..20 và multi/many/both… — nhờ vậy "single_leg_armor" không mất "chiếc".
    word = _vi_word(part)
    if word:
        return word
    quantity = _TAG_VI_QUANTITY_MODIFIERS.get(part)
    if quantity:
        return _lower_first(quantity)
    head = _TAG_VI_COMPOSITE_HEADS.get(part)
    return _lower_first(head) if head else None


# Từ láy tiếng Việt: lặp âm có chủ đích, KHÔNG phải lỗi trùng từ khi ghép.
_TAG_VI_REDPPLICAS = {  # tra thêm giá trị từ điển ở _vi_reduplicatives()
    "lùm lùm", "ròng ròng", "hồng hồng", "giật giật", "xa xa", "gần gần", "man mác",
    "nhè nhẹ", "hiu hiu", "lành lạnh", "se se", "đỏ đỏ", "trắng trắng", "nhỏ nhỏ",
    "lơ thơ", "bồng bềnh", "phất phới", "long lanh", "thoang thoảng", "nho nhỏ",
    "sàn sật", "lách cách", "rào rào", "ào ào", "phập phồng", "bập bềnh",
}


_VI_REDUPLICATIONS = None


def _vi_reduplicatives():
    """Các cặp "X X" ĐƯỢC PHÉP: từ láy khai báo tay + giá trị có sẵn trong từ điển.

    "chuồn chuồn" (dragonfly) hay "bươm bướm" là từ thật, không phải lỗi lặp khi ghép,
    nên chỉ cần một bảng giá trị trong từ điển có dạng trùng nhau là được miễn gộp.
    """
    global _VI_REDUPLICATIONS
    if _VI_REDUPLICATIONS is None:
        pairs = set(_TAG_VI_REDPPLICAS)
        tables = (_TAG_VI_WORDS, _TAG_VI_COMPOSITE_HEADS, _TAG_VI_COLORS, TAG_VI_LABELS,
                  _TAG_VI_ACTION_PREFIXES, _TAG_VI_PAST_STATES, _TAG_VI_TRAILING_ACTIONS,
                  _TAG_VI_TRAILING_STATES, _TAG_VI_LEADING_STATES)
        for table in tables:
            for raw in table.values():
                # Bảng quy tắc có hậu tố "|liên từ": chỉ phần bản dịch mới tính.
                words = raw.split("|")[0].strip().casefold().split()
                for index in range(1, len(words)):
                    if words[index] == words[index - 1]:
                        pairs.add(f"{words[index]} {words[index]}")
        _VI_REDUPLICATIONS = pairs
    return _VI_REDUPLICATIONS


def _tidy_vietnamese_label(label):
    """Gộp từ bị lặp liền nhau do ghép (`tóc tóc`, `hình hình`, `màu màu`).

    Chỉ áp cho nhãn sinh bởi quy tắc; nhãn đã curate giữ nguyên vì người viết có thể
    cố ý lặp ("Doki Doki Literature Club").
    """
    if not label:
        return label
    words = label.split()
    out = []
    for index, word in enumerate(words):
        if index and word == words[index - 1]:
            if f"{word} {word}".casefold() in _vi_reduplicatives():
                out.append(word)
            continue  # bỏ bản sao thứ hai, trừ khi là từ láy thật
        out.append(word)
    return " ".join(out)


def _compose_vietnamese_tag_label(name):
    """Translate only a known modifier sequence followed by a known tag noun."""
    parts = re.split(r"[_-]+", name.casefold())
    if len(parts) < 2 or len(parts) > 7 or any(
        not (part.isalpha() or part.isdigit()) for part in parts
    ):
        return None
    # "covering_nipples": động từ/trạng thái đứng đầu phải do quy tắc cụm xử lý,
    # nếu không sẽ đảo nhầm thành "Núm vú che".
    if parts[0] in _TAG_VI_ACTION_PREFIXES or parts[0] in _TAG_VI_PAST_STATES:
        return None
    # "cloak_lift": HEAD "lift" từng thắng và cho "Nâng lên áo choàng" trong khi quy tắc
    # cụm <món đồ>_<động từ> dịch đúng hơn ("Nhấc áo choàng", "Đang kéo quần short").
    if (len(parts) == 2 and parts[1] in _TAG_VI_GARMENT_ACTION_VERBS and parts[0] in _TAG_VI_GARMENT_TOKENS
            and parts[0] not in _TAG_VI_BODY_TOKENS):
        # Nhường quy tắc cụm cho <món đồ>_<động từ>: "Nhấc áo choàng" đúng hơn
        # "Nâng lên áo choàng", "Đang kéo quần short" đúng hơn "Kéo quần". Bộ phận cơ thể
        # (butt_grab, head_grab) không có trong quy tắc cụm nên compose vẫn lo.
        return None
    if (len(parts) > 2 and parts[-1] in _TAG_VI_TRAILING_ACTIONS
            and parts[-2] in _TAG_VI_ACTION_TARGETS):
        # "pseudo_skirt_lift": động từ đứng sau danh từ thì quy tắc cụm lo thứ tự,
        # còn "nose_piercing"/"butt_grab" (danh từ thật) vẫn phải do compose dịch.
        return None
    for size in range(min(2, len(parts) - 1), 0, -1):
        head_key = "_".join(parts[-size:])
        head = _vi_head(head_key)
        modifiers = parts[:-size]
        if not head or not modifiers or any(_vi_modifier(part) is None for part in modifiers):
            continue
        # Số lượng dạng chữ số chỉ an toàn khi nó là bổ ngữ DUY NHẤT ("9_tails").
        if len(modifiers) > 1 and any(part.isdigit() for part in modifiers):
            continue
        # "three_tails"/"multiple_arms": tiếng Việt đặt số lượng TRƯỚC danh từ.
        if len(modifiers) == 1 and modifiers[0] in _TAG_VI_QUANTITY_MODIFIERS:
            quantity = " ".join(_TAG_VI_QUANTITY_MODIFIERS[part] for part in modifiers)
            noun = _lower_first(head)
            if quantity.split()[-1] in _TAG_VI_COUNT_WORDS:
                classifier = _TAG_VI_NOUN_CLASSIFIERS.get(noun.split()[0], "cái")
                if classifier and " " not in noun:
                    noun = f"{classifier} {noun}"
            return f"{quantity} {noun}".strip()
        colors = [part for part in modifiers if part in _TAG_VI_COLORS]
        shades = [part for part in modifiers if part in {"light", "dark", "bright", "pale"}]
        qualifiers = [part for part in modifiers if part in _TAG_VI_NOUN_MODIFIERS]
        adjectives = [
            part for part in modifiers
            if part not in _TAG_VI_NOUN_MODIFIERS and part not in colors and part not in shades
        ]
        # Tiếng Việt đặt danh từ bổ nghĩa trước tính từ và ngược chuỗi danh từ tiếng Anh.
        if "shaped" in modifiers and qualifiers:
            shape = "hình " + " ".join(_vi_modifier(part) for part in reversed(qualifiers))
            qualifiers = []
            translated = [shape, *(_vi_modifier(part) for part in adjectives if part != "shaped")]
        else:
            translated = [
                *(_vi_modifier(part) for part in reversed(qualifiers)),
                *(_vi_modifier(part) for part in adjectives),
            ]
        translated_colors = [_TAG_VI_COLORS.get(part) or _vi_modifier(part) for part in colors]
        if translated_colors and shades:
            translated_colors.extend(_vi_modifier(part) for part in shades)
        elif shades:
            translated.extend(_vi_modifier(part) for part in shades)
        if translated_colors:
            color_phrase = " ".join(translated_colors)
            if head_key in _TAG_VI_COLOR_MARKER_HEADS:
                color_phrase = "màu " + color_phrase
            translated.append(color_phrase)
        return f"{head} {' '.join(translated)}".strip()
    return None


# --- Chi tiết nhân vật: họ thẻ "<động từ>_<món đồ>", "<A>_<giới từ>_<B>" và "<stem>less".
# Chỉ sinh nhãn khi MỌI mảnh đã có bản dịch — không bao giờ đoán bừa.
_TAG_VI_ACTION_PREFIXES = {
    "holding": "Cầm", "carrying": "Mang theo", "adjusting": "Đang chỉnh", "pulling": "Đang kéo",
    "tugging": "Đang kéo", "pushing": "Đang đẩy", "covering": "Che", "exposed": "Lộ",
    "hidden": "Ẩn", "no": "Không có", "without": "Không có", "wearing": "",
    "brandishing": "Đang giơ", "licking": "Đang liếm", "biting": "Đang cắn",
    "kicking": "Đang đá", "poking": "Đang chỉ vào", "tying": "Đang buộc",
    "untying": "Đang cởi dây",
    # Họ "nhìn/chỉ" + tư thế: tiếng Việt bỏ qua liên từ "at/to" nên có "|" để lược nó.
    "looking": "Nhìn|at", "staring": "Nhìn chằm chằm|at", "glaring": "Nhìn giận dữ|at",
    "glancing": "Liếc nhìn|at", "pointing": "Chỉ vào|at", "smiling": "Cười|at",
    "facing": "Hướng về phía|at,to", "peeking": "Nhòm|through,into,at",
    "sitting": "Ngồi|on,in,at", "lying": "Nằm|on,in", "standing": "Đứng|on,in,at",
    "kneeling": "Quỳ|on,in", "crouching": "Ngồi xổm|on", "perched": "Đậu trên|on",
    "leaning": "Tựa vào|on,against", "resting": "Nghỉ trên|on", "balancing": "Giữ thăng bằng trên|on",
    "stepping": "Bước|on,over", "tripping": "Vấp phải|on",
    "riding": "Cưỡi|on", "touching": "Chạm|", "grabbing": "Nắm|",
    "hugging": "Ôm|", "cuddling": "Ôm ấp|", "kissing": "Hôn|", "nuzzling": "Dụi vào|",
    "sniffing": "Ngửi|", "smelling": "Ngửi|", "tasting": "Nếm|", "swallowing": "Nuốt|",
    "slapping": "Đang tát", "spanking": "Đét|", "hitting": "Đánh|", "patting": "Vỗ|",
    "strangling": "Bóp cổ|", "choking": "Nghẹt cổ|", "reaching": "Với tay tới|",
    "swinging": "Đung đưa|", "throwing": "Đang ném", "catching": "Đang bắt lấy",
    "showing": "Đang cho thấy|", "blowing": "Thổi|on", "tearing": "Đang xé|",
    "removing": "Đang cởi|", "discarding": "Vứt|", "raising": "Nâng|up",
    # Đợt 6 — chi tiết nhân vật:
    "rubbing": "Đang xoa|",
    "chewing": "Đang nhai|",
    "cutting": "Đang cắt|",
    "drying": "Đang sấy|",
    "brushing": "Đang chải|",
    "combing": "Đang chải|",
    "ironing": "Đang ủi|",
    "washing": "Đang giặt|",
    "stretching": "Đang kéo giãn|",
    "bending": "Đang bẻ cong|",
    "twisting": "Đang xoắn|",
    "wrapping": "Đang quấn|",
    "unwrapping": "Đang mở|",
    "painting": "Đang vẽ|",
    "drawing": "Đang vẽ|",
    "writing": "Đang viết|",
    "carving": "Đang khắc|",
    "lighting": "Đang thắp sáng|",
    "counting": "Đang đếm|",
    "measuring": "Đang đo|",
    "squeezing": "Đang bóp|",
    "inflating": "Đang bơm to|",
    "folding": "Đang gấp|",
    "unfolding": "Đang mở ra|",
    "shaking": "Đang lắc|",
    "waving": "Đang vẫy|at,to",
    "spinning": "Đang xoay|",
    "rolling": "Đang cuộn|",
    "gripping": "Đang siết|",
    "kneading": "Đang nhào|",
    "peeling": "Đang bóc|",
    "soaping": "Đang thoa xà phòng|",
    # Đợt 7 — động từ mặc/cởi.
    "lifting": "Đang nhấc|",
    "wringing": "Đang vắt|",
    "dressing": "Đang mặc|",
    "undressing": "Đang cởi|",
    "putting": "Đang mặc|on",
    "stripping": "Đang lột|",
    "zipping": "Đang kéo khóa|",
    "buttoning": "Đang cài cúc|",
    "unbuttoning": "Đang mở cúc|",
    # Đợt 9 — động từ tương tác với người xem.
    "talking": "Nói chuyện với|to,at",
    "aiming": "Chĩa về phía|at,to",
    "smirking": "Nhếch mép nhìn|at,to",
    "grinning": "Cười toe nhìn|at,to",
    "frowning": "Nhăn mặt với|at,to",
    "shouting": "Hét với|at,to",
    "pouting": "Giờn dỗi với|at,to",
    "asking": "Hỏi|at,to",
    "feeding": "Đút cho|at,to",
    "teasing": "Trêu|at,to",
    "attacking": "Tấn công|at,to",
    "blushing": "Đỏ mặt nhìn|at,to",
    "beckoning": "Vẫy gọi|at,to",
    "gesturing": "Ra hiệu|at,to",
    "offering": "Đưa cho|to,at",
    "insulting": "Chửi|at,to",
    "hypnotizing": "Thôi miên|at,to",
    "flirting": "Tán tỉnh|with,to,at",
    "seducing": "Cuốn hút|at,to",
    "inviting": "Mời|at,to",
    "punching": "Đấm|at,to",
    "stomping": "Dẫm lên|at,on",
    "mocking": "Chế nhạo|at,of",
    "yelling": "Hét vào|at,to",
    "laughing": "Cười trước|at,to",
    "farting": "Đánh rắm vào|at,to",
    # Đợt 10 — động từ với tóc.
    "tucking": "Đang giắt|into,in",
    "twirling": "Đang xoắn|",
    "flipping": "Đang hất|",
    "playing_with": "Đang chơi với|",
}
# Trạng thái đã rồi: đặt SAU danh từ theo tiếng Việt ("<áo> được buộc").
_TAG_VI_PAST_STATES = {
    # Đợt 11: dạng bị động của họ thẻ quan hệ.
    "penetrated": "bị thâm nhập",
    "rimmed": "bị liếm hậu môn",
    "mounted": "bị cưỡi lên",
    "tied": "được buộc", "untied": "cởi dây", "unbuttoned": "mở khuy",
    "unzipped": "kéo khóa", "unlaced": "cởi dây buộc", "unfastened": "mở",
    "hidden": "được che", "concealed": "được che", "revealed": "lộ ra",
    "rolled": "cuộn lên", "folded": "gấp lại", "tucked": "giắt vào trong",
    "untucked": "bỏ giắt", "lifted": "nâng lên", "lowered": "hạ xuống",
    "pulled": "kéo", "pushed": "đẩy", "tilted": "nghiêng", "turned": "quay đi",
    "grabbed": "bị nắm", "held": "được cầm", "dropped": "rơi xuống",
    "loosened": "nới lỏng", "tightened": "thắt chặt", "bitten": "bị cắn",
    "torn": "bị rách", "cut": "bị cắt", "flipped": "lật sang bên",
    "removed": "đã tháo ra", "discarded": "đã bỏ đi",
    # Đợt 7 — trạng thái mở/vắt.
    "undone": "chưa cài",
    "wrung": "đã vắt",
}
# Hướng chuyển động "<danh từ>_<hướng>". Tiếng Việt đặt động từ trước nên cần mẫu riêng
# cho từng lớp: trang phục / chi tay chân / đuôi-cánh-tai / còn lại.
_TAG_VI_DIRECTION_GARMENT = {
    "down": "Kéo {} xuống", "up": "Kéo {} lên", "aside": "Kéo {} sang bên",
    "off": "Cởi {}", "on": "Mặc {}", "open": "Mở {}",
}
_TAG_VI_DIRECTION_LIMB = {
    "up": "Giơ {} lên", "down": "Hạ {} xuống", "out": "Dạng {} ra",
    "forward": "Đưa {} về phía trước", "back": "Đưa {} ra sau",
}
_TAG_VI_DIRECTION_BODY = {
    "up": "{} dựng lên", "down": "{} cụp xuống", "upright": "{} dựng đứng",
    "flat": "{} ép phẳng", "back": "{} ép về sau",
}
_TAG_VI_DIRECTION_PLAIN = {
    "down": "{} hướng xuống", "up": "{} hướng lên", "aside": "{} lệch bên",
    "away": "{} quay ra xa", "forward": "{} hướng tới trước", "back": "{} ngả về sau",
    "sideways": "{} nghiêng bên", "open": "{} mở", "closed": "{} khép lại",
    "through": "{} xuyên qua", "out": "{} thò ra",
}
_TAG_VI_GARMENT_TOKENS = frozenset((
    "shirt dress skirt pants shorts underwear topwear bottomwear panties thighhighs kneehighs "
    "socks stockings gloves mittens jacket coat hoodie sweater vest uniform swimsuit swimwear "
    "bikini apron cloak cape legwear armwear footwear headwear headgear eyewear hat cap helmet "
    "shoes boots sandals slippers necklace scarf tie necktie bowtie belt veil mask glasses goggles "
    "choker earring earrings jewelry leotard bodysuit corset bra armor armor_set clothing"
).split())
_TAG_VI_LIMB_TOKENS = frozenset("arm arms hand hands leg legs foot feet finger fingers toe toes".split())
_TAG_VI_BODY_TOKENS = frozenset("tail ear ears wing wings antenna antlers fin fins horn horns".split())
_TAG_VI_ONLY_WEAR = _TAG_VI_GARMENT_TOKENS | frozenset(
    "hat cap glasses goggles mask blindfold collar tie belt choker necklace".split())
# Tân ngữ hợp lệ cho quy tắc "<món đồ|bộ phận>_<động từ>".
_TAG_VI_ACTION_TARGETS = (_TAG_VI_GARMENT_TOKENS | _TAG_VI_LIMB_TOKENS | _TAG_VI_BODY_TOKENS
                          | frozenset("hair fur muzzle snout nipples breasts cheeks hips thighs".split()))
# Nội động từ/trạng thái đứng SAU danh từ tiếng Việt dù tiếng Anh để trước danh từ.
_TAG_VI_LEADING_STATES = {
    "melting": "đang tan chảy", "freezing": "đang đóng băng", "dissolving": "đang tan ra",
    "burning": "đang cháy", "rotting": "đang phân hủy", "rusting": "đang gỉ",
    "withering": "đang héo", "fading": "bị phai", "steaming": "đang bốc khói",
    "smoking": "đang bốc khói", "bleeding": "đang chảy máu", "twitching": "giật giật",
}
# Khung "đồng phục của <tên riêng>": chỉ dịch PHẦN KHUNG, tên riêng giữ nguyên.
# Đây là họ thẻ lớn nhất còn trống của nhóm trang phục (~600 thẻ): tên trường, học viện,
# đội, tổ chức trong anime/game không có bản dịch Việt, nhưng "school_uniform" thì có.
_TAG_VI_UNIFORM_FRAMES = {
    "girls'_academy_school_uniform": "Đồng phục học viện nữ sinh",
    "girls_academy_school_uniform": "Đồng phục học viện nữ sinh",
    "private_academy_school_uniform": "Đồng phục học viện tư thục",
    "private_high_school_uniform": "Đồng phục trường trung học tư thục",
    "junior_high_school_uniform": "Đồng phục trường trung học cơ sở",
    "middle_school_uniform": "Đồng phục trường trung học cơ sở",
    "high_school_uniform": "Đồng phục trường trung học phổ thông",
    "academy_school_uniform": "Đồng phục học viện",
    "gakuen_school_uniform": "Đồng phục học viện",
    "academy_uniform": "Đồng phục học viện",
    "girls_school_uniform": "Đồng phục trường nữ sinh",
    "schoolgirl_uniform": "Đồng phục nữ sinh",
    "schoolboy_uniform": "Đồng phục nam sinh",
    "school_uniform": "Đồng phục trường",
    "military_uniform": "Đồng phục quân đội",
    "naval_uniform": "Đồng phục hải quân",
    "police_uniform": "Đồng phục cảnh sát",
    "nurse_uniform": "Đồng phục y tá",
    "maid_uniform": "Đồng phục hầu gái",
    "butler_uniform": "Đồng phục quản gia",
    "idol_uniform": "Đồng phục thần tượng",
    "cheerleader_uniform": "Đồng phục cổ động viên",
    "training_uniform": "Đồng phục tập luyện",
    "monastery_uniform": "Đồng phục tu viện",
    "dorm_uniform": "Đồng phục ký túc xá",
    "prison_uniform": "Đồng phục tù nhân",
    "national_soccer_team_uniform": "Đồng phục đội tuyển bóng đá quốc gia",
    "squad's_uniform": "Đồng phục đội",
    "team_uniform": "Đồng phục đội",
    "uniform": "Đồng phục",
}
# Sở hữu cách: "của X" đặt SAU danh từ trong tiếng Việt.
_TAG_VI_POSSESSIVES = {
    "another's": "của người khác", "other's": "của người kia", "someone's": "của ai đó",
    "player's": "của người chơi", "owner's": "của chủ nhân", "viewer's": "của người xem",
    "character's": "của nhân vật", "author's": "của tác giả", "artist's": "của họa sĩ",
    "girl's": "của cô gái", "boy's": "của cậu bé", "women's": "của phụ nữ",
    "men's": "của đàn ông", "kid's": "của trẻ em", "cat's": "của mèo", "dog's": "của chó",
    "fox's": "của cáo", "rabbit's": "của thỏ", "dragon's": "của rồng",
}
_TAG_VI_NAME_PUNCTUATION = re.compile(r"^[a-z0-9][a-z0-9.'\-]*$")


def _vi_proper_name(text):
    """Giữ nguyên tên riêng nhưng viết hoa đúng: "st._gloriana's" → "St. Gloriana's"."""
    words = []
    for token in text.replace("_", " ").split():
        if re.fullmatch(r"[a-z](\.[a-z])*\.", token) or re.fullmatch(r"[a-z](\.[a-z])+\.", token):
            words.append(token.upper())  # "u.a." -> "U.A.", "st." -> "St."
            continue
        words.append(token[:1].upper() + token[1:])
    return " ".join(words)


def _tag_vi_uniform_frame(name):
    """Dịch khung "<tên riêng>_<loại>uniform", giữ nguyên tên riêng."""
    lowered = name.casefold()
    for frame in sorted(_TAG_VI_UNIFORM_FRAMES, key=len, reverse=True):
        if not lowered.endswith("_" + frame) and lowered != frame:
            continue
        prefix = name[: len(name) - len(frame) - 1].strip() if lowered != frame else ""
        if not prefix:
            continue
        tokens = [token for token in prefix.replace(" ", "_").split("_") if token]
        if not tokens or any(not _TAG_VI_NAME_PUNCTUATION.fullmatch(token) for token in tokens):
            return None
        # Chỉ màu/sắc thái mới biến "<X>_school_uniform" thành mô tả chung
        # ("red_school_uniform" -> "Đồng phục học đường màu đỏ"). Danh từ chung khác
        # ("dream", "paradise") vẫn là phần tên riêng của học viện trong anime.
        shades = {"light", "dark", "bright", "pale"}
        if all(token in _TAG_VI_COLORS or token in shades for token in tokens):
            return None
        return f"{_TAG_VI_UNIFORM_FRAMES[frame]} {_vi_proper_name(prefix)}"
    return None


# Lượng từ: tiếng Việt đặt TRƯỚC danh từ ("three_tails" → "Ba cái đuôi").
_TAG_VI_QUANTITY_MODIFIERS = {
    "1": "Một", "2": "Hai", "3": "Ba", "4": "Bốn", "5": "Năm", "6": "Sáu", "7": "Bảy",
    "8": "Tám", "9": "Chín", "one": "Một", "two": "Hai", "three": "Ba", "four": "Bốn",
    "five": "Năm", "six": "Sáu", "seven": "Bảy", "eight": "Tám", "nine": "Chín",
    "ten": "Mười", "eleven": "Mười một", "twelve": "Mười hai", "single": "Một",
    "multi": "Nhiều", "multiple": "Nhiều", "many": "Nhiều", "few": "Vài", "extra": "Thêm", "both": "Cả hai",
    "triple": "Ba", "quad": "Bốn", "quadruple": "Bốn", "quintuple": "Năm",
    "sextuple": "Sáu", "septuple": "Bảy", "octuple": "Tám", "nonuple": "Chín",
    "decuple": "Mười",
    # Đợt 10: catalog có thẻ ghi số rời 10..20 ("13_hearts") — ghép tiếp cùng quy tắc.
    "10": "Mười", "11": "Mười một", "12": "Mười hai", "13": "Mười ba", "14": "Mười bốn",
    "15": "Mười lăm", "16": "Mười sáu", "17": "Mười bảy", "18": "Mười tám", "19": "Mười chín",
    "20": "Hai mươi", "thirteen": "Mười ba", "fourteen": "Mười bốn", "fifteen": "Mười lăm",
    "sixteen": "Mười sáu", "seventeen": "Mười bảy", "eighteen": "Mười tám",
    "nineteen": "Mười chín", "twenty": "Hai mươi",
}
# Chỉ số đếm mới cần loại từ "cái"; "Thêm"/"Cả hai" đứng trần trước danh từ.
_TAG_VI_COUNT_WORDS = frozenset(
    "Một Hai Ba Bốn Năm Sáu Bảy Tám Chín Mười Mười một Mười hai Nhiều Vài".split())
# Danh từ đã mang sẵn loại từ thì không thêm "cái"; vài danh từ cần loại từ riêng.
_TAG_VI_NOUN_CLASSIFIERS = {
    "mắt": "con", "tai": "cái", "đuôi": "cái", "sừng": "cái", "mũi": "cái", "váy": "chiếc",
    "áo": "chiếc", "quần": "chiếc", "giày": "đôi", "dép": "đôi", "găng": "đôi",
    "nhẫn": "chiếc", "khăn": "chiếc", "chân": None, "cánh": None, "tóc": None,
    "ngực": None, "tay": None, "miệng": None, "dây": None, "vai": None, "lưng": None, "hoa": "bông", "lá": "chiếc", "sừng": "cái",
}
# Số nhiều bất quy tắc; chỉ tra khi bản gốc không có trong từ điển.
_TAG_VI_PLURAL_IRREGULAR = {
    "feet": "foot", "teeth": "tooth", "geese": "goose", "mice": "mouse", "leaves": "leaf",
    "knives": "knife", "wolves": "wolf", "lives": "life", "halves": "half",
    "shelves": "shelf", "wives": "wife", "calves": "calf", "hooves": "hoof",
    "elves": "elf", "loaves": "loaf", "children": "child", "men": "man", "women": "woman",
    "people": "person", "scarves": "scarf", "lice": "louse", "dice": "die",
}
# Số nhiều mà số ít trong từ điển mang nghĩa khác → không được suy luận.
_TAG_VI_PLURAL_TRAPS = frozenset(
    "shorts pants jeans glasses scissors panties drawers braces tights pumps goods works means "
    "savings thanks tweezers pliers supplies contents riches grounds odds damages outdoors "
    "interiors customs senses measures seasons".split())
# Động từ tiếng Việt đặt TRƯỚC danh từ dù tiếng Anh để sau: "dress_pull" → "Đang kéo váy".
_TAG_VI_TRAILING_ACTIONS = {
    "pull": "Đang kéo", "tug": "Đang kéo", "tugging": "Đang kéo", "yank": "Giật",
    "grab": "Nắm", "grabbing": "Đang nắm", "sniff": "Đang ngửi", "sniffing": "Đang ngửi",
    "lick": "Liếm", "licking": "Đang liếm", "bite": "Cắn", "biting": "Đang cắn",
    "kiss": "Hôn", "kissing": "Đang hôn", "hug": "Ôm", "hugging": "Đang ôm",
    "pat": "Vỗ", "patting": "Vỗ", "pet": "Vuốt", "petting": "Đang vuốt ve",
    "stroke": "Vuốt", "stroking": "Đang vuốt", "adjust": "Chỉnh", "adjusting": "Đang chỉnh",
    "pin": "Ghim", "pinning": "Đang ghim", "clip": "Kẹp", "remove": "Tháo", "tear": "Xé",
    "cut": "Cắt", "wash": "Giặt", "measure": "Đo", "count": "Đếm", "focus": "Nhấn mạnh",
    "theft": "Trộm", "writing": "Viết lên", "drawing": "Vẽ lên", "piercing": "Xâu khuyên",
    "tattoo": "Xăm", "scratch": "Cào", "scratching": "Đang cào", "spank": "Đét",
    "slap": "Tát", "push": "Đẩy", "press": "Ép", "pressing": "Ép lên", "squeeze": "Bóp",
    "squeezing": "Đang bóp", "twist": "Xoắn", "shake": "Lắc", "wave": "Vẫy",
    "waving": "Đang vẫy", "swing": "Đung đưa", "throw": "Ném", "catch": "Bắt",
    "lift": "Nhấc", "lifting": "Đang nhấc", "drop": "Thả rơi", "carry": "Mang",
    "point": "Chỉ vào", "poke": "Chọc vào", "rub": "Xoa", "tying": "Buộc", "untying": "Cởi",
    "opening": "Đang mở", "closing": "Đang đóng", "zipping": "Kéo khóa",
    "buttoning": "Cài khuy", "inflating": "Bơm to", "choking": "Nghẹt cổ",
    "struggling": "V vùng", "burning": "Đốt", "melting": "Làm tan chảy", "soaking": "Ngâm",
    "dipping": "Nhúng", "wrapping": "Quấn", "swinging": "Đung đưa", "raising": "Nâng",
    "lowering": "Hạ", "holding": "Cầm", "snatching": "Giật lấy",
}
def _fill_direction(template, label):
    """Điền nhãn vào mẫu hướng; mẫu bắt đầu bằng {} thì giữ chữ hoa vì danh từ đứng đầu cụm."""
    if template.startswith("{}"):
        return template.format(label)
    return template.format(_lower_first(label))


_ALL_DIRECTIONS = (frozenset(_TAG_VI_DIRECTION_GARMENT) | frozenset(_TAG_VI_DIRECTION_LIMB)
                   | frozenset(_TAG_VI_DIRECTION_BODY) | frozenset(_TAG_VI_DIRECTION_PLAIN))
# "wearing" cần đúng động từ tiếng Việt theo loại món đồ; để trống nghĩa là "Mặc".
_TAG_VI_WEAR_VERBS = {
    "hat": "Đội", "cap": "Đội", "helmet": "Đội", "beret": "Đội", "beanie": "Đội",
    "hood": "Đội", "crown": "Đội", "tiara": "Đội", "mask": "Đeo", "glasses": "Đeo",
    "eyewear": "Đeo", "sunglasses": "Đeo", "goggles": "Đeo", "necklace": "Đeo",
    "earring": "Đeo", "earrings": "Đeo", "bracelet": "Đeo", "anklet": "Đeo", "ring": "Đeo",
    "watch": "Đeo", "piercing": "Đeo", "choker": "Đeo", "headband": "Đeo", "veil": "Đeo",
    "scarf": "Đeo", "tie": "Thắt", "necktie": "Thắt", "bowtie": "Thắt", "belt": "Thắt",
    "shoes": "Đi", "boots": "Đi", "sandals": "Đi", "slippers": "Đi", "sneakers": "Đi",
    "heels": "Đi", "gloves": "Đeo", "mittens": "Đeo", "socks": "Mang", "stockings": "Mang",
    "thighhighs": "Mang", "kneehighs": "Mang", "headphones": "Đeo", "earphones": "Đeo",
}
_TAG_VI_PREPOSITIONS = {
    "on": "trên", "in": "trong", "under": "bên dưới", "over": "phía trên", "above": "bên trên",
    "below": "bên dưới", "through": "xuyên qua", "across": "ngang qua", "around": "quanh",
    "between": "giữa", "behind": "phía sau", "inside": "bên trong", "near": "gần",
    "with": "kèm",
    # Đợt 6 — chi tiết nhân vật:
    "of": "",
    "from": "từ",
    "to": "tới",
    "onto": "lên",
    "upon": "trên",
    "toward": "về phía",
    "towards": "về phía",
    "and": "và",
    "into": "vào trong",
    "beneath": "bên dưới",
    "underneath": "bên dưới",
    "beside": "bên cạnh",
    "beyond": "phía sau",
    "within": "bên trong",
    "along": "dọc theo",
    "atop": "trên đỉnh",
    "via": "qua",
    # Đợt 6 — vòng 2: chất liệu, phụ kiện, kiểu tóc.
    "as": "như là",
    "like": "giống",
    "minus": "không có",
    "plus": "thêm",
}
_TAG_VI_NEGATIVE_SUFFIXES = {"less": "Không có"}
# Trạng thái/mô tả đứng SAU bộ phận: "eyelids_visible" → "Mí mắt nhìn thấy được",
# "tail_raised" → "Đuôi dựng lên". Chỉ chạy khi phần đứng trước kết thúc bằng danh từ đã biết.
_TAG_VI_TRAILING_STATES = {
    "visible": "nhìn thấy được", "invisible": "vô hình", "open": "mở", "closed": "khép",
    "raised": "dựng lên", "lowered": "hạ xuống", "widened": "mở rộng", "narrowed": "thu hẹp",
    "glowing": "phát sáng", "sparkling": "lấp lánh", "shimmering": "lấp lánh",
    "wet": "ướt", "dry": "khô", "damp": "ẩm", "dirty": "bẩn", "clean": "sạch",
    "torn": "rách", "ripped": "rách toạc", "missing": "thiếu", "hidden": "bị che",
    "shown": "được cho thấy", "revealed": "lộ ra", "bare": "trần trụi",
    "bent": "gập lại", "curled": "cuộn lại", "inflated": "phồng lên",
    "deflated": "xẹp xuống", "puffed": "phồng", "swollen": "sưng",
    "twitching": "giật giật", "blinking": "nhấp nháy", "wagging": "vẫy",
    "shaking": "run lên", "trembling": "run rẩy", "flapping": "vỗ",
    "hanging": "thõng xuống", "dangling": "lủng lẳng", "drooping": "cụp xuống",
    "perked": "dựng lên", "pointing": "chĩa ra", "spread": "dạng ra",
    "splayed": "dạng rộng", "crossed": "bắt chéo", "folded": "gấp lại",
    "loose": "thõng", "tight": "bó chặt", "empty": "rỗng", "full": "đầy", "flat": "dẹt",
    # Đợt 6 — chi tiết nhân vật:
    "peek": "lấp ló",
    "stained": "dơ",
    "soaked": "ướt sũng",
    "soiled": "bẩn",
    "sagging": "xệ xuống",
    "quivering": "run lên",
    "glistening": "long lanh",
    "matted": "bết lại",
    "fluffed": "xù lên",
    "tousled": "rối",
    # Đợt 6 — nội động từ/trạng thái đứng sau danh từ.
    "melting": "đang tan chảy", "freezing": "đóng băng", "showing": "lộ ra",
    "wiggle": "lắc lư", "wiggling": "lắc lư", "exposed": "lộ ra",
    # Đợt 9 — trạng thái của thẻ bố cục và bối cảnh.
    "colorized": "được tô màu",
    "uncolored": "chưa tô màu",
    "grayscale": "tông xám",
    "monochrome": "đơn sắc",
    "vintage": "phong cách cổ",
    "abandoned": "bỏ hoang",
    "ruined": "hoang tàn",
    "flooded": "ngập nước",
    "overgrown": "cây mọc um tùm",
    "silhouetted": "thành bóng đen",
    "sharply-focused": "lấy nét sắc",
    # Đợt 10 — trạng thái tóc.
    "unkempt": "bờ phơ",
    "slicked-back": "vuốt ngược",
    "flipped": "bẻ ngược",
    "spiked": "dựng gai",
    "waved": "gợn sóng",
}
# Trạng thái ghép nhiều từ đứng TRƯỚC danh từ: "see_through_<y>" → "<y> xuyên thấu".
_TAG_VI_COMPOUND_PREFIXES = {
    "see_through": "xuyên thấu", "visible_through": "nhìn thấy qua",
    "hidden_under": "giấu dưới", "hidden_in": "giấu trong",
    # Tính từ đứng trước danh từ trong tiếng Anh nhưng tiếng Việt nói "X lơ lửng/bay".
    "floating": "lơ lửng", "levitating": "bay lơ lửng", "disembodied": "tách rời",
    "phantom": "hư ảo", "inanimate": "vô tri", "sentient": "có tri giác",
}
# Phó từ hướng đi ngay sau động từ: "looking_down_at_viewer" → "Nhìn xuống người xem".
_TAG_VI_VERB_DIRECTIONS = {
    "down": "xuống", "up": "lên", "back": "lại phía sau", "aside": "sang bên",
    "away": "đi chỗ khác", "afar": "xa xăm", "forward": "về phía trước",
    "forwards": "về phía trước", "downwards": "xuống", "upwards": "lên",
    "sideways": "sang ngang", "left": "sang trái", "right": "sang phải",
    "out": "ra ngoài", "inside": "vào trong", "outside": "ra ngoài",
    "through": "xuyên qua", "over": "qua", "straight": "thẳng", "directly": "thẳng vào",
}


def _lower_first(label):
    return (label[:1].lower() + label[1:]) if label else label


def _upper_first(label):
    return (label[:1].upper() + label[1:]) if label else label


def _vietnamese_fragment_label(fragment, category, depth=0):
    """Dịch một mảnh tên thẻ: từ điển → hậu tố phủ định → giới từ → động từ → quy tắc ghép."""
    if not fragment:
        return None
    label = TAG_VI_LABELS.get(fragment)
    if label:
        return label
    if "_" not in fragment:
        head = _vi_head(fragment)
        if head:
            return head
        color = _TAG_VI_COLORS.get(fragment)
        if not color:
            singular = _vi_singular(fragment)
            if singular:
                color = _TAG_VI_COLORS.get(singular)
        if color:
            return "màu " + color
        # "-less" được đặt trước tra từ đơn vì nghĩa phủ định chuẩn hơn nghĩa từ điển
        # ("hairless" → "Không có tóc", không phải "không lông" của tính từ hairless_*).
        for suffix, prefix_vi in _TAG_VI_NEGATIVE_SUFFIXES.items():
            if fragment.endswith(suffix) and len(fragment) > len(suffix) + 2:
                stem = fragment[: -len(suffix)]
                stem_label = _TAG_VI_COMPOSITE_HEADS.get(stem) or _TAG_VI_WORDS.get(stem)
                if stem_label:
                    return f"{prefix_vi} {_lower_first(stem_label)}"
        word = _TAG_VI_WORDS.get(fragment) or _vi_word(fragment)
        if word:
            return _upper_first(word)
        return None
    if depth >= 3:
        return None
    tokens = fragment.split("_")
    # "<món đồ>_only" → "Chỉ mặc/Có <món đồ>"
    if len(tokens) > 1 and tokens[-1] == "only":
        inner_label = _vietnamese_fragment_label("_".join(tokens[:-1]), category, depth + 1)
        if inner_label:
            if tokens[-2] in _TAG_VI_ONLY_WEAR:
                wear = _TAG_VI_WEAR_VERBS.get(tokens[-2], "mặc").lower()
                inner = _lower_first(inner_label)
                # Bớt lặp: "Thắt lưng" đã chứa động từ "thắt".
                return f"Chỉ {inner}" if inner.startswith(wear + " ") else f"Chỉ {wear} {inner}"
            return f"Chỉ có {_lower_first(inner_label)}"
    # "<bộ phận|món đồ>_<hướng>" → "Kéo váy xuống", "Tai cụp xuống", ...
    if len(tokens) > 1 and tokens[-1] in _ALL_DIRECTIONS:
        stem_label = _vietnamese_fragment_label("_".join(tokens[:-1]), category, depth + 1)
        if stem_label:
            direction = tokens[-1]
            if direction == "on" and tokens[-2] in _TAG_VI_GARMENT_TOKENS:
                wear = _TAG_VI_WEAR_VERBS.get(tokens[-2], "mặc").lower()
                return f"Đang {wear} {_lower_first(stem_label)}"
            for tokenset, table in (
                (_TAG_VI_GARMENT_TOKENS, _TAG_VI_DIRECTION_GARMENT),
                (_TAG_VI_LIMB_TOKENS, _TAG_VI_DIRECTION_LIMB),
                (_TAG_VI_BODY_TOKENS, _TAG_VI_DIRECTION_BODY),
            ):
                if tokens[-2] in tokenset and direction in table:
                    return _fill_direction(table[direction], stem_label)
            if direction in _TAG_VI_DIRECTION_PLAIN:
                return _fill_direction(_TAG_VI_DIRECTION_PLAIN[direction], stem_label)
    # "<danh từ>_<trạng thái>" (tính từ/phân từ đứng sau bộ phận hoặc món đồ)
    if len(tokens) > 1 and tokens[-1] in _TAG_VI_TRAILING_STATES:
        stem = "_".join(tokens[:-1])
        stem_label = _vietnamese_fragment_label(stem, category, depth + 1)
        if stem_label and (tokens[-2] in _TAG_VI_COMPOSITE_HEADS or tokens[-2] in _TAG_VI_WORDS):
            return f"{stem_label} {_TAG_VI_TRAILING_STATES[tokens[-1]]}"
    # "<danh từ>_<phân từ>" (trạng thái đứng sau): sleeves_rolled → "Tay áo cuộn lên"
    if (len(tokens) > 1 and tokens[-1] in _TAG_VI_PAST_STATES
            and tokens[0] not in _TAG_VI_PAST_STATES):
        stem_label = _vietnamese_fragment_label("_".join(tokens[:-1]), category, depth + 1)
        if stem_label:
            return f"{stem_label} {_TAG_VI_PAST_STATES[tokens[-1]]}"
    for index, token in enumerate(tokens):
        if index == 0 or index == len(tokens) - 1:
            continue
        relation = _TAG_VI_TRANSITIVE_RELATIONS.get(token)
        if relation is not None:
            who = _vietnamese_fragment_label("_".join(tokens[:index]), category, depth + 1)
            whom = _vietnamese_fragment_label("_".join(tokens[index + 1:]), category, depth + 1)
            if who and whom and len(tokens) <= 4:
                return f"{who} {relation} {_lower_first(whom)}"
        preposition = _TAG_VI_PREPOSITIONS.get(token)
        if preposition is None:
            continue
        left = "_".join(tokens[:index])
        right = "_".join(tokens[index + 1:])
        left_label = _vietnamese_fragment_label(left, category, depth + 1)
        right_label = _vietnamese_fragment_label(right, category, depth + 1)
        if left_label and right_label:
            # "shirt_tucked_into_underwear": vế trước đã kết thúc bằng "vào trong" thì
            # không nhắc lại liên từ, nếu không sẽ thành "giắt vào trong vào trong đồ lót".
            if left_label.endswith(" " + preposition):
                return f"{left_label} {_lower_first(right_label)}"
            parts = [left_label, preposition, _lower_first(right_label)]
            return " ".join(part for part in parts if part)
    if tokens[0] in _TAG_VI_PAST_STATES:
        state = _TAG_VI_PAST_STATES[tokens[0]]
        if tokens[0] == "tied" and len(tokens) > 2 and tokens[1] == "to":
            bound = _vietnamese_fragment_label("_".join(tokens[2:]), category, depth + 1)
            if bound:
                return f"Trói vào {_lower_first(bound)}"
        rest = _vietnamese_fragment_label("_".join(tokens[1:]), category, depth + 1)
        if rest:
            return f"{rest} {state}"
    for prefix_key, tail_vi in _TAG_VI_COMPOUND_PREFIXES.items():
        prefix_tokens = prefix_key.split("_")
        if len(tokens) > len(prefix_tokens) and tokens[:len(prefix_tokens)] == prefix_tokens:
            rest_label = _vietnamese_fragment_label("_".join(tokens[len(prefix_tokens):]), category, depth + 1)
            if rest_label:
                return f"{rest_label} {tail_vi}"
    entry = _TAG_VI_ACTION_PREFIXES.get(tokens[0])
    if entry is not None:
        verb, _, links = entry.partition("|")
        droppable = set(links.split(",")) if links else set()
        index = 1
        adverb = ""
        while index < len(tokens) - 1:
            token = tokens[index]
            if token in droppable:
                index += 1
                continue
            if not adverb and token in _TAG_VI_VERB_DIRECTIONS:
                adverb = _TAG_VI_VERB_DIRECTIONS[token]
                index += 1
                continue
            break
        rest = "_".join(tokens[index:])
        rest_label = _vietnamese_fragment_label(rest, category, depth + 1) if rest else None
        if rest_label:
            if not verb:
                verb = _TAG_VI_WEAR_VERBS.get(tokens[-1], "Mặc")
            label = _lower_first(rest_label)
            # Bớt lặp: "Thắt lưng"/"Đeo kính mắt" đã mang nghĩa động từ.
            if not adverb and verb and label.startswith(verb.lower() + " "):
                return _upper_first(label)
            return " ".join(part for part in (verb, adverb, label) if part)
    # Nội động từ dẫn đầu ("melting_tail"): tiếng Việt nói "Đuôi đang tan chảy".
    if tokens[0] in _TAG_VI_LEADING_STATES:
        rest = _vietnamese_fragment_label("_".join(tokens[1:]), category, depth + 1)
        if rest:
            return f"{rest} {_TAG_VI_LEADING_STATES[tokens[0]]}"
    # "<món đồ|bộ phận>_<động từ>" (tiếng Anh đảo động từ ra sau): "dress_pull" → "Đang kéo váy"
    if (len(tokens) > 1 and tokens[-1] in _TAG_VI_TRAILING_ACTIONS
            and tokens[-2] in _TAG_VI_ACTION_TARGETS):
        stem = "_".join(tokens[:-1])
        stem_label = _vietnamese_fragment_label(stem, category, depth + 1)
        if stem_label:
            return f"{_TAG_VI_TRAILING_ACTIONS[tokens[-1]]} {_lower_first(stem_label)}"
    return _compose_vietnamese_tag_label(fragment)



def vietnamese_tag_label(name, category):
    """Return a Vietnamese label or a transparent category/translation fallback."""
    fallback = _TAG_VI_CATEGORY_FALLBACKS.get(category, TAG_VI_TRANSLATION_FALLBACK)
    # Artist, work, character, uploader and lore names are proper names: keep them.
    if category in _TAG_VI_NAME_CATEGORIES:
        return fallback
    translated = TAG_VI_LABELS.get(name)
    if translated:
        return translated
    if re.fullmatch(r"(?:19|20)\d{2}", name):
        return f"Năm {name}"
    if re.fullmatch(r"\d{1,2}:\d{1,2}", name):
        return f"Tỷ lệ {name}"
    count_match = re.fullmatch(r"(\d+)(girls?|boys?|people)", name)
    if count_match:
        number, group = count_match.groups()
        people = "nhân vật nữ" if group.startswith("girl") else "nhân vật nam" if group.startswith("boy") else "người"
        return f"{number} {people}"
    # "t-shirt" có key riêng, nên gạch nối chỉ được đổi thành gạch dưới ở bước dự phòng:
    # thử nguyên bản → chuẩn hoá gạch nối → bỏ "(giải nghĩa)" → cả hai.
    candidates = [name]
    if "-" in name:
        candidates.append(name.replace("-", "_"))
    base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip("_- ")
    if base and base != name:
        candidates.append(base)
        if "-" in base:
            candidates.append(base.replace("-", "_"))
    for candidate in dict.fromkeys(candidates):
        # Ưu tiên 1: nhãn đã curate (kể cả khi chỉ khớp sau khi chuẩn hoá gạch nối).
        curated = TAG_VI_LABELS.get(candidate)
        if curated:
            return curated
    for candidate in dict.fromkeys(candidates):
        translated = _tidy_vietnamese_label(
            _compose_vietnamese_tag_label(candidate) or _vietnamese_fragment_label(candidate, category)
        )
        if translated:
            # 43 nhãn màu ("màu vàng kim") mở đầu bằng chữ thường; trong Kho thẻ và
            # autocomplete nhãn đứng một mình nên viết hoa chữ cái đầu cho nhất quán.
            return translated[:1].upper() + translated[1:]
        framed = _tag_vi_uniform_frame(candidate)
        if framed:
            return framed
    return fallback


def _is_translated_tag_label(label, category):
    return bool(label and label != _TAG_VI_CATEGORY_FALLBACKS.get(category, TAG_VI_TRANSLATION_FALLBACK))


def _prefer_vietnamese_label(supplied_label, name, category):
    """Chọn nhãn Việt mới nhất, ưu tiên từ điển trong mã hơn cột chú giải của CSV.

    CSV được ghim ở một commit nên cột thứ năm đóng băng tại thời điểm tạo tệp. Khi
    từ điển trong file này được bổ sung, thẻ mà CSV vẫn ghi "Chưa có bản dịch" phải
    lấy theo mã — nếu không, Studio và file dịch sẽ tụt hậu so với từ điển ngay cả
    khi bản dịch đã có. Chiều ngược lại cũng đúng: nhãn trong CSV còn được giữ khi
    mã không sinh ra bản dịch nào.
    """
    label = vietnamese_tag_label(name, category)
    if _is_translated_tag_label(label, category):
        return label
    return supplied_label or label


def csv_tag_caption(name, category, vietnamese_label=None):
    """Display a Vietnamese gloss beside the canonical English tag."""
    label = vietnamese_label or vietnamese_tag_label(name, category)
    return f"{label} — {name}"


_TAG_ROWS = None
_TAG_LOCK = threading.Lock()
_TAG_LABEL_WORD_INDEX_ROWS = None
_TAG_LABEL_WORD_INDEX = None
# Tên thẻ đã chuẩn hoá, dựng một lần cho mỗi catalog. Tìm kiểu *đuôi*/*giữa* phải đối
# chiếu khoảng 348k tên; gọi normalize_csv_tag trong vòng lặp mỗi lần gõ là phần tốn nhất.
_TAG_SEARCH_NAMES = None
_TAG_LABEL_INDEX_LOCK = threading.Lock()
# Các lượt tìm trong tab Kho thẻ có thể quét toàn bộ catalog, nhưng không được dùng
# chung slot với một lượt inference GPU kéo dài vài phút.
TAG_CATALOG_CONCURRENCY_ID = "wai_tag_catalog"
TAG_CATALOG_CONCURRENCY_LIMIT = 1
TAG_PAGE_SIZE = 60
# Search-only wording helps natural Vietnamese prompt fragments find canonical tags.
# These synonyms are never displayed or inserted into the model prompt.
TAG_VI_SEARCH_SYNONYMS = {
    "1girl": "girl woman female cô gái con gái nữ nhân vật nữ",
    "2girls": "girls hai cô gái nhiều cô gái",
    "3girls": "ba cô gái nhiều cô gái",
    "4girls": "bốn cô gái nhiều cô gái",
    "multiple_girls": "nhiều cô gái nhiều nhân vật nữ",
    "1boy": "boy man male cậu bé chàng trai nam nhân vật nam",
    "2boys": "hai cậu bé nhiều chàng trai",
    "3boys": "ba cậu bé nhiều chàng trai",
    "4boys": "bốn cậu bé nhiều chàng trai",
    "multiple_boys": "nhiều cậu bé nhiều nhân vật nam",
    "female": "female woman women phụ nữ nữ giới",
    "male": "male man men đàn ông nam giới",
    "long_hair": "mái tóc dài",
    "short_hair": "mái tóc ngắn",
    "black_hair": "mái tóc đen",
    "pink_hair": "mái tóc hồng",
    "blue_hair": "mái tóc xanh dương",
    "looking_at_viewer": "nhìn vào người xem",
}
_TAG_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "of", "on", "or", "the", "to", "with", "without", "wearing",
    "mot", "co", "nhan", "vat", "nguoi", "va", "voi", "cua", "dang", "duoc", "la",
    "nhieu", "nhung", "cac", "cho", "tai", "trong", "tren", "duoi", "phia", "mau",
})
_TAG_QUERY_SINGLETONS = frozenset({
    "girl", "girls", "boy", "boys", "woman", "women", "man", "men",
    "female", "male", "gai", "trai", "nu", "nam",
})


def normalize_csv_tag(value):
    """Normalize English/Vietnamese spelling, accents, hyphens and underscores."""
    value = unicodedata.normalize("NFKD", str(value or "").casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[\s_-]+", " ", value).strip()


_VI_TEXT_CACHE = {}
VI_TEXT_CACHE_LIMIT = 600_000


def _normalize_tag_text_preserving_accents(value):
    """Normalize separators/case but keep Vietnamese marks to disambiguate captions.

    Có memo: hàm này chạy trên khoảng 348k nhãn khi dựng index từ khóa và chạy lại cho từng
    ứng viên mỗi lần tìm. Nhãn trong catalog lặp lại rất nhiều (cùng một cụm cho hàng
    nghìn thẻ) nên bộ nhớ chỉ vài MB mà tiết kiệm phần lớn thời gian.
    """
    text = value if isinstance(value, str) else None
    if text is not None:
        cached = _VI_TEXT_CACHE.get(text)
        if cached is not None:
            return cached
    normalized = unicodedata.normalize("NFC", str(value or "").casefold())
    normalized = re.sub(r"[\s_-]+", " ", normalized).strip()
    if text is not None:
        if len(_VI_TEXT_CACHE) >= VI_TEXT_CACHE_LIMIT:
            _VI_TEXT_CACHE.clear()
        _VI_TEXT_CACHE[text] = normalized
    return normalized


def _tag_query_pattern(query, allow_partial_last=False, preserve_accents=False):
    """Match whole words/phrases, optionally treating the last typed word as a prefix."""
    normalize = (
        _normalize_tag_text_preserving_accents if preserve_accents else normalize_csv_tag
    )
    words = normalize(query).split()
    if not words:
        return None
    escaped = [re.escape(word) for word in words]
    if allow_partial_last:
        escaped[-1] += r"\w*"
    return re.compile(r"(?<!\w)" + r"\s+".join(escaped) + r"(?!\w)")


def _tag_field_pattern(query):
    """Match an exact comma-delimited name/alias in the normalized search index."""
    return re.compile(r"(?:^|,)\s*" + re.escape(query) + r"\s*(?:,|$)")


def _build_tag_label_word_index(rows):
    r"""from → danh sách thẻ, lập chỉ mục trên NHÃN TIẾNG VIỆT của catalog.

    Tách từ bằng ``\w+`` (không phải split theo khoảng trắng) vì nhãn có dấu chấm và
    dấu phẩy — ví dụ "U.A." phải cho hai từ "u" và "a" như cách ``_tag_query_pattern``
    ghép regex ``(?<!\w)u\s+a(?!\w)``.
    """
    word_index = {}
    for row in rows:
        label = _normalize_tag_text_preserving_accents(row[5])
        for word in set(re.findall(r"\w+", label)):
            word_index.setdefault(word, []).append(row)
    return word_index


def tag_label_word_index(rows):
    """Index nhãn Việt đã dựng sẵn cho catalog đang nạp; dựng một lần nếu chưa có.

    Trả ``None`` khi ``rows`` không phải catalog chung — khi đó nơi tìm kiếm quét trực
    tiếp, giữ nguyên hành vi cho các bộ dữ liệu nhỏ (test, catalog tự nạp lại).
    """
    if rows is not _TAG_ROWS:
        return None
    global _TAG_LABEL_WORD_INDEX_ROWS, _TAG_LABEL_WORD_INDEX
    with _TAG_LABEL_INDEX_LOCK:
        if _TAG_LABEL_WORD_INDEX_ROWS is not rows or _TAG_LABEL_WORD_INDEX is None:
            _TAG_LABEL_WORD_INDEX = _build_tag_label_word_index(rows)
            _TAG_LABEL_WORD_INDEX_ROWS = rows
    return _TAG_LABEL_WORD_INDEX


def tag_normalized_names(rows=None):
    """Danh sách tên thẻ đã chuẩn hoá, trùng thứ tự với ``rows`` — cache theo catalog."""
    global _TAG_SEARCH_NAMES
    rows = load_csv_tags() if rows is None else rows
    cached = _TAG_SEARCH_NAMES
    if cached is None or cached[0] is not rows:
        cached = (rows, tuple(normalize_csv_tag(row[0]) for row in rows))
        _TAG_SEARCH_NAMES = cached
    return cached[1]


def prime_tag_label_index(rows=None):
    """Dựng index NGAY LÚC KHỞI ĐỘNG.

    Trước đây index được dựng bên trong `_caption_search_matches`, tức là cú gõ tiếng
    Việt đầu tiên phải trả ~1 s CPU (đo trên máy test; Colab chậm hơn và còn đang tải
    model) — chính là khoảng "đơ" mà người dùng gặp sau khi thêm tra nhãn tiếng Việt.
    Dựng ở ô 7/8 thì người dùng thấy dòng tiến trình, còn sự kiện UI chỉ việc tra.
    """
    try:
        # Nhận sẵn rows từ nơi gọi: khởi động chỉ được phép nạp catalog MỘT lần,
        # nếu gọi lại load_csv_tags() thì một lỗi mạng nhỏ cũng thành hai lượt thử.
        rows = load_csv_tags() if rows is None else rows
        return len(tag_label_word_index(rows) or {})
    except Exception:
        return 0


def _caption_search_matches(rows, raw_query, pattern):
    """Use a cached caption-word index for the shared catalog; preserve query accents."""
    query_words = re.findall(r"\w+", _normalize_tag_text_preserving_accents(raw_query))
    candidates = rows
    if query_words:
        index = tag_label_word_index(rows)
        if index is not None:
            candidates = index.get(query_words[0], ())
    return [
        row for row in candidates
        if pattern.search(_normalize_tag_text_preserving_accents(row[5]))
    ]


def parse_tag_csv(text):
    import csv
    import io

    patterns = [(name, re.compile(pattern)) for name, pattern in TAG_THEMES.items()]
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) not in (4, 5):
            continue
        name, category, count, aliases = fields[:4]
        if not name or category not in TAG_CATEGORIES or not count.isdigit():
            continue
        supplied_label = fields[4].strip() if len(fields) == 5 else ""
        label = _prefer_vietnamese_label(supplied_label, name, category)
        search_label = label if _is_translated_tag_label(label, category) else ""
        themes = tuple(label for label, pattern in patterns if pattern.search(name))
        search_synonyms = TAG_VI_SEARCH_SYNONYMS.get(name, "")
        search_index = normalize_csv_tag(",".join((name, aliases, search_label, search_synonyms)))
        rows.append((name, category, int(count), search_index, themes, label))
    if not rows:
        raise ValueError("CSV không chứa thẻ hợp lệ.")
    return tuple(rows)


def load_csv_tags():
    """Shared read-only catalog, not per-session prompt state. Failed loads can retry."""
    import hashlib
    import urllib.request

    global _TAG_ROWS
    with _TAG_LOCK:
        if _TAG_ROWS is not None:
            return _TAG_ROWS
        accepted_hashes = {TAG_CSV_SHA256, TAG_CSV_LEGACY_SHA256}
        candidates = [Path(TAG_CSV_NAME), Path("/content") / TAG_CSV_NAME]
        data = None
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size <= TAG_CSV_MAX_BYTES:
                content = candidate.read_bytes()
                if hashlib.sha256(content).hexdigest() in accepted_hashes:
                    data = content
                    break
        if data is None:
            with urllib.request.urlopen(TAG_CSV_URL, timeout=45) as response:
                data = response.read(TAG_CSV_MAX_BYTES + 1)
            if len(data) > TAG_CSV_MAX_BYTES or hashlib.sha256(data).hexdigest() not in accepted_hashes:
                raise ValueError("CSV tải về không khớp SHA-256 đã xác minh.")
        rows = parse_tag_csv(data.decode("utf-8-sig"))
        if Path("/content").is_dir():
            (Path("/content") / TAG_CSV_NAME).write_bytes(data)
        _TAG_ROWS = rows
        return rows


def cached_tag_catalog():
    """Return the already verified in-memory catalog for read-only UI searches.

    ``build_app`` calls :func:`prime_prompt_tag_catalog` before registering the UI,
    so normal Kho thẻ searches take this fast path and do not re-open, hash, or parse
    the CSV.  The fallback is intentional: if startup was offline, the first search
    may make the one allowed retry; ``load_csv_tags`` still serializes and caches it.
    """
    rows = _TAG_ROWS
    return rows if rows is not None else load_csv_tags()


def search_csv_tags(
    rows, query="", category="", theme="", sort="Phổ biến nhất", page=1,
    allow_partial_last=False,
):
    # Mỗi cụm cách nhau bằng dấu phẩy/chấm phẩy/xuống dòng được tìm độc lập.
    raw_queries = [
        part.strip() for part in re.split(r"[,;\n]+", str(query or ""))
        if normalize_csv_tag(part)
    ]
    query_specs = []
    for query_id, raw_query in enumerate(raw_queries):
        normalized = normalize_csv_tag(raw_query)
        query_specs.append({
            "id": query_id,
            "raw": raw_query,
            "normalized": normalized,
            "exact_field": _tag_field_pattern(normalized),
            "phrase": _tag_query_pattern(normalized),
            "partial": (
                _tag_query_pattern(normalized, allow_partial_last=True)
                if allow_partial_last else None
            ),
            "caption": (
                _tag_query_pattern(raw_query, preserve_accents=True)
                if (
                    _normalize_tag_text_preserving_accents(raw_query) != normalized
                    and len(normalized.split()) <= 3
                )
                else None
            ),
        })

    def passes_filters(row):
        return (
            (not category or row[1] == category)
            and (not theme or (not row[4] if theme == "Chưa phân nhóm" else theme in row[4]))
        )

    # Khi cụm ngắn có dấu, ưu tiên nhãn Việt khớp chính tả; tránh va chạm
    # sau khi bỏ dấu như “đỏ”/“độ”. Những cụm câu dài dùng nhánh tìm kiếm thường.
    caption_matches = {}
    for spec in query_specs:
        if spec["caption"] is None:
            continue
        matches = {
            (row[0], row[1])
            for row in _caption_search_matches(rows, spec["raw"], spec["caption"])
            if passes_filters(row)
        }
        if matches:
            caption_matches[spec["id"]] = matches

    found_by_key = {}
    match_scores = {}
    matched_queries = set()
    for row in rows:
        if not passes_filters(row):
            continue
        key = (row[0], row[1])
        if not query_specs:
            found_by_key[key] = row
            continue
        score = 0
        for spec in query_specs:
            raw_caption_hits = caption_matches.get(spec["id"])
            if raw_caption_hits is not None:
                query_score = 4 if key in raw_caption_hits else 0
            elif spec["exact_field"].search(row[3]):
                query_score = 3
            elif spec["phrase"].search(row[3]):
                query_score = 2
            elif spec["partial"] and spec["partial"].search(row[3]):
                query_score = 1
            else:
                query_score = 0
            if query_score:
                matched_queries.add(spec["id"])
                score = max(score, query_score)
        if score:
            found_by_key[key] = row
            match_scores[key] = score

    # Nếu cả cụm tự nhiên không khớp nguyên văn, tìm các cụm con có nghĩa riêng.
    # Ví dụ “một cô gái tóc dài” khớp “1girl” và “long_hair”.
    missing_queries = [
        spec["normalized"] for spec in query_specs if spec["id"] not in matched_queries
    ]
    fallback_terms = []
    for needle in missing_queries:
        phrase_words = [
            term for term in needle.split()
            if term not in _TAG_QUERY_STOPWORDS or term == "co"
        ]
        for width in (3, 2):
            fallback_terms.extend(
                " ".join(phrase_words[index:index + width])
                for index in range(len(phrase_words) - width + 1)
            )
        fallback_terms.extend(
            term for term in phrase_words if term in _TAG_QUERY_SINGLETONS
        )
    fallback_patterns = [
        pattern for term in dict.fromkeys(fallback_terms)
        if (pattern := _tag_query_pattern(term)) is not None
    ]
    if fallback_patterns:
        for row in rows:
            key = (row[0], row[1])
            if key in found_by_key or not passes_filters(row):
                continue
            if any(pattern.search(row[3]) for pattern in fallback_patterns):
                found_by_key[key] = row
                match_scores[key] = 0.5

    found = list(found_by_key.values())
    if sort == "Khớp nhất":
        # Exact catalog entries outrank words that merely occur within another
        # tag or alias. For a very short exact query, hide weaker prefix hits.
        best_score = max(match_scores.values(), default=0)
        short_query = max((len(spec["normalized"]) for spec in query_specs), default=0) <= 3
        if best_score >= 3 and short_query:
            found = [row for row in found if match_scores.get((row[0], row[1])) == best_score]
        found.sort(key=lambda row: (
            -match_scores.get((row[0], row[1]), 0), -row[2], row[0]
        ))
    elif sort == "Tên A–Z":
        found.sort(key=lambda row: row[0])
    else:
        found.sort(key=lambda row: (-row[2], row[0]))
    pages = max(1, math.ceil(len(found) / TAG_PAGE_SIZE))
    page = min(pages, max(1, int(page or 1)))
    return found[(page - 1) * TAG_PAGE_SIZE:page * TAG_PAGE_SIZE], len(found), page, pages


def browse_csv_tags(query, category, theme, sort, page):
    import gradio as gr

    try:
        rows = cached_tag_catalog()
        found, total, current, pages = search_csv_tags(rows, query, category, theme, sort, page)
        choices = [(f"{csv_tag_caption(row[0], row[1], row[5])} · {TAG_CATEGORIES[row[1]]} · {row[2]:,} lượt", row[0]) for row in found]
        return gr.update(choices=choices, value=[]), current, (
            f"**{total:,} thẻ phù hợp / {len(rows):,} thẻ** · Trang {current}/{pages}. "
            "Chọn thẻ rồi nhấn Thêm. Đổi bộ lọc và nhấn Tìm để cập nhật; lựa chọn cũ sẽ được xóa."
        )
    except Exception as exc:
        return gr.update(choices=[], value=[]), 1, (
            f"Không nạp được kho thẻ ({type(exc).__name__}). Nhấn Tìm để thử lại, "
            f"hoặc tải đúng file `{TAG_CSV_NAME}` lên thư mục `/content` bằng bảng Files của Colab. "
            "Bạn vẫn có thể viết prompt và tạo ảnh bình thường."
        )


_CSV_TAG_NAMES = None
_CSV_TAG_ROWS = None


def csv_tag_names(rows=None):
    """Tập tên thẻ hợp lệ của catalog — dựng một lần cho mỗi bộ dữ liệu đang nạp.

    Dựng lại set 349k phần tử mỗi lần bấm tốn ~90 ms CPU; khi đang tạo ảnh, lượng CPU
    đó phải chờ GIL nên tính bằng giây. Khóa theo chính đối tượng rows (``is``) nên
    một catalog khác — ví dụ bản vá trong test — không bao giờ dùng nhầm set cũ.
    """
    global _CSV_TAG_NAMES, _CSV_TAG_ROWS
    rows = load_csv_tags() if rows is None else rows
    if _CSV_TAG_ROWS is not rows:
        _CSV_TAG_NAMES = frozenset(row[0] for row in rows)
        _CSV_TAG_ROWS = rows
    return _CSV_TAG_NAMES


def _canonical_selected_csv_tags(selected, valid):
    """Lấy tên canonical từ mọi dạng payload mà Dropdown Gradio có thể gửi."""
    if selected is None:
        return []
    if isinstance(selected, (str, int, float)):
        selected = [selected]
    elif isinstance(selected, dict):
        selected = [selected]
    names = []
    for item in selected:
        if isinstance(item, dict):
            item = item.get("value", item.get("name", item.get("label")))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            item = item[1]
        if isinstance(item, str) and item in valid and item not in names:
            names.append(item)
    return names


def apply_csv_tags(positive, negative, selected, destination):
    if not selected:
        return positive, negative, "Hãy tìm và chọn ít nhất một thẻ."
    # Only canonical names in the verified catalog may be inserted.
    valid = csv_tag_names()
    selected = _canonical_selected_csv_tags(selected, valid)
    current = negative if destination == "Negative prompt" else positive
    existing = {normalize_csv_tag(tag) for tag in split_tags(current or "")}
    additions = [name for name in selected if normalize_csv_tag(name) not in existing]
    result = _add_prompt_tags(current or "", additions)
    if destination == "Negative prompt":
        negative = result
    else:
        positive = result
    return positive, negative, f"Đã thêm {len(additions)} thẻ vào {destination}; bỏ qua thẻ trùng. Bạn có thể sửa/xóa trực tiếp trong ô prompt."


# ---------------------------------------------------------------------------
# DỊCH PROMPT TIẾNG VIỆT → TIẾNG ANH (offline, ưu tiên tag canonical trong catalog).
#
# Không gửi prompt ra dịch vụ ngoài. Bộ dịch dùng nhãn Việt đã xác minh trong catalog,
# từ điển Studio và một số cụm thông dụng; phần không nhận diện được được giữ nguyên để
# người dùng còn nhìn thấy và sửa trước khi gửi model.
# ---------------------------------------------------------------------------
_VI_PROMPT_TRANSLATION_CACHE = None  # (rows, exact accent, exact không dấu, phrase indexes)
_VI_PROMPT_TRANSLATION_LOCK = threading.Lock()

# Cụm thường xuất hiện khi người dùng mô tả prompt bằng tiếng Việt nhưng không phải
# lúc nào cũng có đúng một nhãn trong CSV. Giá trị là tag canonical hoặc cụm English
# hợp lệ; không phải bản dịch ẩn — nút dịch ghi kết quả thẳng vào ô prompt.
_VI_PROMPT_PHRASES = {
    "một cô gái": "1girl",
    "cô gái": "1girl",
    "nhiều cô gái": "multiple girls",
    "một chàng trai": "1boy",
    "chàng trai": "1boy",
    "nhiều chàng trai": "multiple boys",
    "một nhân vật nữ": "1girl",
    "một nhân vật nam": "1boy",
    "tóc dài": "long_hair",
    "mái tóc dài": "long_hair",
    "tóc ngắn": "short_hair",
    "mắt xanh": "blue_eyes",
    "mắt xanh dương": "blue_eyes",
    "mắt xanh lá": "green_eyes",
    "mắt đỏ": "red_eyes",
    "váy đỏ": "red_dress",
    "váy trắng": "white_dress",
    "áo trắng": "white_shirt",
    "hoa anh đào": "cherry_blossoms",
    "mỉm cười": "smile",
    "nụ cười dịu dàng": "gentle smile",
    "đang đứng": "standing",
    "đứng": "standing",
    "đang ngồi": "sitting",
    "ngồi": "sitting",
    "ánh sáng mềm": "soft lighting",
    "ánh sáng dịu": "soft lighting",
    "ánh sáng ấm": "warm lighting",
    "bối cảnh": "background",
    "phông nền": "background",
    "trang phục": "clothing",
    "phong cách": "style",
    "dưới": "under",
    "trên": "on",
    "trong": "in",
    "với": "with",
    "và": "and",
    "không có": "no",
    "mắt": "eyes",
    "tóc": "hair",
    "dài": "long",
    "ngắn": "short",
    "đỏ": "red",
    "xanh dương": "blue",
    "xanh lá": "green",
    "trắng": "white",
    "đen": "black",
}
_VI_PROMPT_IGNORED_WORDS = frozenset({
    "một", "một cô", "đang", "có", "là", "ở", "cho", "của", "này", "nọ",
})


def _translation_key(value, preserve_accents=True):
    if preserve_accents:
        return _normalize_tag_text_preserving_accents(value)
    # `unicodedata` không tách chữ đ; coi đ/d tương đương khi người dùng gõ không dấu.
    return normalize_csv_tag(value).replace("đ", "d")


def _choose_translation_candidate(candidates, allow_ambiguous=False):
    """Choose one canonical tag, preferring popularity but rejecting collisions by default."""
    if not candidates:
        return None
    names = {name for _, name in candidates}
    if len(names) > 1 and not allow_ambiguous:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _build_prompt_translation_index(rows):
    exact_accent = {}
    exact_plain = {}
    phrases_accent = {}
    phrases_plain = {}
    valid_names = {row[0] for row in rows}

    def add(table, key, name, count):
        if key:
            table.setdefault(key, []).append((int(count or 0), name))

    for name, category, count, _search_index, _themes, label in rows:
        if not _is_translated_tag_label(label, category):
            continue
        accent = _translation_key(label)
        plain = _translation_key(label, preserve_accents=False)
        add(exact_accent, accent, name, count)
        add(exact_plain, plain, name, count)
        if len(accent.split()) >= 2:
            add(phrases_accent, accent, name, count)
        if len(plain.split()) >= 2:
            add(phrases_plain, plain, name, count)

    def finalize(table, ambiguous=False):
        return {
            key: _choose_translation_candidate(value, allow_ambiguous=ambiguous)
            for key, value in table.items()
            if _choose_translation_candidate(value, allow_ambiguous=ambiguous)
        }

    exact_accent = finalize(exact_accent, ambiguous=True)
    # Không dấu dễ va chạm (`cô gái`/`có gai`), chỉ dùng khi một tên canonical duy nhất.
    exact_plain = finalize(exact_plain, ambiguous=False)
    phrases_accent = finalize(phrases_accent, ambiguous=True)
    phrases_plain = finalize(phrases_plain, ambiguous=False)

    manual_accent = {}
    manual_plain = {}
    for phrase, english in _VI_PROMPT_PHRASES.items():
        if english in valid_names or " " in english or "_" not in english:
            manual_accent[_translation_key(phrase)] = english
            manual_plain[_translation_key(phrase, preserve_accents=False)] = english
    # Manual phrase thắng nhãn CSV xung đột, ví dụ `cô gái` không bị hiểu thành `có gai`.
    phrases_accent.update({key: value for key, value in manual_accent.items() if len(key.split()) >= 2})
    phrases_plain.update({key: value for key, value in manual_plain.items() if len(key.split()) >= 2})
    exact_accent.update(manual_accent)
    exact_plain.update(manual_plain)

    def first_word_index(table):
        index = {}
        for phrase, name in table.items():
            words = phrase.split()
            if len(words) >= 2 and name:
                index.setdefault(words[0], []).append((tuple(words), name))
        for values in index.values():
            values.sort(key=lambda item: -len(item[0]))
        return index

    valid_keys = {_translation_key(name, preserve_accents=False) for name in valid_names}
    return (
        exact_accent,
        exact_plain,
        first_word_index(phrases_accent),
        first_word_index(phrases_plain),
        valid_keys,
    )


def _prompt_translation_index(rows):
    global _VI_PROMPT_TRANSLATION_CACHE
    cached = _VI_PROMPT_TRANSLATION_CACHE
    if cached is not None and cached[0] is rows:
        return cached[1]
    with _VI_PROMPT_TRANSLATION_LOCK:
        cached = _VI_PROMPT_TRANSLATION_CACHE
        if cached is None or cached[0] is not rows:
            cached = (rows, _build_prompt_translation_index(rows))
            _VI_PROMPT_TRANSLATION_CACHE = cached
        return cached[1]


def _translated_weight_wrapper(original, translated):
    """Keep a simple weight/bracket wrapper around the translated phrase."""
    text = str(original or "").strip()
    match = WEIGHT_RE.fullmatch(text)
    if match:
        return f"({translated}:{match.group('weight')})"
    if len(text) >= 2 and text[0] in "([" and text[-1] == {")": "]"}.get(text[0]):
        return f"{text[0]}{translated}{text[-1]}"
    return translated


def _find_translation_phrases(words, phrase_index):
    matches = []
    for start, word in enumerate(words):
        for phrase_words, value in phrase_index.get(word, ()):
            end = start + len(phrase_words)
            if tuple(words[start:end]) == phrase_words:
                matches.append((start, end, value))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen = []
    occupied = set()
    for start, end, value in matches:
        if any(index in occupied for index in range(start, end)):
            continue
        chosen.append((start, end, value))
        occupied.update(range(start, end))
    return chosen


def _prompt_word_fallback(word):
    key = _translation_key(word)
    if not key or key in _VI_PROMPT_IGNORED_WORDS:
        return ""
    direct = _VI_PROMPT_PHRASES.get(key)
    if direct:
        return direct
    # Dùng từ điển nhãn đã có trong Studio, chỉ khi bản dịch ngắn và không mơ hồ.
    candidates = []
    for english, vietnamese in _TAG_VI_WORDS.items():
        if _translation_key(vietnamese) == key:
            candidates.append(english.replace("_", " "))
    if candidates:
        return sorted(set(candidates), key=lambda value: (len(value), value))[0]
    # Từ English đang có trong prompt được giữ nguyên; từ Việt chưa biết cũng giữ
    # nguyên để người dùng không mất nội dung khi bản dịch offline không đủ dữ liệu.
    if re.fullmatch(r"[A-Za-z0-9_.'-]+", str(word)):
        return str(word)
    return str(word)


def _translate_prompt_segment(segment, index):
    original = str(segment or "").strip()
    if not original:
        return "", False
    core = tag_core(original)
    accent = _translation_key(core)
    plain = _translation_key(core, preserve_accents=False)
    exact_accent, exact_plain, phrases_accent, phrases_plain, valid_keys = index
    if plain in valid_keys:
        return original, False
    translated = exact_accent.get(accent) or exact_plain.get(plain)
    if translated:
        return _translated_weight_wrapper(original, translated), translated != core

    words = accent.split()
    matches = _find_translation_phrases(words, phrases_accent)
    if not matches:
        matches = _find_translation_phrases(words, phrases_plain)
    if not matches:
        # Fallback cho đoạn mô tả tự do: thay từng từ đã biết, không xóa từ lạ.
        fallback = [_prompt_word_fallback(word) for word in words]
        fallback = [word for word in fallback if word]
        translated = " ".join(fallback)
        return _translated_weight_wrapper(original, translated), translated != core

    pieces = []
    cursor = 0
    for start, end, value in matches:
        pieces.extend(_prompt_word_fallback(word) for word in words[cursor:start])
        if value:
            pieces.append(value)
        cursor = end
    pieces.extend(_prompt_word_fallback(word) for word in words[cursor:])
    pieces = [piece for piece in pieces if piece]
    translated = ", ".join(dict.fromkeys(pieces))
    return _translated_weight_wrapper(original, translated), translated != core


def _translate_prompt_text(text, rows):
    if not isinstance(text, str) or not text.strip():
        return text or "", 0, []
    index = _prompt_translation_index(rows)
    translated_segments = []
    changed = 0
    untouched = []
    for segment in re.split(r"[,;\n]+", text):
        translated, did_change = _translate_prompt_segment(segment, index)
        if not translated:
            continue
        translated_segments.append(translated)
        if did_change:
            changed += 1
        elif translated == segment.strip() and any("\u0080" <= char for char in segment):
            untouched.append(segment.strip())
    return ", ".join(translated_segments), changed, untouched


def translate_vietnamese_prompt(text, rows=None):
    """Translate one editable prompt to English using the verified local tag catalog."""
    rows = cached_tag_catalog() if rows is None else rows
    return _translate_prompt_text(text, rows)[0]


def translate_prompts_to_english(positive, negative):
    """UI handler: translate both editable fields and report what stayed unchanged."""
    try:
        rows = cached_tag_catalog()
        positive_en, positive_changed, positive_unknown = _translate_prompt_text(positive, rows)
        negative_en, negative_changed, negative_unknown = _translate_prompt_text(negative, rows)
    except Exception as exc:
        return positive, negative, (
            f"⚠️ Không dịch được prompt ({type(exc).__name__}). "
            "Nội dung hai ô được giữ nguyên; bạn vẫn có thể viết prompt English trực tiếp."
        )
    changed = positive_changed + negative_changed
    unknown = positive_unknown + negative_unknown
    note = (
        "Bộ dịch chạy offline bằng catalog/tag tiếng Việt đã xác minh; phần không nhận diện "
        "được được giữ nguyên để bạn sửa trước khi tạo ảnh."
    )
    if unknown:
        note += f" Còn giữ nguyên {len(unknown)} cụm chưa chắc nghĩa."
    return positive_en, negative_en, f"✅ Đã dịch {changed} cụm sang English. {note}"


# ---------------------------------------------------------------------------
# KIỂM TRA THẺ TRONG PROMPT VỚI KHO THẺ (CSV Danbooru + e621 đã xác minh SHA-256).
#
# Nút "🧪 Kiểm tra thẻ với kho thẻ" ở tab 🧭 Quy trình tách từng thẻ trong hai ô
# prompt/negative rồi đối chiếu với tên thẻ, alias và nhãn tiếng Việt trong catalog:
# thẻ nào đúng tên trong kho, thẻ nào chỉ là alias/nhãn Việt của một thẻ khác, thẻ
# nào không có trong kho (kèm gợi ý gần nhất cho trường hợp gõ sai). Chỉ đọc và
# báo cáo — không sửa nội dung hai ô prompt.
# ---------------------------------------------------------------------------
_CATALOG_TAG_FIELDS = None  # (rows, {đã chuẩn hoá: (tên thẻ chính, kind)})
_CATALOG_NAME_BUCKETS = None  # (rows, {ký tự đầu: (tên thẻ đã chuẩn hoá, ...)})


def catalog_tag_fields(rows=None):
    """Bảng tên/alias/nhãn đã chuẩn hoá → (tên thẻ chính, kind), dựng một lần/catalog.

    kind ∈ {"name", "alias", "label"}: "name" là đúng tên thẻ trong kho, "alias" là
    tên phụ ở cột alias của CSV, "label" là nhãn tiếng Việt của thẻ — hai loại sau
    đều có trong kho nhưng nên viết bằng tên thẻ chính tiếng Anh. Khi một chuỗi vừa
    là tên thẻ vừa là alias của thẻ khác thì "name" thắng. Khóa theo đúng đối tượng
    rows (``is``) như csv_tag_names nên một catalog khác (bản vá trong test) không
    bao giờ dùng nhầm bảng cũ.
    """
    global _CATALOG_TAG_FIELDS
    rows = load_csv_tags() if rows is None else rows
    cached = _CATALOG_TAG_FIELDS
    if cached is not None and cached[0] is rows:
        return cached[1]
    names = {}
    fields = {}
    for row in rows:
        name = row[0]
        name_key = normalize_csv_tag(name)
        names[name_key] = name
        label_key = normalize_csv_tag(row[5])
        synonym_key = normalize_csv_tag(TAG_VI_SEARCH_SYNONYMS.get(name, ""))
        for field in row[3].split(","):
            field = field.strip()
            if not field or field == name_key:
                continue
            kind = "label" if field in (label_key, synonym_key) else "alias"
            fields.setdefault(field, (name, kind))
    index = {key: (name, "name") for key, name in names.items()}
    for key, value in fields.items():
        index.setdefault(key, value)
    _CATALOG_TAG_FIELDS = (rows, index)
    return index


def _catalog_name_buckets(rows):
    """Tên thẻ đã chuẩn hoá nhóm theo ký tự đầu — ứng viên gợi ý khi gõ sai chính tả."""
    global _CATALOG_NAME_BUCKETS
    cached = _CATALOG_NAME_BUCKETS
    if cached is not None and cached[0] is rows:
        return cached[1]
    buckets = {}
    for name in tag_normalized_names(rows):
        if name:
            buckets.setdefault(name[0], []).append(name)
    buckets = {key: tuple(value) for key, value in buckets.items()}
    _CATALOG_NAME_BUCKETS = (rows, buckets)
    return buckets


def _catalog_check_exempt_tags():
    """Thẻ hợp lệ nhưng không có trong kho CSV — chính Studio đề xuất hoặc chèn.

    Gồm thẻ chất lượng chuẩn WAI v17, toàn bộ thẻ của 8 bộ negative, thẻ neo của
    khung prompt/phong cách, thẻ kỹ thuật `highres`/`absurdres`, thẻ gợi ý sửa vùng,
    thẻ của nhóm Chi tiết mắt & móng, trigger LoRA ``perfect eyes`` và các từ khóa
    ``BREAK``/``AND`` của SDXL. Một số tag kỹ thuật này nằm ở nhóm Metadata của nguồn
    nên đã được lọc khỏi CSV tối ưu, nhưng vẫn là prompt hợp lệ của Studio. Kiểm tra
    trong kho được ưu tiên hơn: thẻ vừa có trong CSV vừa ở đây vẫn báo "đúng tên thẻ"
    chứ không phải "thẻ chuẩn ngoài kho".
    """
    tags = set(QUALITY_TAG_SET)
    tags.update(QUALITY_TAIL)
    tags.update(STYLE_TAG_HINTS)
    tags.update(("highres", "hi_res", "absurdres", "absurd_res"))
    tags.update(("perfect eyes", "BREAK", "AND"))
    for scaffold in PROMPT_SCAFFOLDS.values():
        for anchor_tags in scaffold["anchors"].values():
            tags.update(anchor_tags)
    for preset in NEGATIVE_PRESETS:
        tags.update(preset["tags"])
    for positive, negative in REPAIR_HINTS.values():
        tags.update(split_tags(positive))
        tags.update(split_tags(negative))
    for _field, _label, choices in LOOK_FIELDS:
        for option in choices.values():
            tags.update(option)
    return frozenset(key for key in (normalize_csv_tag(tag) for tag in tags) if key)


CATALOG_CHECK_EXEMPT = _catalog_check_exempt_tags()


def _catalog_tag_suggestions(core, fields, rows, limit=3):
    """Gợi ý tên thẻ trong kho cho thẻ không khớp: tìm trong kho trước, rồi đoán chính tả."""
    found, _, _, _ = search_csv_tags(rows, query=core, sort="Khớp nhất")
    names = []
    for row in found:
        if row[0] not in names:
            names.append(row[0])
        if len(names) >= limit:
            return names
    if names:
        return names
    import difflib

    key = normalize_csv_tag(core)
    bucket = _catalog_name_buckets(rows).get(key[:1], ())
    close = difflib.get_close_matches(key, bucket, n=limit, cutoff=0.72)
    if not close:
        # Gõ sai thẻ chuẩn ngoài kho (vd. "worst qualit"): so với chính tập thẻ đó —
        # nhỏ hơn nhiều nên rẻ, và tên chuẩn của chúng đã đúng dạng cần gợi ý.
        close = difflib.get_close_matches(
            key, sorted(CATALOG_CHECK_EXEMPT), n=limit, cutoff=0.72
        )
    suggestions = []
    for match in close:
        canonical = fields.get(match, (match, "name"))[0]
        if canonical not in suggestions:
            suggestions.append(canonical)
    return suggestions


def check_prompt_tag(tag, fields, rows):
    """Đối chiếu MỘT thẻ với kho — trả (status, tên thẻ chính, gợi ý).

    status ∈ {"ok", "alias", "label", "known", "unknown", "skip"}: "ok" = đúng tên
    thẻ trong kho, "alias"/"label" = khớp alias/nhãn tiếng Việt của một thẻ (nên đổi
    sang tên chính), "known" = thẻ chuẩn ngoài kho, "unknown" = không có trong kho
    (mô tả tự do hoặc gõ sai), "skip" = thẻ rỗng sau khi gỡ cú pháp trọng số.
    """
    core = tag_core(tag)
    key = normalize_csv_tag(core)
    if not key:
        return "skip", None, ()
    if core == "BREAK":
        # Viết hoa đúng chuẩn là từ khóa tách khối của SDXL, không phải thẻ "break"
        # (nghỉ giải lao) trong kho Danbooru — đừng gợi ý đổi thành tên thẻ.
        return "known", None, ()
    if key in fields:
        canonical, kind = fields[key]
        return {"name": "ok", "alias": "alias", "label": "label"}[kind], canonical, ()
    if key in CATALOG_CHECK_EXEMPT:
        return "known", None, ()
    return "unknown", None, tuple(_catalog_tag_suggestions(core, fields, rows))


def validate_prompt_tags(text, rows=None):
    """Kiểm tra từng thẻ của một ô prompt với kho thẻ.

    Trả (total, results): total là số thẻ trong ô (kể cả trùng), results là danh
    sách (thẻ gốc, status, tên thẻ chính, gợi ý) cho từng thẻ duy nhất — thẻ trùng
    đã được 🩺 Kiểm tra prompt & thông số báo nên không lặp lại ở đây.
    """
    rows = load_csv_tags() if rows is None else rows
    fields = catalog_tag_fields(rows)
    tags = split_tags(text if isinstance(text, str) else "")
    results = []
    seen = set()
    for tag in tags:
        key = normalize_csv_tag(tag_core(tag))
        if not key or key in seen:
            continue
        seen.add(key)
        status, canonical, suggestions = check_prompt_tag(tag, fields, rows)
        if status == "skip":
            continue
        results.append((tag, status, canonical, suggestions))
    return len(tags), results


TAG_CHECK_REPORT_LIMIT = 12  # mỗi nhóm liệt kê tối đa 12 thẻ, phần còn lại gộp theo số


def _format_tag_check_group(icon, title, entries):
    shown = entries[:TAG_CHECK_REPORT_LIMIT]
    line = f"{icon} **{title} — {len(entries)}:** " + " · ".join(shown)
    hidden = len(entries) - len(shown)
    if hidden > 0:
        line += f" · … và {hidden} thẻ khác"
    return line


def format_tag_check_report(prompt_data, negative_data, catalog_size):
    """Kết quả kiểm tra thẻ với kho thẻ thành Markdown cho tab 🧭 Quy trình."""
    lines = [
        "**🧪 Kiểm tra thẻ với kho thẻ** · "
        f"kho: {catalog_size:,} thẻ Danbooru + e621 (đã xác minh SHA-256)"
    ]
    unknown_total = 0
    for title, (total, results) in (
        ("Prompt", prompt_data),
        ("Negative", negative_data),
    ):
        lines.append("")
        if not results:
            state = "ô trống." if total == 0 else f"{total} thẻ."
            lines.append(f"**{title}** · {state}")
            continue
        note = f" ({total - len(results)} thẻ trùng đã gộp)" if total > len(results) else ""
        lines.append(f"**{title}** · {total} thẻ{note}:")
        groups = {"ok": [], "alias": [], "label": [], "known": [], "unknown": []}
        for tag, status, canonical, suggestions in results:
            groups[status].append((tag, canonical, suggestions))
        unknown_total += len(groups["unknown"])
        if groups["ok"]:
            entries = []
            for tag, canonical, _ in groups["ok"]:
                core = tag_core(tag)
                entries.append(f"`{core}`" if core == canonical else f"`{core}` → `{canonical}`")
            lines.append(_format_tag_check_group("✅", "Đúng tên thẻ trong kho", entries))
        if groups["alias"]:
            entries = [
                f"`{tag_core(tag)}` → `{canonical}`"
                for tag, canonical, _ in groups["alias"]
            ]
            lines.append(_format_tag_check_group("🔀", "Alias của thẻ trong kho", entries))
        if groups["label"]:
            entries = [
                f"`{tag_core(tag)}` → nên dùng `{canonical}`"
                for tag, canonical, _ in groups["label"]
            ]
            lines.append(_format_tag_check_group("ℹ️", "Nhãn tiếng Việt", entries))
        if groups["known"]:
            entries = [f"`{tag_core(tag)}`" for tag, _, _ in groups["known"]]
            lines.append(
                _format_tag_check_group(
                    "ℹ️", "Thẻ chuẩn ngoài kho (chất lượng/negative/LoRA/BREAK)", entries
                )
            )
        if groups["unknown"]:
            entries = []
            for tag, _, suggestions in groups["unknown"]:
                entry = f"`{tag_core(tag)}`"
                if suggestions:
                    entry += " → gợi ý: " + ", ".join(f"`{name}`" for name in suggestions)
                else:
                    entry += " (không có gợi ý)"
                entries.append(entry)
            lines.append(
                _format_tag_check_group(
                    "⚠️", "Không có trong kho — mô tả tự do hoặc gõ sai", entries
                )
            )
    lines.append("")
    if unknown_total:
        lines.append(
            f"⚠️ {unknown_total} thẻ không có trong kho **không phải lỗi**: Illustrious "
            "vẫn đọc mô tả tự do — nhưng nếu bạn định dùng thẻ Danbooru/e621 thì có thể "
            "đã gõ sai chính tả; đối chiếu gợi ý rồi sửa trong ô prompt. Báo cáo này chỉ "
            "đọc hai ô prompt/negative, **không sửa gì**."
        )
    else:
        lines.append(
            "✅ Mọi thẻ trong hai ô đều có trong kho hoặc là thẻ chuẩn ngoài kho. "
            "Báo cáo này chỉ đọc hai ô prompt/negative, **không sửa gì**."
        )
    return "\n".join(lines)


def run_tag_check(prompt, negative):
    """Sự kiện UI: kiểm tra thẻ trong hai ô prompt với kho thẻ, không đổi giá trị nào."""
    try:
        rows = load_csv_tags()
        prompt_data = validate_prompt_tags(prompt, rows)
        negative_data = validate_prompt_tags(negative, rows)
    except Exception as exc:
        return (
            f"⚠️ Chưa nạp được kho thẻ ({type(exc).__name__}). Bấm **Tìm trong kho thẻ** "
            "ở tab 🏷️ Kho thẻ để thử lại — kiểm tra này cần kho thẻ đã xác minh "
            "SHA-256. Viết prompt và tạo ảnh vẫn hoạt động bình thường."
        )
    return format_tag_check_report(prompt_data, negative_data, len(rows))


# ---------------------------------------------------------------------------
# SỬA PROMPT THÀNH THẺ CHUẨN (dùng chung kho thẻ với 🧪 Kiểm tra thẻ).
#
# Nút "🛠️ Sửa prompt thành thẻ chuẩn" phân tích từng thẻ trong ô prompt: thẻ chưa
# đúng tên chuẩn (alias, nhãn tiếng Việt, hoặc không có trong kho nhưng có gợi ý)
# được đề xuất đổi sang tên thẻ chính; thẻ đã đúng tên kho thì liệt kê các thẻ có
# chung từ trong kho để đổi TÙY CHỌN. Người dùng chọn (các thẻ cần sửa đã được chọn
# sẵn) rồi bấm "✅ Tạo prompt hoàn chỉnh" — kết quả chỉ được GHI VÀO Ô HIỂN THỊ để
# sửa/xóa, không có thẻ nào được ghép ngầm khi tạo ảnh.
# ---------------------------------------------------------------------------
_CATALOG_NAME_WORD_INDEX = None  # (rows, {từ: (tên thẻ đã chuẩn hoá, ...)})


def _catalog_name_word_index(rows):
    """Từ tiếng Anh → các tên thẻ đã chuẩn hoá có chứa từ đó, dựng một lần/catalog.

    Dùng để tìm thẻ liên quan khi gợi ý đổi tùy chọn mà không phải quét regex toàn
    bộ catalog cho từng thẻ. Khóa theo đúng đối tượng rows (``is``) như các index khác.
    """
    global _CATALOG_NAME_WORD_INDEX
    cached = _CATALOG_NAME_WORD_INDEX
    if cached is not None and cached[0] is rows:
        return cached[1]
    index = {}
    for name in tag_normalized_names(rows):
        for word in set(name.split()):
            index.setdefault(word, []).append(name)
    index = {key: tuple(value) for key, value in index.items()}
    _CATALOG_NAME_WORD_INDEX = (rows, index)
    return index


def _related_catalog_tags(core_key, rows, fields, present, limit=3):
    """Các tên thẻ trong kho có chung từ với thẻ này — gợi ý đổi tùy chọn.

    Chỉ đọc index theo từ (không quét toàn bộ catalog): ưu tiên thẻ chung nhiều từ
    nhất, rồi theo thứ tự first-seen (cũng là thứ tự phổ biến trong kho). Bỏ qua
    chính nó và các thẻ đang có trong prompt. Trả về tên thẻ chính (dạng gốc).
    """
    tag_words = set(core_key.split())
    if not tag_words:
        return []
    index = _catalog_name_word_index(rows)
    scored = {}
    for word in tag_words:
        for name in index.get(word, ()):
            if name == core_key or name in present or name in scored:
                continue
            shared = len(tag_words & set(name.split()))
            if shared:
                scored[name] = shared
    related = sorted(scored, key=lambda name: -scored[name])  # ổn định: giữ first-seen
    return [
        fields.get(name, (name, "name"))[0]
        for name in related[:limit]
    ]


def propose_prompt_rewrites(text, rows=None):
    """Phân tích prompt và đề xuất cách sửa từng thẻ theo kho thẻ.

    Trả về (segments, proposals):
    - segments: [(thẻ gốc, status, tên thẻ chính, gợi ý)] theo đúng thứ tự ô prompt
      (kể cả thẻ trùng), status như check_prompt_tag.
    - proposals: danh sách dict cho dropdown thay thế, mỗi dict có
      {"core", "raw", "kind", "replacement", "label", "value"} với kind ∈ {"fix",
      "swap"}: "fix" là thẻ chưa đúng tên chuẩn (alias/nhãn Việt/gõ sai nhưng có
      gợi ý) — đổi sang tên thẻ chính, chọn sẵn; "swap" là thẻ đã đúng tên kho —
      các thẻ có chung từ để đổi TÙY CHỌN, không chọn sẵn. value có dạng
      "core<TAB>replacement" để nút áp dụng không cần giữ state.
    """
    rows = load_csv_tags() if rows is None else rows
    fields = catalog_tag_fields(rows)
    tags = split_tags(text if isinstance(text, str) else "")
    segments = []
    for tag in tags:
        core = tag_core(tag)
        key = normalize_csv_tag(core)
        if not key:
            continue
        status, canonical, suggestions = check_prompt_tag(tag, fields, rows)
        segments.append((tag, status, canonical, suggestions))
    present = {normalize_csv_tag(tag_core(tag)) for tag, _, _, _ in segments}
    proposals = []
    seen_cores = set()
    for tag, status, canonical, suggestions in segments:
        core = tag_core(tag)
        key = normalize_csv_tag(core)
        if key in seen_cores:
            continue
        seen_cores.add(key)
        if status in ("alias", "label"):
            proposals.append({
                "core": key,
                "raw": core,
                "kind": "fix",
                "replacement": canonical,
                "label": f"`{core}` → `{canonical}`",
                "value": f"{key}\t{canonical}",
            })
        elif status == "unknown":
            for rank, name in enumerate(suggestions[:3]):
                proposals.append({
                    "core": key,
                    "raw": core,
                    "kind": "fix",
                    "replacement": name,
                    "label": (
                        f"`{core}` → `{name}`"
                        + ("" if rank == 0 else " (lựa chọn khác)")
                    ),
                    "value": f"{key}\t{name}",
                })
        elif status == "ok":
            for name in _related_catalog_tags(key, rows, fields, present):
                proposals.append({
                    "core": key,
                    "raw": core,
                    "kind": "swap",
                    "replacement": name,
                    "label": f"`{core}` → `{name}` (đổi tùy chọn)",
                    "value": f"{key}\t{name}",
                })
    return segments, proposals


def format_rewrite_report(segments, proposals):
    """Tóm tắt kết quả phân tích của 🛠️ Sửa prompt thành thẻ chuẩn."""
    if not segments:
        return (
            "**🛠️ Sửa prompt thành thẻ chuẩn** · ô prompt đang trống — hãy viết "
            "hoặc nạp prompt trước rồi bấm lại."
        )
    counts = {"ok": 0, "alias": 0, "label": 0, "known": 0, "unknown": 0}
    raw_by_core = {}
    cores_by_status = {}
    for tag, status, _, _ in segments:
        counts[status] = counts.get(status, 0) + 1
        key = normalize_csv_tag(tag_core(tag))
        raw_by_core.setdefault(key, tag_core(tag))
        cores_by_status.setdefault(status, set()).add(key)
    lines = [
        f"**🛠️ Sửa prompt thành thẻ chuẩn** · {len(segments)} thẻ trong ô prompt:",
        f"- Đúng tên kho: {counts['ok']} · Thẻ chuẩn ngoài kho: {counts['known']}",
    ]
    fixes = []
    seen = set()
    for proposal in proposals:
        if proposal["kind"] == "fix" and proposal["core"] not in seen:
            seen.add(proposal["core"])
            fixes.append(proposal)
    if fixes:
        lines.append(
            f"- Sẽ đổi sang tên thẻ chính ({len(fixes)} thẻ, đã chọn sẵn trong ô "
            "dưới đây):"
        )
        for proposal in fixes[:10]:
            lines.append(f"  - {proposal['label']}")
        if len(fixes) > 10:
            lines.append(f"  - … và {len(fixes) - 10} thẻ khác")
    fix_cores = {proposal["core"] for proposal in proposals if proposal["kind"] == "fix"}
    unfixed = sorted(cores_by_status.get("unknown", set()) - fix_cores)
    if unfixed:
        shown = ", ".join(f"`{raw_by_core[core]}`" for core in unfixed[:8])
        more = f" … và {len(unfixed) - 8} thẻ khác" if len(unfixed) > 8 else ""
        lines.append(
            f"- ⚠️ Không có gợi ý trong kho, giữ nguyên ({len(unfixed)} thẻ): {shown}{more}"
        )
    swaps = [proposal for proposal in proposals if proposal["kind"] == "swap"]
    if swaps:
        lines.append(
            f"- ⇄ {len(swaps)} thay thế TÙY CHỌN cho thẻ đã đúng tên kho (chưa chọn "
            "sẵn) — đánh dấu trong ô dưới đây nếu muốn đổi."
        )
    lines.append("")
    lines.append(
        "Bấm **✅ Tạo prompt hoàn chỉnh** để ghi kết quả vào ô *Prompt gửi model* — "
        "bạn vẫn sửa/xóa được trước khi tạo ảnh."
    )
    return "\n".join(lines)


def run_prompt_rewrite(prompt):
    """Sự kiện UI: phân tích ô prompt, trả báo cáo + dropdown các thẻ đề xuất thay thế."""
    import gradio as gr

    try:
        rows = load_csv_tags()
        segments, proposals = propose_prompt_rewrites(prompt, rows)
    except Exception as exc:
        return (
            f"⚠️ Chưa nạp được kho thẻ ({type(exc).__name__}). Bấm **Tìm trong kho thẻ** "
            "ở tab 🏷️ Kho thẻ để thử lại — công cụ này cần kho thẻ đã xác minh "
            "SHA-256. Viết prompt và tạo ảnh vẫn hoạt động bình thường.",
            gr.update(choices=[], value=[]),
        )
    selected = []
    seen = set()
    for proposal in proposals:
        if proposal["kind"] == "fix" and proposal["core"] not in seen:
            seen.add(proposal["core"])
            selected.append(proposal["value"])
    return (
        format_rewrite_report(segments, proposals),
        gr.update(
            choices=[(proposal["label"], proposal["value"]) for proposal in proposals],
            value=selected,
        ),
    )


def apply_prompt_rewrite(prompt, selected):
    """Sự kiện UI: áp dụng các thẻ thay thế đã chọn, ghi prompt hoàn chỉnh vào ô.

    Chỉ đổi những thẻ có trong danh sách đã chọn (giá trị dạng "core<TAB>replacement"),
    giữ nguyên cú pháp nhấn mạnh của thẻ gốc (vd. ``(long_hari:1.2)`` →
    ``(long_hair:1.2)``). Kết quả chỉ ghi vào ô hiển thị để người dùng sửa/xóa.
    """
    tags = split_tags(prompt if isinstance(prompt, str) else "")
    if not tags:
        raise ValueError("Prompt đang trống — hãy viết hoặc nạp prompt trước.")
    replacements = {}
    for choice in selected or []:
        if not isinstance(choice, str) or "\t" not in choice:
            continue
        core, replacement = choice.split("\t", 1)
        if core and replacement:
            # Các phương án cùng một thẻ nằm cạnh nhau trong CheckboxGroup; nếu
            # người dùng đánh dấu thêm phương án khác, lựa chọn sau sẽ thay thế
            # phương án mặc định đầu tiên thay vì âm thầm giữ lỗi gõ cũ.
            replacements[core] = replacement
    if not replacements:
        return prompt, (
            "Chưa có thẻ nào được chọn — prompt giữ nguyên. Bấm **🛠️ Sửa prompt "
            "thành thẻ chuẩn** để phân tích và chọn thẻ thay thế trước."
        )
    out = []
    changed = []
    for tag in tags:
        raw_core = tag_core(tag)
        key = normalize_csv_tag(raw_core)
        if key in replacements and raw_core:
            replacement = replacements[key]
            new_tag = tag.replace(raw_core, replacement, 1)
            if new_tag != tag:
                changed.append(f"`{raw_core}` → `{replacement}`")
            out.append(new_tag)
        else:
            out.append(tag)
    if not changed:
        return prompt, (
            "Không có thẻ nào trong lựa chọn còn khớp với prompt hiện tại — prompt giữ "
            "nguyên. Bạn có thể bấm lại **🛠️ Sửa prompt thành thẻ chuẩn** để làm mới đề xuất."
        )
    note = (
        f"Đã tạo prompt hoàn chỉnh — đổi {len(changed)} thẻ: "
        + "; ".join(changed[:8])
        + ("…" if len(changed) > 8 else "")
        + ". Prompt chỉ được ghi vào ô hiển thị; bạn sửa/xóa được trước khi tạo ảnh."
    )
    return ", ".join(out), note


PROMPT_TAG_SUGGESTION_LIMIT = 16
# Danh sách gợi ý là dropdown Gradio, giới hạn để không nghẽn DOM trên điện thoại;
# tab Kho thẻ mới là nơi xem toàn bộ kết quả (lọc nhóm/chủ đề + phân trang).
PROMPT_TAG_SUGGESTION_CHOICES = (16, 32, 64, 150)


def _prompt_tag_suggestion_limit(value):
    """Ép giới hạn gợi ý về một trong các mức hợp lệ (mặc định 16)."""
    try:
        limit = int(float(value))
    except (TypeError, ValueError):
        return PROMPT_TAG_SUGGESTION_LIMIT
    return min(max(limit, 1), PROMPT_TAG_SUGGESTION_CHOICES[-1])

PROMPT_TAG_SUGGESTION_MIN_CHARS = 2
PROMPT_TAG_SUGGESTION_CACHE_LIMIT = 128
PROMPT_TAG_WEIGHT_STEP = 0.1
PROMPT_TAG_WEIGHT_RANGE = (0.1, 2.0)
PROMPT_TAG_WEIGHT_SELECTION_START = "\ue000"
PROMPT_TAG_WEIGHT_SELECTION_END = "\ue001"
PROMPT_TAG_WEIGHT_SELECTION_JS = r"""(promptText) => {
    const field = document.querySelector("#studio-prompt textarea");
    if (!field || typeof promptText !== "string") return [promptText];

    // Dấu phẩy bên trong nhóm "(a, b:1.1)" không được tách cụm; cùng quy tắc với
    // _prompt_weight_boundary() trong Python nên hai bên luôn chọn một phạm vi.
    const separators = [];
    let depth = 0;
    for (let i = 0; i < promptText.length; i++) {
        const ch = promptText[i];
        if (ch === "(") depth += 1;
        else if (ch === ")") depth = Math.max(0, depth - 1);
        else if (depth === 0 && (ch === "," || ch === ";" || ch === "\n")) separators.push(i);
    }
    const phraseStart = (position) => {
        let found = -1;
        for (const index of separators) {
            if (index < position) found = index;
            else break;
        }
        return found + 1;
    };
    const phraseEnd = (position) => {
        for (const index of separators) if (index >= position) return index;
        return promptText.length;
    };

    let start;
    let end;
    if (field !== document.activeElement) {
        // Bấm nút +/-: chỉnh cụm cuối prompt, nguyên cả một nhóm có dấu phẩy.
        start = phraseStart(promptText.length);
        end = promptText.length;
    } else {
        // Phím tắt: giữ đoạn đang bôi đen, nếu không thì lấy trọn tag quanh con trỏ.
        start = field.selectionStart;
        end = field.selectionEnd;
        if (start === end) {
            start = phraseStart(start);
            end = phraseEnd(field.selectionStart);
        }
    }

    while (start < end && /\s/.test(promptText[start])) start += 1;
    while (end > start && /\s/.test(promptText[end - 1])) end -= 1;
    if (start === end) return [promptText];

    return [
        promptText.slice(0, start) + "\uE000" +
        promptText.slice(start, end) + "\uE001" +
        promptText.slice(end)
    ];
}"""
PROMPT_TAG_WEIGHT_SHORTCUT_JS = r"""() => {
    // Phím tắt Ctrl+↑/↓ cho ô prompt. Observer CHỈ sống tới khi gắn được listener:
    // quan sát document.body với subtree:true sẽ chạy callback cho MỌI thay đổi DOM
    // (chuyển tab, gallery cập nhật, ảnh tải xong) và mỗi lần lại querySelector toàn
    // trang — trên điện thoại đó là giật lag lặp lại vô thời hạn khi thao tác.
    let observer = null;
    const cleanup = () => {
        if (observer) { observer.disconnect(); observer = null; }
    };
    const attachShortcut = () => {
        const field = document.querySelector("#studio-prompt textarea");
        if (!field) return false;
        if (field.dataset.waiWeightShortcuts !== "ready") {
            field.dataset.waiWeightShortcuts = "ready";
            field.addEventListener("keydown", (event) => {
                if (!event.ctrlKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
                event.preventDefault();
                event.stopPropagation();
                const id = event.key === "ArrowUp" ? "prompt-weight-up" : "prompt-weight-down";
                const element = document.getElementById(id);
                const button = element?.matches("button") ? element : element?.querySelector("button");
                button?.click();
            });
        }
        cleanup();
        return true;
    };

    if (!attachShortcut()) {
        observer = new MutationObserver(attachShortcut);
        // Quan sát vùng chứa ô prompt (nơi textarea thực sự xuất hiện) thay vì cả body.
        observer.observe(document.querySelector("#studio-prompt") || document.body,
                         { childList: true, subtree: true });
        // Phòng khi Gradio đổi cấu trúc DOM: đừng để observer sống mãi.
        setTimeout(cleanup, 30000);
    }
}"""
STUDIO_OFFLINE_WATCHDOG_JS = r"""() => {
    // Trang kẹt vĩnh viễn (phải tải lại) thường là mất đường hầm giữa chừng: Gradio
    // không phát lại sự kiện đã rơi, nên người dùng chỉ thấy màn hình đứng. Vòng ping
    // nhẹ này giữ cho kết nối còn "sống" sau thời gian dài không có dữ liệu (lượt tạo
    // ảnh có thể kéo dài nhiều phút) và báo rõ ràng + cho nút tải lại khi mất kết nối,
    // thay vì để trang treo im lặng.
    if (window.__waiOfflineWatchdog) return;
    window.__waiOfflineWatchdog = true;
    let failures = 0;
    const banner = () => {
        const existing = document.getElementById("studio-offline-banner");
        if (existing) return existing;
        const node = document.createElement("div");
        node.id = "studio-offline-banner";
        node.setAttribute("role", "alert");
        const text = document.createElement("span");
        text.textContent = "Mất kết nối với phiên Colab — giao diện đang chờ phản hồi.";
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Tải lại trang";
        button.addEventListener("click", () => window.location.reload());
        node.appendChild(text);
        node.appendChild(button);
        Object.assign(node.style, {
            position: "fixed", left: "50%", bottom: "16px", transform: "translateX(-50%)",
            zIndex: "9999", display: "flex", gap: "10px", alignItems: "center",
            padding: "10px 14px", borderRadius: "12px", background: "#2b2140", color: "#ffffff",
            boxShadow: "0 6px 18px rgba(0,0,0,.28)", font: "600 13px system-ui, sans-serif",
            maxWidth: "92vw"
        });
        Object.assign(button.style, {
            minHeight: "30px", padding: "4px 10px", borderRadius: "8px", border: "0",
            background: "#8a74ef", color: "#ffffff", fontWeight: "700", cursor: "pointer"
        });
        document.body.appendChild(node);
        return node;
    };
    const show = (offline) => {
        const node = document.getElementById("studio-offline-banner");
        if (!node) return;
        node.style.display = offline ? "flex" : "none";
    };
    // Chậm UI và mất đường hầm là hai việc khác nhau. Dải này không có nút tải lại
    // và không được gọi banner() — nếu không, một cú bấm > 0,4 s bị báo là mất kết nối.
    const blockBanner = (message) => {
        let node = document.getElementById("studio-block-banner");
        if (!node) {
            node = document.createElement("div");
            node.id = "studio-block-banner";
            node.setAttribute("role", "status");
            Object.assign(node.style, {
                position: "fixed", left: "50%", bottom: "72px", transform: "translateX(-50%)",
                zIndex: "9998", display: "none",
                padding: "10px 14px", borderRadius: "12px", background: "#3a3428", color: "#fff8e8",
                boxShadow: "0 6px 18px rgba(0,0,0,.28)", font: "600 13px system-ui, sans-serif",
                maxWidth: "92vw"
            });
            document.body.appendChild(node);
        }
        node.textContent = message;
        node.style.display = "flex";
        clearTimeout(node._hide);
        node._hide = setTimeout(() => { node.style.display = "none"; }, 25000);
    };
    const root = (window.gradio_config && window.gradio_config.root) || ".";
    const url = root.replace(/\/+$/, "") + "/config";
    const ping = async () => {
        if (document.visibilityState !== "visible") return;
        try {
            const response = await fetch(url + "?wai=" + Date.now(), {cache: "no-store"});
            failures = response.ok ? 0 : failures + 1;
        } catch (error) {
            failures += 1;
        }
        if (failures >= 2) banner();
        show(failures >= 2);
    };
    window.setInterval(ping, 20000);

    // Bằng chứng "đơ" phải lấy từ chính trình duyệt: đo khoảng thời gian từ cú bấm
    // tới lần vẽ kế tiếp. Nếu > 0,4 s thì trang thật sự bị chặn (không phải cảm giác).
    const blocks = [];
    window.__waiBlockLog = blocks;
    const note = (ms, what, how) => {
        blocks.push({at: new Date().toISOString(), ms: Math.round(ms), what, how});
        while (blocks.length > 12) blocks.shift();
        console.warn(`[wai] chặn ${Math.round(ms)} ms sau khi ${how}: ${what}`);
        blockBanner(
            `Trang bị chặn ${Math.round(ms / 100) / 10} s khi ${how} „${what}”. `
            + `Không phải mất kết nối — đừng tải lại chỉ vì dòng này. `
            + `Xem window.__waiBlockLog trong DevTools để lấy nhật ký.`
        );
    };
    let pending = null;
    document.addEventListener("click", (event) => {
        const node = event.target && event.target.closest
            ? event.target.closest("button, [role='tab'], label, a, input")
            : null;
        pending = {
            at: performance.now(),
            what: ((node && node.innerText) || (event.target && event.target.tagName) || "?")
                .replace(/\s+/g, " ").trim().slice(0, 60),
            how: node && node.matches("[role='tab']") ? "chuyển tab" : "bấm nút",
        };
        // Hai vòng rAF: trang chỉ được coi là phản hồi sau khi vẽ xong.
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (!pending) return;
            const ms = performance.now() - pending.at;
            if (ms > 400) note(ms, pending.what, pending.how);
            pending = null;
        }));
    }, true);
    try {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.duration >= 700) {
                    note(entry.duration, pending ? pending.what : "tác vụ dài trên luồng chính", "chặn");
                }
            }
        });
        observer.observe({entryTypes: ["longtask"]});
    } catch (error) {
        // Safari iOS không có longtask — đã có đo click→paint ở trên.
    }
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") ping();
    });
}"""

PROMPT_TAG_CATEGORY_MARKS = {
    "0": "[G]", "1": "[A]", "3": "[©]", "4": "[C]", "5": "[M]",
    "7": "<G>", "8": "<A>", "9": "<U>", "10": "<©>",
    "11": "<C>", "12": "<S>", "14": "<M>", "15": "<L>",
}
_PROMPT_TAG_SUGGESTION_CACHE = {}
_PROMPT_TAG_SUGGESTION_LOCK = threading.Lock()


def prompt_tag_fragment(prompt_text):
    """Return the final comma/semicolon/newline-delimited phrase being written."""
    if not isinstance(prompt_text, str):
        return ""
    return re.split(r"[,;\n]+", prompt_text)[-1].strip()


def _prompt_tag_search_mode(fragment):
    """Parse SAA-style wildcard and artist-filter prefixes from the final phrase."""
    phrase = str(fragment or "").strip()
    if phrase.startswith("@"):
        return "artist", phrase[1:].strip()
    if phrase.startswith("*"):
        if len(phrase) > 1 and phrase.endswith("*"):
            return "contains", phrase[1:-1].strip()
        return "suffix", phrase[1:].strip()
    return "catalog", phrase


def prompt_tag_choice(row):
    """Build a compact marked English option, preserving the CSV canonical value."""
    mark = PROMPT_TAG_CATEGORY_MARKS.get(row[1], f"[{row[1]}]")
    return f"{mark} {row[0].replace('_', ' ')}", row[0]


def prime_prompt_tag_catalog():
    """Download/verify and parse the suggestion CSV as the UI starts."""
    print("⏳ Đang nạp và xác minh CSV kho gợi ý tag...")
    try:
        rows = load_csv_tags()
    except Exception as exc:
        message = (
            f"⚠️ Chưa nạp được CSV kho gợi ý ({type(exc).__name__}); "
            "Studio sẽ thử lại khi bạn gõ hoặc tìm trong tab Kho thẻ. "
            "Tạo ảnh bình thường vẫn hoạt động."
        )
        print(message)
        return message
    message = f"✅ CSV kho gợi ý đã sẵn sàng: {len(rows):,} tag, đã kiểm SHA-256."
    tag_normalized_names(rows)  # làm ấm bảng tên thẻ cho tìm *đuôi*/*giữa*
    indexed = prime_tag_label_index(rows)
    if indexed:
        message += f" Đã dựng index {indexed:,} từ khóa từ nhãn tiếng Việt."
        print(message)
    else:
        print(message)
    return message


class PromptTagSearchSuperseded(Exception):
    """Lượt tìm cũ đã bị lần gõ mới ghi đè — không cần quét nốt catalog."""


_PROMPT_TAG_REQUEST_SEQ = 0


def prompt_tag_request_begin():
    """Ghi nhận một lượt tìm thẻ mới, trả số thứ tự để lượt cũ tự dừng."""
    global _PROMPT_TAG_REQUEST_SEQ
    with _PROMPT_TAG_SUGGESTION_LOCK:
        _PROMPT_TAG_REQUEST_SEQ += 1
        return _PROMPT_TAG_REQUEST_SEQ


def prompt_tag_search_alive(token):
    return token == _PROMPT_TAG_REQUEST_SEQ


def _prompt_tag_suggestion_results(rows, fragment, token=None):
    """Search the verified catalog and memoize results across keystrokes.

    ``token`` (số thứ tự của lượt gõ) cho phép quét catalog theo từng cụm 8192 dòng và
    dừng sớm nếu người dùng đã gõ tiếp: một lượt tìm đầy tốn tới ~1,5 s CPU, nên nếu để
    chạy trọn vẹn cho từng phím thì máy — đang gồng GPU — sẽ không còn phản hồi.
    """
    mode, query = _prompt_tag_search_mode(fragment)
    stale = (lambda: not prompt_tag_search_alive(token)) if token is not None else None
    cache_key = (
        id(rows), mode,
        _normalize_tag_text_preserving_accents(query), normalize_csv_tag(query),
    )
    with _PROMPT_TAG_SUGGESTION_LOCK:
        cached = _PROMPT_TAG_SUGGESTION_CACHE.get(cache_key)
        if cached is not None and cached[0] is rows:
            return cached[1], cached[2]

    normalized_query = normalize_csv_tag(query)
    if mode in ("contains", "suffix"):
        found = []
        names = tag_normalized_names(rows)
        for index, row in enumerate(rows):
            if stale and not index % 8192 and stale():
                raise PromptTagSearchSuperseded()
            name = names[index]
            matches = normalized_query in name if mode == "contains" else name.endswith(normalized_query)
            if normalized_query and matches:
                found.append(row)
        found.sort(key=lambda row: (-row[2], row[0]))
        total = len(found)
    elif mode == "artist":
        # `@` is a search filter, not part of the inserted WAI-Illustrious tag.
        pattern = _tag_query_pattern(query)
        found = [
            row for row in rows
            if row[1] in {"1", "8"} and pattern and pattern.search(row[3])
        ]
        found.sort(key=lambda row: (-row[2], row[0]))
        total = len(found)
    else:
        preserved = _normalize_tag_text_preserving_accents(query)
        if preserved != normalized_query:
            pattern = _tag_query_pattern(query, preserve_accents=True)
            found = _caption_search_matches(rows, query, pattern)
            found.sort(key=lambda row: (-row[2], row[0]))
            total = len(found)
        else:
            found, total, _, _ = search_csv_tags(
                rows, query, sort="Khớp nhất", page=1
            )

    if stale and stale():
        raise PromptTagSearchSuperseded()
    result = (rows, tuple(found), total)
    with _PROMPT_TAG_SUGGESTION_LOCK:
        cached = _PROMPT_TAG_SUGGESTION_CACHE.get(cache_key)
        if cached is not None and cached[0] is rows:
            return cached[1], cached[2]
        if len(_PROMPT_TAG_SUGGESTION_CACHE) >= PROMPT_TAG_SUGGESTION_CACHE_LIMIT:
            _PROMPT_TAG_SUGGESTION_CACHE.clear()
        _PROMPT_TAG_SUGGESTION_CACHE[cache_key] = result
    return result[1], result[2]


def _replace_prompt_tag_fragment(text, replacement):
    """Replace only the last comma/semicolon/newline-delimited prompt phrase."""
    last_separator = max(text.rfind(","), text.rfind(";"), text.rfind("\n"))
    prefix = text[:last_separator + 1]
    tail = text[last_separator + 1:]
    leading = tail[:len(tail) - len(tail.lstrip())]
    trailing = tail[len(tail.rstrip()):]
    return prefix + leading + replacement + trailing


def get_keyword_tag_suggestions(
    prompt_text, rows=None, limit=PROMPT_TAG_SUGGESTION_LIMIT, token=None
):
    """Search the preloaded CSV and return English tags with Danbooru/e621 marks."""
    limit = _prompt_tag_suggestion_limit(limit)
    fragment = prompt_tag_fragment(prompt_text)
    mode, query = _prompt_tag_search_mode(fragment)
    normalized = normalize_csv_tag(query)
    if len(normalized) < PROMPT_TAG_SUGGESTION_MIN_CHARS:
        if mode == "artist":
            return [], "Gõ `@` rồi ít nhất 2 ký tự để chỉ tìm tag họa sĩ Danbooru/e621."
        if mode in ("contains", "suffix"):
            return [], "Dùng `*đuôi` để tìm tag kết thúc bằng cụm đó hoặc `*giữa*` để tìm bên trong tag."
        return [], "Gõ ít nhất 2 ký tự ở cuối prompt (ví dụ `mắt` hoặc `eyes`) để tìm trong CSV."

    rows = load_csv_tags() if rows is None else rows
    found, total = _prompt_tag_suggestion_results(rows, fragment, token=token)
    choices = [prompt_tag_choice(row) for row in found[:max(1, int(limit))]]
    if not total:
        if mode == "artist":
            return [], f"CSV chưa tìm thấy tag họa sĩ cho `{fragment}`."
        return [], f"CSV chưa tìm thấy tag cho `{fragment}`; thử từ khóa tiếng Việt hoặc English khác."
    shown = len(choices)
    more = (
        f" trong {total:,} kết quả (tăng **Số gợi ý** để xem tiếp)"
        if total > shown else ""
    )
    if mode == "artist":
        detail = "`@` chỉ lọc họa sĩ Danbooru/e621; dấu `@` không được chèn vào prompt WAI."
    elif mode == "suffix":
        detail = "Tìm tag kết thúc bằng cụm nhập sau `*`."
    elif mode == "contains":
        detail = "Tìm tag có chứa cụm nằm giữa hai dấu `*`."
    else:
        detail = "Chọn một dòng để thay cụm từ khóa cuối prompt."
    return choices, f"**{shown} tag từ CSV**{more} cho `{fragment}`. {detail}"


def _prompt_weight_boundary(text):
    """Split at the last top-level separator, keeping a `(a, b:1.1)` group whole.

    Splitting only on commas would cut a weighted group in half, so separators
    inside parentheses are ignored. The scan runs forward to stay correct even
    when a bracket is left open while the user is still typing.
    Returns ``(prefix, region)``.
    """
    depth = 0
    split = -1
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and char in ",;\n":
            split = index
    return text[:split + 1], text[split + 1:]


def _explicit_prompt_weight(fragment):
    """(lõi, trọng số) của ``(thẻ:1.1)``, kể cả khi đang bị bọc ``((thẻ:1.1))``.

    ``((blue eyes))`` không phải trọng số. Gỡ lớp nhấn rồi mới tăng, nếu không
    nút +1 biến nó thành ``(((blue eyes)):1.1)``. Dấu ngoặc lệch như
    ``((blue_eyes:1.2)`` không được đọc thành trọng số 1.2.
    """
    text = str(fragment or "").strip()

    def explicit(value):
        match = WEIGHT_RE.fullmatch(value)
        if not match:
            return None
        body = match.group("body").strip()
        if not body or body.count("(") != body.count(")"):
            return None
        return body, float(match.group("weight"))

    found = explicit(text)
    if found:
        return found
    body = text
    for _ in range(8):
        if len(body) < 2 or {"(": ")", "[": "]"}.get(body[0]) != body[-1]:
            break
        inner = body[1:-1].strip()
        if not inner:
            break
        body = inner
        found = explicit(body)
        if found:
            return found
    return body, 1.0


def adjust_prompt_tag_weight(prompt_text, direction):
    """Adjust the selected/caret tag, or the final phrase, in 0.1 increments.

    The browser marks the current selection with the two private-use characters
    ``PROMPT_TAG_WEIGHT_SELECTION_START/END`` before calling this function; those
    markers are always stripped, including on the paths that change nothing.
    """
    raw = prompt_text if isinstance(prompt_text, str) else ""
    start = raw.find(PROMPT_TAG_WEIGHT_SELECTION_START)
    end = raw.find(PROMPT_TAG_WEIGHT_SELECTION_END, start + 1) if start >= 0 else -1
    has_selection = start >= 0 and end > start
    if has_selection:
        before = raw[:start]
        region = raw[start + 1:end]
        after = raw[end + 1:]
    else:
        before = ""
        region = ""
        after = ""
    # Marker-free text is what the prompt becomes when nothing can be adjusted.
    text = (before + region + after) if has_selection else raw.replace(
        PROMPT_TAG_WEIGHT_SELECTION_START, ""
    ).replace(PROMPT_TAG_WEIGHT_SELECTION_END, "")

    if direction not in (-1, 1):
        return text, "Hướng chỉnh trọng số không hợp lệ."

    if has_selection:
        fragment = region.strip()
    else:
        before, region = _prompt_weight_boundary(text)
        after = ""
        fragment = region.strip()
    if not fragment:
        return text, "Đặt con trỏ trong một tag hoặc chọn cụm tag cần chỉnh trọng số."

    # A weighted group is kept intact: "(long hair, blue_eyes:1.1)" is adjusted as
    # a whole, the same way ComfyUI and WebUI wrap a selected fragment. Emphasis
    # wrappers are removed first so "((blue eyes))" becomes "(blue eyes:1.1)".
    tag, current = _explicit_prompt_weight(fragment)
    low, high = PROMPT_TAG_WEIGHT_RANGE
    updated_weight = round(min(high, max(low, current + direction * PROMPT_TAG_WEIGHT_STEP)), 1)
    leading = region[:len(region) - len(region.lstrip())]
    trailing = region[len(region.rstrip()):]
    if updated_weight == current:
        return text, (
            f"Trọng số đã ở giới hạn {updated_weight:.1f} "
            f"(dải cho phép {low:.1f}–{high:.1f})."
        )
    replacement = tag if updated_weight == 1.0 else f"({tag}:{updated_weight:.1f})"
    updated = before + leading + replacement + trailing + after
    verb = "tăng" if direction > 0 else "giảm"
    scope = "đang chọn/con trỏ" if has_selection else "cuối prompt"
    return updated, f"Đã {verb} trọng số tag {scope} lên `{updated_weight:.1f}`: `{replacement}`."


def adjust_prompt_tag_weight_ui(prompt_text, direction):
    """Gradio adapter for visible +/- weight controls."""
    import gradio as gr

    updated, message = adjust_prompt_tag_weight(prompt_text, direction)
    return updated, message, gr.update(choices=[], value=None)


def update_keyword_tag_suggestions(prompt_text, limit=PROMPT_TAG_SUGGESTION_LIMIT):
    """Gradio adapter for live suggestions from the preloaded CSV catalog."""
    import gradio as gr

    token = prompt_tag_request_begin()
    try:
        choices, message = get_keyword_tag_suggestions(prompt_text, limit=limit, token=token)
        return gr.update(choices=choices, value=None), message
    except PromptTagSearchSuperseded:
        # Người dùng đã gõ tiếp: giữ nguyên danh sách cũ, lượt mới sẽ trả lời.
        return gr.update(), gr.update()
    except Exception as exc:
        return gr.update(choices=[], value=None), (
            f"Không đọc được CSV kho gợi ý ({type(exc).__name__}). Mở tab **🏷️ Kho thẻ** "
            "để thử tải lại; bạn vẫn có thể viết prompt và tạo ảnh."
        )


def apply_keyword_tag_suggestion(prompt_text, selected, rows=None):
    """Replace the final typed phrase with a selected tag from the CSV."""
    text = prompt_text if isinstance(prompt_text, str) else ""
    if not isinstance(selected, str) or not selected:
        return text, "Chọn một tag tiếng Anh trong danh sách."

    try:
        rows = load_csv_tags() if rows is None else rows
        # Đối chiếu với TOÀN BỘ kết quả (đã cache), không chỉ top-N đang hiển thị,
        # nên người dùng chọn gợi ý thứ 40 khi giới hạn là 16 vẫn hợp lệ.
        found, _total = _prompt_tag_suggestion_results(rows, prompt_tag_fragment(text))
    except Exception as exc:
        return text, f"Không đọc được CSV kho gợi ý ({type(exc).__name__}); hãy thử lại."
    allowed = {row[0] for row in found}
    if normalize_csv_tag(selected) not in {normalize_csv_tag(tag) for tag in allowed}:
        return text, "Lựa chọn không còn trong CSV; hãy gõ lại từ khóa để làm mới."

    fragment = prompt_tag_fragment(text)
    if normalize_csv_tag(fragment) == normalize_csv_tag(selected):
        return text, f"`{selected}` đã là tag hiện tại."

    updated = _replace_prompt_tag_fragment(text, selected)
    return updated, f"Đã thay `{fragment}` bằng tag tiếng Anh `{selected}` từ CSV."


def apply_keyword_tag_suggestion_ui(prompt_text, selected):
    """Apply a selected CSV tag immediately and clear the old choices."""
    import gradio as gr

    updated, message = apply_keyword_tag_suggestion(prompt_text, selected)
    return updated, message, gr.update(choices=[], value=None)


# --- Chọn ảnh nguồn để sửa / phóng -------------------------------------------------
# Ba nút ở khung Kết quả trước đây luôn lấy ảnh mới nhất nên không cách nào chọn ảnh
# khác. Thư viện ảnh giờ là danh sách nguồn: bấm vào ảnh là chọn, và ảnh của các lượt
# tạo trước vẫn nạp được bằng nút ↻ (đọc thư mục xuất).
GALLERY_SOURCE_LIMIT = 40


def gallery_item_path(item):
    """(đường dẫn, chú thích) của một phần tử gr.Gallery: (path, caption) | dict | path."""
    caption = ""
    if isinstance(item, (tuple, list)):
        path, *rest = item
        caption = str(rest[0]) if rest and rest[0] else ""
    elif isinstance(item, dict):
        image = item.get("image", item)
        path = image.get("path") if isinstance(image, dict) else image
        caption = str(item.get("caption") or "")
    else:
        path = item
    return (str(path) if path else ""), caption


def gallery_entries(gallery_value, verify=True):
    """Ảnh trong một giá trị thư viện, giữ nguyên thứ tự hiển thị.

    ``verify=False`` bỏ qua bước ``is_file()`` (stat trên ổ mạng) khi ảnh vừa được
    server gửi về thư viện — ảnh đó chắc chắn còn; nếu không, ``selected_source_image``
    sẽ báo rõ khi nạp.
    """
    entries = []
    for item in gallery_value or []:
        path, caption = gallery_item_path(item)
        if path and (not verify or Path(path).is_file()):
            entries.append((path, caption))
    return entries


def tapped_gallery_path(gallery_value, index):
    """Đường dẫn ở vị trí thứ `index` trong thư viện (Gradio đánh số theo thứ tự hiển thị)."""
    items = list(gallery_value or [])
    try:
        position = index[0] if isinstance(index, (list, tuple)) else index
        path, _ = gallery_item_path(items[int(position)])
    except (IndexError, TypeError, ValueError):
        return None
    return path or None


_OUTPUT_SCAN_CACHE = {}
OUTPUT_SCAN_TTL_SECONDS = 5


def output_timestamp(name):
    """Chuỗi thời gian UTC nằm trong tên ảnh, đã bỏ phần mode.

    Ảnh có tên ``wai_<mode>_<YYYYmmdd>_<HHMMSS>_<micro>_<seed>.png``; xếp theo cả tên
    thì ``upscale`` luôn đứng trước ``t2i`` bất kể thời gian, còn lấy ``st_mtime`` thì
    phải ``stat()`` từng file — trên ổ mạng ``/content`` của Colab mỗi lần stat mất
    hàng chục ms. Mốc thời gian trong tên cho đúng thứ tự mà không chạm đĩa.
    """
    parts = Path(name).stem.split("_")
    for index, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            return "_".join(parts[index:index + 3])
    return Path(name).stem


def output_dir_entries(directory, limit=GALLERY_SOURCE_LIMIT, ttl=OUTPUT_SCAN_TTL_SECONDS):
    """Mọi ảnh PNG trong thư mục xuất, mới nhất trước — gồm cả ảnh của lượt tạo trước.

    Tên file đã chứa mốc thời gian tạo (``wai_<ngày>_<giờ>_<seed>_<i>.png``) nên xếp
    theo tên là đúng thứ tự thời gian: tránh ``stat()`` từng file, vốn trên ổ mạng
    ``/content`` của Colab (gcsfuse) tốn vài chục ms mỗi ảnh và từng làm nghẽn cả app.
    Kết quả được giữ lại ``ttl`` giây để nút ↻ / mở lại trang không quét lại đĩa liên tục.
    """
    if not directory:
        return []
    key = str(Path(directory))
    now = time.time()
    cached = _OUTPUT_SCAN_CACHE.get(key)
    if ttl and cached and now - cached[0] < ttl:
        return list(cached[1][:limit])
    try:
        files = sorted(
            Path(key).glob("*.png"), key=lambda path: output_timestamp(path.name), reverse=True
        )
    except OSError:
        return []
    entries = [(str(path), "") for path in files[:limit]]
    _OUTPUT_SCAN_CACHE[key] = (now, entries)
    return list(entries)


def source_entries(directory, gallery_value=None, verify=True, ttl=OUTPUT_SCAN_TTL_SECONDS):
    """Danh sách cho ô chọn ảnh: cả thư mục xuất (mới nhất trước), kèm chú thích của
    ảnh đang hiện trong thư viện. Rơi xuống thư viện khi không đọc được thư mục."""
    captions = dict(gallery_entries(gallery_value, verify=verify))
    entries = [(path, captions.get(path, ""))
               for path, _ in output_dir_entries(directory, ttl=ttl)]
    if not entries:
        entries = [(path, captions.get(path, "")) for path, _ in reversed(captions.items())]
    return entries


def source_picker_state(entries, keep=None, pick=None):
    """(choices, value) cho ô chọn ảnh nguồn — phần thuần logic, không cần Gradio.

    Ưu tiên ảnh vừa bấm trong thư viện (`pick`), rồi tới lựa chọn đang giữ (`keep`),
    cuối cùng mới tới ảnh mới nhất; nhờ vậy hành vi cũ "luôn lấy ảnh mới nhất" vẫn còn
    mà người dùng vẫn chỉ định được bức cụ thể.
    """
    choices = [
        (f"{index + 1} · {Path(path).name}" + (f" · {caption}" if caption else ""), path)
        for index, (path, caption) in enumerate(entries)
    ]
    paths = [path for path, _ in entries]
    if pick in paths:
        value = pick
    elif keep in paths:
        value = keep
    else:
        value = paths[0] if paths else None
    return choices, value


def merge_source_entries(remembered, fresh, limit=GALLERY_SOURCE_LIMIT):
    """Gộp ảnh thư viện với danh sách đã nạp bằng ↻, không đọc đĩa.

    Ảnh mới trong thư viện đứng trước. Ảnh cũ đã nạp không bị xóa chỉ vì người dùng
    bấm một ảnh hoặc vì lượt tạo vừa thay cả thư viện. Trùng đường dẫn thì giữ chú
    thích không rỗng.
    """
    captions = {}
    for bucket in (remembered or [], fresh or []):
        if isinstance(bucket, (str, bytes)):
            continue
        for item in bucket:
            if not isinstance(item, (list, tuple)) or not item or not item[0]:
                continue
            path = str(item[0])
            caption = str(item[1]) if len(item) > 1 and item[1] else ""
            # Ảnh thư viện đi sau nên chú thích mới (seed, kích thước) thắng chú thích cũ.
            if caption or path not in captions:
                captions[path] = caption
    ordered = []
    seen = set()
    for bucket in (fresh or [], remembered or []):
        if isinstance(bucket, (str, bytes)):
            continue
        for item in bucket:
            if not isinstance(item, (list, tuple)) or not item or not item[0]:
                continue
            path = str(item[0])
            if path in seen:
                continue
            seen.add(path)
            ordered.append([path, captions.get(path, "")])
            if len(ordered) >= limit:
                return ordered
    return ordered


def source_picker_update(entries, keep=None, pick=None):
    """Giá trị gr.update cho ô chọn ảnh nguồn."""
    import gradio as gr

    choices, value = source_picker_state(entries, keep=keep, pick=pick)
    return gr.update(choices=choices, value=value)


def selected_source_image(path):
    """Ảnh người dùng chọn; báo lỗi rõ thay vì âm thầm lấy ảnh mới nhất."""
    import gradio as gr
    from PIL import Image

    if not path:
        raise gr.Error("Chưa có ảnh để nạp — hãy tạo ảnh trước, hoặc bấm ↻ để đọc thư mục xuất.")
    if not Path(path).is_file():
        raise gr.Error("Ảnh đã chọn không còn trên đĩa (Colab có thể đã đổi phiên).")
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # file đang ghi dở, dung lượng 0, hoặc không phải ảnh
        raise gr.Error(
            f"Không đọc được ảnh đã chọn ({type(exc).__name__}): {Path(path).name}"
        ) from exc


def editor_value_for(image):
    """Giá trị cho gr.ImageEditor: chỉ có lớp nền là ảnh đã chọn, chưa có nét vẽ.

    Để ``composite`` trống vì Gradio tự vẽ composite ở trình duyệt khi tô, còn tuyến
    sửa vùng đọc ``background`` + ``layers`` (xem ``_editor_mask``). Trước đây hàm này
    ``convert("RGBA")`` rồi nhét cùng một ảnh vào cả background lẫn composite, nên mỗi lần nạp ảnh máy phải
    mã hóa thêm một PNG RGBA full-size và tab ✎ Sửa vùng phải tải về đúng file đó khi
    được mở — trên điện thoại (và qua đường hầm Colab) đó là khoảng một giây đứng hình.
    Trả ảnh gốc còn nguyên ``filename`` giúp cả ba khung ◈ / ⤢ / ✎ dùng chung MỘT file
    đã cache, trình duyệt chỉ tải một lần.
    """
    return {"background": image, "layers": [], "composite": None}


SLOW_EVENT_SECONDS = 0.3
SLOW_EVENT_WARNING_SECONDS = 10.0


def instrument_ui_events(demo):
    """In ra notebook mỗi sự kiện UI tốn quá `SLOW_EVENT_SECONDS`, và báo trước nếu
    một sự kiện vẫn còn chạy sau 10 giây.

    Mục đích: khi người dùng báo "trang đơ", ô chạy Gradio cho biết ĐÚNG sự kiện nào
    và mất bao lâu — hết phải đoán xem là tab, hàng đợi hay catalog. Chỉ in, không đổi
    hành vi; sự kiện tạo ảnh (generator) được tính tới khi kết thúc.
    """
    import functools
    import inspect

    labels = {}
    for index, block_fn in list(getattr(demo, "fns", {}).items()):
        fn = getattr(block_fn, "fn", None)
        if fn is None or getattr(fn, "_wai_timed", False):
            continue
        name = labels.setdefault(
            index, _ui_event_label(demo, block_fn, index)
        )

        @functools.wraps(fn)
        def wrapper(*args, _fn=fn, _name=name, **kwargs):
            started = time.perf_counter()
            alarm = threading.Timer(SLOW_EVENT_WARNING_SECONDS, _slow_event_started, args=(_name,))
            alarm.daemon = True
            alarm.start()
            try:
                result = _fn(*args, **kwargs)
            except BaseException:
                alarm.cancel()
                _slow_event_done(_name, started, failed=True)
                raise
            if inspect.isgenerator(result):
                return _timed_generator(result, _name, started, alarm)
            if inspect.isasyncgen(result):
                return _timed_async_generator(result, _name, started, alarm)
            alarm.cancel()
            _slow_event_done(_name, started)
            return result

        wrapper._wai_timed = True
        # Gradio chỉ tiêm SelectData khi inspect.signature thấy annotation. wraps
        # không đổi chữ ký *args của wrapper, nên phải gán lại chữ ký gốc.
        try:
            wrapper.__signature__ = inspect.signature(fn)
        except (TypeError, ValueError):
            pass
        block_fn.fn = wrapper
    return demo


def _ui_event_label(demo, block_fn, index):
    """'gallery.change «Ảnh đã tạo»' — đủ để đọc trong log mà không cần biết internals."""
    event = "?"
    title = ""
    try:
        targets = list(getattr(block_fn, "targets", []) or [])
        if targets:
            renderable_id, event = targets[0][0], targets[0][1]
            block = getattr(demo, "blocks", {}).get(renderable_id)
            if block is not None:
                title = getattr(block, "label", None) or type(block).__name__
                event = f"{type(block).__name__.lower()}.{event}"
    except Exception:
        pass
    return f"#{index} {event}" + (f" «{title}»" if title else "")


def _slow_event_started(name):
    print(f"⏳ Sự kiện {name} vẫn đang chạy sau {SLOW_EVENT_WARNING_SECONDS:g} s — "
          "đây là thứ đang giữ giao diện chờ.", flush=True)


def _slow_event_done(name, started, failed=False):
    took = time.perf_counter() - started
    if took >= SLOW_EVENT_SECONDS or failed:
        flag = "⚠️" if failed else "⏱"
        print(f"{flag} {name} hết {took:,.2f} s" + (" (lỗi)" if failed else ""), flush=True)


def _timed_generator(gen, name, started, alarm):
    try:
        for item in gen:
            yield item
    finally:
        alarm.cancel()
        _slow_event_done(name, started)


def _timed_async_generator(agen, name, started, alarm):
    async def run():
        try:
            async for item in agen:
                yield item
        finally:
            alarm.cancel()
            _slow_event_done(name, started)

    return run()


def warn_if_gradio_freezes_tabs():
    """Gradio 6.11–6.15.2 khóa trình duyệt khi bấm tab. Ô 2 phải cài 6.17.3."""
    try:
        from importlib.metadata import version
        from packaging.version import Version
        current = Version(version("gradio"))
    except Exception:
        return
    if current < Version("6.16.0"):
        print(
            f"⚠️ Gradio {current} làm trình duyệt báo «trang không phản hồi» khi bấm "
            "✦ Tạo ảnh / 🏷️ Kho thẻ / 📚 Thư viện. Runtime → Restart runtime → Run all "
            "để ô 2 cài Gradio 6.17.3. Chỉ chạy lại ô 8 hoặc ô 9 không hết đơ."
        )


def build_app(runtime):
    """Build Gradio Blocks without opening a public tunnel until launch cell runs."""
    warn_if_gradio_freezes_tabs()
    tag_catalog_status = prime_prompt_tag_catalog()
    import gradio as gr
    from PIL import Image

    css = """
    #wai-studio {
        max-width: 1480px !important;
        margin: 0 auto !important;
        padding: 16px clamp(10px, 2vw, 24px) 30px !important;
        --studio-accent: #6c4be8;
        --studio-accent-dark: #4733ad;
        --studio-ink: #25243a;
        --studio-muted: #77758c;
        --studio-border: #e4e2ef;
        --studio-surface: #ffffff;
        --studio-surface-soft: #f8f7fc;
    }
    #wai-studio .form {gap: 0.65rem;}
    #wai-studio .block {border-radius: 12px;}
    #wai-studio span[data-testid="block-label"], #wai-studio .label-text {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--studio-ink);
    }
    #wai-studio .row {gap: 0.7rem;}
    .studio-hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 14px 24px;
        position: relative;
        overflow: hidden;
        padding: 20px 24px;
        margin-bottom: 14px;
        border: 1px solid #382c68;
        border-radius: 18px;
        color: #fff;
        background:
            radial-gradient(ellipse at 85% 0%, #7f63ea 0%, transparent 38%),
            linear-gradient(118deg, #19172a 0%, #30234f 58%, #4b2861 100%);
        box-shadow: 0 14px 32px #29204420;
    }
    .studio-brand {display: flex; align-items: center; gap: 14px; min-width: 0;}
    .studio-mark {
        display: grid; place-items: center; flex: 0 0 48px; width: 48px; height: 48px;
        border: 1px solid #ffffff35; border-radius: 15px; color: #fff;
        background: linear-gradient(145deg, #a68bff, #6547d7 68%, #d660a2);
        font-size: 1.5rem; box-shadow: 0 8px 22px #100c233d;
    }
    .studio-brand-copy {min-width: 0;}
    .studio-kicker {
        color: #d5c8ff; font-size: 0.64rem; font-weight: 800;
        letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 3px;
    }
    .studio-kicker span {color: #f2bedf; margin-left: 7px;}
    .studio-hero h1 {
        color: #fff !important; font-size: clamp(1.25rem, 2.2vw, 1.8rem);
        line-height: 1.15; font-weight: 750; margin: 0;
    }
    .studio-hero p {color: #d6d0e8; font-size: 0.82rem; margin: 5px 0 0;}
    .studio-meta {display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px;}
    .studio-chip {
        display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
        padding: 8px 11px; border: 1px solid #ffffff2c; border-radius: 999px;
        color: #f4f0ff; background: #ffffff12; font-size: 0.72rem; font-weight: 650;
    }
    .studio-chip-dot {
        width: 7px; height: 7px; border-radius: 50%; background: #7de0ad;
        box-shadow: 0 0 0 3px #7de0ad25;
    }
    .studio-privacy {
        grid-column: 1 / -1; display: flex; align-items: flex-start; gap: 9px;
        padding-top: 11px; border-top: 1px solid #ffffff25;
        color: #e8e3f1; font-size: 0.76rem; line-height: 1.45;
    }
    .studio-privacy-mark {
        display: grid; place-items: center; flex: 0 0 18px; height: 18px;
        border-radius: 50%; color: #241832; background: #f0c66b; font-size: 0.7rem;
        font-weight: 900;
    }
    .studio-privacy strong {color: #fff;}
    .studio-panel {
        min-width: 0 !important; padding: clamp(12px, 1.5vw, 20px) !important;
        border: 1px solid var(--studio-border) !important;
        border-radius: 18px !important; background: var(--studio-surface) !important;
        box-shadow: 0 8px 24px #2920440b;
    }
    .studio-layout {
        display: grid !important;
        grid-template-columns: minmax(0, 1.08fr) minmax(320px, 0.92fr);
        align-items: start; gap: 16px !important;
    }
    .studio-layout > .studio-panel {width: auto !important; flex: initial !important;}
    .studio-editor-column {gap: 0.72rem !important;}
    .studio-output-column {gap: 0.8rem !important;}
    /* ImageEditor places the brush toolbar inside the canvas. Keep it visible on
       narrow/mobile layouts where the surrounding Gradio block may clip overflow. */
    #studio-inpaint-editor,
    #studio-inpaint-editor [class*="image-container"],
    #studio-inpaint-editor [class*="toolbar-wrap"] {
        overflow: visible !important;
    }
    #studio-inpaint-editor [class*="toolbar-wrap"] {
        z-index: 1000 !important;
        visibility: visible !important;
    }
    .studio-section-heading, .studio-section-heading p {
        color: var(--studio-ink) !important; font-size: 1rem !important;
        font-weight: 750 !important; line-height: 1.25; margin: 0 !important;
    }
    .studio-section-subtitle, .studio-section-subtitle p {
        color: var(--studio-muted) !important; font-size: 0.76rem !important;
        line-height: 1.45; margin: 2px 0 0 !important;
    }
    .studio-hint, .studio-hint p {
        color: var(--studio-muted) !important; font-size: 0.76rem !important;
        line-height: 1.45; margin: 0 !important;
    }
    #wai-studio textarea {line-height: 1.55;}
    #wai-studio input, #wai-studio textarea {border-radius: 10px !important;}
    #wai-studio button {
        min-height: 40px; border-radius: 10px; font-weight: 650;
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
    }
    #wai-studio button:hover {filter: brightness(1.025);}
    #wai-studio button:active {transform: translateY(1px);}
    #wai-studio button:focus-visible, #wai-studio input:focus-visible,
    #wai-studio textarea:focus-visible {
        outline: 3px solid #8a74ef70 !important; outline-offset: 2px;
    }
    #wai-studio .studio-primary button {
        min-height: 46px; font-size: 0.9rem; font-weight: 750;
        box-shadow: 0 6px 14px #6247d522;
    }
    .studio-hires, .studio-detailer, .studio-look {
        padding: 14px; border: 1px solid var(--studio-border);
        border-radius: 14px; background: var(--studio-surface-soft);
    }
    /* Navigation is compact, touch-scrollable and has one consistent active state. */
    #studio-workspace-tabs > .tab-nav, #studio-mode-tabs > .tab-nav,
    #studio-workspace-tabs > [role="tablist"], #studio-mode-tabs > [role="tablist"] {
        display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 5px;
        scrollbar-width: thin; -webkit-overflow-scrolling: touch;
        padding: 5px 1px 9px; max-width: 100%; border-bottom: 1px solid var(--studio-border);
    }
    #studio-workspace-tabs > .tab-nav button, #studio-mode-tabs > .tab-nav button,
    #studio-workspace-tabs > [role="tablist"] button, #studio-mode-tabs > [role="tablist"] button {
        flex: 0 0 auto; white-space: nowrap; min-height: 42px;
        border-radius: 10px; font-weight: 650;
    }
    #studio-workspace-tabs button[role="tab"][aria-selected="true"] {
        color: var(--studio-accent-dark); background: #eeeaff; font-weight: 750;
        box-shadow: inset 0 -2px 0 var(--studio-accent);
    }
    #studio-mode-tabs button[role="tab"][aria-selected="true"] {
        color: var(--studio-accent-dark); background: #f3f0ff; font-weight: 750;
    }
    #studio-workspace-tabs, #studio-mode-tabs {min-width: 0;}
    .studio-gallery {
        min-height: 380px !important;
        border: 1px solid var(--studio-border) !important;
        border-radius: 14px !important; background: var(--studio-surface-soft) !important;
    }
    /* Prompt list stays touch-scrollable and rows are easy to tap. */
    .studio-prompt-list {
        max-height: 44vh; overflow-y: auto; -webkit-overflow-scrolling: touch;
        border: 1px solid var(--studio-border); border-radius: 12px;
        padding: 5px 10px; background: var(--studio-surface-soft);
    }
    .studio-prompt-list label {
        min-height: 44px; display: flex; align-items: center; padding: 3px 0;
        font-size: 0.86rem; line-height: 1.35;
    }
    @media (max-width: 900px) {
        #wai-studio {padding: 10px !important;}
        .studio-layout {grid-template-columns: minmax(0, 1fr);}
        .studio-hero {padding: 16px; grid-template-columns: minmax(0, 1fr);}
        .studio-meta {justify-content: flex-start;}
        .studio-privacy {grid-column: 1;}
        #wai-studio .studio-panel {padding: 12px !important;}
    }
    @media (max-width: 520px) {
        #wai-studio {padding: 6px !important;}
        .studio-gallery {height: min(58vh, 390px) !important; min-height: 250px !important;}
        .studio-hero {padding: 14px; border-radius: 14px;}
        .studio-brand {align-items: flex-start; gap: 10px;}
        .studio-mark {flex-basis: 40px; width: 40px; height: 40px; border-radius: 12px;}
        .studio-meta {gap: 6px;}
        .studio-chip {padding: 7px 9px; font-size: 0.67rem;}
        #studio-workspace-tabs > .tab-nav button, #studio-mode-tabs > .tab-nav button,
        #studio-workspace-tabs > [role="tablist"] button, #studio-mode-tabs > [role="tablist"] button {
            min-height: 44px; padding-left: 11px; padding-right: 11px;
        }
    }
    """
    with gr.Blocks(
        title="WAI Studio · Colab GPU",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
        elem_id="wai-studio",
    ) as demo:
        gr.HTML(
            f"<header class='studio-hero'>"
            "<div class='studio-brand'>"
            "<div class='studio-mark' aria-hidden='true'>✦</div>"
            "<div class='studio-brand-copy'>"
            "<div class='studio-kicker'>WAI STUDIO <span>COLAB GPU</span></div>"
            "<h1>Studio tạo ảnh anime</h1>"
            "<p>WAI-illustrious v17 · LoRA tùy chọn được kiểm SHA-256</p>"
            "</div></div>"
            "<div class='studio-meta'>"
            "<span class='studio-chip'><span class='studio-chip-dot'></span>Model đã xác minh</span>"
            "<span class='studio-chip'>GPU Colab</span>"
            f"<span class='studio-chip' id='studio-build-chip' "
            f"title='Mã bản dựng của ô 7. Không thấy dòng này, hoặc bấm tab vẫn báo "
            f"trang không phản hồi: Restart runtime rồi Run all — ô 2 phải cài Gradio 6.17.3, "
            f"chỉ chạy lại ô 8/9 không đủ'>Bản dựng {STUDIO_BUILD}</span></div>"
            "<div class='studio-privacy'>"
            "<span class='studio-privacy-mark' aria-hidden='true'>!</span>"
            "<span><strong>Liên kết không có đăng nhập:</strong> bất kỳ ai có link đều "
            "có thể dùng GPU của bạn. Đừng chia sẻ; dừng runtime để thu hồi link. "
            "Ảnh lưu trong /content và sẽ mất khi runtime kết thúc.</span>"
            "</div></header>"
        )
        with gr.Row(equal_height=False, elem_classes="studio-layout"):
            with gr.Column(
                scale=6,
                min_width=340,
                elem_classes=["studio-panel", "studio-editor-column"],
            ):
                gr.Markdown("### 01 · Soạn prompt", elem_classes="studio-section-heading")
                gr.Markdown(
                    "Mô tả chủ thể, ngoại hình, bối cảnh và phong cách; "
                    "nội dung hai ô được gửi nguyên văn.",
                    elem_classes="studio-section-subtitle",
                )
                prompt = gr.Textbox(
                    label="Prompt gửi model · tự viết phong cách của bạn",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    max_lines=8,
                    placeholder=(
                        "Mô tả nhân vật, trang phục, khung cảnh, ánh sáng và phong cách "
                        "vẽ bạn muốn (anime illustration, cel shading, watercolor...)"
                    ),
                    elem_id="studio-prompt",
                )
                with gr.Row(equal_height=False):
                    keyword_tag_suggestion = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Semi-auto tag · ví dụ mắt / eyes · ↑↓ + Enter/Tab để chọn",
                        allow_custom_value=False,
                        scale=1,
                    )
                    keyword_tag_weight_down = gr.Button(
                        "−0,1", size="sm", scale=0, min_width=58, variant="secondary",
                        elem_id="prompt-weight-down",
                    )
                    keyword_tag_weight_up = gr.Button(
                        "+0,1", size="sm", scale=0, min_width=58, variant="secondary",
                        elem_id="prompt-weight-up",
                    )
                    keyword_tag_limit = gr.Dropdown(
                        choices=list(PROMPT_TAG_SUGGESTION_CHOICES),
                        value=PROMPT_TAG_SUGGESTION_LIMIT,
                        label="Số gợi ý",
                        scale=0,
                        min_width=104,
                        elem_id="prompt-tag-limit",
                    )
                keyword_tag_status = gr.Markdown(
                    f"{tag_catalog_status} Gõ từ khóa cuối như `mắt`/`eyes`; dùng `*đuôi`, "
                    "`*giữa*` để tìm tag theo hậu tố/nội dung, hoặc `@họa sĩ` để lọc nhóm "
                    "Artist. Danh mục: `[G/A/©/C/M]` Danbooru · `<G/A/©/C/S/M/L>` e621. "
                    "Chọn gợi ý để thay cụm cuối; Ctrl+↑/↓ hoặc nút ± chỉnh tag ở con trỏ/đoạn chọn. "
                    "Chỉ hiện N gợi ý hot nhất (chọn N ở ô **Số gợi ý**); xem toàn bộ kết quả ở tab **🏷️ Kho thẻ**.",
                    elem_classes="studio-hint",
                )
                negative = gr.Textbox(
                    label="Negative gửi model · ngón tay / ngón chân",
                    value=DEFAULT_NEGATIVE,
                    lines=2,
                    max_lines=6,
                )
                with gr.Row(equal_height=False):
                    translate_prompt_button = gr.Button(
                        "🇻🇳 → 🇬🇧 Dịch prompt Việt sang English",
                        size="sm",
                        scale=2,
                    )
                    translate_prompt_status = gr.Markdown(
                        "Dịch offline bằng nhãn/tag trong catalog; phần chưa nhận diện được sẽ giữ nguyên để bạn sửa.",
                        elem_classes="studio-hint",
                        scale=3,
                    )
                with gr.Row():
                    eyes_trigger_button = gr.Button(
                        "Thêm trigger `perfect eyes` cho LoRA mắt (sửa/xóa được)",
                        size="sm",
                        scale=2,
                    )
                    gr.HTML(
                        "<p class='studio-hint'>Viết phong cách ngay trong prompt hoặc "
                        "nạp từ tab Thư viện; không có selector phong cách.</p>"
                    )
                with gr.Tabs(selected="create", elem_id="studio-workspace-tabs") as workspace_tabs:
                    with gr.Tab("✦ Tạo ảnh", id="create"):
                        with gr.Tabs(selected="text", elem_id="studio-mode-tabs") as mode_tabs:
                            with gr.Tab("✦ Văn bản → ảnh", id="text"):
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
                            with gr.Tab("◈ Ảnh → ảnh", id="image"):
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
                            with gr.Tab("⤢ Phóng to ảnh", id="upscale"):
                                upscale_source = gr.Image(
                                    label="Ảnh cần phóng to (PNG/JPG/WebP)",
                                    type="pil",
                                    sources=["upload"],
                                    image_mode="RGB",
                                    height=230,
                                )
                                with gr.Row():
                                    upscale_scale = gr.Dropdown(
                                        choices=[k for k in HIRES_SCALES if k != HIRES_OFF],
                                        value="2×",
                                        label="Hệ số phóng",
                                    )
                                    upscale_strength = gr.Slider(
                                        *HIRES_STRENGTH_RANGE,
                                        value=HIRES_DEFAULT_STRENGTH,
                                        step=0.05,
                                        label="Hires strength",
                                    )
                                upscale_button = gr.Button(
                                    "Phóng to ảnh",
                                    variant="primary",
                                    elem_classes="studio-primary",
                                )
                                gr.Markdown(
                                    "Không tạo lại ảnh gốc: Real-ESRGAN 4x+ Anime6B phóng "
                                    "to rồi WAI img2img tinh chỉnh theo prompt hiện tại. "
                                    "Chọn 1.25×, 1.5×, 1.75× hoặc 2×; nếu vượt ≈4,2 MP, "
                                    "hệ số sẽ tự giảm. Lần đầu tải weight ~18 MB; pin SHA từ "
                                    "mirror chưa được đối chiếu độc lập với file chính thức, "
                                    "nên sai hash là tính năng sẽ dừng.",
                                    elem_classes="studio-hint",
                                )
                            with gr.Tab("✎ Sửa vùng ảnh", id="inpaint"):
                                editor = gr.ImageEditor(
                                    label="Tải ảnh vào đây và dùng cọ tô vùng cần sửa",
                                    type="pil",
                                    image_mode="RGBA",
                                    sources=["upload"],
                                    height=380,
                                    format="png",
                                    transforms=(),
                                    interactive=True,
                                    brush=gr.Brush(
                                        default_size=40,
                                        colors=["#ffffff"],
                                        default_color="#ffffff",
                                        color_mode="fixed",
                                    ),
                                    eraser=gr.Eraser(default_size=40),
                                    # Giữ đúng một layer vẽ nhưng không vô hiệu hóa layer
                                    # manager; ImageEditor cần layer đang active để nhận nét cọ.
                                    layers=gr.LayerOptions(allow_additional_layers=False),
                                    elem_id="studio-inpaint-editor",
                                )
                                mask_file = gr.Image(
                                    label="Hoặc tải mask trắng/đen PNG (ưu tiên hơn vùng tô)",
                                    type="pil",
                                    image_mode="L",
                                    sources=["upload"],
                                    height=150,
                                )
                                with gr.Row():
                                    target = gr.Dropdown(
                                        choices=[
                                            (label, key) for key, label in REPAIR_LABELS.items()
                                        ],
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
                                    "được thu về cạnh dài tối đa 1024 px cùng mask (ảnh hires lớn hơn sẽ bị thu nhỏ "
                                    "khi sửa vùng). Chọn "
                                    "tay/chân/mắt không tự thêm từ vào prompt.",
                                    elem_classes="studio-hint",
                                )
                    with gr.Tab("🏷️ Kho thẻ", id="tags"):
                        gr.Markdown(
                            "Nguồn CSV 01/10/2026 · 348.716 tag prompt (đã bỏ metadata). Catalog được nạp và kiểm SHA-256 một lần khi Studio khởi động; nhấn **Tìm trong kho thẻ** để lọc nhanh, không tải lại CSV. "
                            "Tìm tiếng Việt hoặc English trên toàn bộ kho; có thể dán prompt nhiều cụm bằng dấu phẩy/xuống dòng, không cần dấu tiếng Việt. Cụm dài không khớp nguyên văn sẽ tìm theo từng từ khóa. "
                            "Nhãn Việt đứng trước thẻ gốc; tên riêng được giữ nguyên và mục chưa dịch được ghi rõ. "
                            "Khi thêm vào prompt chỉ dùng tên thẻ tiếng Anh gốc. "
                            "Kho có thể chứa thẻ nhạy cảm. Đây là từ khóa, không phải model hay ảnh huấn luyện.",
                            elem_classes="studio-hint",
                        )
                        tag_query = gr.Textbox(
                            label="Tìm prompt tiếng Việt / English trên toàn bộ kho",
                            placeholder="tóc dài, mắt xanh, red dress, blue eyes…",
                            lines=2,
                            max_lines=4,
                        )
                        with gr.Row():
                            tag_category = gr.Dropdown(
                                choices=[("Tất cả danh mục", "")] + [(label, key) for key, label in TAG_CATEGORIES.items()],
                                value="", label="Danh mục",
                            )
                            tag_theme = gr.Dropdown(
                                choices=[("Tất cả chủ đề", "")] + [(name, name) for name in TAG_THEMES] + [("Chưa phân nhóm", "Chưa phân nhóm")],
                                value="", label="Chủ đề",
                            )
                        with gr.Row():
                            tag_sort = gr.Radio(choices=["Phổ biến nhất", "Tên A–Z"], value="Phổ biến nhất", label="Sắp xếp")
                            tag_page = gr.Number(value=1, minimum=1, precision=0, label="Trang (60 thẻ / trang)")
                        tag_search = gr.Button("Tìm trong kho thẻ", size="sm")
                        # Event Thêm thẻ chạy qua queue nên Gradio đôi khi kiểm input
                        # theo một snapshot choices cũ sau khi trang kết quả vừa đổi.
                        # Cho phép nhận giá trị cũ rồi apply_csv_tags vẫn whitelist chính
                        # xác tên canonical trong catalog trước khi ghi vào prompt.
                        tag_selection = gr.Dropdown(choices=[], multiselect=True, label="Chọn thẻ trong trang kết quả", allow_custom_value=True)
                        with gr.Row():
                            tag_destination = gr.Radio(choices=["Prompt", "Negative prompt"], value="Prompt", label="Thêm vào")
                            tag_add = gr.Button("Thêm thẻ đã chọn", size="sm")
                        tag_status = gr.Markdown(tag_catalog_status, elem_classes="studio-hint")
                    with gr.Tab("📚 Thư viện", id="library"):
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
                        prompt_choice = gr.Radio(
                            choices=(),
                            value=None,
                            interactive=False,
                            label="Chọn prompt để nạp (chạm một dòng)",
                            elem_classes="studio-prompt-list",
                        )
                        load_choice_button = gr.Button(
                            "⬇️ Nạp prompt đã chọn vào ô prompt", size="sm"
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
                            "dụng (kẹp về dải của giao diện). Chạm một dòng trong danh "
                            "sách là nạp ngay; trên điện thoại có thể chọn rồi bấm nút "
                            "**⬇️ Nạp prompt đã chọn**.",
                            elem_classes="studio-hint",
                        )
                    with gr.Tab("🧭 Quy trình", id="workflow"):
                        with gr.Row():
                            scaffold_kind = gr.Dropdown(
                                choices=list(SCAFFOLD_CHOICES),
                                value="character",
                                label="Khung prompt theo loại ảnh",
                            )
                            scaffold_button = gr.Button(
                                "Sắp xếp prompt theo thứ tự chuẩn", size="sm"
                            )
                        with gr.Row():
                            negative_choice = gr.Dropdown(
                                choices=list(negative_preset_choices()),
                                value=None,
                                label="Negative tối ưu theo mục đích",
                            )
                            negative_mode = gr.Radio(
                                choices=list(NEGATIVE_MODES),
                                value=NEGATIVE_REPLACE,
                                label="Cách áp dụng",
                            )
                        with gr.Row():
                            negative_apply_button = gr.Button(
                                "Nạp negative đã chọn", size="sm", scale=1
                            )
                            check_button = gr.Button(
                                "🩺 Kiểm tra prompt & thông số", size="sm", scale=1
                            )
                            tag_check_button = gr.Button(
                                "🧪 Kiểm tra thẻ với kho thẻ", size="sm", scale=1
                            )
                        with gr.Row():
                            rewrite_button = gr.Button(
                                "🛠️ Sửa prompt thành thẻ chuẩn", size="sm", scale=1
                            )
                            rewrite_apply_button = gr.Button(
                                "✅ Tạo prompt hoàn chỉnh", size="sm", scale=1,
                                variant="primary",
                            )
                        workflow_status = gr.Markdown(
                            "**Quy trình gợi ý:** 1) sắp xếp prompt theo thứ tự chuẩn → "
                            "2) chọn negative đúng mục đích → 3) kiểm tra prompt/thông số "
                            "+ thẻ với kho thẻ → 4) sửa thẻ, chọn thay thế nếu cần → 5) "
                            "tạo ở ~1 MP, dò 3–4 seed → 6) hires 1.5–2× → 7) inpaint "
                            "vùng tay/mắt còn lỗi. Mọi nút ở đây chỉ ghi nội dung **hiển thị** "
                            "vào hai ô prompt; không có thẻ nào được thêm ngầm.",
                            elem_classes="studio-hint",
                        )
                        prompt_report = gr.Markdown("", elem_classes="studio-hint")
                        tag_check_report = gr.Markdown("", elem_classes="studio-hint")
                        prompt_rewrite_report = gr.Markdown("", elem_classes="studio-hint")
                        prompt_rewrite_choices = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Thẻ thay thế · mục bắt buộc đã chọn sẵn, mục tùy chọn có thể đánh dấu",
                            info=(
                                "Mỗi dòng có dạng thẻ đang dùng → tên canonical trong kho. "
                                "Bỏ chọn hoặc chọn một phương án khác trước khi tạo prompt hoàn chỉnh."
                            ),
                            elem_id="prompt-rewrite-choices",
                        )
                        prompt_rewrite_status = gr.Markdown("", elem_classes="studio-hint")
                        gr.Markdown(
                            "Thứ tự chuẩn: chủ thể → nhãn phân loại → ngoại hình/chi tiết "
                            "nhân vật → trang phục → tư thế → bố cục → bối cảnh → ánh sáng "
                            "→ phong cách → thẻ khác → chất lượng → `absurdres`. Negative "
                            "chia theo mục đích và cố "
                            "ý ngắn: nhà phát hành WAI v17 cảnh báo negative quá dài làm "
                            "giảm chất lượng ảnh.",
                            elem_classes="studio-hint",
                        )
                    with gr.Tab("⚙️ Thông số", id="settings"):
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
                    with gr.Tab("✨ Chi tiết", id="details"):
                        with gr.Group(elem_classes="studio-hires"):
                            gr.Markdown(
                                "### 🔍 Ảnh độ phân giải cao", elem_classes="studio-hint"
                            )
                            with gr.Row():
                                hires_scale = gr.Dropdown(
                                    choices=list(HIRES_SCALES),
                                    value=HIRES_OFF,
                                    label="Độ phân giải cao (hires fix)",
                                )
                                hires_strength = gr.Slider(
                                    *HIRES_STRENGTH_RANGE,
                                    value=HIRES_DEFAULT_STRENGTH,
                                    step=0.05,
                                    label="Hires strength (chi tiết thêm vào)",
                                )
                            gr.Markdown(
                                "Chọn **1.25×, 1.5×, 1.75× hoặc 2×** để ảnh lớn hơn "
                                "kích thước đã chọn (tối đa ≈4,2 MP): Real-ESRGAN Anime6B "
                                "phóng ảnh rồi img2img tinh chỉnh theo prompt. Áp dụng cho "
                                "*Văn bản → ảnh* và *Ảnh → ảnh*; tab *Phóng to ảnh* có "
                                "strength riêng. Lần đầu tải/xác minh weight ~18 MB; pin SHA "
                                "theo metadata mirror chưa được đối chiếu độc lập với file "
                                "chính thức, nên sai hash là tính năng sẽ dừng. Strength thấp "
                                "giữ bố cục, cao thêm chi tiết nhưng dễ đổi nét.",
                                elem_classes="studio-hint",
                            )
                        with gr.Group(elem_classes="studio-detailer"):
                            gr.Markdown(
                                "### 🔎 Tự sửa mặt / bàn tay", elem_classes="studio-hint"
                            )
                            with gr.Row():
                                detailer_target = gr.Dropdown(
                                    choices=list(DETAILER_TARGETS),
                                    value=DETAILER_OFF,
                                    label="Tự sửa mặt/tay (auto-detailer)",
                                )
                                detailer_strength = gr.Slider(
                                    *DETAILER_STRENGTH_RANGE,
                                    value=DETAILER_DEFAULT_STRENGTH,
                                    step=0.05,
                                    label="Detailer strength (mức thay đổi vùng)",
                                )
                            with gr.Row():
                                detailer_conf = gr.Slider(
                                    *DETAILER_CONF_RANGE,
                                    value=DETAILER_DEFAULT_CONF,
                                    step=0.05,
                                    label="Ngưỡng phát hiện (thấp = dễ tìm hơn)",
                                )
                                detailer_max = gr.Slider(
                                    *DETAILER_MAX_RANGE,
                                    value=DETAILER_DEFAULT_MAX,
                                    step=1,
                                    label="Số vùng tối đa sửa mỗi ảnh",
                                )
                            gr.Markdown(
                                "Tự tìm **mặt** và **bàn tay** trong ảnh, cắt từng vùng ra, "
                                "vẽ lại vùng đó bằng chính prompt/negative bạn đang thấy "
                                "(không thêm thẻ ngầm), rồi dán lại với viền mềm. Weight dò "
                                "YOLOv8 (~6 MB/file) tải một lần từ `Bingsu/adetailer`, kiểm "
                                "SHA-256 như checkpoint. Áp dụng cho *Văn bản → ảnh* và *Ảnh "
                                "→ ảnh*; mỗi vùng tốn thêm một lượt inpaint nên hãy bắt đầu "
                                "với 1–2 vùng. Strength thấp giữ nét, cao sửa mạnh nhưng dễ "
                                "lệch nét mặt.",
                                elem_classes="studio-hint",
                            )
                        with gr.Group(elem_classes="studio-look"):
                            gr.Markdown(
                                "### 💅 Chi tiết mắt & móng", elem_classes="studio-hint"
                            )
                            with gr.Row():
                                eye_color = gr.Dropdown(
                                    choices=list(LOOK_OPTIONS["eye_color"]),
                                    value=LOOK_OFF,
                                    label=LOOK_LABELS["eye_color"],
                                )
                                nail_shape = gr.Dropdown(
                                    choices=list(LOOK_OPTIONS["nail_shape"]),
                                    value=LOOK_OFF,
                                    label=LOOK_LABELS["nail_shape"],
                                )
                            with gr.Row():
                                nail_color = gr.Dropdown(
                                    choices=list(LOOK_OPTIONS["nail_color"]),
                                    value=LOOK_OFF,
                                    label=LOOK_LABELS["nail_color"],
                                )
                                toenail_color = gr.Dropdown(
                                    choices=list(LOOK_OPTIONS["toenail_color"]),
                                    value=LOOK_OFF,
                                    label=LOOK_LABELS["toenail_color"],
                                )
                            look_button = gr.Button(
                                "Thêm chi tiết mắt/móng vào prompt đang hiển thị", size="sm"
                            )
                            gr.Markdown(
                                "Chọn màu mắt, **kiểu dáng móng tay** và màu sơn móng tay/móng "
                                "chân rồi bấm nút: các thẻ Danbooru tương ứng (`blue eyes`, "
                                "`almond-shaped nails`, `red nails, nail polish`, `painted "
                                "toenails`…) được **ghi thẳng vào ô *Prompt gửi model*** để bạn "
                                "sửa hoặc xóa — không có thẻ nào được thêm ngầm khi tạo ảnh. "
                                "Bấm lại không tạo thẻ trùng. Móng/mắt chỉ hiện rõ khi tay hoặc "
                                "mặt đủ lớn trong khung: với ảnh cận tay/cận mặt hãy dùng kèm "
                                "*Tự sửa mặt/tay* hoặc tab *Sửa vùng ảnh*; móng chân cần thấy "
                                "bàn chân (khung *Chân / bàn chân*).",
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
                hires = [hires_scale, hires_strength]
                detailer = [
                    detailer_target,
                    detailer_strength,
                    detailer_conf,
                    detailer_max,
                ]
            with gr.Column(
                scale=5,
                min_width=320,
                elem_classes=["studio-panel", "studio-output-column"],
            ):
                gr.Markdown("### 02 · Kết quả", elem_classes="studio-section-heading")
                gr.Markdown(
                    "Ảnh tạo xong xuất hiện tại đây. Chọn ảnh để mở lớn hoặc tải xuống; bấm một "
                    "ảnh cũng là **chỉ định nó làm ảnh nguồn** cho tab *Sửa vùng* / *Phóng to* / "
                    "*Biến đổi*.",
                    elem_classes="studio-section-subtitle",
                )
                gallery = gr.Gallery(
                    label="Ảnh đầu ra · nhấn để xem lớn",
                    columns=2,
                    height=520,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                    elem_classes="studio-gallery",
                )
                status = gr.Markdown(
                    f"Ảnh đầu tiên sẽ xuất hiện tại đây. Model đã nạp trong Google Colab. "
                    f"**Chế độ:** {runtime.execution_mode}"
                )
                with gr.Row():
                    source_choice = gr.Dropdown(
                        label="Ảnh sẽ nạp vào tab sửa / phóng",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                        info="Mặc định là ảnh vừa tạo; bấm ảnh trong thư viện để chọn ảnh đó "
                             "mà không xóa danh sách đã nạp bằng ↻.",
                        scale=4,
                    )
                    source_refresh = gr.Button("↻", scale=1, variant="secondary")
                with gr.Row():
                    to_all = gr.Button(
                        "↪ Nạp ảnh đã chọn vào cả ba tab", size="sm", variant="primary", scale=3
                    )
                    to_image = gr.Button("→ ◈ Biến đổi", size="sm", scale=1)
                    to_upscale = gr.Button("→ ⤢ Phóng to", size="sm", scale=1)
                    to_inpaint = gr.Button("→ ✎ Sửa vùng", size="sm", scale=1)
                downloads = gr.File(
                    label="Tải ảnh PNG", file_count="multiple", interactive=False
                )
                latest = gr.State(None)
                # Danh sách ↻ / mở trang. Bấm ảnh chỉ gộp bộ nhớ này, không quét đĩa.
                source_memory = gr.State([])
                gr.Markdown(
                    "Ảnh chỉ nằm ở `/content/wai_outputs` của phiên Colab (không lưu "
                    "Drive) — **tải xuống trước khi runtime hết hạn**.",
                    elem_classes="studio-hint",
                )

        outputs = [gallery, downloads, status, latest]
        events = (
            (
                text_button,
                runtime.text_to_image,
                [text_size, *shared, *hires, *detailer],
            ),
            (
                image_button,
                runtime.image_to_image,
                [
                    image_source,
                    image_size,
                    image_strength,
                    *shared,
                    *hires,
                    *detailer,
                ],
            ),
            (
                upscale_button,
                runtime.upscale,
                [upscale_source, upscale_scale, upscale_strength, *shared],
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

        def load_selected(path):
            """(ảnh RGB, ảnh RGB cho phóng to, ảnh RGBA cho sửa vùng, dòng trạng thái)."""
            image = selected_source_image(path)
            return (
                image,
                image.copy(),
                editor_value_for(image),
                f"↪ Đã nạp **{Path(path).name}** · {image.width}×{image.height} px",
            )

        def load_into_all(path):
            image, upscale_value, editor_value, note = load_selected(path)
            return image, upscale_value, editor_value, (
                f"{note} → **◈ Biến đổi**, **⤢ Phóng to** và **✎ Sửa vùng** — chọn tab rồi chạy "
                "tiếp, thông số đang giữ nguyên."
            )

        def load_into_image_tab(path):
            image, _, _, note = load_selected(path)
            return image, f"{note} → **◈ Biến đổi**."

        def load_into_upscale_tab(path):
            _, upscale_value, _, note = load_selected(path)
            return upscale_value, f"{note} → **⤢ Phóng to**."

        def load_into_inpaint_tab(path):
            _, _, editor_value, note = load_selected(path)
            return editor_value, f"{note} → **✎ Sửa vùng**."

        # Ảnh vừa tạo xong tự cập nhật vào ô chọn; bấm ảnh trong thư viện là chọn ảnh đó.
        # Hai sự kiện bấm/đổi thư viện chỉ đọc bộ nhớ (giá trị gallery do server gửi về
        # và source_memory) — tuyệt đối không quét đĩa ở đây, vì mỗi lần cập nhật thư viện
        # lại kéo theo một lượt glob/stat trên /content là nguyên nhân khiến trang đứng hình.
        def picker_entries(gallery_value=None, ttl=OUTPUT_SCAN_TTL_SECONDS):
            return source_entries(
                getattr(runtime, "output_dir", None), gallery_value, verify=False, ttl=ttl
            )

        def on_gallery_change(value, remembered, keep):
            fresh = source_entries(None, value, verify=False)
            merged = merge_source_entries(remembered, fresh)
            newest = fresh[0][0] if fresh else None
            return source_picker_update(merged, keep=keep, pick=newest), merged

        def on_gallery_select(value, remembered, data: gr.SelectData):
            # Annotation bắt buộc: Gradio 6 chỉ tiêm SelectData khi tham số là subclass
            # của EventData. Lambda không gắn được hint, nên data không bao giờ tới.
            fresh = source_entries(None, value, verify=False)
            merged = merge_source_entries(remembered, fresh)
            return (
                source_picker_update(
                    merged, pick=tapped_gallery_path(value, getattr(data, "index", None))
                ),
                merged,
            )

        def refresh_source_picker(keep):
            entries = picker_entries(ttl=0)
            return source_picker_update(entries, keep=keep), entries

        def load_source_picker():
            entries = picker_entries(ttl=0)
            return source_picker_update(entries), entries

        gallery.change(
            fn=on_gallery_change,
            inputs=[gallery, source_memory, source_choice],
            outputs=[source_choice, source_memory],
            api_visibility="private",
            queue=False,
        )
        gallery.select(
            fn=on_gallery_select,
            inputs=[gallery, source_memory],
            outputs=[source_choice, source_memory],
            api_visibility="private",
            queue=False,
        )
        # Chỉ nút ↻ mới thật sự đọc thư mục xuất; ttl=0 bỏ cache 5 giây. Nó chạy trong
        # queue (không chiếm slot "wai_gpu") nên bấm lúc đang tạo ảnh vẫn có phản hồi.
        source_refresh.click(
            fn=refresh_source_picker,
            inputs=source_choice,
            outputs=[source_choice, source_memory],
            api_visibility="private",
            show_progress="minimal",
        )
        to_all_event = to_all.click(
            fn=load_into_all,
            inputs=source_choice,
            outputs=[image_source, upscale_source, editor, status],
            api_visibility="private",
        )
        to_image_event = to_image.click(
            fn=load_into_image_tab,
            inputs=source_choice,
            outputs=[image_source, status],
            api_visibility="private",
        )
        to_upscale_event = to_upscale.click(
            fn=load_into_upscale_tab,
            inputs=source_choice,
            outputs=[upscale_source, status],
            api_visibility="private",
        )
        to_inpaint_event = to_inpaint.click(
            fn=load_into_inpaint_tab,
            inputs=source_choice,
            outputs=[editor, status],
            api_visibility="private",
        )
        # Gõ một chủ đề/từ khóa ở cuối prompt để tìm tag English trong catalog CSV;
        # chọn một tag xác thực từ kết quả sẽ thay thế từ khóa ngay.
        # Gõ prompt là sự kiện lặp lại liên tục nên bắt buộc chạy trong queue với
        # trigger_mode="always_last": nếu queue=False, Gradio chạy handler ngay trên
        # event loop và mỗi phím (quét 349k dòng, tới ~1,5 s) chặn toàn bộ HTTP/SSE
        # — trang đứng hẳn cho tới khi xong.
        prompt.input(
            fn=update_keyword_tag_suggestions,
            inputs=[prompt, keyword_tag_limit],
            outputs=[keyword_tag_suggestion, keyword_tag_status],
            api_visibility="private",
            show_progress="minimal",
            trigger_mode="always_last",
        )
        keyword_tag_limit.change(
            fn=update_keyword_tag_suggestions,
            inputs=[prompt, keyword_tag_limit],
            outputs=[keyword_tag_suggestion, keyword_tag_status],
            api_visibility="private",
            show_progress="minimal",
            trigger_mode="always_last",
        )
        keyword_tag_suggestion.input(
            fn=apply_keyword_tag_suggestion_ui,
            inputs=[prompt, keyword_tag_suggestion],
            outputs=[prompt, keyword_tag_status, keyword_tag_suggestion],
            api_visibility="private",
            show_progress="hidden",
        )
        # Dịch hai ô đang hiển thị, không thêm thẻ ngầm lúc tạo ảnh. Catalog đã nạp
        # sẵn nên callback chỉ tra index cục bộ; vẫn qua queue để không chặn event loop.
        translate_prompt_button.click(
            fn=translate_prompts_to_english,
            inputs=[prompt, negative],
            outputs=[prompt, negative, translate_prompt_status],
            api_visibility="private",
            show_progress="minimal",
            concurrency_id=TAG_CATALOG_CONCURRENCY_ID,
            concurrency_limit=TAG_CATALOG_CONCURRENCY_LIMIT,
        )
        keyword_tag_weight_down.click(
            fn=lambda text: adjust_prompt_tag_weight_ui(text, -1),
            inputs=prompt,
            outputs=[prompt, keyword_tag_status, keyword_tag_suggestion],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
            js=PROMPT_TAG_WEIGHT_SELECTION_JS,
        )
        keyword_tag_weight_up.click(
            fn=lambda text: adjust_prompt_tag_weight_ui(text, 1),
            inputs=prompt,
            outputs=[prompt, keyword_tag_status, keyword_tag_suggestion],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
            js=PROMPT_TAG_WEIGHT_SELECTION_JS,
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
        look_button.click(
            fn=apply_look_tags,
            inputs=[prompt, negative, eye_color, nail_shape, nail_color, toenail_color],
            outputs=[prompt, negative],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        # Quy trình chuẩn: các nút đều chỉ ghi nội dung HIỂN THỊ vào hai ô prompt /
        # ô báo cáo, người dùng xem và sửa được trước khi bấm tạo ảnh.
        scaffold_button.click(
            fn=structure_prompt,
            inputs=[prompt, scaffold_kind],
            outputs=[prompt, workflow_status],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        negative_apply_button.click(
            fn=apply_negative_preset,
            inputs=[negative_choice, negative, negative_mode],
            outputs=[negative, workflow_status],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        check_button.click(
            fn=run_prompt_check,
            inputs=[
                prompt,
                negative,
                steps,
                cfg,
                text_size,
                hires_scale,
                hires_strength,
            ],
            outputs=prompt_report,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        # Kiểm tra thẻ với kho thẻ: lần đầu bấm còn phải dựng index tên/alias của
        # ~350k thẻ nên bắt buộc qua queue giống các sự kiện tìm trong tab Kho thẻ —
        # không được chạy trên event loop (xem UiResponsivenessTests).
        tag_check_button.click(
            fn=run_tag_check,
            inputs=[prompt, negative],
            outputs=tag_check_report,
            api_visibility="private",
            show_progress="minimal",
        )
        # Sửa prompt thành thẻ chuẩn cũng dựng index catalog và tìm gợi ý nên chạy
        # trong queue; các thẻ alias/nhãn/gõ sai có gợi ý được chọn sẵn, còn các
        # phương án đổi tùy chọn cho thẻ đã đúng tên thì để người dùng đánh dấu.
        rewrite_button.click(
            fn=run_prompt_rewrite,
            inputs=prompt,
            outputs=[prompt_rewrite_report, prompt_rewrite_choices],
            api_visibility="private",
            show_progress="minimal",
        )
        rewrite_apply_button.click(
            fn=apply_prompt_rewrite,
            inputs=[prompt, prompt_rewrite_choices],
            outputs=[prompt, prompt_rewrite_status],
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )
        # Thư viện prompt: đọc danh sách từ file hoặc đoạn văn bản đã dán, rồi
        # chọn một dòng để nạp thẳng vào ô prompt gửi model.
        library_outputs = [prompt_library_state, prompt_choice, library_status]

        def library_loaded(library):
            return (
                library["items"],
                gr.Radio(**prompt_library_choices(library["items"])),
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
            items, picker, message = prompt_library_reset()
            return items, gr.Radio(**picker), message

        # Đọc file/dán danh sách prompt là thao tác với dữ liệu tới 2 MB — để trong queue
        # để không chặn event loop khi Colab đang chạy suy luận.
        prompt_file.upload(
            fn=load_library_from_file,
            inputs=prompt_file,
            outputs=library_outputs,
            api_visibility="private",
            show_progress="minimal",
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
            show_progress="minimal",
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
        load_choice_button.click(
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
        # Kho thẻ có hàng đợi CPU riêng: tìm/lọc vẫn ở worker (không chặn event loop
        # khi quét 348k dòng), nhưng không phải chờ slot `wai_gpu` đang bận tạo ảnh.
        # `always_last` bỏ các lượt tìm cũ khi người dùng đổi bộ lọc liên tiếp.
        tag_search.click(
            fn=browse_csv_tags,
            inputs=[tag_query, tag_category, tag_theme, tag_sort, tag_page],
            outputs=[tag_selection, tag_page, tag_status],
            api_visibility="private",
            queue=True,
            trigger_mode="always_last",
            concurrency_id=TAG_CATALOG_CONCURRENCY_ID,
            concurrency_limit=TAG_CATALOG_CONCURRENCY_LIMIT,
            show_progress="minimal",
        )
        tag_add.click(
            fn=apply_csv_tags,
            inputs=[prompt, negative, tag_selection, tag_destination],
            outputs=[prompt, negative, tag_status],
            api_visibility="private",
            queue=True,
            concurrency_id=TAG_CATALOG_CONCURRENCY_ID,
            concurrency_limit=TAG_CATALOG_CONCURRENCY_LIMIT,
            show_progress="minimal",
        )
        # Only open a destination tab after the *selected* image really loaded (a missing
        # file raises gr.Error and must not yank the user into an empty tab).
        for event, destination in (
            (to_all_event, "image"),
            (to_image_event, "image"),
            (to_upscale_event, "upscale"),
            (to_inpaint_event, "inpaint"),
        ):
            event.success(
                fn=lambda destination=destination: (
                    gr.update(selected="create"), gr.update(selected=destination)
                ),
                inputs=[], outputs=[workspace_tabs, mode_tabs],
                api_visibility="private", queue=False,
            )
        # Mở lại trang là danh sách ảnh trên đĩa vẫn chọn được (không chỉ ảnh vừa tạo).
        demo.load(
            fn=load_source_picker,
            outputs=[source_choice, source_memory],
            api_visibility="private",
            show_progress="minimal",
        )
        demo.load(fn=None, js=PROMPT_TAG_WEIGHT_SHORTCUT_JS)
        # Giám sát kết nối: trang cho biết đang mất phiên thay vì đứng im vĩnh viễn.
        demo.load(fn=None, js=STUDIO_OFFLINE_WATCHDOG_JS)
        # max_size phải đủ lớn: mỗi lượt tạo ảnh giữ một slot hàng đợi trong nhiều phút,
        # với max_size nhỏ thì mọi thao tác khác bị từ chối (HTTP 429) và giao diện
        # kẹt ở trạng thái chờ — người dùng thấy là trang "treo đơ". Job GPU đã tự
        # giới hạn bằng `wai_gpu`; Kho thẻ có nhóm `wai_tag_catalog` riêng, còn lại
        # mặc định 1 event/sự kiện.
        demo.queue(max_size=64, default_concurrency_limit=1, api_open=False)
    instrument_ui_events(demo)
    # Gradio 6 applies CSS and themes at launch, not in the Blocks constructor.
    demo.studio_theme = gr.themes.Soft(
        primary_hue="purple", secondary_hue="pink", neutral_hue="slate"
    )
    demo.studio_css = css
    return demo
