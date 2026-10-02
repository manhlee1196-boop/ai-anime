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
# Hires fix: SDXL chỉ học tốt quanh ~1 MP, nên ảnh lớn hơn được tạo theo 2 bước —
# tạo ở kích thước gốc rồi phóng to (Lanczos) và tinh chỉnh bằng ảnh → ảnh ở
# denoise thấp để thêm chi tiết thật thay vì chỉ làm mờ/nhòe.
HIRES_OFF = "Tắt"
HIRES_SCALES = {HIRES_OFF: 1.0, "1.5×": 1.5, "2×": 2.0}
HIRES_MAX_PIXELS = 4_200_000  # ≈ 2048×2048; giới hạn để không tràn VRAM/RAM Colab
HIRES_STRENGTH_RANGE = (0.2, 0.7)
HIRES_DEFAULT_STRENGTH = 0.4
# Prompt mẫu viết theo THỨ TỰ CHUẨN của dân chuyên nghiệp (chất lượng → chủ thể
# → ngoại hình → tư thế → bối cảnh → ánh sáng → phong cách → độ nét). Bạn sửa/xóa
# tùy ý; Studio gửi đúng nội dung hai ô prompt/negative cho model.
DEFAULT_PROMPT = (
    "masterpiece, best quality, amazing quality, 1girl, solo, adult woman, "
    "long dark hair, gentle smile, standing under cherry blossoms, petals falling, "
    "spring, soft sunlight, cel shading, anime illustration, absurdres"
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
# bộ kiểm tra prompt/thông số).
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
QUALITY_HEAD = ("masterpiece", "best quality", "amazing quality")
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

# Thứ tự thẻ chuẩn: thẻ đứng trước được CLIP chú ý nhiều hơn, nên chất lượng và
# chủ thể đi đầu, bối cảnh/ánh sáng/phong cách đi sau, thẻ độ nét chốt cuối.
PROMPT_SECTIONS = (
    "quality",
    "rating",
    "subject",
    "appearance",
    "outfit",
    "pose",
    "extra",
    "composition",
    "background",
    "lighting",
    "style",
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
        r"\bshort hair\b|\bblue eyes\b|\bbody\b|\bproportions\b|\bnails?\b|\bfingernails?\b|\btoenails?\b|\bnail polish\b|\bnail art\b|\bpupils?\b|\bheterochromia\b|\beyelashes\b|\beyebrows\b|\birides\b|\biris\b)",
        # Móng và chi tiết mắt được xếp vào nhóm Ngoại hình thay vì 'extra'.
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
            "quality": QUALITY_HEAD,
            "subject": ("1girl", "solo"),
            "style": ("anime illustration", "cel shading"),
            "tail": QUALITY_TAIL,
        },
    },
    "portrait": {
        "label": "Chân dung cận mặt",
        "anchors": {
            "quality": QUALITY_HEAD,
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
            "quality": QUALITY_HEAD,
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
            "quality": QUALITY_HEAD,
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
    """Bỏ cú pháp nhấn mạnh ((thẻ), [thẻ], (thẻ:1.2)) để so khớp nội dung thẻ."""
    core = tag.strip()
    match = WEIGHT_RE.match(core)
    if match:
        core = match.group("body").strip()
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
    """Xếp một thẻ vào nhóm trong PROMPT_SECTIONS (không khớp → 'extra')."""
    core = tag_core(tag)
    for section, pattern in _SECTION_RES:
        if pattern.search(core):
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
        "thứ tự: chất lượng → chủ thể → ngoại hình → trang phục → tư thế → bố cục → "
        "bối cảnh → ánh sáng → phong cách → độ nét."
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
                "Chưa có thẻ chất lượng. WAI v17 khuyến nghị mở đầu bằng "
                "`masterpiece, best quality, amazing quality` (không cần nhiều hơn).",
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
        """Upscale a finished base image, then let img2img add real detail."""
        from PIL import Image

        # VAE tiling giữ đỉnh VRAM thấp khi giải mã ảnh > 1 MP (pipe SDXL thật có
        # hàm này; dùng chung VAE với pipeline ảnh → ảnh nên bật một lần là đủ).
        enable_tiling = getattr(self.pipe, "enable_vae_tiling", None)
        if callable(enable_tiling):
            enable_tiling()
        upscaled = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
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
                raise ValueError("Chọn hệ số phóng 1.5× hoặc 2×.")
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
        """Phóng to một ảnh có sẵn (Lanczos + ảnh → ảnh), không tạo lại từ đầu."""
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
TAG_CSV_SHA256 = "287bb5ad86fcc56f535b9ebae8d3e696e5ff3fa5c057d6fa6884282fb083e738"
TAG_CSV_URL = (
    "https://raw.githubusercontent.com/manhlee1196-boop/ai-anime/"
    "0a0d3b87a4f7fa77da3274674c4d89649f7c3657/" + TAG_CSV_NAME
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
_TAG_ROWS = None
_TAG_LOCK = threading.Lock()
TAG_PAGE_SIZE = 60


def normalize_csv_tag(value):
    return re.sub(r"[_\s]+", " ", value.strip().lower())


def parse_tag_csv(text):
    import csv
    import io

    patterns = [(name, re.compile(pattern)) for name, pattern in TAG_THEMES.items()]
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) != 4:
            continue
        name, category, count, aliases = fields
        if not name or category not in TAG_CATEGORIES or not count.isdigit():
            continue
        themes = tuple(label for label, pattern in patterns if pattern.search(name))
        rows.append((name, category, int(count), normalize_csv_tag(name + "," + aliases), themes))
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
        candidates = [Path(TAG_CSV_NAME), Path("/content") / TAG_CSV_NAME]
        data = None
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size <= 12_000_000:
                content = candidate.read_bytes()
                if hashlib.sha256(content).hexdigest() == TAG_CSV_SHA256:
                    data = content
                    break
        if data is None:
            with urllib.request.urlopen(TAG_CSV_URL, timeout=45) as response:
                data = response.read(12_000_001)
            if len(data) > 12_000_000 or hashlib.sha256(data).hexdigest() != TAG_CSV_SHA256:
                raise ValueError("CSV tải về không khớp SHA-256 đã xác minh.")
        rows = parse_tag_csv(data.decode("utf-8-sig"))
        if Path("/content").is_dir():
            (Path("/content") / TAG_CSV_NAME).write_bytes(data)
        _TAG_ROWS = rows
        return rows


def search_csv_tags(rows, query="", category="", theme="", sort="Phổ biến nhất", page=1):
    query = normalize_csv_tag(query or "")
    found = [row for row in rows if
             (not query or query in row[3]) and
             (not category or row[1] == category) and
             (not theme or (not row[4] if theme == "Chưa phân nhóm" else theme in row[4]))]
    found.sort(key=(lambda row: row[0]) if sort == "Tên A–Z" else (lambda row: (-row[2], row[0])))
    pages = max(1, math.ceil(len(found) / TAG_PAGE_SIZE))
    page = min(pages, max(1, int(page or 1)))
    return found[(page - 1) * TAG_PAGE_SIZE:page * TAG_PAGE_SIZE], len(found), page, pages


def browse_csv_tags(query, category, theme, sort, page):
    import gradio as gr

    try:
        rows = load_csv_tags()
        found, total, current, pages = search_csv_tags(rows, query, category, theme, sort, page)
        choices = [(f"{row[0]} · {TAG_CATEGORIES[row[1]]} · {row[2]:,} lượt", row[0]) for row in found]
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


def apply_csv_tags(positive, negative, selected, destination):
    if not selected:
        return positive, negative, "Hãy tìm và chọn ít nhất một thẻ."
    # Only canonical names in the verified catalog may be inserted.
    valid = {row[0] for row in load_csv_tags()}
    selected = list(dict.fromkeys(name for name in selected if name in valid))
    current = negative if destination == "Negative prompt" else positive
    existing = {normalize_csv_tag(tag) for tag in split_tags(current or "")}
    additions = [name for name in selected if normalize_csv_tag(name) not in existing]
    result = _add_prompt_tags(current or "", additions)
    if destination == "Negative prompt":
        negative = result
    else:
        positive = result
    return positive, negative, f"Đã thêm {len(additions)} thẻ vào {destination}; bỏ qua thẻ trùng. Bạn có thể sửa/xóa trực tiếp trong ô prompt."


def build_app(runtime):
    """Build Gradio Blocks without opening a public tunnel until launch cell runs."""
    import gradio as gr
    from PIL import Image

    css = """
    #wai-studio {max-width: 1220px !important; margin: auto !important;}
    #wai-studio .form {gap: 0.5rem;}
    #wai-studio .block {border-radius: 10px;}
    #wai-studio span[data-testid="block-label"], #wai-studio .label-text {font-size: 0.82rem;}
    .studio-hero {display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px;
        padding: 10px 16px; border-radius: 12px; color: #fff;
        background: linear-gradient(118deg, #21182f, #382641 64%, #704065);
        box-shadow: 0 6px 20px #140f2026;}
    .studio-hero h1 {color: #fff !important; font-size: 1.12rem; font-weight: 700; margin: 0;}
    .studio-hero p {color: #e4d2e1; font-size: 0.8rem; margin: 0;}
    .studio-badge {font-size: 0.6rem; letter-spacing: 0.12rem; font-weight: bold; color: #f4b3dc;}
    .studio-warn {flex-basis: 100%; font-size: 0.74rem; color: #f2d5e9;
        border-top: 1px solid #ffffff2b; padding-top: 5px; margin-top: 2px;}
    .studio-hint, .studio-hint p {font-size: 0.74rem !important; line-height: 1.35;
        color: #7a7386; margin: 0 !important;}
    .studio-hires {border: 1px solid #b9a3e0; border-radius: 10px; background: #f7f3fd;
        padding: 6px 10px;}
    /* Keep both tab bars on one horizontal, touch-scrollable row. */
    #studio-workspace-tabs > .tab-nav, #studio-mode-tabs > .tab-nav,
    #studio-workspace-tabs > [role="tablist"], #studio-mode-tabs > [role="tablist"] {
        display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 4px;
        scrollbar-width: thin; -webkit-overflow-scrolling: touch;
        padding: 4px 0 8px; max-width: 100%;
    }
    #studio-workspace-tabs > .tab-nav button, #studio-mode-tabs > .tab-nav button,
    #studio-workspace-tabs > [role="tablist"] button, #studio-mode-tabs > [role="tablist"] button {
        flex: 0 0 auto; white-space: nowrap; min-height: 44px; border-radius: 9px;
    }
    #studio-workspace-tabs button[role="tab"][aria-selected="true"] {
        background: #704065; color: #fff; font-weight: 700;
    }
    #studio-workspace-tabs, #studio-mode-tabs {min-width: 0;}
    @media (max-width: 640px) {
        #wai-studio {padding: 8px !important;}
        .studio-hero {padding: 10px;}
    }
    /* Danh sách prompt: cuộn được trong khung, mỗi dòng là một ô chạm cao. */
    .studio-prompt-list {max-height: 44vh; overflow-y: auto; -webkit-overflow-scrolling: touch;
        border: 1px solid #e2dced; border-radius: 10px; padding: 4px 8px; background: #fbfaff;}
    .studio-prompt-list label {min-height: 42px; font-size: 0.86rem; line-height: 1.3;
        display: flex; align-items: center; padding: 2px 0;}
    .studio-primary button {min-height: 42px; font-weight: 600;}
    """
    with gr.Blocks(
        title="WAI Studio · Colab GPU",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
        elem_id="wai-studio",
    ) as demo:
        gr.HTML(
            "<div class='studio-hero'>"
            "<span class='studio-badge'>✦ WAI · COLAB GPU</span>"
            "<h1>Biến ý tưởng thành thế giới anime</h1>"
            "<p>WAI-illustrious v17 · LoRA tay/chân/mắt đã xác minh</p>"
            "<span class='studio-warn'>Link tạm thời <b>không cần đăng nhập</b>: ai có "
            "link đều dùng được GPU Colab của bạn — đừng chia sẻ, dừng runtime để thu "
            "hồi. Ảnh lưu dưới /content.</span>"
            "</div>"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=300):
                prompt = gr.Textbox(
                    label="Prompt gửi model · tự viết phong cách của bạn",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    max_lines=8,
                    placeholder=(
                        "Mô tả nhân vật, trang phục, khung cảnh, ánh sáng và phong cách "
                        "vẽ bạn muốn (anime illustration, cel shading, watercolor...)"
                    ),
                )
                negative = gr.Textbox(
                    label="Negative gửi model · ngón tay / ngón chân",
                    value=DEFAULT_NEGATIVE,
                    lines=2,
                    max_lines=6,
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
                                    "Không tạo lại ảnh: phóng to bằng Lanczos rồi tinh chỉnh "
                                    "bằng ảnh → ảnh theo prompt hiện tại (nên mô tả đúng nội "
                                    "dung ảnh). Kết quả tối đa ≈4,2 MP; ảnh đã lớn hơn sẽ bị "
                                    "giảm hệ số hoặc từ chối.",
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
                                    brush=gr.Brush(
                                        default_size=40, colors=["#ffffff"], color_mode="fixed"
                                    ),
                                    eraser=gr.Eraser(default_size=40),
                                    layers=False,
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
                            "Nguồn CSV 01/10/2026 · 349.714 thẻ. Nhấn **Tìm / tải kho thẻ** để nạp lần đầu (~9 MB). "
                            "Chủ đề được nhóm tự động theo tên, có thể chồng lặp; danh mục giữ theo nguồn. "
                            "Kho có thể chứa thẻ nhạy cảm. Đây là từ khóa, không phải model hay ảnh huấn luyện.",
                            elem_classes="studio-hint",
                        )
                        tag_query = gr.Textbox(label="Tìm tên thẻ hoặc bí danh", placeholder="long hair, smile…")
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
                        tag_search = gr.Button("Tìm / tải kho thẻ", size="sm")
                        tag_selection = gr.Dropdown(choices=[], multiselect=True, label="Chọn thẻ trong trang kết quả", allow_custom_value=False)
                        with gr.Row():
                            tag_destination = gr.Radio(choices=["Prompt", "Negative prompt"], value="Prompt", label="Thêm vào")
                            tag_add = gr.Button("Thêm thẻ đã chọn", size="sm")
                        tag_status = gr.Markdown("Kho thẻ chưa được tải.", elem_classes="studio-hint")
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
                        workflow_status = gr.Markdown(
                            "**Quy trình gợi ý:** 1) sắp xếp prompt theo thứ tự chuẩn → "
                            "2) chọn negative đúng mục đích → 3) kiểm tra prompt/thông số → "
                            "4) tạo ở ~1 MP, dò 3–4 seed → 5) hires 1.5–2× → 6) inpaint "
                            "vùng tay/mắt còn lỗi. Mọi nút ở đây chỉ ghi nội dung **hiển "
                            "thị** vào hai ô prompt; không có thẻ nào được thêm ngầm.",
                            elem_classes="studio-hint",
                        )
                        prompt_report = gr.Markdown("", elem_classes="studio-hint")
                        gr.Markdown(
                            "Thứ tự chuẩn: chất lượng → nhãn phân loại → chủ thể → ngoại "
                            "hình → trang phục → tư thế → bố cục → bối cảnh → ánh sáng → "
                            "phong cách → `absurdres`. Negative chia theo mục đích và cố "
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
                                "Chọn **1.5× hoặc 2×** để ảnh ra lớn hơn kích thước đã chọn "
                                "(tối đa ≈4,2 MP, ví dụ 1024×1024 → 2048×2048): ảnh được tạo, "
                                "phóng to rồi tinh chỉnh để thêm chi tiết. Áp dụng cho *Văn "
                                "bản → ảnh* và *Ảnh → ảnh* (tab *Phóng to ảnh* có cài đặt "
                                "riêng); tốn thêm thời gian và VRAM. Strength thấp giữ bố "
                                "cục, cao thêm chi tiết nhưng dễ đổi nét.",
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
            with gr.Column(scale=5, min_width=300):
                gallery = gr.Gallery(
                    label="Kết quả · nhấn để xem lớn",
                    columns=2,
                    height=560,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                )
                status = gr.Markdown(
                    f"Ảnh đầu tiên sẽ xuất hiện tại đây. Model đã nạp trong Google Colab. "
                    f"**Chế độ:** {runtime.execution_mode}"
                )
                with gr.Row():
                    to_image = gr.Button(
                        "Dùng ảnh mới nhất để biến đổi", size="sm", scale=1
                    )
                    to_upscale = gr.Button(
                        "Dùng ảnh mới nhất để phóng to", size="sm", scale=1
                    )
                    to_inpaint = gr.Button(
                        "Dùng ảnh mới nhất để sửa vùng", size="sm", scale=1
                    )
                downloads = gr.File(
                    label="Tải ảnh PNG", file_count="multiple", interactive=False
                )
                latest = gr.State(None)
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

        def load_last(path):
            if not path or not Path(path).is_file():
                raise gr.Error("Hãy tạo ít nhất một ảnh trước.")
            return Image.open(path).convert("RGB")

        def edit_last(path):
            image = load_last(path).convert("RGBA")
            return {"background": image, "layers": [], "composite": image}

        to_image_event = to_image.click(
            fn=load_last,
            inputs=latest,
            outputs=image_source,
            api_visibility="private",
        )
        to_upscale_event = to_upscale.click(
            fn=load_last,
            inputs=latest,
            outputs=upscale_source,
            api_visibility="private",
        )
        to_inpaint_event = to_inpaint.click(
            fn=edit_last, inputs=latest, outputs=editor, api_visibility="private"
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
        # Quy trình chuẩn: ba nút đều chỉ ghi nội dung HIỂN THỊ vào hai ô prompt /
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
        tag_search.click(
            fn=browse_csv_tags,
            inputs=[tag_query, tag_category, tag_theme, tag_sort, tag_page],
            outputs=[tag_selection, tag_page, tag_status],
            api_visibility="private",
        )
        tag_add.click(
            fn=apply_csv_tags,
            inputs=[prompt, negative, tag_selection, tag_destination],
            outputs=[prompt, negative, tag_status],
            api_visibility="private", queue=False,
        )
        # Open the destination only when loading the latest image succeeded.
        for event, destination in (
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
        demo.queue(max_size=4, default_concurrency_limit=1, api_open=False)
    # Gradio 6 applies CSS and themes at launch, not in the Blocks constructor.
    demo.studio_theme = gr.themes.Soft(
        primary_hue="purple", secondary_hue="pink", neutral_hue="slate"
    )
    demo.studio_css = css
    return demo
