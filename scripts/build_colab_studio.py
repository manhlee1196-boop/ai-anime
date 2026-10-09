"""Regenerate the self-contained, one-click Colab studio from verified setup cells.

Run from the repo root: python scripts/build_colab_studio.py
The generated notebook does not fetch or execute repository Python files at runtime.
"""

import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "WAI_Illustrious_Colab.ipynb"
MODULE = ROOT / "colab" / "studio.py"
OUTPUT = ROOT / "WAI_Illustrious_Studio_Colab.ipynb"

# Colab thu gọn ô code thành một thanh tiêu đề (kèm nút Run) khi dòng đầu là
# `# @title ... { display-mode: "form" }`; code vẫn xem được bằng biểu tượng
# ">_" ở góc ô. Nhờ vậy notebook đọc như một bảng 8 bước, không phơi cả
# nghìn dòng mã của ô giao diện.
FORM_MARKER = ' { display-mode: "form" }'
TITLE_LINE = re.compile(r"^#\s*@title\s+(.+?)\s*$")


def as_form(text):
    """Ensure a cell's `# @title` line collapses the cell into a compact form."""
    lines = text.split("\n")
    title = TITLE_LINE.match(lines[0])
    if not title:
        raise AssertionError(
            "Ô code phải bắt đầu bằng '# @title ...' thì mới thu gọn được."
        )
    if "display-mode" not in title.group(1):
        lines[0] = f"# @title {title.group(1)}{FORM_MARKER}"
    return "\n".join(lines)


def markdown(text, cell_id):
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": text.splitlines(keepends=True),
    }


def code(text, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id},
        "outputs": [],
        "source": as_form(text).splitlines(keepends=True),
    }


STUDIO_INSTALL_CELL = """# @title 2. Tự cài thư viện còn thiếu (giữ PyTorch/CUDA của Colab) { display-mode: "form" }
# Gradio 6.11–6.15.2 khóa trình duyệt khi bấm tab (trang “không phản hồi”).
# 6.17.3 có bản vá đó, vẫn dùng gradio-client 2.5.0 và huggingface-hub 0.36.2.
%pip -q install "diffusers==0.35.2" "transformers==4.52.4" "accelerate==1.10.1" "peft==0.17.1" "safetensors>=0.4.5,<1" "huggingface-hub==0.36.2" "hf-xet>=1.1.3,<2" "gradio==6.17.3" "pydantic>=2.12.5,<3" "starlette>=1.3.1,<2"

# ultralytics chỉ phục vụ tính năng *tự sửa mặt/tay* (auto-detailer) nên cài
# riêng và chỉ cảnh báo khi thất bại: thiếu nó thì Studio vẫn chạy bình thường,
# riêng auto-detailer báo lỗi rõ ràng lúc được bật.
try:
    get_ipython().run_line_magic("pip", '-q install "ultralytics==8.4.170"')
except Exception as exc:
    print("⚠️ Chưa cài được ultralytics nên auto-detailer sẽ tắt:", exc)

from importlib.metadata import PackageNotFoundError, version
from packaging.specifiers import SpecifierSet
import gradio, gradio_client, pydantic, starlette, huggingface_hub

versions = {
    "gradio": gradio.__version__,
    "gradio-client": gradio_client.__version__,
    "pydantic": pydantic.__version__,
    "starlette": starlette.__version__,
    "huggingface-hub": huggingface_hub.__version__,
}
for package in ("diffusers", "transformers", "accelerate", "peft", "safetensors", "hf-xet"):
    try:
        versions[package] = version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Thiếu {package}. Cài đặt ô 2 chưa hoàn tất; xem lỗi pip ở phía trên.") from exc
required = {
    "gradio": "==6.17.3",
    "gradio-client": "==2.5.0",
    "pydantic": ">=2.12.5,<3",
    "starlette": ">=1.3.1,<2",
    "huggingface-hub": "==0.36.2",
    "diffusers": "==0.35.2",
    "transformers": "==4.52.4",
    "accelerate": "==1.10.1",
    "peft": "==0.17.1",
    "safetensors": ">=0.4.5,<1",
    "hf-xet": ">=1.1.3,<2",
}
for package, constraint in required.items():
    if versions[package] not in SpecifierSet(constraint):
        raise RuntimeError(f"{package} đang là {versions[package]}, cần {constraint}. Chọn Runtime → Restart runtime rồi Run all.")
# Tính năng tuỳ chọn: lệch phiên bản thì chỉ nhắc, không chặn Studio.
optional = {"ultralytics": ">=8.4,<9"}
for package, constraint in optional.items():
    try:
        installed = version(package)
    except PackageNotFoundError:
        print(f"⚠️ Không có {package}: auto-detailer sẽ báo lỗi khi được bật, các chế độ khác vẫn chạy.")
        continue
    if installed not in SpecifierSet(constraint):
        print(f"⚠️ {package} đang là {installed}, khác {constraint}: auto-detailer có thể lỗi. Có thể để tính năng này ở mức Tắt.")
    versions[package] = installed
# Bắt lỗi import trước khi tải checkpoint 6,94 GB ở ô 4.
try:
    from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting
except Exception as exc:
    raise RuntimeError("Diffusers không import được. Chọn Runtime → Restart runtime rồi Run all; nếu vẫn lỗi, gửi traceback ô 2 (che thông tin riêng).") from exc
print("✅ Thư viện Studio đã sẵn sàng:", versions)
"""


def build():
    # Some repository snapshots keep only the self-contained Studio notebook;
    # in that case, reuse its six verified setup cells as the regeneration base.
    using_legacy_base = BASE.is_file()
    source = BASE if using_legacy_base else OUTPUT
    if not source.is_file():
        raise FileNotFoundError(f"Thiếu notebook nguồn: {BASE} và {OUTPUT}")
    original = json.loads(source.read_text(encoding="utf-8"))
    setup = copy.deepcopy(original["cells"][1:7])
    install = "".join(setup[1]["source"])
    if using_legacy_base:
        assert install.count("%pip -q install") == 1 and "gradio==" not in install
    setup[1]["source"] = STUDIO_INSTALL_CELL.splitlines(keepends=True)

    # The original notebook allows a custom SDXL checkpoint. This personal WAI
    # studio is advertised as *v17*, so fail early if a supplied file does not
    # match the official release. Reuse hashes already computed by cell 4; do
    # not read the 6.94 GB file a second time just to show the interface.
    prepare = "".join(setup[3]["source"])

    def insert_once(old, new):
        nonlocal prepare
        if new in prepare:
            return
        assert prepare.count(old) == 1, old
        prepare = prepare.replace(old, new)

    insert_once(
        "if source_model.exists() and not source_model.is_file():",
        "WAI_STUDIO_VERSION_VERIFIED = False\nif source_model.exists() and not source_model.is_file():",
    )
    insert_once(
        "        checkpoint_hash = HF_SHA256\n        print(",
        "        checkpoint_hash = HF_SHA256\n        WAI_STUDIO_VERSION_VERIFIED = True\n        print(",
    )
    insert_once(
        "        inspect_checkpoint(saved_model, verify_official=True)\n        checkpoint = saved_model",
        "        inspect_checkpoint(saved_model, verify_official=True)\n        WAI_STUDIO_VERSION_VERIFIED = True\n        checkpoint = saved_model",
    )
    insert_once(
        "        inspect_checkpoint(downloaded, verify_official=True)\n        checkpoint = downloaded",
        "        inspect_checkpoint(downloaded, verify_official=True)\n        WAI_STUDIO_VERSION_VERIFIED = True\n        checkpoint = downloaded",
    )
    if "if not WAI_STUDIO_VERSION_VERIFIED:" not in prepare:
        prepare += '\nif not WAI_STUDIO_VERSION_VERIFIED:\n    raise RuntimeError("Studio chỉ nhận checkpoint WAI-illustrious v17 đúng SHA-256. Đổi MODEL_PATH sang bản gốc, rồi chạy lại ô 4.")\n'
    setup[3]["source"] = prepare.splitlines(keepends=True)

    introduction = """# WAI Studio · tạo ảnh anime trên Google Colab bằng liên kết tạm thời

> **Không cần Google Drive, tài khoản hay API token.** Link Gradio ở ô 8 không cần Cloudflare; ô 9 dùng Cloudflare Quick Tunnel tùy chọn làm link dự phòng. Colab chạy checkpoint WAI-illustrious v17 + LoRA đã kiểm SHA-256; Gradio chỉ hiển thị giao diện.

### Chạy trong 3 bước

> **Notebook đã thu gọn:** mỗi ô code chỉ hiện một thanh tiêu đề kèm nút **Run** (Colab form), kể cả ô 7 chứa toàn bộ mã giao diện và ô 9 mở giao diện dự phòng — nên trang rất ngắn và dễ chạy tuần tự. Muốn xem hoặc sửa code của ô nào, bấm biểu tượng `>_` (hay ⋮ → *Show code*) ở góc ô đó; kết quả in ra vẫn hiển thị bình thường.
1. Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU mạnh hơn), rồi **Runtime → Run all**. Checkpoint ~6,94 GB và tối đa ~457 MB LoRA được tải **trực tiếp vào `/content`**, kiểm tra SHA-256 trước khi nạp, không gắn hay sao chép sang Drive. Nếu file còn trong cùng runtime sẽ dùng lại; phiên Colab mới phải tải lại. Cần ~9 GiB đĩa trống. Ô 4 kiểm toàn bộ hash, có thể mất một lúc.
2. Ô 8 mở Gradio trên `127.0.0.1:7860` (mặc định **không** bật Gradio Share `gradio.live`, vì mọi request — kể cả ảnh — phải đi qua máy chủ trung gian công cộng, hay nghẽn và kẹt hẳn giao diện), rồi **ô 9 tạo URL tạm `https://….trycloudflare.com` — đường khuyến nghị để mở Studio, không cần tài khoản/mật khẩu**. Muốn dùng link `gradio.live` làm đường chính thì đặt `GRADIO_SHARE = True` ở đầu ô 8. Ô 9 chạy tự động khi chọn **Runtime → Run all**, tải cloudflared một lần rồi lưu binary trong `/content`; giữ runtime hoạt động. Link `trycloudflare.com` cũng là URL công khai, không có đăng nhập — đừng chia sẻ. **Không có selector phong cách: bạn tự viết phong cách ngay trong prompt** (`anime illustration, cel shading`, `watercolor`, `cinematic lighting`…). Khi Studio khởi động ở ô 8, CSV tag được đọc/tải và xác minh SHA-256 một lần; gõ từ khóa chủ đề ở cuối prompt (ví dụ `mắt`/`eyes`) sẽ tìm tag tiếng Anh canonical trong catalog dùng chung với tab Kho thẻ. Chọn một tag để thay cụm cuối; nếu tải catalog thất bại, tạo ảnh vẫn hoạt động và tab Kho thẻ có thể thử lại. Hai ô **Prompt gửi model** / **Negative gửi model** là đúng những gì được gửi ở cả ba chế độ, không thêm thẻ ẩn theo LoRA hay vùng sửa khi bấm tạo. Cần trigger cho LoRA mắt thì bấm nút **Thêm `perfect eyes`**; tab sửa vùng có nút **Thêm gợi ý sửa vùng vào prompt** — cả hai đều hiển thị trong ô để bạn sửa hoặc xóa trước khi tạo.
   - **Nạp nhiều prompt một lúc:** mở accordion **📚 Thư viện prompt · nạp danh sách từ file text**, tải file `.txt`/`.md`/`.json` (hoặc dán nội dung) có dạng `PROMPT 01 - Tên tiếng Việt` rồi đoạn prompt bên dưới (cũng đọc được bảng `Tên tiếng Việt | Nội dung prompts tiếng Anh`, JSON `[{"title","prompt"}]`, hoặc các đoạn prompt cách nhau dòng trống). Danh sách hiện ra dưới dạng **danh sách chạm cuộn được** (dễ dùng trên điện thoại) và **chạm một dòng là hệ thống tự nạp prompt** vào ô *Prompt gửi model*, hoặc chọn dòng rồi bấm **⬇️ Nạp prompt đã chọn**; các dòng `Negative:`, `Steps:`, `CFG:`, `Size:`, `Seed:` trong file cũng được áp dụng. Chưa có file thì bấm **Nạp thư viện mẫu** để xem định dạng.
   - **Làm theo quy trình chuyên nghiệp:** accordion **🧭 Quy trình chuẩn · khung prompt + negative tối ưu** có bốn nút. **Sắp xếp prompt theo thứ tự chuẩn** xếp lại thẻ của bạn theo thứ tự nút đang dùng (chủ thể → nhãn phân loại → ngoại hình → trang phục → tư thế → bố cục → bối cảnh → ánh sáng → phong cách → thẻ khác → chất lượng → `absurdres`), bỏ thẻ trùng và thêm thẻ neo còn thiếu của khung đã chọn. **Nạp negative đã chọn** đưa một trong 8 bộ negative tối ưu theo mục đích (chuẩn nhà phát hành WAI v17, Illustrious chuẩn, tay/chân, giữ chất 2D, chân dung, phong cảnh, an toàn nội dung, inpaint) vào ô negative — ghi đè hoặc nối thêm. **🩺 Kiểm tra prompt & thông số** báo prompt ≈ bao nhiêu token so với khối 75 token của SDXL, thẻ chất lượng thừa, thẻ trùng, thẻ vừa dương vừa âm, cú pháp Pony, trọng số quá 1.2, negative quá dài và steps/CFG/kích thước/hires ngoài khuyến nghị. **🧪 Kiểm tra thẻ với kho thẻ** đối chiếu từng thẻ trong hai ô prompt/negative với kho CSV Danbooru + e621 đã xác minh SHA-256: thẻ nào đúng tên trong kho, thẻ nào là alias/nhãn tiếng Việt của một thẻ khác, thẻ nào không có trong kho (kèm gợi ý sửa chính tả). Cả bốn nút **chỉ ghi nội dung hiển thị** vào hai ô prompt/ô báo cáo để bạn sửa; không có thẻ nào được thêm ngầm khi tạo ảnh. Quy trình đầy đủ: [docs/QUY_TRINH_TAO_ANH.md](https://github.com/manhlee1196-boop/ai-anime/blob/main/docs/QUY_TRINH_TAO_ANH.md).
**Hires fix / phóng to ảnh:** tab **✨ Chi tiết** có `1.25×`, `1.5×`, `1.75×`, `2×`; Real-ESRGAN 4x+ Anime6B (tải khoảng 18 MB khi dùng lần đầu) chạy theo tile rồi WAI img2img tinh chỉnh. Kết quả tối đa ≈4,2 MP; tùy chọn scale tự giảm khi cần. SHA-256 đang ghim theo metadata mirror Hugging Face nhưng **chưa được đối chiếu độc lập với asset GitHub chính thức** do lỗi TLS khi tải trong sandbox; file sai hash sẽ bị từ chối, chế độ tạo ảnh thường không hires vẫn dùng được. Xem `VERIFICATION.md` để biết giới hạn.

3. **Giữ Colab kết nối.** Link chỉ tồn tại khi phiên Colab/Gradio còn chạy; dừng runtime để ngắt link. Ảnh chỉ nằm tại `/content/wai_outputs` hoặc đường dẫn cục bộ đã đặt ở ô 3: **tải PNG về trước khi phiên hết**, nếu không sẽ mất cả model, LoRA và ảnh. Lần sau Run all để có link mới.

**Bảo mật:** link `gradio.live` và URL Cloudflare Quick Tunnel (`trycloudflare.com`) đều là URL Internet *không có đăng nhập*. Bất kỳ ai biết link đều có thể dùng GPU Colab của bạn; không chia sẻ URL hoặc notebook có output chứa URL ở nơi công khai. File checkpoint/LoRA bị chặn khỏi đường tải file Gradio; URL không phải cơ chế xác thực. [Tài liệu share link](https://www.gradio.app/guides/understanding-gradio-share-links). Metadata prompt trong PNG chỉ được nhúng khi bạn bật tùy chọn tương ứng.

**RAM gần đầy nhưng VRAM còn trống?** `VRAM_MODE=auto` ở ô 3 ưu tiên GPU trực tiếp khi VRAM trống lúc nạp ≥ 12,5 GiB + 0,4 GiB/LoRA (bật cả hai: 13,3 GiB); thiếu VRAM/OOM mới thử CPU offload. Ô 6 và UI cho biết chế độ thực tế. CPU offload vẫn tính toán từng phần trên GPU, nhưng tốn RAM hệ thống. Nếu OOM, giảm kích thước ảnh, chọn `low_vram` hoặc tắt LoRA rồi **Restart runtime → Run all**. GPU trống khi không tạo ảnh là bình thường. Đổi đường dẫn, chế độ bộ nhớ hoặc danh sách LoRA cũng cần restart và Run all để không giữ pipeline cũ.
"""
    ui = (
        """# @title 7. Chuẩn bị giao diện WAI bằng model đã xác minh
"""
        + MODULE.read_text(encoding="utf-8")
        + """
if "pipe" not in globals():
    raise RuntimeError("Chưa có pipeline. Chạy các ô 1–6 theo thứ tự.")
studio_runtime = StudioRuntime(
    torch=torch, pipe=pipe, create_pipeline=create_pipeline,
    checkpoint=checkpoint, lora_paths=lora_paths, lora_manifest=lora_manifest,
    vram_mode=VRAM_MODE, use_offload=use_offload, output_dir=output_dir,
)
del pipe  # runtime owns the only reference, allowing OOM recovery to free VRAM
print("Đã chuẩn bị WAI Studio. Ô 8 sẽ tạo liên kết giao diện không cần đăng nhập.")
"""
    )
    launch = """# @title 8. Mở liên kết giao diện tạm thời (không cần đăng nhập)
from pathlib import Path

if "studio_runtime" not in globals():
    raise RuntimeError("Chưa nạp model. Chạy các ô 1–7 trước khi tạo link.")
# Đóng tunnel cũ khi cần tạo lại link, nhưng không nạp lại model.
if "studio_app" in globals():
    studio_app.close()
studio_app = build_app(studio_runtime)
allowed_outputs = [str(studio_runtime.output_dir.resolve()), str(studio_runtime.backup_dir.resolve())]
blocked_weights = [
    str(studio_runtime.checkpoint.resolve()),
    str(local_cache_root.resolve()), str(local_lora_cache.resolve()),
    *(str(Path(path).resolve()) for path in studio_runtime.lora_paths.values()),
]
# Mặc định KHÔNG dùng Gradio Share (gradio.live): mọi request — kể cả ảnh — phải đi
# qua thiết bị công cộng trung gian của gradio.live, hay nghẽn khi Colab đang tải và
# là nguồn gây "treo hẳn, phải tải lại". Ô 9 mở Cloudflare Quick Tunnel thẳng tới
# Gradio đang chạy, đó là đường khuyến nghị. Đặt GRADIO_SHARE = True nếu muốn dùng
# link gradio.live làm đường chính.
GRADIO_SHARE = globals().get("GRADIO_SHARE", False)
_, studio_local_url, share_url = studio_app.launch(
    share=GRADIO_SHARE, inline=False, prevent_thread_lock=True,
    server_name="127.0.0.1", max_file_size="12mb", footer_links=[],
    theme=studio_app.studio_theme, css=studio_app.studio_css,
    allowed_paths=allowed_outputs, blocked_paths=blocked_weights,
    enable_monitoring=False, show_error=True,
)
if GRADIO_SHARE and not share_url:
    print("⚠️ Gradio Share chưa tạo được link công khai; giao diện nội bộ vẫn chạy. Ô 9 sẽ tạo link Cloudflare Quick Tunnel.")
elif GRADIO_SHARE:
    print("Mở link:", share_url)
    print("Không cần tài khoản/mật khẩu. Ai có link đều có thể dùng GPU Colab của bạn.")
else:
    print("↪ Bỏ qua Gradio Share (mặc định) để tránh nghẽn qua máy chủ trung gian gradio.live.")
    print("↪ Chạy tiếp Ô 9 để lấy link https://…trycloudflare.com tới đúng giao diện này.")
    print("↪ Cần link gradio.live thì đặt GRADIO_SHARE = True ở đầu ô này rồi chạy lại.")
print("Link ngừng hoạt động khi Colab dừng/ngắt. KHÔNG chia sẻ link; dừng runtime để thu hồi.")
"""
    quick_tunnel = """# @title 9. Tạo link Cloudflare Quick Tunnel (dự phòng Gradio Live)
import platform
import re
import subprocess
import time
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from IPython.display import HTML, display

if "studio_app" not in globals() or not globals().get("studio_local_url"):
    raise RuntimeError("Chạy ô 8 trước để khởi động giao diện WAI Studio.")


def _stop_cloudflare_tunnel(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


# Nếu chạy lại ô này, thu hồi tunnel cũ trước khi kiểm tra/tạo URL mới.
_stop_cloudflare_tunnel(globals().get("cloudflare_tunnel_process"))
cloudflare_tunnel_process = None
cloudflare_tunnel_url = None

local_url = studio_local_url.rstrip("/")
try:
    local_parts = urlparse(local_url)
    local_port = local_parts.port
except ValueError as exc:
    raise RuntimeError(f"URL Gradio nội bộ không hợp lệ: {studio_local_url!r}.") from exc
if (local_parts.scheme != "http"
        or local_parts.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not local_port):
    raise RuntimeError(f"Không nhận ra URL Gradio nội bộ: {studio_local_url!r}.")
# Bỏ qua HTTP_PROXY của Colab khi kiểm tra localhost; nếu không, yêu cầu có
# thể bị gửi nhầm qua proxy Internet và báo timeout dù Gradio vẫn đang chạy.
try:
    with build_opener(ProxyHandler({})).open(local_url + "/config", timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"Gradio trả HTTP {response.status} ở giao diện nội bộ.")
except Exception as exc:
    raise RuntimeError(
        "Không kết nối được Gradio nội bộ. Kiểm tra ô 8; chưa khởi động Cloudflare tunnel."
    ) from exc

machine = platform.machine().lower()
architecture = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
if architecture is None:
    raise RuntimeError(f"cloudflared chưa hỗ trợ kiến trúc Colab này: {machine}.")
cloudflared_path = Path("/content/cloudflared")
download_url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{architecture}"

# Tải bản nhị phân chính chủ một lần vào /content; các lần chạy sau dùng cache.
cloudflared_ready = False
if cloudflared_path.is_file() and cloudflared_path.stat().st_size > 1_000_000:
    cloudflared_path.chmod(0o755)
    try:
        version_check = subprocess.run(
            [str(cloudflared_path), "--version"], capture_output=True,
            text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        version_check = None
    cloudflared_ready = version_check is not None and version_check.returncode == 0
if not cloudflared_ready:
    download_path = cloudflared_path.with_name("cloudflared.download")
    print("Đang tải cloudflared từ bản phát hành chính thức của Cloudflare…")
    try:
        subprocess.run(
            ["wget", "-q", "-O", str(download_path), download_url],
            check=True, timeout=180,
        )
        if not download_path.is_file() or download_path.stat().st_size < 1_000_000:
            raise RuntimeError("File tải về quá nhỏ hoặc không phải binary cloudflared.")
        download_path.chmod(0o755)
        download_path.replace(cloudflared_path)
    except Exception as exc:
        download_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Không tải được cloudflared. Kiểm tra Internet của Colab rồi chạy lại ô 9."
        ) from exc
    try:
        version_check = subprocess.run(
            [str(cloudflared_path), "--version"], capture_output=True, text=True,
            timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        cloudflared_path.unlink(missing_ok=True)
        raise RuntimeError("Bản cloudflared tải về không chạy được; chạy lại ô 9 để thử tải lại.") from exc
cloudflared_path.chmod(0o755)
version_output = (version_check.stdout or version_check.stderr).strip()
print("cloudflared:", version_output.splitlines()[0] if version_output else "binary sẵn sàng")

cloudflare_log_path = Path("/tmp/wai-studio-cloudflared.log")
command = [
    str(cloudflared_path), "tunnel", "--no-autoupdate", "--protocol", "http2",
    "--url", local_url,
]
with cloudflare_log_path.open("w", encoding="utf-8") as tunnel_log:
    cloudflare_tunnel_process = subprocess.Popen(
        command, stdout=tunnel_log, stderr=subprocess.STDOUT, start_new_session=True
    )
cloudflare_tunnel_log_path = str(cloudflare_log_path)

# URL Quick Tunnel được cloudflared in ra trong log. Chờ tối đa 90 giây và
# dừng tiến trình nếu Cloudflare không cấp được URL.
deadline = time.monotonic() + 90
log_text = ""
try:
    while time.monotonic() < deadline:
        if cloudflare_tunnel_process.poll() is not None:
            break
        log_text = cloudflare_log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log_text, re.IGNORECASE)
        if matches and cloudflare_tunnel_process.poll() is None:
            cloudflare_tunnel_url = matches[-1]
            break
        time.sleep(0.5)
except KeyboardInterrupt:
    _stop_cloudflare_tunnel(cloudflare_tunnel_process)
    cloudflare_tunnel_process = None
    raise

if not cloudflare_tunnel_url:
    log_text = cloudflare_log_path.read_text(encoding="utf-8", errors="replace")
    log_tail = "\\n".join(log_text.splitlines()[-20:])[-3000:]
    _stop_cloudflare_tunnel(cloudflare_tunnel_process)
    cloudflare_tunnel_process = None
    raise RuntimeError(
        "Cloudflare chưa tạo được URL trycloudflare.com trong 90 giây. "
        "Kiểm tra mạng hoặc log bên dưới rồi chạy lại ô 9.\\n" + log_tail
    )

print("✅ Mở WAI Studio qua Cloudflare Quick Tunnel:", cloudflare_tunnel_url)
print("Không cần tài khoản/API token. URL công khai, không có đăng nhập — đừng chia sẻ.")
print("Giữ Colab hoạt động. Chạy lại ô 9 để đổi URL; dừng tunnel: _stop_cloudflare_tunnel(cloudflare_tunnel_process).")
display(HTML(
    f'<p><a href="{escape(cloudflare_tunnel_url, quote=True)}" target="_blank" '
    'rel="noopener">Mở WAI Studio trong tab mới</a></p>'
))
"""

    def replace_setup(index, old, new, label):
        """Vá ô setup một lần. build() đọc lại notebook vừa ghi, nên lần sau phải no-op."""
        current = "".join(setup[index]["source"])
        if new in current:
            return
        if old not in current:
            raise AssertionError(
                f"Không tìm thấy đoạn cần vá ở ô setup[{index}] ({label})"
            )
        setup[index]["source"] = current.replace(old, new, 1).splitlines(keepends=True)

    replace_setup(
        3,
        'print("Checkpoint có sẵn: chỉ kiểm tra định dạng SDXL, KHÔNG xác thực là WAI v17; LoRA chỉ hợp với họ Illustrious.")',
        'print("Checkpoint có sẵn không trùng SHA-256 của WAI v17. Studio sẽ dừng, không nạp file khác.")',
        "thông báo checkpoint có sẵn",
    )
    replace_setup(
        4,
        '    except Exception as exc:\n'
        '        raise ValueError(f"Không đọc được header LoRA: {path}") from exc',
        '    except ValueError:\n'
        '        raise\n'
        '    except Exception as exc:\n'
        '        raise ValueError(f"Không đọc được header LoRA: {path}") from exc',
        "inspect_lora không nuốt ValueError",
    )
    replace_setup(
        5,
        'if "pipe" in globals():\n'
        '    del pipe\n'
        '    gc.collect()\n'
        '    torch.cuda.empty_cache()',
        '# Chạy lại ô này không được giữ pipeline cũ trong studio_runtime, nếu không\n'
        '# Colab nạp hai checkpoint cùng lúc và hết VRAM. Ô 7 đã xóa tên `pipe`.\n'
        'if "studio_app" in globals():\n'
        '    try:\n'
        '        studio_app.close()\n'
        '    except Exception:\n'
        '        pass\n'
        '    del studio_app\n'
        'if "studio_runtime" in globals():\n'
        '    try:\n'
        '        studio_runtime.pipe = None\n'
        '    except Exception:\n'
        '        pass\n'
        '    del studio_runtime\n'
        'if "pipe" in globals():\n'
        '    del pipe\n'
        'gc.collect()\n'
        'if "torch" in globals() and torch.cuda.is_available():\n'
        '    torch.cuda.empty_cache()',
        "nhả studio_runtime trước khi nạp lại",
    )
    replace_setup(
        5,
        'print("Model đã sẵn sàng. Chạy ô 7 để tạo ảnh; ô 8 sửa vùng lỗi nếu cần.")',
        'print("Model đã sẵn sàng. Chạy tiếp ô 7, rồi ô 8 và ô 9 để mở Studio. Tạo ảnh và sửa vùng nằm trong giao diện, không phải trong ô này.")',
        "hướng dẫn ô sau khi nạp",
    )

    # Thu gọn mọi ô code thành form: notebook có 9 thanh tiêu đề + nút Run.
    for cell in setup:
        if cell["cell_type"] == "code":
            cell["source"] = (
                as_form("".join(cell["source"])).splitlines(keepends=True)
            )

    cells = [
        markdown(introduction, "studio-intro"),
        *setup,
        code(ui, "studio-ui"),
        code(launch, "studio-launch"),
        code(quick_tunnel, "studio-cloudflare-tunnel"),
        markdown(
            "**Khi gặp lỗi:** nếu ô 2 chỉ hiện dòng `ERROR: pip's dependency resolver...`, hãy xem ô đó có in `✅ Thư viện Studio đã sẵn sàng` không; riêng dòng này có thể là cảnh báo không chặn cài đặt. Nếu không có dấu ✅ hoặc có traceback, mở lại notebook mới nhất, chọn *Runtime → Restart runtime → Run all* và gửi đầy đủ traceback nếu vẫn lỗi (che link Gradio và thông tin riêng). Nếu không có GPU, chọn GPU trong Runtime; nếu RAM gần đầy nhưng VRAM còn trống, để `VRAM_MODE=auto` rồi xem VRAM và `Chế độ sau khi nạp` ở ô 6 (CPU offload vẫn dùng GPU từng phần). Nếu OOM, giảm kích thước ảnh; nếu cần, chỉnh `VRAM_MODE=low_vram` hoặc tắt LoRA ở ô 3 rồi khởi động lại runtime. Nếu **bấm ✦ Tạo ảnh / 🏷️ Kho thẻ / 📚 Thư viện mà trình duyệt báo trang không phản hồi**: đó là lỗi Gradio 6.15.2. **Runtime → Restart runtime → Run all** để ô 2 cài Gradio 6.17.3; chỉ chạy lại ô 8 hoặc ô 9 thì trang vẫn đơ. Nếu **giao diện đứng im và phải tải lại trang**: thường là đường hầm bị ngắt giữa lượt tạo (một lượt có thể chạy nhiều phút) — trang sẽ hiện dải “Mất kết nối với phiên Colab” kèm nút **Tải lại trang**, ảnh đã tạo vẫn nằm trong `/content/wai_outputs` và trong danh sách tải. Nếu trang chỉ chậm, dải riêng “Trang bị chặn … s” hiện lên — đó không phải mất kết nối và không có nút tải lại; đừng tải lại chỉ vì dòng đó. Chạy lại ô 9 nếu link `trycloudflare.com` chết hẳn; không cần nạp lại model. Nếu bật `GRADIO_SHARE = True` mà `gradio.live` báo 504/nghẽn, để mặc định và dùng ô 9. Cả hai link đều công khai và không có đăng nhập — đừng chia sẻ. Notebook cơ sở và hash tài nguyên xem [README của dự án](https://github.com/manhlee1196-boop/ai-anime/blob/main/README.md).",
            "studio-help",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    OUTPUT.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Generated", OUTPUT)
