"""Regenerate the self-contained, one-click Colab studio from verified setup cells.

Run from the repo root: python scripts/build_colab_studio.py
The generated notebook does not fetch or execute repository Python files at runtime.
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "WAI_Illustrious_Colab.ipynb"
MODULE = ROOT / "colab" / "studio.py"
OUTPUT = ROOT / "WAI_Illustrious_Studio_Colab.ipynb"


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
        "source": text.splitlines(keepends=True),
    }


def build():
    original = json.loads(BASE.read_text(encoding="utf-8"))
    setup = copy.deepcopy(original["cells"][1:7])
    install = "".join(setup[1]["source"])
    assert install.count("%pip -q install") == 1 and "gradio==" not in install
    setup[1]["source"] = (
        install.rstrip()
        + ' "gradio==6.15.2" "pydantic>=2.12.5,<3" "starlette>=1.3.1,<2"\n'
        + """
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
    "gradio": "==6.15.2",
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
# Bắt lỗi import trước khi tải checkpoint 6,94 GB ở ô 4.
try:
    from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting
except Exception as exc:
    raise RuntimeError("Diffusers không import được. Chọn Runtime → Restart runtime rồi Run all; nếu vẫn lỗi, gửi traceback ô 2 (che thông tin riêng).") from exc
print("✅ Thư viện Studio đã sẵn sàng:", versions)
"""
    ).splitlines(keepends=True)

    # The original notebook allows a custom SDXL checkpoint. This personal WAI
    # studio is advertised as *v17*, so fail early if a supplied file does not
    # match the official release. Reuse hashes already computed by cell 4; do
    # not read the 6.94 GB file a second time just to show the interface.
    prepare = "".join(setup[3]["source"])

    def insert_once(old, new):
        nonlocal prepare
        assert prepare.count(old) == 1, old
        prepare = prepare.replace(old, new)

    insert_once(
        "if source_model.exists() and not source_model.is_file():",
        "WAI_STUDIO_VERSION_VERIFIED = False\nif source_model.exists() and not source_model.is_file():",
    )
    insert_once(
        "        if checkpoint_hash == HF_SHA256:\n            print(",
        "        if checkpoint_hash == HF_SHA256:\n            WAI_STUDIO_VERSION_VERIFIED = True\n            print(",
    )
    insert_once(
        "        inspect_checkpoint(downloaded, verify_official=True)\n        checkpoint = downloaded",
        "        inspect_checkpoint(downloaded, verify_official=True)\n        WAI_STUDIO_VERSION_VERIFIED = True\n        checkpoint = downloaded",
    )
    insert_once(
        "            inspect_checkpoint(downloaded, verify_official=True)\n            if downloaded != source_model:",
        "            inspect_checkpoint(downloaded, verify_official=True)\n            WAI_STUDIO_VERSION_VERIFIED = True\n            if downloaded != source_model:",
    )
    prepare += '\nif not WAI_STUDIO_VERSION_VERIFIED:\n    raise RuntimeError("Studio chỉ nhận checkpoint WAI-illustrious v17 đúng SHA-256. Đổi MODEL_PATH sang bản gốc, rồi chạy lại ô 4.")\n'
    setup[3]["source"] = prepare.splitlines(keepends=True)

    introduction = """# WAI Studio · tạo ảnh anime từ Google Colab bằng liên kết tạm thời

> **Không cần Cloudflare, API token hay máy chủ GPU khác.** Colab chạy checkpoint WAI-illustrious v17 + LoRA đã kiểm SHA-256; Gradio chỉ hiển thị giao diện. Ảnh mẫu web Cloudflare không liên quan tới model này.

### Chạy trong 3 bước
1. Trong Colab chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU mạnh hơn). Bấm **Runtime → Run all**. Nếu được hỏi, cho phép gắn Google Drive để lưu checkpoint/ảnh cho các phiên sau. Lần đầu cần tải ~6,94 GB model + tối đa ~457 MB LoRA và có thể mất một lúc; những lần sau dùng lại các file đã xác minh.
2. Ở ô cuối, chờ dòng `Running on public URL` rồi mở liên kết `https://....gradio.live` — **không cần tài khoản hay mật khẩu**. Chọn **Anime chuẩn**, **Bán thực 2.5D** hoặc **Tùy chỉnh**, rồi dùng giao diện để tạo ảnh, ảnh → ảnh, tô mask sửa tay/chân/mắt và tải PNG.
3. **Giữ notebook Colab đang kết nối.** Link này chỉ tồn tại khi phiên Colab/Gradio còn chạy (có thể hết hạn sớm khi runtime bị ngắt). Đóng phiên bằng cách dừng runtime; lần sau chạy notebook để có link mới. Nếu dùng Drive, ảnh nằm ở `MyDrive/AI/outputs`; nếu không lưu được Drive, hãy tải từ giao diện hoặc `/content/wai_outputs` trước khi hết phiên.

**Lưu ý bảo mật:** `share=True` tạo URL *truy cập được từ Internet* qua proxy Gradio và **không có đăng nhập**. Bất kỳ ai biết URL đều có thể dùng giao diện và GPU Colab của bạn. Không chia sẻ URL hoặc lưu notebook có output chứa URL ở nơi công khai; dừng runtime để ngắt link. File checkpoint/LoRA bị chặn khỏi đường tải file Gradio, nhưng URL không phải cơ chế xác thực. Gradio nhận dữ liệu qua đường hầm TLS; CPU/GPU và checkpoint vẫn chạy trên Colab. [Giải thích share link](https://www.gradio.app/guides/understanding-gradio-share-links). Ảnh có thể chứa thông tin trong metadata nếu bạn tự bật tùy chọn này.

Hai phong cách chỉ bổ sung từ khóa trên cùng checkpoint WAI v17; Bán thực 2.5D là minh họa lai anime, không phải ảnh chụp hay model khác. Negative mặc định nhắm lỗi ngón tay/ngón chân thừa, thiếu hoặc dính, nhưng **không bảo đảm** sửa đúng mọi ảnh; hãy inpaint vùng nhỏ nếu cần. Chỉ bật LoRA đã chọn trước khi nạp model (ô 3–6). Trong giao diện có thể tắt/bật và đổi cường độ **các LoRA đã nạp** mà không cần tải lại. Inpainting từ checkpoint SDXL gốc định hướng vùng trắng, ghép lại để giữ pixel đen. Colab không bảo đảm GPU liên tục hay đủ RAM/ổ đĩa. Để đổi đường dẫn, chế độ bộ nhớ hoặc danh sách LoRA, hãy **khởi động lại runtime và Run all**; không chạy lại riêng ô nạp model khi giao diện còn giữ pipeline cũ.
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
    drive_root=drive_root,
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
    str((studio_runtime.drive_root / "MyDrive/AI/models").resolve()),
    str((studio_runtime.drive_root / "MyDrive/AI/loras").resolve()),
    "/content/wai_model_cache", "/content/wai_lora_cache",
    *(str(Path(path).resolve()) for path in studio_runtime.lora_paths.values()),
]
_, _, share_url = studio_app.launch(
    share=True, inline=False, prevent_thread_lock=True,
    server_name="127.0.0.1", max_file_size="12mb", footer_links=[],
    theme=studio_app.studio_theme, css=studio_app.studio_css,
    allowed_paths=allowed_outputs, blocked_paths=blocked_weights,
    enable_monitoring=False, show_error=True,
)
if not share_url:
    studio_app.close()
    raise RuntimeError("Gradio chưa tạo được link. Kiểm tra mạng Colab rồi chạy lại ô 8.")
print("Mở link:", share_url)
print("Không cần tài khoản/mật khẩu. Ai có link đều có thể dùng GPU Colab của bạn.")
print("Link ngừng hoạt động khi Colab dừng/ngắt. KHÔNG chia sẻ link; dừng runtime để thu hồi.")
"""

    cells = [
        markdown(introduction, "studio-intro"),
        *setup,
        code(ui, "studio-ui"),
        code(launch, "studio-launch"),
        markdown(
            "**Khi gặp lỗi:** nếu ô 2 chỉ hiện dòng `ERROR: pip's dependency resolver...`, hãy xem ô đó có in `✅ Thư viện Studio đã sẵn sàng` không; riêng dòng này có thể là cảnh báo không chặn cài đặt. Nếu không có dấu ✅ hoặc có traceback, mở lại notebook mới nhất, chọn *Runtime → Restart runtime → Run all* và gửi đầy đủ traceback nếu vẫn lỗi (che link Gradio và thông tin riêng). Nếu không có GPU, chọn GPU trong Runtime; nếu OOM, chỉnh `VRAM_MODE=low_vram` hoặc tắt LoRA ở ô 3 rồi khởi động lại runtime. Nếu không tạo được URL, kiểm tra kết nối Colab/Gradio và chạy lại ô 8. **Link không có đăng nhập; đừng chia sẻ.** Notebook cơ sở và hash tài nguyên xem [README của dự án](https://github.com/manhlee1196-boop/ai-anime/tree/arena/01a0d84b-ai-anime).",
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
