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
    prepare += '\nif not WAI_STUDIO_VERSION_VERIFIED:\n    raise RuntimeError("Studio chỉ nhận checkpoint WAI-illustrious v17 đúng SHA-256. Đổi MODEL_PATH sang bản gốc, rồi chạy lại ô 4.")\n'
    setup[3]["source"] = prepare.splitlines(keepends=True)

    introduction = """# WAI Studio · tạo ảnh anime trên Google Colab bằng liên kết tạm thời

> **Không cần Google Drive, Cloudflare, tài khoản hay API token.** Colab chạy checkpoint WAI-illustrious v17 + LoRA đã kiểm SHA-256; Gradio chỉ hiển thị giao diện. Ảnh mẫu web Cloudflare không liên quan tới model này.

### Chạy trong 3 bước
1. Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU mạnh hơn), rồi **Runtime → Run all**. Checkpoint ~6,94 GB và tối đa ~457 MB LoRA được tải **trực tiếp vào `/content`**, kiểm tra SHA-256 trước khi nạp, không gắn hay sao chép sang Drive. Nếu file còn trong cùng runtime sẽ dùng lại; phiên Colab mới phải tải lại. Cần ~9 GiB đĩa trống. Ô 4 kiểm toàn bộ hash, có thể mất một lúc.
2. Chờ ô cuối in `Running on public URL`, mở liên kết `https://....gradio.live` — **không cần tài khoản/mật khẩu**. Chọn **Anime chuẩn**, **Bán thực 2.5D**, **Tùy chỉnh** hoặc **Anime NSFW 18+** (chỉ nhân vật trưởng thành, cần xác nhận riêng). Phong cách và ý tưởng gốc **điền** hai ô prompt/negative **gửi model có thể sửa trực tiếp**; bấm tạo sẽ gửi chính xác nội dung hiện tại, không tự thêm thẻ ẩn. Đổi preset/ý tưởng gốc/LoRA mắt sẽ ghi đè chỉnh sửa trong hai ô đó; dùng nút **Áp dụng lại phong cách** nếu muốn reset. Tab sửa vùng có nút **Thêm gợi ý sửa vùng vào prompt**: bấm để xem, sửa hoặc xóa gợi ý trước khi tạo.
3. **Giữ Colab kết nối.** Link chỉ tồn tại khi phiên Colab/Gradio còn chạy; dừng runtime để ngắt link. Ảnh chỉ nằm tại `/content/wai_outputs` hoặc đường dẫn cục bộ đã đặt ở ô 3: **tải PNG về trước khi phiên hết**, nếu không sẽ mất cả model, LoRA và ảnh. Lần sau Run all để có link mới.

**Bảo mật:** `share=True` tạo URL Internet *không có đăng nhập*. Bất kỳ ai biết URL đều có thể dùng GPU Colab của bạn; không chia sẻ URL hoặc notebook có output chứa URL ở nơi công khai. File checkpoint/LoRA bị chặn khỏi đường tải file Gradio; URL không phải cơ chế xác thực. [Tài liệu share link](https://www.gradio.app/guides/understanding-gradio-share-links). Metadata prompt trong PNG chỉ được nhúng khi bạn bật tùy chọn tương ứng.

Preset chỉ điều hướng cùng **một** checkpoint WAI v17, không phải model 2.5D chuyên dụng. Negative mặc định nhắm lỗi ngón tay/ngón chân; preset thường điền `nsfw, explicit` vào negative, preset **Anime NSFW 18+** thì không, nhưng vẫn yêu cầu xác nhận 18+ và từ chối một số từ khóa trẻ em trong prompt gửi model (không phải bộ lọc hoàn chỉnh). LoRA mắt có gợi ý `perfect eyes` trong prompt khi áp dụng preset và bật LoRA; bạn có thể sửa/xóa từ đó. Inpaint dùng checkpoint SDXL hiện có, tô vùng trắng, ghép để giữ pixel đen; không bảo đảm sửa được mọi lỗi. LoRA phải chọn và tải ở ô 3–6 trước khi mở giao diện; trong UI có thể tắt/bật và đổi cường độ **các LoRA đã nạp**. Tuân thủ giấy phép model và điều khoản Colab/Gradio.

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
            "**Khi gặp lỗi:** nếu ô 2 chỉ hiện dòng `ERROR: pip's dependency resolver...`, hãy xem ô đó có in `✅ Thư viện Studio đã sẵn sàng` không; riêng dòng này có thể là cảnh báo không chặn cài đặt. Nếu không có dấu ✅ hoặc có traceback, mở lại notebook mới nhất, chọn *Runtime → Restart runtime → Run all* và gửi đầy đủ traceback nếu vẫn lỗi (che link Gradio và thông tin riêng). Nếu không có GPU, chọn GPU trong Runtime; nếu RAM gần đầy nhưng VRAM còn trống, để `VRAM_MODE=auto` rồi xem VRAM và `Chế độ sau khi nạp` ở ô 6 (CPU offload vẫn dùng GPU từng phần). Nếu OOM, giảm kích thước ảnh; nếu cần, chỉnh `VRAM_MODE=low_vram` hoặc tắt LoRA ở ô 3 rồi khởi động lại runtime. Nếu không tạo được URL, kiểm tra kết nối Colab/Gradio và chạy lại ô 8. **Link không có đăng nhập; đừng chia sẻ.** Notebook cơ sở và hash tài nguyên xem [README của dự án](https://github.com/manhlee1196-boop/ai-anime/tree/arena/01a0d84b-ai-anime).",
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
