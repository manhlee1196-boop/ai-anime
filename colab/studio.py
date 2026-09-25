"""Personal WAI-illustrious Colab studio. Inlined into the standalone notebook.

The notebook's existing setup cells verify checkpoint and LoRA hashes before this
module is used. The model stays in Colab: Gradio only provides an authenticated UI.
"""

import gc
import json
import math
import os
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
DEFAULT_PROMPT = (
    "general, 1girl, solo, cherry blossoms, spring, soft sunlight, "
    "detailed eyes, anime illustration, masterpiece, best quality"
)
DEFAULT_NEGATIVE = "nsfw, explicit, lowres, worst quality, bad anatomy, blurry"
REPAIR_HINTS = {
    "hands": (
        "natural hands, correct number of fingers, detailed fingers",
        "extra fingers, missing fingers, fused fingers, deformed hands",
    ),
    "legs": (
        "natural leg anatomy, well-formed feet, balanced pose",
        "extra legs, broken legs, deformed feet, extra toes",
    ),
    "eyes": (
        "symmetrical eyes, detailed irises, perfect eyes",
        "misaligned eyes, deformed eyes, extra eyes",
    ),
    "custom": ("", ""),
}


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
        drive_root,
        backup_dir="/content/wai_outputs",
    ):
        if pipe is None or not Path(checkpoint).is_file():
            raise RuntimeError(
                "Model chưa sẵn sàng; chạy lại các ô chuẩn bị checkpoint và nạp model."
            )
        self.torch = torch
        self.pipe = pipe
        self.create_pipeline = create_pipeline
        self.checkpoint = Path(checkpoint)
        self.lora_paths = dict(lora_paths)
        self.lora_manifest = dict(lora_manifest)
        self.vram_mode = vram_mode
        self.use_offload = bool(use_offload)
        self.output_dir = Path(output_dir)
        self.drive_root = Path(drive_root)
        self.backup_dir = Path(backup_dir)
        self.lock = threading.Lock()

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
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError("Prompt phải có từ 1 đến 2000 ký tự.")
        if not isinstance(negative, str) or len(negative) > 1500:
            raise ValueError("Negative prompt tối đa 1500 ký tự.")
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
        positive = prompt.strip()
        if loras["eyes"][0] and "perfect eyes" not in positive.lower():
            positive += ", perfect eyes"
        return positive, negative.strip(), steps, cfg, seed, count, loras

    def _apply_loras(self, choices):
        if self.lora_paths:
            names = list(self.lora_paths)
            # Weight zero disables a previously loaded adapter without deleting
            # the verified file or reloading the 6.94 GB checkpoint.
            weights = [choices[name][1] if choices[name][0] else 0.0 for name in names]
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
            if (
                self.output_dir == self.drive_root
                or self.drive_root in self.output_dir.parents
            ) and not (self.drive_root / "MyDrive").is_dir():
                raise OSError("Google Drive đã ngắt kết nối")
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
            hint_pos, hint_neg = REPAIR_HINTS[target]
            if hint_pos:
                positive += ", " + hint_pos
            if hint_neg:
                negative = ", ".join(part for part in (negative, hint_neg) if part)
        else:
            raise ValueError("Chế độ tạo ảnh không được hỗ trợ.")
        if len(positive) > 2200 or len(negative) > 1700:
            raise ValueError("Prompt quá dài sau khi thêm gợi ý; hãy rút ngắn.")

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
        status = f"✅ Đã tạo {len(paths)} ảnh · seed: {', '.join(selected)} · đã lưu: {Path(paths[0]).parent}"
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
    .gradio-container {max-width: 1400px !important; margin: auto !important;}
    .studio-hero {padding: 25px 30px; border-radius: 18px; background: linear-gradient(118deg,#21182f,#382641 64%,#704065); color: #fff; margin-bottom: 16px; box-shadow: 0 12px 40px #140f202c;}
    .studio-hero h1 {color:#fff !important; font-size: 2rem; margin: 4px 0 8px;}
    .studio-hero p {color:#ecdce8; margin:0;}
    .studio-badge {font-size: 0.75rem; letter-spacing: 0.13rem; font-weight: bold; color:#f4b3dc;}
    .studio-notice {border-left: 3px solid #d891c2; padding: 9px 14px; background: #b088b61b; border-radius: 5px;}
    """
    with gr.Blocks(
        title="WAI Studio · Colab GPU",
        css=css,
        theme=gr.themes.Soft(
            primary_hue="purple", secondary_hue="pink", neutral_hue="slate"
        ),
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(
            "<div class='studio-hero'><span class='studio-badge'>✦ WAI · COLAB GPU · PRIVATE STUDIO</span><h1>Biến ý tưởng thành thế giới anime.</h1><p>WAI-illustrious v17 · LoRA tay/chân/mắt đã xác minh · Ảnh lưu ở Drive hoặc /content.</p></div>"
        )
        gr.Markdown(
            "**Chỉ dành cho bạn:** liên kết tạm thời vẫn đi qua máy chủ trung gian Gradio; hãy giữ bí mật URL và mật khẩu. Đóng Colab là dừng tạo ảnh. Model không chạy trên Cloudflare.",
            elem_classes="studio-notice",
        )
        with gr.Row():
            with gr.Column(scale=5, min_width=360):
                prompt = gr.Textbox(
                    label="Ý tưởng / prompt",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    max_lines=6,
                    placeholder="Mô tả nhân vật, khung cảnh, ánh sáng, phong cách...",
                )
                negative = gr.Textbox(
                    label="Negative prompt", value=DEFAULT_NEGATIVE, lines=2
                )
                with gr.Accordion("⚙️ Thông số ảnh và LoRA", open=True):
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
                    eyes = gr.Checkbox(
                        label="Perfect Eyes · mắt",
                        value="eyes" in runtime.lora_paths,
                        interactive="eyes" in runtime.lora_paths,
                    )
                    eyes_weight = gr.Slider(
                        0.1,
                        1,
                        value=runtime.lora_manifest.get("eyes", {}).get("weight", 0.45),
                        step=0.05,
                        label="Cường độ Eyes",
                    )
                    embed = gr.Checkbox(
                        label="Nhúng prompt vào metadata PNG (tắt nếu chia sẻ ảnh)",
                        value=False,
                    )
                    gr.Markdown(
                        "LoRA chỉ có thể bật nếu đã chọn và xác minh ở ô cấu hình trước khi mở giao diện. Tắt/bật và đổi cường độ ở đây **không** tải lại checkpoint."
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
                with gr.Tabs():
                    with gr.Tab("✦ Văn bản → ảnh"):
                        text_size = gr.Dropdown(
                            choices=list(SIZE_PRESETS),
                            value="1024x1024",
                            label="Kích thước",
                        )
                        text_button = gr.Button("Tạo ảnh từ prompt", variant="primary")
                    with gr.Tab("◈ Ảnh → ảnh"):
                        image_source = gr.Image(
                            label="Ảnh nguồn",
                            type="pil",
                            sources=["upload"],
                            image_mode="RGB",
                        )
                        image_size = gr.Dropdown(
                            choices=list(SIZE_PRESETS),
                            value="1024x1024",
                            label="Kích thước đầu ra",
                        )
                        image_strength = gr.Slider(
                            0.2, 0.85, value=0.45, step=0.05, label="Denoise strength"
                        )
                        gr.Markdown(
                            "Nếu ảnh nguồn có tỷ lệ khác kích thước đầu ra, giao diện sẽ cắt giữa ảnh để không làm méo."
                        )
                        image_button = gr.Button("Biến đổi ảnh", variant="primary")
                    with gr.Tab("✎ Sửa vùng ảnh"):
                        editor = gr.ImageEditor(
                            label="Tải ảnh vào đây và dùng cọ tô vùng cần sửa",
                            type="pil",
                            image_mode="RGBA",
                            sources=["upload"],
                            height=460,
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
                        gr.Markdown(
                            "**Vùng trắng / nét cọ = sửa; vùng đen = giữ nguyên.** Ảnh tải lên được thu về cạnh dài tối đa 1024 px cùng mask. Tránh tô toàn bộ ảnh."
                        )
                        inpaint_button = gr.Button("Sửa vùng đã tô", variant="primary")
            with gr.Column(scale=4, min_width=330):
                gallery = gr.Gallery(
                    label="Kết quả · nhấn để xem lớn",
                    columns=2,
                    height=550,
                    object_fit="contain",
                    format="png",
                    show_share_button=False,
                )
                status = gr.Markdown(
                    "Ảnh đầu tiên sẽ xuất hiện tại đây. Model đã nạp trong Google Colab."
                )
                downloads = gr.File(
                    label="Tải ảnh PNG", file_count="multiple", interactive=False
                )
                latest = gr.State(None)
                with gr.Row():
                    to_image = gr.Button("Dùng ảnh mới nhất để biến đổi")
                    to_inpaint = gr.Button("Dùng ảnh mới nhất để sửa vùng")
                gr.Markdown(
                    "Ảnh được lưu tại Drive (nếu đã gắn) hoặc `/content/wai_outputs`. Tải xuống trước khi phiên Colab hết hạn nếu không dùng Drive."
                )

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
                api_name=False,
                show_api=False,
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

        to_image.click(
            fn=load_last,
            inputs=latest,
            outputs=image_source,
            api_name=False,
            show_api=False,
        )
        to_inpaint.click(
            fn=edit_last, inputs=latest, outputs=editor, api_name=False, show_api=False
        )
        demo.queue(max_size=4, default_concurrency_limit=1, api_open=False)
    return demo
