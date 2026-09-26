"""CPU tests for the personal Colab WAI studio (public, unlisted URL).

Run with Pillow/Gradio installed for image and component tests; standard-library
setup tests run even without them. No checkpoint/GPU/network required.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from colab import studio
from scripts.build_colab_studio import build

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "WAI_Illustrious_Studio_Colab.ipynb"
BASE = ROOT / "WAI_Illustrious_Colab.ipynb"

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


class FakeGenerator:
    def __init__(self, device):
        assert device == "cpu"
        self.seed = None

    def manual_seed(self, value):
        self.seed = value
        return self


class FakeTorch:
    Generator = FakeGenerator
    inference_mode = staticmethod(contextlib.nullcontext)
    cuda = types.SimpleNamespace(
        OutOfMemoryError=type("FakeOOM", (RuntimeError,), {}),
        empty_cache=lambda: None,
    )


class FakePipe:
    calls = []

    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.weights = []
        self.hooks_removed = 0
        self.offloaded = False

    def set_adapters(self, names, adapter_weights):
        self.weights.append((names, adapter_weights))

    def remove_all_hooks(self):
        self.hooks_removed += 1

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def __call__(self, **kwargs):
        FakePipe.calls.append((self, kwargs))
        if self.fail_once:
            self.fail_once = False
            raise FakeTorch.cuda.OutOfMemoryError("GPU test OOM")
        return types.SimpleNamespace(
            images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")]
        )


class FakeDerived:
    @classmethod
    def from_pipe(cls, base):
        return base


class NotebookTests(unittest.TestCase):
    def test_notebook_is_self_contained_clean_and_reuses_verified_setup(self):
        n = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        old = json.loads(BASE.read_text(encoding="utf-8"))
        self.assertEqual(
            n, build(), "Regenerate with python scripts/build_colab_studio.py"
        )
        self.assertEqual(n["nbformat"], 4)
        self.assertEqual(n["nbformat_minor"], 0)
        self.assertEqual(len(n["cells"]), 10)
        for index in (1, 3, 5, 6):
            self.assertEqual(n["cells"][index]["source"], old["cells"][index]["source"])
        prepare = "".join(n["cells"][4]["source"])
        self.assertIn("WAI_STUDIO_VERSION_VERIFIED = False", prepare)
        self.assertIn("if not WAI_STUDIO_VERSION_VERIFIED:", prepare)
        install = "".join(n["cells"][2]["source"])
        for requirement in (
            '"gradio==6.15.2"',
            '"pydantic>=2.12.5,<3"',
            '"starlette>=1.3.1,<2"',
        ):
            self.assertIn(requirement, install)
        self.assertIn("✅ Thư viện Studio đã sẵn sàng:", install)
        ui_source = "".join(n["cells"][7]["source"])
        self.assertIn((ROOT / "colab/studio.py").read_text(encoding="utf-8"), ui_source)
        self.assertIn("del pipe", ui_source)
        executable = "\n".join(
            "".join(cell["source"])
            for cell in n["cells"]
            if cell["cell_type"] == "code"
        )
        for removed in (
            "drive.mount(",
            "MOUNT_DRIVE",
            "CACHE_MODEL_LOCAL",
            "PERSIST_MODEL_TO_DRIVE",
            "PERSIST_LORAS_TO_DRIVE",
        ):
            self.assertNotIn(removed, executable)
        for cell in n["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["execution_count"], None)
                self.assertEqual(cell["outputs"], [])
        compile(ui_source, "studio-ui", "exec")
        compile("".join(n["cells"][8]["source"]), "studio-launch", "exec")
        try:
            import nbformat
        except ImportError:
            pass
        else:
            nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))

    @unittest.skipIf(
        not importlib.util.find_spec("packaging"), "packaging needed for cell 2 check"
    )
    def test_install_cell_reports_stale_colab_dependencies(self):
        install = "".join(
            json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][2]["source"]
        )
        start = "from importlib.metadata import PackageNotFoundError, version"
        check = install[install.index(start) :]
        versions = {
            "gradio": "6.15.2",
            "gradio_client": "2.5.0",
            "pydantic": "2.12.5",
            "starlette": "1.3.1",
            "huggingface_hub": "0.36.2",
        }
        installed = {
            "diffusers": "0.35.2",
            "transformers": "4.52.4",
            "accelerate": "1.10.1",
            "peft": "0.17.1",
            "safetensors": "0.8.0",
            "hf-xet": "1.6.0",
        }
        modules = {
            name: types.SimpleNamespace(__version__=number)
            for name, number in versions.items()
        }
        modules["diffusers"] = types.SimpleNamespace(
            StableDiffusionXLPipeline=object,
            AutoPipelineForImage2Image=object,
            AutoPipelineForInpainting=object,
        )
        from importlib.metadata import PackageNotFoundError

        def fake_version(name):
            if name not in installed:
                raise PackageNotFoundError(name)
            return installed[name]

        with (
            patch.dict(sys.modules, modules),
            patch("importlib.metadata.version", side_effect=fake_version),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            exec(check, {})
            self.assertIn("✅ Thư viện Studio đã sẵn sàng:", output.getvalue())
            modules["gradio_client"].__version__ = "1.14.0"
            with self.assertRaisesRegex(RuntimeError, "gradio-client đang là 1.14.0"):
                exec(check, {})
            modules["gradio_client"].__version__ = "2.5.0"
            installed.pop("diffusers")
            with self.assertRaisesRegex(RuntimeError, "Thiếu diffusers"):
                exec(check, {})
            installed["diffusers"] = "0.35.2"
            installed["transformers"] = "4.53.0"
            with self.assertRaisesRegex(RuntimeError, "transformers đang là 4.53.0"):
                exec(check, {})
            installed["transformers"] = "4.52.4"
            del modules["diffusers"].AutoPipelineForInpainting
            with self.assertRaisesRegex(RuntimeError, "Diffusers không import được"):
                exec(check, {})

    def test_studio_rejects_wrong_version_before_loading_gpu(self):
        source = "".join(
            json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][4]["source"]
        )
        good_bytes = b"small but correctly hashed v17 test checkpoint"
        source = source.replace(
            "HF_MODEL_BYTES = 6_938_040_682", f"HF_MODEL_BYTES = {len(good_bytes)}"
        )
        source = source.replace(
            "MIN_CHECKPOINT_BYTES = 100 * 2**20", "MIN_CHECKPOINT_BYTES = 1"
        )
        source = source.replace(
            "DISK_RESERVE_BYTES = 2 * 2**30", "DISK_RESERVE_BYTES = 1"
        )
        source = source.replace(
            'HF_SHA256 = "f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04"',
            f'HF_SHA256 = "{hashlib.sha256(good_bytes).hexdigest()}"',
        )
        self.assertIn("MIN_CHECKPOINT_BYTES = 1", source)

        class Header:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def keys(self):
                return [
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "conditioner.embedders.0.transformer.token_embedding.weight",
                ]

        modules = {
            "huggingface_hub": types.SimpleNamespace(hf_hub_download=lambda **_: None),
            "safetensors": types.SimpleNamespace(safe_open=lambda *_, **__: Header()),
        }
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules):
            root = Path(temp)
            model = root / "provided.safetensors"
            base = dict(
                Path=Path,
                source_model=model,
                output_dir=root / "images",
                local_cache_root=root / "cache",
                AUTO_DOWNLOAD=False,
            )
            model.write_bytes(good_bytes)
            ns = dict(base)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, ns)
            self.assertTrue(ns["WAI_STUDIO_VERSION_VERIFIED"])
            model.write_bytes(b"wrong" + good_bytes[5:])  # same length, different SHA
            with self.assertRaisesRegex(RuntimeError, "đúng SHA-256"):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(source, dict(base))
            model.unlink()
            downloaded = root / "cache" / "waiIllustriousSDXL_v170.safetensors"

            def download(**kwargs):
                self.assertEqual(kwargs["local_dir"], str(root / "cache"))
                downloaded.write_bytes(good_bytes)
                return str(downloaded)

            modules["huggingface_hub"].hf_hub_download = download
            with contextlib.redirect_stdout(io.StringIO()):
                ns = dict(base, AUTO_DOWNLOAD=True)
                exec(source, ns)
            self.assertTrue(ns["WAI_STUDIO_VERSION_VERIFIED"])
            self.assertEqual(ns["checkpoint"], downloaded)
            modules["huggingface_hub"].hf_hub_download = lambda **_: (
                _ for _ in ()
            ).throw(AssertionError("Do not redownload"))
            with contextlib.redirect_stdout(io.StringIO()):
                ns = dict(base, AUTO_DOWNLOAD=True)
                exec(source, ns)  # cached branch also verifies and marks v17
            self.assertTrue(ns["WAI_STUDIO_VERSION_VERIFIED"])
            self.assertEqual(ns["checkpoint"], downloaded)

    def test_launch_without_login_still_blocks_checkpoint_and_lora_files(self):
        launch = "".join(
            json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][8]["source"]
        )
        self.assertNotIn("getpass", launch)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ck = root / "weights.safetensors"
            ck.touch()
            runtime = types.SimpleNamespace(
                output_dir=root / "outputs",
                backup_dir=root / "backup",
                checkpoint=ck,
                lora_paths={"anatomy": ck},
            )
            calls = []

            class FakeApp:
                studio_theme = "test-theme"
                studio_css = "test-css"

                def __init__(self):
                    self.closed = False

                def launch(self, **kwargs):
                    calls.append(kwargs)
                    return (None, None, "https://temporary.gradio.live")

                def close(self):
                    self.closed = True

            ns = {
                "studio_runtime": runtime,
                "build_app": lambda _: FakeApp(),
                "local_cache_root": root / "wai_model_cache",
                "local_lora_cache": root / "wai_lora_cache",
            }
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                exec(launch, ns)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("auth", calls[0])
            self.assertNotIn("auth_message", calls[0])
            self.assertEqual(calls[0]["share"], True)
            self.assertEqual(calls[0]["theme"], "test-theme")
            self.assertEqual(calls[0]["css"], "test-css")
            self.assertEqual(calls[0]["footer_links"], [])
            self.assertIn(str(ck.resolve()), calls[0]["blocked_paths"])
            self.assertIn(str(ns["local_cache_root"]), calls[0]["blocked_paths"])
            self.assertIn(str(ns["local_lora_cache"]), calls[0]["blocked_paths"])
            self.assertNotIn(str(ck.parent), calls[0]["allowed_paths"])
            self.assertIn("Ai có link đều có thể dùng GPU", text.getvalue())
            old_app = ns["studio_app"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(launch, ns)
            self.assertTrue(old_app.closed)
            self.assertEqual(len(calls), 2)


class RuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        root_patch = patch.object(studio, "CONTENT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.ck = self.root / "verified.safetensors"
        self.ck.touch()
        self.pipe = FakePipe()
        self.runtime = studio.StudioRuntime(
            torch=FakeTorch,
            pipe=self.pipe,
            create_pipeline=lambda offload: FakePipe(),
            checkpoint=self.ck,
            lora_paths={"anatomy": self.ck, "eyes": self.ck},
            lora_manifest={
                "anatomy": {"weight": 0.55, "version": "v1", "sha256": "hash"},
                "eyes": {"weight": 0.45, "version": "v1", "sha256": "hash"},
            },
            vram_mode="auto",
            use_offload=False,
            output_dir=self.root / "output",
            backup_dir=self.root / "backup",
        )
        FakePipe.calls = []

    def params(self, **change):
        values = dict(
            prompt="anime portrait",
            negative="bad anatomy",
            steps=25,
            cfg=6.0,
            seed=-1,
            count=1,
            anatomy_enabled=True,
            anatomy_weight=0.55,
            eyes_enabled=True,
            eyes_weight=0.45,
        )
        values.update(change)
        return self.runtime._parameters(**values)

    def test_style_presets_populate_editable_fields_without_overriding_edits(self):
        negative = studio.DEFAULT_NEGATIVE
        for anomaly in (
            "extra fingers",
            "missing fingers",
            "fused fingers",
            "extra toes",
            "missing toes",
            "fused toes",
        ):
            self.assertIn(anomaly, negative)
        self.assertNotIn("nsfw", negative)
        anime_pos, anime_neg = studio.compose_style_prompts(
            "anime illustration, portrait", negative, "Anime chuẩn", True
        )
        self.assertTrue(anime_pos.startswith("anime illustration, clean lineart"))
        self.assertEqual(anime_pos.count("anime illustration"), 1)
        self.assertIn("perfect eyes", anime_pos)
        self.assertTrue(anime_neg.startswith("nsfw, explicit"))
        self.assertIn("photorealistic", anime_neg)
        semi_pos, semi_neg = studio.compose_style_prompts(
            "portrait", "", "Bán thực 2.5D", False
        )
        self.assertTrue(
            semi_pos.startswith("semi-realistic anime art, 2.5d illustration")
        )
        self.assertIn("volumetric lighting", semi_pos)
        self.assertIn("flat cel shading", semi_neg)
        self.assertNotIn("perfect eyes", semi_pos)
        long_prompt = "portrait, " + "soft sunlight, " * 90 + "semi-realistic anime art"
        long_pos, long_neg = studio.compose_style_prompts(
            long_prompt, negative, "Bán thực 2.5D", True
        )
        self.assertEqual(long_pos.count("semi-realistic anime art"), 1)
        self.assertTrue(long_pos.startswith("semi-realistic anime art"))
        self.assertTrue(long_neg.startswith("nsfw, explicit, flat cel shading"))
        # Runtime uses only what remains in the editable fields, even if the
        # selected style and enabled eye LoRA have different suggested tags.
        edited = "  my own portrait, no preset tags  "
        self.assertEqual(
            self.params(
                prompt=edited, negative="  custom negative ", style="Bán thực 2.5D"
            )[:2],
            (edited, "  custom negative "),
        )
        with self.assertRaisesRegex(ValueError, "Chọn phong cách"):
            self.params(style="unknown")
        with self.assertRaisesRegex(ValueError, "Chọn phong cách"):
            studio.compose_style_prompts("portrait", "", "unknown", False)

    def test_adult_style_is_opt_in_and_rejects_obvious_underage_prompts(self):
        with self.assertRaisesRegex(ValueError, "cần xác nhận"):
            self.params(style=studio.ADULT_STYLE, adult_confirmed=False)
        for prompt in (
            "underage character",
            "school girl portrait",
            "teenage",
            "17-year-old",
            "vị thành niên",
        ):
            with (
                self.subTest(prompt=prompt),
                self.assertRaisesRegex(ValueError, "vị thành niên"),
            ):
                self.params(
                    style=studio.ADULT_STYLE, adult_confirmed=True, prompt=prompt
                )
        preset_pos, preset_neg = studio.compose_style_prompts(
            "adult, woman portrait", studio.DEFAULT_NEGATIVE, studio.ADULT_STYLE, True
        )
        self.assertIn("erotic anime illustration", preset_pos)
        self.assertEqual(preset_pos.count("adult"), 1)
        self.assertIn("underage", preset_neg)
        self.assertNotIn("nsfw", preset_neg)
        self.assertIn("missing toes", preset_neg)
        self.assertEqual(
            self.params(
                style=studio.ADULT_STYLE,
                adult_confirmed=True,
                prompt=preset_pos,
                negative=preset_neg,
            )[:2],
            (preset_pos, preset_neg),
        )
        # Underage terms in the *negative* should not trigger the positive guard.
        self.assertEqual(
            self.params(
                style=studio.ADULT_STYLE,
                adult_confirmed=True,
                prompt="adult",
                negative="underage, child",
            )[1],
            "underage, child",
        )

    def test_validation_and_eye_trigger(self):
        self.assertEqual(self.params()[:2], ("anime portrait", "bad anatomy"))
        suggested, _ = studio.compose_style_prompts(
            "anime portrait", "", "Tùy chỉnh", True
        )
        self.assertIn("perfect eyes", suggested)
        self.assertEqual(self.params(eyes_enabled=True)[0], "anime portrait")
        self.assertEqual(self.params(eyes_enabled=False)[0], "anime portrait")
        for setting in (
            dict(prompt=" "),
            dict(prompt="x" * 2201),
            dict(negative="x" * 1701),
            dict(steps=46),
            dict(cfg=float("nan")),
            dict(seed=-2),
            dict(seed=2**32),
            dict(count=5),
            dict(anatomy_weight=1.5),
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.params(**setting)
        self.runtime.lora_paths.pop("eyes")
        with self.assertRaisesRegex(ValueError, "chưa được nạp"):
            self.params(eyes_enabled=True)

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_text_to_image_batch_seed_save_and_disabled_adapter(self):
        gallery, paths, status, latest = self.runtime.text_to_image(
            "512x512",
            "anime cat",
            "bad paws",
            20,
            5.5,
            42,
            2,
            False,
            0.55,
            True,
            0.45,
            False,
        )
        self.assertEqual(len(gallery), 2)
        self.assertEqual(len(paths), 2)
        self.assertEqual(latest, paths[-1])
        self.assertIn("42, 43", status)
        self.assertIn("GPU trực tiếp", status)
        self.assertEqual(self.pipe.weights[-1], (["anatomy", "eyes"], [0.0, 0.45]))
        self.assertEqual(
            [call[1]["generator"].seed for call in FakePipe.calls], [42, 43]
        )
        self.assertTrue(
            all(
                call[1]["prompt"] == "anime cat"
                and call[1]["negative_prompt"] == "bad paws"
                for call in FakePipe.calls
            )
        )
        with Image.open(paths[0]) as result:
            self.assertEqual(result.size, (512, 512))
            self.assertNotIn("parameters", result.info)  # private by default
        # Style is metadata only: user-edited prompts are the exact pipe inputs.
        self.runtime.text_to_image(
            "512x512",
            "  portrait without style tags  ",
            "  my negative  ",
            20,
            6,
            2,
            1,
            True,
            0.4,
            False,
            0.45,
            True,
            "Bán thực 2.5D",
        )
        image = next((self.root / "output").glob("*_*_2.png"))
        with Image.open(image) as result:
            parameters = json.loads(result.info["parameters"])
            self.assertEqual(parameters["seed"], 2)
            self.assertEqual(parameters["style"], "Bán thực 2.5D")
            self.assertEqual(parameters["prompt"], "  portrait without style tags  ")
            self.assertEqual(parameters["negative_prompt"], "  my negative  ")
            self.assertEqual(FakePipe.calls[-1][1]["prompt"], parameters["prompt"])
            self.assertEqual(
                FakePipe.calls[-1][1]["negative_prompt"], parameters["negative_prompt"]
            )

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_img2img_and_painted_inpainting_keep_unmasked_pixels(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            source = Image.new("RGB", (600, 500), "navy")
            self.runtime.image_to_image(
                source,
                "512x512",
                0.5,
                "forest",
                "",
                20,
                6,
                9,
                1,
                True,
                0.55,
                False,
                0.45,
                False,
            )
            self.assertEqual(FakePipe.calls[-1][1]["image"].size, (512, 512))
            self.assertEqual(FakePipe.calls[-1][1]["strength"], 0.5)
            self.assertEqual(FakePipe.calls[-1][1]["generator"].seed, 9)
            self.assertEqual(
                (
                    FakePipe.calls[-1][1]["prompt"],
                    FakePipe.calls[-1][1]["negative_prompt"],
                ),
                ("forest", ""),
            )
            painted = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            ImageDraw.Draw(painted).rectangle(
                (100, 100, 199, 199), fill=(255, 255, 255, 255)
            )
            editor = {
                "background": Image.new("RGB", (512, 512), "navy"),
                "layers": [painted],
                "composite": None,
            }
            gallery, paths, _, _ = self.runtime.inpaint(
                editor,
                None,
                "hands",
                0.45,
                0,
                "natural portrait",
                "",
                20,
                6,
                11,
                1,
                True,
                0.55,
                False,
                0.45,
                False,
            )
            with Image.open(paths[0]) as result:
                self.assertEqual(result.getpixel((0, 0)), (0, 0, 128))
                self.assertEqual(result.getpixel((150, 150)), (255, 255, 255))
                self.assertEqual(result.getpixel((250, 250)), (0, 0, 128))
            self.assertEqual(
                FakePipe.calls[-1][1]["mask_image"].getpixel((150, 150)), 255
            )
            self.assertEqual(FakePipe.calls[-1][1]["prompt"], "natural portrait")
            self.assertEqual(FakePipe.calls[-1][1]["negative_prompt"], "")
            self.assertEqual(len(gallery), 1)
            for target in ("hands", "legs", "eyes"):
                with self.subTest(target=target):
                    suggested = studio.apply_repair_hints(
                        "natural portrait", "bad anatomy", target
                    )
                    self.assertNotEqual(suggested, ("natural portrait", "bad anatomy"))
                    self.assertEqual(
                        studio.apply_repair_hints(*suggested, target), suggested
                    )
            self.assertEqual(
                studio.apply_repair_hints("as-is", "", "custom"), ("as-is", "")
            )
            with self.assertRaisesRegex(ValueError, "Chọn vùng sửa"):
                studio.apply_repair_hints("as-is", "", "wrong")
            with self.assertRaisesRegex(ValueError, "vùng nhỏ"):
                studio._editor_mask({"background": source, "layers": []}, None)
            with self.assertRaisesRegex(ValueError, "cùng kích thước"):
                studio._editor_mask(editor, Image.new("L", (1, 1), 255))
            with self.assertRaisesRegex(ValueError, "vùng nhỏ"):
                studio._editor_mask(editor, Image.new("L", (512, 512), 255))

    @unittest.skipIf(Image is None, "Pillow needed for inference-mode smoke test")
    def test_second_image_changes_adapters_inside_inference_mode(self):
        active = {"inference": False}

        @contextlib.contextmanager
        def inference_mode():
            active["inference"] = True
            try:
                yield
            finally:
                active["inference"] = False

        class GuardPipe(FakePipe):
            def set_adapters(self, names, adapter_weights):
                if not active["inference"]:
                    raise AssertionError("LoRA adapter changed outside InferenceMode")
                super().set_adapters(names, adapter_weights)

        self.runtime.pipe = GuardPipe()
        self.runtime.torch = types.SimpleNamespace(
            Generator=FakeGenerator, inference_mode=inference_mode, cuda=FakeTorch.cuda
        )
        self.runtime.use_offload = True
        for seed in (42, 43):
            self.runtime.text_to_image(
                "512x512",
                "anime",
                "",
                20,
                6,
                seed,
                1,
                True,
                0.55,
                True,
                0.45,
                False,
            )
        self.assertEqual(len(self.runtime.pipe.weights), 2)
        self.assertFalse(active["inference"])

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("torch"),
        "PyTorch and Pillow needed to reproduce the second-image inference tensor error",
    )
    def test_second_image_can_reactivate_loras_with_cpu_offload(self):
        import torch

        class InferenceAdapterPipe(FakePipe):
            def __init__(self):
                super().__init__()
                self.adapter_tensor = None

            def set_adapters(self, names, adapter_weights):
                # PEFT toggles requires_grad when selecting adapters. An
                # offloaded parameter may be an inference tensor after image 1.
                if self.adapter_tensor is not None:
                    self.adapter_tensor.requires_grad_(True)
                super().set_adapters(names, adapter_weights)

            def __call__(self, **kwargs):
                if not torch.is_inference_mode_enabled():
                    raise AssertionError("Inference must run inside InferenceMode")
                self.adapter_tensor = torch.ones(1)
                return super().__call__(**kwargs)

        pipe = InferenceAdapterPipe()
        self.runtime.pipe = pipe
        self.runtime.torch = torch
        self.runtime.use_offload = True
        for index, (anatomy_weight, eyes_enabled) in enumerate(
            ((0.55, True), (0.4, True), (0.4, False))
        ):
            _, paths, _, _ = self.runtime.text_to_image(
                "512x512",
                "anime",
                "",
                20,
                6,
                42 + index,
                1,
                True,
                anatomy_weight,
                eyes_enabled,
                0.45,
                False,
            )
            self.assertTrue(Path(paths[0]).is_file())
            self.assertTrue(pipe.adapter_tensor.is_inference())
            if index == 0:
                with self.assertRaisesRegex(RuntimeError, "outside InferenceMode"):
                    pipe.adapter_tensor.requires_grad_(True)
        self.assertEqual(
            pipe.weights,
            [
                (["anatomy", "eyes"], [0.55, 0.45]),
                (["anatomy", "eyes"], [0.4, 0.45]),
                (["anatomy", "eyes"], [0.4, 0.0]),
            ],
        )

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_gpu_oom_retries_with_offload_and_same_seed(self):
        self.runtime.pipe = FakePipe(fail_once=True)
        creations = []

        def make(offload):
            creations.append(offload)
            return FakePipe()

        self.runtime.create_pipeline = make
        _, _, status, _ = self.runtime.text_to_image(
            "512x512", "anime", "", 20, 6, 77, 1, False, 0.55, False, 0.45, False
        )
        self.assertIn("CPU offload", status)
        self.assertEqual(creations, [True])
        self.assertTrue(self.runtime.use_offload)
        self.assertEqual(
            [call[1]["generator"].seed for call in FakePipe.calls], [77, 77]
        )

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_offload_hooks_restored_even_after_image_inference_error(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        self.runtime.use_offload = True
        self.runtime.vram_mode = "low_vram"
        source = Image.new("RGB", (512, 512), "navy")
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            self.runtime.image_to_image(
                source,
                "512x512",
                0.5,
                "anime",
                "",
                20,
                6,
                10,
                1,
                False,
                0.55,
                False,
                0.45,
                False,
            )
            self.assertEqual(self.runtime.pipe.hooks_removed, 2)
            self.assertTrue(self.runtime.pipe.offloaded)
            self.runtime.pipe.fail_once = True
            with self.assertRaisesRegex(RuntimeError, "Hết VRAM"):
                self.runtime.image_to_image(
                    source,
                    "512x512",
                    0.5,
                    "anime",
                    "",
                    20,
                    6,
                    10,
                    1,
                    False,
                    0.55,
                    False,
                    0.45,
                    False,
                )
            self.assertEqual(self.runtime.pipe.hooks_removed, 4)
            self.assertTrue(self.runtime.pipe.offloaded)

    def test_runtime_rejects_nonlocal_and_drive_output_directories(self):
        (self.root / "drive" / "MyDrive").mkdir(parents=True)
        (self.root / "symlinked-output").symlink_to(
            self.root / "drive" / "MyDrive", target_is_directory=True
        )
        for output in (
            self.root / "drive" / "MyDrive" / "images",
            self.root / "symlinked-output",
            Path("/tmp/outside-wai"),
        ):
            with (
                self.subTest(output=output),
                self.assertRaisesRegex(ValueError, "/content"),
            ):
                studio.StudioRuntime(
                    torch=FakeTorch,
                    pipe=self.pipe,
                    create_pipeline=lambda offload: FakePipe(),
                    checkpoint=self.ck,
                    lora_paths={},
                    lora_manifest={},
                    vram_mode="auto",
                    use_offload=False,
                    output_dir=output,
                    backup_dir=self.root / "backup",
                )

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_output_falls_back_to_local_when_requested_folder_is_unwritable(self):
        self.runtime.output_dir = self.root / "blocked-output"
        self.runtime.output_dir.write_bytes(b"not a directory")
        _, paths, status, _ = self.runtime.text_to_image(
            "512x512",
            "anime",
            "",
            20,
            6,
            42,
            1,
            False,
            0.55,
            False,
            0.45,
            False,
        )
        self.assertEqual(Path(paths[0]).parent, self.runtime.backup_dir)
        self.assertIn(str(self.runtime.backup_dir), status)
        self.assertTrue(self.runtime.output_dir.is_file())

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("gradio"),
        "Gradio and Pillow needed for UI construction",
    )
    def test_gradio_components_build_without_gpu_or_public_share(self):
        demo = studio.build_app(self.runtime)
        config = demo.get_config_file()
        self.assertEqual(
            len([c for c in config["components"] if c["type"] == "imageeditor"]), 1
        )
        self.assertEqual(
            len([c for c in config["components"] if c["type"] == "gallery"]), 1
        )
        style = [
            c
            for c in config["components"]
            if c["type"] == "dropdown"
            and c["props"].get("label") == "Phong cách hình ảnh"
        ]
        self.assertEqual(len(style), 1)
        self.assertEqual(style[0]["props"]["value"], "Anime chuẩn")
        self.assertIn("Bán thực 2.5D", str(style[0]["props"]["choices"]))
        self.assertIn(studio.ADULT_STYLE, str(style[0]["props"]["choices"]))
        confirmations = [
            c
            for c in config["components"]
            if c["type"] == "checkbox" and "18 tuổi" in str(c["props"].get("label"))
        ]
        self.assertEqual(len(confirmations), 1)
        self.assertFalse(confirmations[0]["props"]["value"])
        self.assertTrue(
            any(
                c["type"] == "markdown"
                and "Chế độ:** GPU trực tiếp" in str(c["props"].get("value"))
                for c in config["components"]
            )
        )
        effective = [
            c
            for c in config["components"]
            if c["type"] == "textbox"
            and "gửi model · sửa được" in c["props"].get("label", "")
        ]
        self.assertEqual(len(effective), 2)
        self.assertTrue(all(c["props"]["interactive"] for c in effective))
        self.assertTrue(
            any("anime illustration" in c["props"]["value"] for c in effective)
        )
        reapply = next(
            c
            for c in config["components"]
            if c["type"] == "button"
            and "Áp dụng lại phong cách" in c["props"].get("value", "")
        )
        preset_event = next(
            d
            for d in config["dependencies"]
            if (style[0]["id"], "change") in d["targets"]
        )
        self.assertIn((reapply["id"], "click"), preset_event["targets"])
        self.assertEqual(set(preset_event["outputs"]), {c["id"] for c in effective})
        self.assertFalse(preset_event["queue"])
        repair_event = config["dependencies"][-1]
        self.assertEqual(set(repair_event["outputs"]), {c["id"] for c in effective})
        self.assertEqual(repair_event["inputs"][:2], [c["id"] for c in effective])
        self.assertEqual(len(config["dependencies"]), 7)
        for dep in config["dependencies"][:3]:  # text, img2img, inpaint
            self.assertEqual(dep["inputs"][-13:-11], [c["id"] for c in effective])
        self.assertTrue(
            all(x["api_visibility"] == "private" for x in config["dependencies"])
        )
        self.assertTrue(demo.studio_css)
        self.assertIsNotNone(demo.studio_theme)

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("gradio"),
        "Gradio and Pillow needed for image event smoke test",
    )
    def test_gradio_events_preprocess_and_return_downloadable_pngs(self):
        """UI event route: exact edited fields reach all three pipeline modes."""
        import asyncio
        from gradio.state_holder import SessionState

        def file_data(path):
            return {"path": str(path), "meta": {"_type": "gradio.FileData"}}

        source = self.root / "upload.png"
        Image.new("RGB", (512, 512), "navy").save(source)
        layer = self.root / "paint.png"
        painted = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(painted).rectangle((50, 50, 120, 120), fill="white")
        painted.save(layer)
        mask = self.root / "mask.png"
        black_white = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(black_white).rectangle((60, 60, 100, 100), fill=255)
        black_white.save(mask)

        demo = studio.build_app(self.runtime)
        state = SessionState(demo)
        editor = {
            "background": file_data(source),
            "layers": [file_data(layer)],
            "composite": file_data(source),
        }

        async def process(index, inputs):
            return (
                await demo.process_api(index, inputs, state=state, explicit_call=True)
            )["data"]

        def shared(positive, negative, eyes=False):
            return [positive, negative, 20, 6, 42, 1, False, 0.55, eyes, 0.45, False]

        async def smoke():
            presets = {}
            for choice in ("Anime chuẩn", "Bán thực 2.5D", studio.ADULT_STYLE):
                presets[choice] = await process(
                    5, [studio.DEFAULT_PROMPT, studio.DEFAULT_NEGATIVE, choice, False]
                )
            self.assertNotEqual(presets["Anime chuẩn"], presets["Bán thực 2.5D"])
            self.assertTrue(
                presets["Bán thực 2.5D"][0].startswith("semi-realistic anime art")
            )
            self.assertIn("flat cel shading", presets["Bán thực 2.5D"][1])
            self.assertNotIn("nsfw", presets["Anime chuẩn"][0])
            self.assertIn("nsfw", presets[studio.ADULT_STYLE][0])
            self.assertNotIn("nsfw", presets[studio.ADULT_STYLE][1])
            eyes_preset = await process(
                5,
                [studio.DEFAULT_PROMPT, studio.DEFAULT_NEGATIVE, "Bán thực 2.5D", True],
            )
            self.assertIn("perfect eyes", eyes_preset[0])
            self.assertNotIn("perfect eyes", presets["Bán thực 2.5D"][0])
            # The repair button only changes the visible editable fields.
            leg_prompts = await process(6, [*presets["Bán thực 2.5D"], "legs"])
            self.assertIn("natural toes", leg_prompts[0])
            self.assertIn("broken legs", leg_prompts[1])
            self.assertEqual(await process(6, [*leg_prompts, "legs"]), leg_prompts)
            for index, inputs, expected, style in (
                (
                    0,
                    [
                        "512x512",
                        *shared("  my own portrait  ", "  no hidden tags  ", eyes=True),
                        "Anime chuẩn",
                        False,
                    ],
                    ("  my own portrait  ", "  no hidden tags  "),
                    "Anime chuẩn",
                ),
                (
                    1,
                    [
                        file_data(source),
                        "512x512",
                        0.45,
                        *shared("paint this picture", "my bad quality"),
                        "Bán thực 2.5D",
                        False,
                    ],
                    ("paint this picture", "my bad quality"),
                    "Bán thực 2.5D",
                ),
                (
                    2,
                    [
                        editor,
                        None,
                        "hands",
                        0.45,
                        8,
                        *shared("no repair suggestions", "bad hands"),
                        "Anime chuẩn",
                        False,
                    ],
                    ("no repair suggestions", "bad hands"),
                    "Anime chuẩn",
                ),
                (
                    2,
                    [
                        editor,
                        file_data(mask),
                        "legs",
                        0.45,
                        8,
                        *shared(*leg_prompts),
                        "Bán thực 2.5D",
                        False,
                    ],
                    tuple(leg_prompts),
                    "Bán thực 2.5D",
                ),
                (
                    0,
                    [
                        "512x512",
                        *shared(*presets[studio.ADULT_STYLE]),
                        studio.ADULT_STYLE,
                        True,
                    ],
                    tuple(presets[studio.ADULT_STYLE]),
                    studio.ADULT_STYLE,
                ),
            ):
                data = await process(index, inputs)
                self.assertIn("✅ Đã tạo 1 ảnh", data[2])
                self.assertIn(style, data[2])
                kwargs = FakePipe.calls[-1][1]
                self.assertEqual(
                    (kwargs["prompt"], kwargs["negative_prompt"]), expected
                )
                self.assertEqual(len(data[0]), 1)  # Gradio Gallery
                self.assertTrue(Path(data[0][0]["image"]["path"]).is_file())
                png = Path(data[1][0]["path"])  # Gradio File download
                self.assertTrue(png.is_file())
                with Image.open(png) as result:
                    self.assertEqual(result.format, "PNG")
            self.assertEqual(len(FakePipe.calls), 5)
            self.assertTrue(Path((await process(3, [None]))[0]["path"]).is_file())
            self.assertTrue(
                Path((await process(4, [None]))[0]["background"]["path"]).is_file()
            )

        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            asyncio.run(smoke())

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("gradio"),
        "Gradio needed for public URL smoke test",
    )
    def test_gradio_public_link_serves_results_but_blocks_weights(self):
        import httpx
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        saved = self.runtime.output_dir / "available.png"
        saved.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "navy").save(saved)
        demo = studio.build_app(self.runtime)
        try:
            _, url, _ = demo.launch(
                share=False,
                inline=False,
                quiet=True,
                prevent_thread_lock=True,
                server_name="127.0.0.1",
                server_port=port,
                footer_links=[],
                theme=demo.studio_theme,
                css=demo.studio_css,
                allowed_paths=[
                    str(self.runtime.output_dir),
                    str(self.runtime.backup_dir),
                ],
                blocked_paths=[str(self.ck)],
                enable_monitoring=False,
            )
            with httpx.Client(timeout=15) as client:
                base = url.rstrip("/")
                self.assertEqual(client.get(base + "/config").status_code, 200)
                info = client.get(base + "/gradio_api/info")
                self.assertEqual(info.status_code, 200)
                self.assertEqual(info.json()["named_endpoints"], {})
                png = client.get(base + "/gradio_api/file=" + str(saved))
                self.assertEqual(png.status_code, 200)
                self.assertTrue(png.content.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(
                    client.get(base + "/gradio_api/file=" + str(self.ck)).status_code,
                    403,
                )
        finally:
            demo.close()


if __name__ == "__main__":
    unittest.main()
