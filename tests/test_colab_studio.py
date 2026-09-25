"""CPU tests for the personal, password-protected Colab WAI studio.

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
                drive_root=root / "drive",
                local_cache_root=root / "cache",
                CACHE_MODEL_LOCAL=False,
                AUTO_DOWNLOAD=False,
                PERSIST_MODEL_TO_DRIVE=False,
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
            downloaded = root / "downloaded.safetensors"
            downloaded.write_bytes(good_bytes)
            modules["huggingface_hub"].hf_hub_download = lambda **_: str(downloaded)
            with contextlib.redirect_stdout(io.StringIO()):
                ns = dict(base, AUTO_DOWNLOAD=True)
                exec(source, ns)
            self.assertTrue(ns["WAI_STUDIO_VERSION_VERIFIED"])
            self.assertEqual(ns["checkpoint"], downloaded)

    def test_launch_requires_password_and_never_exposes_model_dir(self):
        launch = "".join(
            json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][8]["source"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ck = root / "weights.safetensors"
            ck.touch()
            runtime = types.SimpleNamespace(
                output_dir=root / "outputs",
                backup_dir=root / "backup",
                checkpoint=ck,
                drive_root=root / "drive",
                lora_paths={"anatomy": ck},
            )
            calls = []

            class FakeApp:
                studio_theme = "test-theme"
                studio_css = "test-css"

                def launch(self, **kwargs):
                    calls.append(kwargs)
                    return (None, None, "https://temporary.gradio.live")

                def close(self):
                    pass

            ns = {"studio_runtime": runtime, "build_app": lambda _: FakeApp()}
            with (
                patch("getpass.getpass", return_value="short"),
                self.assertRaisesRegex(ValueError, "ít nhất 16"),
            ):
                exec(launch, ns)
            self.assertEqual(calls, [])
            ns = {"studio_runtime": runtime, "build_app": lambda _: FakeApp()}
            text = io.StringIO()
            with (
                patch("getpass.getpass", return_value="my-private-long-password"),
                contextlib.redirect_stdout(text),
            ):
                exec(launch, ns)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["auth"], ("owner", "my-private-long-password"))
            self.assertEqual(calls[0]["share"], True)
            self.assertEqual(calls[0]["theme"], "test-theme")
            self.assertEqual(calls[0]["css"], "test-css")
            self.assertEqual(calls[0]["footer_links"], [])
            self.assertIn(str(ck.resolve()), calls[0]["blocked_paths"])
            self.assertNotIn(str(ck.parent), calls[0]["allowed_paths"])
            self.assertNotIn("my-private-long-password", text.getvalue())
            self.assertNotIn("password", ns)


class RuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
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
            drive_root=self.root / "drive",
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

    def test_validation_and_eye_trigger(self):
        params = self.params()
        self.assertEqual(params[0], "anime portrait, perfect eyes")
        self.assertEqual(
            self.params(prompt="perfect eyes, soft light")[0].count("perfect eyes"), 1
        )
        self.assertEqual(self.params(eyes_enabled=False)[0], "anime portrait")
        for setting in (
            dict(prompt=" "),
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
        self.assertEqual(self.pipe.weights[-1], (["anatomy", "eyes"], [0.0, 0.45]))
        self.assertEqual(
            [call[1]["generator"].seed for call in FakePipe.calls], [42, 43]
        )
        with Image.open(paths[0]) as result:
            self.assertEqual(result.size, (512, 512))
            self.assertNotIn("parameters", result.info)  # private by default
        self.runtime.text_to_image(
            "512x512", "anime", "", 20, 6, 2, 1, True, 0.4, False, 0.45, True
        )
        image = next((self.root / "output").glob("*_*_2.png"))
        with Image.open(image) as result:
            self.assertIn('"seed": 2', result.info["parameters"])

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
            self.assertIn("natural hands", FakePipe.calls[-1][1]["prompt"])
            self.assertEqual(len(gallery), 1)
            with self.assertRaisesRegex(ValueError, "vùng nhỏ"):
                studio._editor_mask({"background": source, "layers": []}, None)
            with self.assertRaisesRegex(ValueError, "cùng kích thước"):
                studio._editor_mask(editor, Image.new("L", (1, 1), 255))
            with self.assertRaisesRegex(ValueError, "vùng nhỏ"):
                studio._editor_mask(editor, Image.new("L", (512, 512), 255))

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_gpu_oom_retries_with_offload_and_same_seed(self):
        self.runtime.pipe = FakePipe(fail_once=True)
        creations = []

        def make(offload):
            creations.append(offload)
            return FakePipe()

        self.runtime.create_pipeline = make
        self.runtime.text_to_image(
            "512x512", "anime", "", 20, 6, 77, 1, False, 0.55, False, 0.45, False
        )
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

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_output_falls_back_to_local_when_drive_disconnects(self):
        self.runtime.output_dir = self.root / "drive" / "MyDrive" / "AI" / "outputs"
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
        self.assertFalse((self.root / "drive").exists())

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
        self.assertEqual(
            len(
                [x for x in config["dependencies"] if x["api_visibility"] == "private"]
            ),
            5,
        )
        self.assertTrue(demo.studio_css)
        self.assertIsNotNone(demo.studio_theme)

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("gradio"),
        "Gradio and Pillow needed for image event smoke test",
    )
    def test_gradio_events_preprocess_and_return_downloadable_pngs(self):
        """Exercise all UI events through Gradio 6, not only the runtime methods."""
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
        common = [
            "anime portrait",
            "bad anatomy",
            20,
            6,
            42,
            1,
            False,
            0.55,
            False,
            0.45,
            False,
        ]
        editor = {
            "background": file_data(source),
            "layers": [file_data(layer)],
            "composite": file_data(source),
        }

        async def process(index, inputs):
            return (
                await demo.process_api(index, inputs, state=state, explicit_call=True)
            )["data"]

        async def smoke():
            for index, inputs in (
                (0, ["512x512", *common]),
                (1, [file_data(source), "512x512", 0.45, *common]),
                (2, [editor, None, "hands", 0.45, 8, *common]),
                (2, [editor, file_data(mask), "eyes", 0.45, 8, *common]),
            ):
                data = await process(index, inputs)
                self.assertIn("✅ Đã tạo 1 ảnh", data[2])
                self.assertEqual(len(data[0]), 1)  # Gradio Gallery
                self.assertTrue(Path(data[0][0]["image"]["path"]).is_file())
                png = Path(data[1][0]["path"])  # Gradio File download
                self.assertTrue(png.is_file())
                with Image.open(png) as result:
                    self.assertEqual(result.format, "PNG")
            self.assertEqual(len(FakePipe.calls), 4)
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
        "Gradio needed for local auth smoke test",
    )
    def test_gradio_private_login_rejects_unauthenticated_api_and_weights(self):
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
                auth=("owner", "a-strong-test-password"),
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
                for path in (
                    "/config",
                    "/gradio_api/info",
                    "/gradio_api/file=" + str(self.ck),
                    "/gradio_api/file=" + str(saved),
                ):
                    self.assertEqual(client.get(base + path).status_code, 401)
                self.assertEqual(
                    client.post(
                        base + "/login",
                        data={
                            "username": "owner",
                            "password": "a-strong-test-password",
                        },
                    ).status_code,
                    200,
                )
                self.assertEqual(client.get(base + "/config").status_code, 200)
                self.assertEqual(client.get(base + "/gradio_api/info").status_code, 200)
                self.assertEqual(
                    client.get(base + "/gradio_api/file=" + str(saved)).status_code,
                    200,
                )
                self.assertEqual(
                    client.get(base + "/gradio_api/file=" + str(self.ck)).status_code,
                    403,
                )
        finally:
            demo.close()


if __name__ == "__main__":
    unittest.main()
