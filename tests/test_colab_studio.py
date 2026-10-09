"""CPU tests for the personal Colab WAI studio (public, unlisted URL).

Run with Pillow/Gradio installed for image and component tests; standard-library
setup tests run even without them. No checkpoint/GPU/network required.
"""

import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from colab import studio
from scripts.build_colab_studio import as_form, build

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


class RealESRGANDownloadTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch needed for RRDBNet shape check")
    def test_anime6b_rrdbnet_layer_names_match_the_pinned_checkpoint(self):
        import torch

        model = studio._build_realesrgan_model(torch)
        weights = model.state_dict()
        self.assertEqual(tuple(weights["conv_first.weight"].shape), (64, 3, 3, 3))
        self.assertIn("body.0.rdb1.conv1.weight", weights)
        self.assertIn("body.5.rdb3.conv5.weight", weights)
        self.assertEqual(tuple(weights["conv_last.weight"].shape), (3, 64, 3, 3))

    def test_weight_download_is_atomic_and_rejects_corrupt_cache_or_response(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            payload = b"pinned anime upscaler fixture"
            spec = {
                "file": "anime.pth",
                "url": "https://example.invalid/anime.pth",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "version": "test-fixture",
            }
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as fetch:
                cached = studio.ensure_realesrgan_weight(folder, spec)
            self.assertEqual(cached.read_bytes(), payload)
            fetch.assert_called_once_with(spec["url"], timeout=90)
            with patch("urllib.request.urlopen", side_effect=AssertionError("cache miss")):
                self.assertEqual(studio.ensure_realesrgan_weight(folder, spec), cached)

            cached.write_bytes(b"corrupt")
            with patch("urllib.request.urlopen", side_effect=AssertionError("overwrite")):
                with self.assertRaisesRegex(RuntimeError, "sai kích thước/SHA-256"):
                    studio.ensure_realesrgan_weight(folder, spec)

            cached.unlink()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(RuntimeError, "không khớp"):
                    studio.ensure_realesrgan_weight(folder, spec)
            self.assertFalse(cached.exists())
            self.assertFalse(cached.with_name(cached.name + ".partial").exists())


class NotebookTests(unittest.TestCase):
    def test_notebook_is_self_contained_clean_and_reuses_verified_setup(self):
        n = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(
            n, build(), "Regenerate with python scripts/build_colab_studio.py"
        )
        self.assertEqual(n["nbformat"], 4)
        self.assertEqual(n["nbformat_minor"], 0)
        self.assertEqual(len(n["cells"]), 11)
        if BASE.is_file():
            old = json.loads(BASE.read_text(encoding="utf-8"))
            for index in (1, 3, 5, 6):
                self.assertEqual(n["cells"][index]["source"], old["cells"][index]["source"])
        prepare = "".join(n["cells"][4]["source"])
        self.assertIn("WAI_STUDIO_VERSION_VERIFIED = False", prepare)
        self.assertIn("if not WAI_STUDIO_VERSION_VERIFIED:", prepare)
        install = "".join(n["cells"][2]["source"])
        for requirement in (
            '"gradio==6.17.3"',
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
        launch_source = "".join(n["cells"][8]["source"])
        self.assertIn("studio_local_url, share_url", launch_source)
        self.assertIn("giao diện nội bộ vẫn chạy", launch_source)
        self.assertIn("Cloudflare Quick Tunnel", launch_source)
        compile(launch_source, "studio-launch", "exec")
        tunnel_source = "".join(n["cells"][9]["source"])
        self.assertEqual(
            hashlib.sha256(tunnel_source.encode()).hexdigest(),
            "2aa0d01a2a8974246eebf4ccf5878fc65cb79afabf8e560f6ae6aea4e545ecfc",
        )
        self.assertIn("trycloudflare.com", tunnel_source)
        self.assertIn("cloudflared-linux-{architecture}", tunnel_source)
        self.assertIn("--protocol", tunnel_source)
        self.assertIn("cloudflare_tunnel_process", tunnel_source)
        self.assertIn("ProxyHandler({})", tunnel_source)
        self.assertNotIn("google.colab.kernel.proxyPort", tunnel_source)
        compile(tunnel_source, "studio-cloudflare-tunnel", "exec")
        intro = "".join(n["cells"][0]["source"])
        self.assertIn("chủ thể → nhãn phân loại → ngoại hình", intro)
        self.assertNotIn("chất lượng → chủ thể → ngoại hình", intro)
        prepare = "".join(n["cells"][4]["source"])
        self.assertIn("Studio sẽ dừng, không nạp file khác", prepare)
        self.assertNotIn("KHÔNG xác thực là WAI v17", prepare)
        lora = "".join(n["cells"][5]["source"])
        self.assertIn("except ValueError:\n        raise", lora)
        loaded = "".join(n["cells"][6]["source"])
        self.assertIn("Chạy tiếp ô 7, rồi ô 8 và ô 9", loaded)
        self.assertIn("del studio_runtime", loaded)
        self.assertNotIn("ô 8 sửa vùng", loaded)
        try:
            import nbformat
        except ImportError:
            pass
        else:
            nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))

    def test_every_code_cell_collapses_into_a_compact_form(self):
        """Cả hai notebook phải gọn: mỗi ô code là một form tiêu đề + nút Run."""
        notebooks = [NOTEBOOK, *([BASE] if BASE.is_file() else [])]
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            self.assertTrue(code_cells, path.name)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                first = "".join(cell["source"]).splitlines()[0]
                with self.subTest(notebook=path.name, cell=index):
                    self.assertRegex(first, r"^# @title \d+\.")
                    self.assertIn('{ display-mode: "form" }', first)

    def test_as_form_marks_a_cell_once_and_requires_a_title(self):
        text = "# @title 9. Ô mới\nprint(1)\n"
        marked = as_form(text)
        self.assertEqual(
            marked, '# @title 9. Ô mới { display-mode: "form" }\nprint(1)\n'
        )
        self.assertEqual(as_form(marked), marked)  # không lặp marker
        with self.assertRaises(AssertionError):
            as_form("print('ô không có tiêu đề')\n")

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
            "gradio": "6.17.3",
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
            # Mặc định KHÔNG bật Gradio Share: mọi request (cả ảnh) phải đi qua relay
            # công cộng gradio.live, nguồn gây nghẽn/kẹt hẳn phải tải lại trang.
            self.assertIs(calls[0]["share"], False)
            self.assertEqual(calls[0]["theme"], "test-theme")
            self.assertEqual(calls[0]["css"], "test-css")
            self.assertEqual(calls[0]["footer_links"], [])
            self.assertIn(str(ck.resolve()), calls[0]["blocked_paths"])
            self.assertIn(str(ns["local_cache_root"]), calls[0]["blocked_paths"])
            self.assertIn(str(ns["local_lora_cache"]), calls[0]["blocked_paths"])
            self.assertNotIn(str(ck.parent), calls[0]["allowed_paths"])
            self.assertNotIn("Ai có link đều có thể dùng GPU", text.getvalue())
            self.assertIn("trycloudflare.com", text.getvalue())
            old_app = ns["studio_app"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(launch, ns)
            self.assertTrue(old_app.closed)
            self.assertEqual(len(calls), 2)

    def test_launch_can_opt_back_into_gradio_share(self):
        launch = "".join(
            json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][8]["source"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ck = root / "weights.safetensors"
            ck.touch()
            runtime = types.SimpleNamespace(
                output_dir=root / "outputs", backup_dir=root / "backup",
                checkpoint=ck, lora_paths={"anatomy": ck},
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

            ns = {
                "studio_runtime": runtime,
                "build_app": lambda _: FakeApp(),
                "local_cache_root": root / "wai_model_cache",
                "local_lora_cache": root / "wai_lora_cache",
                "GRADIO_SHARE": True,
            }
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                exec(launch, ns)
            self.assertIs(calls[0]["share"], True)
            self.assertIn("https://temporary.gradio.live", text.getvalue())
            self.assertIn("Ai có link đều có thể dùng GPU", text.getvalue())


class PromptLibraryTests(unittest.TestCase):
    """Nạp danh sách prompt từ file text rồi chọn một dòng để nạp vào UI."""

    LIBRARY_TEXT = (
        "=== 50 PROMPTS – ÁO SƠ MI VĂN PHÒNG ===\n"
        "\n"
        "Tên tiếng Việt | Nội dung prompts tiếng Anh\n"
        "\n"
        "Nhân vật công sở, trang phục lịch sự, ánh sáng trong trẻo\n"
        "\n"
        "PROMPT 01 - Nữ thư ký ngồi bàn làm việc mỉm cười\n"
        "Masterpiece, best quality, ultra-detailed anime style, side view,\n"
        "office worker at a tidy desk, white shirt, glasses, high detail.\n"
        "\n"
        "PROMPT 02 - Nữ thư ký đứng bên cửa sổ\n"
        "Prompt: Masterpiece, best quality, front view, worker by a window.\n"
        "Negative: lowres, bad hands, extra fingers\n"
        "Steps: 60\n"
        "CFG: 20\n"
        "Size: 832x1216\n"
        "Seed: 424242\n"
        "\n"
        "03 - Chuyến tàu cuối ngày\n"
        "Masterpiece, best quality, office worker on a night train.\n"
    )

    def test_parses_the_prompt_nn_file_format_and_clamps_parameters(self):
        library = studio.parse_prompt_library(self.LIBRARY_TEXT, name="50-prompts.txt")
        self.assertEqual(library["name"], "50 PROMPTS – ÁO SƠ MI VĂN PHÒNG")
        self.assertEqual([item["index"] for item in library["items"]], [1, 2, 3])
        self.assertEqual(
            library["items"][0]["title"], "Nữ thư ký ngồi bàn làm việc mỉm cười"
        )
        self.assertEqual(
            library["items"][0]["label"], "01 · Nữ thư ký ngồi bàn làm việc mỉm cười"
        )
        self.assertIn("Nhân vật công sở", library["description"])
        self.assertNotIn("\n", library["items"][0]["prompt"])
        tuned = library["items"][1]
        self.assertEqual(tuned["negative"], "lowres, bad hands, extra fingers")
        # Ngoài dải giao diện cho phép (10–45 steps, 1–12 CFG) thì bị kẹp lại.
        self.assertEqual((tuned["steps"], tuned["cfg"]), (45, 12.0))
        self.assertEqual(tuned["size"], "832x1216")
        self.assertEqual(tuned["seed"], 424242)
        self.assertTrue(
            tuned["prompt"].startswith("Masterpiece, best quality, front view"),
            'dòng "Prompt:" là nội dung, chỉ bỏ tiền tố',
        )
        self.assertNotIn("Negative:", tuned["prompt"])
        self.assertEqual(library["items"][2]["label"], "03 · Chuyến tàu cuối ngày")

    def test_reads_table_rows_json_and_plain_paragraphs(self):
        table = studio.parse_prompt_library(
            "=== 3 PROMPTS – BẢNG ===\n"
            "Tên tiếng Việt | Nội dung prompts tiếng Anh\n"
            "Cô gái bên hồ | 1girl, solo, reading by a lake at dawn, masterpiece\n"
            "Phố mưa neon | 1girl, solo, rainy neon street, reflective asphalt\n"
            "Vườn trên mây | 1girl, solo, floating garden above the clouds\n"
        )
        self.assertEqual(len(table["items"]), 3)
        self.assertEqual(table["items"][0]["title"], "Cô gái bên hồ")
        self.assertIn("floating garden", table["items"][2]["prompt"])

        parsed = studio.parse_prompt_library(
            json.dumps(
                [
                    {
                        "title": "Chân dung",
                        "prompt": "1girl, portrait, soft light",
                        "negative": "lowres, bad hands",
                        "steps": 28,
                        "cfg": 6,
                        "size": "832x1216",
                        "seed": 7,
                    },
                    {"name": "Phong cảnh", "en": "1girl, landscape, wide sky"},
                ]
            )
        )
        self.assertEqual([i["title"] for i in parsed["items"]], ["Chân dung", "Phong cảnh"])
        tuned_json = parsed["items"][0]
        self.assertEqual(tuned_json["negative"], "lowres, bad hands")
        self.assertEqual(tuned_json["steps"], 28)
        self.assertEqual(tuned_json["cfg"], 6.0)
        self.assertEqual(tuned_json["size"], "832x1216")
        self.assertEqual(tuned_json["seed"], 7)
        self.assertNotIn("Negative:", tuned_json["prompt"])

        plain = studio.parse_prompt_library(
            "1girl, under cherry blossoms, masterpiece\n\n"
            "1boy, rooftop at sunset, masterpiece"
        )
        self.assertEqual(len(plain["items"]), 2)
        self.assertIn("rooftop at sunset", plain["items"][1]["prompt"])

    def test_rejects_empty_and_unparseable_input(self):
        with self.assertRaisesRegex(ValueError, "File trống"):
            studio.parse_prompt_library("   ")
        with self.assertRaisesRegex(ValueError, "Không tìm thấy prompt nào"):
            studio.parse_prompt_library("Tên tiếng Việt | Nội dung prompts tiếng Anh")
        with self.assertRaisesRegex(ValueError, "Hãy dán nội dung"):
            studio.load_prompt_library_text("")
        with self.assertRaisesRegex(ValueError, "Hãy chọn file"):
            studio.read_prompt_library(None)

    def test_long_prompts_are_trimmed_to_the_ui_limit(self):
        library = studio.parse_prompt_library(
            "PROMPT 01 - Prompt rất dài\n" + "masterpiece, " * 400
        )
        self.assertEqual(len(library["items"][0]["prompt"]), studio.PROMPT_LIBRARY_LIMIT)
        self.assertTrue(library["items"][0]["truncated"])
        self.assertIn("đã bị cắt bớt", studio.prompt_library_status(library))

    def test_reads_uploaded_file_and_refuses_oversized_or_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            uploaded = root / "50-prompts.txt"
            uploaded.write_text(self.LIBRARY_TEXT, encoding="utf-8")
            library = studio.read_prompt_library(uploaded)
            # Tiêu đề "=== ... ===" trong file được dùng làm tên thư viện.
            self.assertEqual(library["name"], "50 PROMPTS – ÁO SƠ MI VĂN PHÒNG")
            self.assertEqual(len(library["items"]), 3)
            # Không có tiêu đề thì lấy tên file làm tên thư viện.
            anonymous = root / "danh-sach-rieng.txt"
            anonymous.write_text(
                "PROMPT 01 - Một prompt\n1girl, masterpiece, best quality",
                encoding="utf-8",
            )
            self.assertEqual(
                studio.read_prompt_library(anonymous)["name"], "danh-sach-rieng.txt"
            )
            with self.assertRaisesRegex(ValueError, "Không mở được file"):
                studio.read_prompt_library(root / "khong-ton-tai.txt")
            with patch.object(studio, "PROMPT_LIBRARY_MAX_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "quá lớn"):
                    studio.read_prompt_library(uploaded)

    def test_selection_fills_editable_fields_and_keeps_missing_parameters(self):
        items = studio.parse_prompt_library(self.LIBRARY_TEXT)["items"]
        current = ("prompt cũ", "negative cũ", 25, 6.0, -1, "1024x1024", "1024x1024")
        filled = studio.apply_prompt_choice(items[1]["label"], items, *current)
        self.assertEqual(filled[0], items[1]["prompt"])
        self.assertEqual(filled[1], "lowres, bad hands, extra fingers")
        self.assertEqual(filled[2:5], (45, 12.0, 424242))
        self.assertEqual(filled[5:7], ("832x1216", "832x1216"))
        self.assertIn("Đã nạp", filled[7])
        self.assertIn("Steps 45", filled[7])
        # Prompt không kèm thông số thì giữ nguyên thông số đang có.
        plain = studio.apply_prompt_choice(items[0]["label"], items, *current)
        self.assertEqual(plain[0], items[0]["prompt"])
        self.assertEqual(plain[1:7], current[1:])
        # Chưa chọn dòng nào hoặc thư viện rỗng: không đổi gì, chỉ báo trạng thái.
        for choice, entries in ((None, items), (items[0]["label"], ())):
            untouched = studio.apply_prompt_choice(choice, entries, *current)
            self.assertEqual(untouched[:7], current)
            self.assertIn("Chưa nạp thư viện", untouched[7])

    def test_sample_library_is_bundled_and_reparseable(self):
        library = studio.load_sample_prompt_library()
        self.assertEqual(len(library["items"]), 12)
        self.assertEqual(
            studio.prompt_library_choices(library["items"])["interactive"], True
        )
        self.assertEqual(studio.prompt_library_choices(())["choices"], ())
        self.assertEqual(studio.prompt_library_reset()[0], ())
        # Kích thước trong file mẫu phải là preset hợp lệ của giao diện.
        sized = next(item for item in library["items"] if item["size"])
        self.assertIn(sized["size"], studio.SIZE_PRESETS)
        # Prompt mẫu phải qua được bộ kiểm tra độ dài của runtime.
        for item in library["items"]:
            self.assertLessEqual(len(item["prompt"]), 2200)


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
        self.realesrgan_calls = []

        def fake_realesrgan_upscaler(image, size):
            self.realesrgan_calls.append((image.size, tuple(size)))
            return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)

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
            realesrgan_upscaler=fake_realesrgan_upscaler,
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

    def test_prompt_and_negative_reach_the_model_verbatim(self):
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
        # Không còn selector phong cách: prompt người dùng tự viết chính là prompt
        # gửi model, không bị ghép thêm thẻ nào.
        edited = "  my own portrait, watercolor style, no preset tags  "
        self.assertEqual(
            self.params(prompt=edited, negative="  custom negative  ")[:2],
            (edited, "  custom negative  "),
        )
        # Trigger LoRA mắt là nút bấm tường minh, thêm đúng một lần, sửa/xóa được.
        triggered, kept = studio.add_eyes_trigger("anime portrait", negative)
        self.assertIn("perfect eyes", triggered)
        self.assertEqual(triggered.count("perfect eyes"), 1)
        self.assertEqual(kept, negative)
        self.assertEqual(studio.add_eyes_trigger(triggered, negative)[0], triggered)
        # Trọng số và lớp nhấn không được tính là thẻ khác, nếu không nút chèn thêm bản trần.
        for already in ("(perfect eyes:1.1)", "((perfect eyes))", "((perfect eyes:1.2))"):
            again, _ = studio.add_eyes_trigger(f"{already}, anime portrait", negative)
            self.assertEqual(again, f"{already}, anime portrait")
            self.assertEqual(again.casefold().count("perfect eyes"), 1)
        # Gợi ý sửa vùng cũng là hành động tường minh của người dùng.
        repaired, repaired_neg = studio.apply_repair_hints(
            "portrait", "bad hands", "eyes"
        )
        self.assertIn("perfect eyes", repaired)
        self.assertIn("misaligned eyes", repaired_neg)
        with self.assertRaisesRegex(ValueError, "Chọn vùng sửa"):
            studio.apply_repair_hints("portrait", "", "unknown")

    def test_prompt_and_negative_have_no_keyword_or_length_filters(self):
        """Runtime không lọc từ khóa/độ dài: hai ô gửi model nguyên văn.

        Giao diện cũng không còn ô tick xác nhận 18+ (đã gỡ 02/10/2026), nên
        không có tham số nội dung nào trong `_parameters`.
        """
        self.assertFalse(hasattr(studio, "UNDERAGE_PROMPT"))
        self.assertFalse(hasattr(studio, "ADULT_PROMPT"))
        adult = "1girl, adult woman, nsfw, explicit, detailed anatomy"
        self.assertEqual(self.params(prompt=adult)[0], adult)
        # Không có hàng rào từ khóa: hai ô vẫn được truyền nguyên văn.
        formerly_blocked = "school girl portrait, teen"
        self.assertEqual(self.params(prompt=formerly_blocked)[0], formerly_blocked)
        self.assertNotIn("adult_confirmed", inspect.signature(
            studio.StudioRuntime._parameters
        ).parameters)
        self.assertEqual(
            self.params(prompt="adult", negative="underage, child")[1],
            "underage, child",
        )

    def test_validation_and_eye_trigger(self):
        self.assertEqual(self.params()[:2], ("anime portrait", "bad anatomy"))
        suggested, _ = studio.add_eyes_trigger("anime portrait", "")
        self.assertIn("perfect eyes", suggested)
        # Bật/tắt LoRA mắt không tự sửa prompt: chỉ nút bấm mới thêm trigger.
        self.assertEqual(self.params(eyes_enabled=True)[0], "anime portrait")
        self.assertEqual(self.params(eyes_enabled=False)[0], "anime portrait")
        for setting in (
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
        # Không còn preset: prompt người dùng sửa là đúng input của pipe.
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
        )
        image = next((self.root / "output").glob("*_*_2.png"))
        with Image.open(image) as result:
            parameters = json.loads(result.info["parameters"])
            self.assertEqual(parameters["seed"], 2)
            self.assertNotIn("style", parameters)
            self.assertEqual(parameters["prompt"], "  portrait without style tags  ")
            self.assertEqual(parameters["negative_prompt"], "  my negative  ")
            self.assertEqual(FakePipe.calls[-1][1]["prompt"], parameters["prompt"])
            self.assertEqual(
                FakePipe.calls[-1][1]["negative_prompt"], parameters["negative_prompt"]
            )

    def test_upscaler_output_must_match_selected_dimensions(self):
        class FakePILImage:
            def __init__(self, size):
                self.size = tuple(size)

            def convert(self, mode):
                self.converted_to = mode
                return self

        pil_api = types.ModuleType("PIL")
        pil_api.Image = types.SimpleNamespace(Image=FakePILImage)
        source = FakePILImage((512, 512))
        expected = (640, 640)
        with patch.dict(sys.modules, {"PIL": pil_api}):
            self.runtime.realesrgan_upscaler = lambda image, size: FakePILImage(size)
            result = self.runtime._upscale_with_realesrgan(source, expected)
            self.assertEqual(result.size, expected)
            self.assertEqual(result.converted_to, "RGB")
            self.runtime.realesrgan_upscaler = lambda image, size: FakePILImage(
                (512, 512)
            )
            with self.assertRaisesRegex(RuntimeError, "đúng kích thước"):
                self.runtime._upscale_with_realesrgan(source, expected)

    def test_realesrgan_cuda_oom_falls_back_to_cpu_and_cpu_oom_is_reported(self):
        class FakeModel:
            def to(self, device):
                self.device = device
                return self

            def float(self):
                self.is_float = True
                return self

        model = FakeModel()
        calls = []
        result = object()
        self.runtime._load_realesrgan_model = lambda: (model, "cuda")

        def infer(image, selected_model, device):
            calls.append(device)
            if device == "cuda":
                raise FakeTorch.cuda.OutOfMemoryError("fake GPU OOM")
            return result

        self.runtime._realesrgan_x4_on_device = infer
        self.assertIs(self.runtime._realesrgan_x4(object()), result)
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(self.runtime._realesrgan_device, "cpu")
        self.assertIs(self.runtime._realesrgan_model, model)

        self.runtime._load_realesrgan_model = lambda: (model, "cpu")
        self.runtime._realesrgan_x4_on_device = lambda *args: (_ for _ in ()).throw(
            FakeTorch.cuda.OutOfMemoryError("fake CPU OOM")
        )
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            self.runtime._realesrgan_x4(object())

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_realesrgan_tile_upscale_defines_result_buffer(self):
        """Phóng to phải ghi buffer `result_bgr`. Lỗi tên này làm tab phóng to chết."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy needed to exercise the Real-ESRGAN tile path")

        class FakeTensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            @property
            def shape(self):
                return self.array.shape

            def unsqueeze(self, axis):
                return FakeTensor(np.expand_dims(self.array, axis))

            def to(self, device=None, dtype=None):
                return self

            def div_(self, value):
                self.array = self.array / value
                return self

            def __getitem__(self, item):
                return FakeTensor(self.array[item])

            def detach(self):
                return self

            def float(self):
                return FakeTensor(self.array.astype(np.float32))

            def clamp_(self, low, high):
                self.array = np.clip(self.array, low, high)
                return self

            def mul_(self, value):
                self.array = self.array * value
                return self

            def round_(self):
                self.array = np.rint(self.array)
                return self

            def permute(self, *axes):
                return FakeTensor(np.transpose(self.array, axes))

            def numpy(self):
                return np.ascontiguousarray(self.array)

        def pad(tensor, pads, mode="reflect"):
            left, right, top, bottom = pads
            return FakeTensor(
                np.pad(
                    tensor.array,
                    ((0, 0), (0, 0), (top, bottom), (left, right)),
                    mode=mode,
                )
            )

        def identity_x4(tile):
            return FakeTensor(np.repeat(np.repeat(tile.array, 4, axis=-1), 4, axis=-2))

        torch_api = types.SimpleNamespace(
            float16=np.float16,
            float32=np.float32,
            uint8=np.uint8,
            from_numpy=FakeTensor,
            inference_mode=staticmethod(contextlib.nullcontext),
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(pad=pad)),
        )
        source = Image.new("RGB", (32, 24), "white")
        source.putpixel((1, 2), (10, 20, 30))
        source.putpixel((31, 23), (200, 100, 50))
        self.runtime.torch = torch_api
        result = self.runtime._realesrgan_x4_on_device(source, identity_x4, "cpu")
        self.assertEqual(result.size, (128, 96))
        self.assertEqual(result.getpixel((4, 8)), (10, 20, 30))
        self.assertEqual(result.getpixel((124, 92)), (200, 100, 50))
        self.assertNotIn("resultresult_bgr", inspect.getsource(studio.StudioRuntime._realesrgan_x4_on_device))

    def test_hires_size_is_multiple_of_8_and_clamped_to_pixel_budget(self):
        self.assertIsNone(studio._hires_size(1024, 1024, studio.HIRES_OFF))
        self.assertEqual(studio._hires_size(512, 512, "1.25×"), (640, 640, False))
        self.assertEqual(studio._hires_size(512, 512, "1.75×"), (896, 896, False))
        self.assertEqual(studio._hires_size(512, 512, "2×"), (1024, 1024, False))
        self.assertEqual(studio._hires_size(832, 1216, "1.5×"), (1248, 1824, False))
        self.assertEqual(studio._hires_size(1024, 1024, "2×")[:2], (2048, 2048))
        width, height, clamped = studio._hires_size(1344, 1024, "2×")  # 5,5 MP
        self.assertTrue(clamped)
        self.assertEqual((width % 8, height % 8), (0, 0))
        self.assertLessEqual(width * height, studio.HIRES_MAX_PIXELS)
        self.assertGreater(width, 1344)
        with self.assertRaisesRegex(ValueError, "độ phân giải cao"):
            studio._hires_size(512, 512, "4×")

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_hires_fix_upscales_then_refines_with_img2img(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        tiling = []
        self.pipe.enable_vae_tiling = lambda: tiling.append(True)
        args = ("anime cat", "bad paws", 20, 5.5, 7, 1, True, 0.55, False, 0.45, True)
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            # Tắt: một lượt, không tiling, kích thước gốc.
            _, paths, status, _ = self.runtime.text_to_image("512x512", *args)
            self.assertEqual(len(FakePipe.calls), 1)
            self.assertFalse(tiling)
            self.assertNotIn("hires", status)
            FakePipe.calls = []
            gallery, paths, status, _ = self.runtime.text_to_image(
                "512x512", *args, hires_scale="2×", hires_strength=0.3
            )
        self.assertEqual(len(FakePipe.calls), 2)
        base, refine = (call[1] for call in FakePipe.calls)
        self.assertNotIn("image", base)
        self.assertEqual((base["width"], base["height"]), (512, 512))
        self.assertEqual(refine["image"].size, (1024, 1024))
        self.assertEqual((refine["width"], refine["height"]), (1024, 1024))
        self.assertEqual(refine["strength"], 0.3)
        self.assertEqual((refine["prompt"], refine["negative_prompt"]), ("anime cat", "bad paws"))
        self.assertEqual(refine["generator"].seed, 7)  # cùng seed, tái lập được
        self.assertEqual(tiling, [True])
        self.assertIn("512×512 → 1024×1024", status)
        self.assertIn("1024×1024", gallery[0][1])
        with Image.open(paths[0]) as result:
            self.assertEqual(result.size, (1024, 1024))
            metadata = json.loads(result.info["parameters"])
        self.assertEqual((metadata["width"], metadata["height"]), (1024, 1024))
        self.assertEqual(
            metadata["hires"],
            {
                "scale": "2×",
                "upscaler": {
                    "name": studio.REAL_ESRGAN_MODEL["name"],
                    "version": studio.REAL_ESRGAN_MODEL["version"],
                    "sha256": studio.REAL_ESRGAN_MODEL["sha256"],
                },
                "base_width": 512,
                "base_height": 512,
                "strength": 0.3,
            },
        )
        self.assertEqual(
            self.realesrgan_calls[-1], ((512, 512), (1024, 1024))
        )
        self.assertIn("RealESRGAN_x4plus_anime_6B", status)
        for scale, expected in (("1.25×", (640, 640)), ("1.75×", (896, 896))):
            FakePipe.calls = []
            with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
                self.runtime.text_to_image(
                    "512x512", *args, hires_scale=scale, hires_strength=0.3
                )
            self.assertEqual(len(FakePipe.calls), 2)
            self.assertEqual(self.realesrgan_calls[-1], ((512, 512), expected))
            self.assertEqual(FakePipe.calls[-1][1]["image"].size, expected)
        with self.assertRaisesRegex(ValueError, "Hires strength"):
            self.runtime.text_to_image(
                "512x512", *args, hires_scale="2×", hires_strength=0.95
            )
        with self.assertRaisesRegex(ValueError, "độ phân giải cao"):
            self.runtime.text_to_image("512x512", *args, hires_scale="8×")

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_upscale_existing_image_skips_base_generation(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        args = ("portrait", "bad", 20, 6, 5, 1, False, 0.55, False, 0.45, True)
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            # 605×803 → làm tròn bội số 8 (608×800) rồi 2× = 1216×1600.
            gallery, paths, status, _ = self.runtime.upscale(
                Image.new("RGB", (605, 803), "teal"), "2×", 0.35, *args
            )
            self.assertEqual(len(FakePipe.calls), 1)  # chỉ lượt tinh chỉnh
            call = FakePipe.calls[0][1]
            self.assertEqual(call["image"].size, (1216, 1600))
            self.assertEqual(call["strength"], 0.35)
            self.assertIn("phóng to 608×800 → 1216×1600", status)
            with Image.open(paths[0]) as result:
                self.assertEqual(result.size, (1216, 1600))
                metadata = json.loads(result.info["parameters"])
            self.assertEqual(metadata["operation"], "upscale")
            self.assertIsNone(metadata["strength"])
            self.assertEqual(metadata["hires"]["base_width"], 608)
            self.assertEqual(
                metadata["hires"]["upscaler"],
                {
                    "name": studio.REAL_ESRGAN_MODEL["name"],
                    "version": studio.REAL_ESRGAN_MODEL["version"],
                    "sha256": studio.REAL_ESRGAN_MODEL["sha256"],
                },
            )
            self.assertEqual(self.realesrgan_calls[-1], ((608, 800), (1216, 1600)))
            self.assertIn("upscale", Path(paths[0]).name)
            for scale, expected in (("1.25×", (760, 1000)), ("1.75×", (1064, 1400))):
                FakePipe.calls = []
                self.runtime.upscale(
                    Image.new("RGB", (605, 803), "teal"), scale, 0.35, *args
                )
                self.assertEqual(self.realesrgan_calls[-1], ((608, 800), expected))
                self.assertEqual(FakePipe.calls[0][1]["image"].size, expected)
            for source, scale, message in (
                (None, "2×", "Tải ảnh"),
                (Image.new("RGB", (512, 512)), studio.HIRES_OFF, "Chọn hệ số"),
                (Image.new("RGB", (128, 512)), "2×", "cạnh ngắn"),
                (Image.new("RGB", (2048, 2048)), "2×", "giới hạn"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    self.runtime.upscale(source, scale, 0.4, *args)

    @unittest.skipIf(Image is None, "Pillow needed for raster smoke tests")
    def test_hires_fix_works_for_img2img_and_is_rejected_for_inpaint(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            _, paths, status, _ = self.runtime.image_to_image(
                Image.new("RGB", (600, 500), "navy"),
                "768x768",
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
                hires_scale="1.5×",
            )
            self.assertEqual(
                [call[1]["image"].size for call in FakePipe.calls],
                [(768, 768), (1152, 1152)],
            )
            self.assertEqual(FakePipe.calls[0][1]["strength"], 0.5)
            self.assertEqual(
                FakePipe.calls[1][1]["strength"], studio.HIRES_DEFAULT_STRENGTH
            )
            with Image.open(paths[0]) as result:
                self.assertEqual(result.size, (1152, 1152))
            region = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(region).rectangle((10, 10, 60, 60), fill=255)
            with self.assertRaisesRegex(ValueError, "không áp dụng"):
                self.runtime._generate(
                    "inpaint",
                    Image.new("RGB", (512, 512)),
                    region,
                    "hands",
                    0,
                    0.5,
                    None,
                    "x",
                    "",
                    20,
                    6,
                    1,
                    1,
                    False,
                    0.5,
                    False,
                    0.5,
                    False,
                    hires_scale="2×",
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
        with patch.object(
            studio, "prime_prompt_tag_catalog", return_value="✅ CSV kho gợi ý đã sẵn sàng."
        ) as preload:
            demo = studio.build_app(self.runtime)
        preload.assert_called_once_with()
        config = demo.get_config_file()
        self.assertEqual(
            len([c for c in config["components"] if c["type"] == "imageeditor"]), 1
        )
        self.assertEqual(
            len([c for c in config["components"] if c["type"] == "gallery"]), 1
        )
        # Không còn selector phong cách: người dùng tự viết prompt phong cách.
        labels = [str(c["props"].get("label") or "") for c in config["components"]]
        self.assertNotIn("Phong cách hình ảnh", labels)
        self.assertFalse(any("Ý tưởng gốc" in label for label in labels))
        self.assertFalse(any("sửa được" in label for label in labels))
        self.assertFalse(hasattr(studio, "STYLE_PRESETS"))
        self.assertFalse(hasattr(studio, "compose_style_prompts"))
        fields = [
            c
            for c in config["components"]
            if c["type"] == "textbox" and "gửi model" in str(c["props"].get("label"))
        ]
        self.assertEqual(len(fields), 2)
        prompt_field, negative_field = fields
        self.assertTrue(all(c["props"].get("interactive", True) for c in fields))
        self.assertEqual(prompt_field["props"]["value"], studio.DEFAULT_PROMPT)
        self.assertEqual(negative_field["props"]["value"], studio.DEFAULT_NEGATIVE)
        field_ids = {c["id"] for c in fields}

        inline_suggestions = next(
            c for c in config["components"]
            if c["type"] == "dropdown"
            and "Semi-auto tag" in str(c["props"].get("label"))
        )
        inline_status = next(
            c for c in config["components"]
            if c["type"] == "markdown"
            and "`*giữa*` để tìm tag theo hậu tố/nội dung" in str(c["props"].get("value"))
        )
        self.assertFalse(any(
            c["type"] == "button"
            and "thêm tag đã chọn" in c["props"].get("value", "").casefold()
            for c in config["components"]
        ))
        suggestion_limit = next(
            c for c in config["components"]
            if c["props"].get("elem_id") == "prompt-tag-limit"
        )
        self.assertEqual(suggestion_limit["type"], "dropdown")
        self.assertEqual(suggestion_limit["props"]["label"], "Số gợi ý")
        self.assertEqual(
            [choice[1] for choice in suggestion_limit["props"]["choices"]],
            list(studio.PROMPT_TAG_SUGGESTION_CHOICES),
        )
        self.assertEqual(suggestion_limit["props"]["value"], studio.PROMPT_TAG_SUGGESTION_LIMIT)
        limit_change_event = next(
            d for d in config["dependencies"]
            if (suggestion_limit["id"], "change") in d["targets"]
        )
        self.assertEqual(
            limit_change_event["outputs"], [inline_suggestions["id"], inline_status["id"]]
        )
        live_suggestion_event = next(
            d for d in config["dependencies"]
            if (prompt_field["id"], "input") in d["targets"]
        )
        self.assertEqual(
            live_suggestion_event["inputs"], [prompt_field["id"], suggestion_limit["id"]]
        )
        self.assertEqual(
            live_suggestion_event["outputs"], [inline_suggestions["id"], inline_status["id"]]
        )
        # queue=True: quét catalog 349k dòng tốn tới ~1,5 s CPU, nếu chạy trên
        # event loop (queue=False) thì mọi request khác của trang phải chờ => treo.
        self.assertTrue(live_suggestion_event["queue"])
        self.assertTrue(live_suggestion_event["trigger_mode"], "always_last")
        apply_suggestion_event = next(
            d for d in config["dependencies"]
            if (inline_suggestions["id"], "input") in d["targets"]
        )
        self.assertEqual(
            apply_suggestion_event["inputs"], [prompt_field["id"], inline_suggestions["id"]]
        )
        self.assertEqual(
            apply_suggestion_event["outputs"],
            [prompt_field["id"], inline_status["id"], inline_suggestions["id"]],
        )
        self.assertTrue(apply_suggestion_event["queue"])

        eyes_button = next(
            c
            for c in config["components"]
            if c["type"] == "button"
            and "perfect eyes" in c["props"].get("value", "")
        )
        eyes_event = next(
            d
            for d in config["dependencies"]
            if (eyes_button["id"], "click") in d["targets"]
        )
        self.assertEqual(set(eyes_event["outputs"]), field_ids)
        self.assertFalse(eyes_event["queue"])
        self.assertTrue(
            any(
                c["type"] == "markdown"
                and "Chế độ:** GPU trực tiếp" in str(c["props"].get("value"))
                for c in config["components"]
            )
        )
        repair_event = next(
            d
            for d in config["dependencies"]
            if set(d["outputs"]) == field_ids and len(d["targets"]) == 1
            and len(d["inputs"]) == 3
        )
        self.assertEqual(
            repair_event["inputs"][:2],
            [prompt_field["id"], negative_field["id"]],
        )
        # Gồm 2 sự kiện gợi ý tag inline, 2 chỉnh trọng số, 2 kho thẻ,
        # 3 chuyển tab ảnh, các sự kiện của ô chọn ảnh nguồn (gallery.change/select, ↻,
        # 4 nút nạp + 4 bước chuyển tab sau khi nạp, demo.load) và một listener Ctrl+↑/↓.
        # Con số này là "mọi thứ phải có", không phải số sự kiện tối đa: thêm handler ở
        # test này để bắt buộc cập nhật dòng trên khi giao diện đổi.
        self.assertEqual(len(config["dependencies"]), 37)
        self.assertEqual(prompt_field["props"].get("elem_id"), "studio-prompt")
        for elem_id, target in (
            ("prompt-weight-down", "click"),
            ("prompt-weight-up", "click"),
        ):
            button = next(
                c for c in config["components"] if c["props"].get("elem_id") == elem_id
            )
            event = next(
                d for d in config["dependencies"]
                if (button["id"], target) in d["targets"]
            )
            self.assertEqual(event["inputs"], [prompt_field["id"]])
            self.assertEqual(
                event["outputs"],
                [prompt_field["id"], inline_status["id"], inline_suggestions["id"]],
            )
            self.assertFalse(event["queue"])
            # JS phải được Gradio gửi xuống trình duyệt, không chỉ nằm trong Python.
            self.assertEqual(event["js"], studio.PROMPT_TAG_WEIGHT_SELECTION_JS)
        self.assertIn("selectionStart", studio.PROMPT_TAG_WEIGHT_SELECTION_JS)
        self.assertIn("ArrowUp", studio.PROMPT_TAG_WEIGHT_SHORTCUT_JS)
        self.assertIn("ArrowDown", studio.PROMPT_TAG_WEIGHT_SHORTCUT_JS)
        # Ba listener lúc mở trang: Ctrl+↑/↓ (chỉ JS), giám sát kết nối (chỉ JS),
        # và ô chọn ảnh nguồn.
        load_events = [
            d for d in config["dependencies"]
            if any(target[1] == "load" for target in d["targets"])
        ]
        self.assertEqual(len(load_events), 3)
        self.assertEqual(
            sum(1 for d in load_events if not d.get("js") and d["queue"]), 1,
            "chỉ sự kiện nạp danh sách ảnh nguồn là gọi handler",
        )
        load_event = next(d for d in load_events if d.get("js"))
        self.assertEqual(load_event["js"], studio.PROMPT_TAG_WEIGHT_SHORTCUT_JS)
        self.assertTrue(any(not d.get("js") and d["queue"] for d in load_events),
                        "ô chọn ảnh nguồn phải nạp danh sách ảnh qua queue")
        # Six horizontal workspace tabs plus four generation modes.
        tabs = [c for c in config["components"] if c["type"] == "tabitem"]
        self.assertEqual(
            [c["props"].get("id") for c in tabs],
            ["create", "text", "image", "upscale", "inpaint", "tags",
             "library", "workflow", "settings", "details"],
        )
        tab_groups = [c for c in config["components"] if c["type"] == "tabs"]
        self.assertEqual([c["props"].get("selected") for c in tab_groups], ["create", "text"])
        self.assertFalse(any(c["type"] == "accordion" for c in config["components"]))
        self.assertIn("overflow-x: auto", demo.studio_css)
        # Không còn cảnh báo/ô tick xác nhận nội dung trong giao diện.
        self.assertFalse(
            any("18 tuổi" in str(c["props"]) for c in config["components"])
        )
        detailer_dropdown = next(
            c
            for c in config["components"]
            if c["type"] == "dropdown" and "auto-detailer" in str(c["props"].get("label"))
        )
        self.assertEqual(
            [choice[0] for choice in detailer_dropdown["props"]["choices"]],
            list(studio.DETAILER_TARGETS),
        )
        self.assertEqual(detailer_dropdown["props"]["value"], studio.DETAILER_OFF)
        detailer_sliders = [
            c
            for c in config["components"]
            if c["type"] == "slider"
            and any(
                key in str(c["props"].get("label"))
                for key in ("Detailer strength", "Ngưỡng phát hiện", "Số vùng tối đa")
            )
        ]
        self.assertEqual(len(detailer_sliders), 3)
        detailer_ids = [detailer_dropdown["id"]] + [c["id"] for c in detailer_sliders]
        # 8 bộ negative tối ưu + khung prompt + nút kiểm tra prompt đều có mặt.
        negative_choice = next(
            c
            for c in config["components"]
            if c["type"] == "dropdown"
            and "Negative tối ưu" in str(c["props"].get("label"))
        )
        self.assertEqual(
            [choice[0] for choice in negative_choice["props"]["choices"]],
            [label for label, _ in studio.negative_preset_choices()],
        )
        self.assertTrue(
            any(
                c["type"] == "dropdown"
                and "Khung prompt" in str(c["props"].get("label"))
                for c in config["components"]
            )
        )
        self.assertTrue(
            any(
                c["type"] == "button" and "Kiểm tra prompt" in str(c["props"].get("value"))
                for c in config["components"]
            )
        )
        hires_fields = [
            c
            for c in config["components"]
            if "hires" in str(c["props"].get("label") or "").lower()
        ]
        # Match roles, not visual order: generation modes now precede Details.
        hires_fields.sort(key=lambda c: {
            "Độ phân giải cao (hires fix)": 0,
            "Hires strength (chi tiết thêm vào)": 1,
            "Hires strength": 2,
        }[c["props"]["label"]])
        # Dropdown + slider dùng chung cho 2 tab tạo ảnh; slider thứ hai thuộc tab
        # Phóng to ảnh (dropdown "Hệ số phóng" không chứa chữ hires).
        self.assertEqual(
            [c["type"] for c in hires_fields], ["dropdown", "slider", "slider"]
        )
        hires_ids = [c["id"] for c in hires_fields[:2]]
        self.assertEqual(
            [choice[0] for choice in hires_fields[0]["props"]["choices"]],
            list(studio.HIRES_SCALES),
        )
        self.assertEqual(hires_fields[0]["props"]["value"], studio.HIRES_OFF)
        upscale_scale_field = next(
            c
            for c in config["components"]
            if c["type"] == "dropdown" and c["props"].get("label") == "Hệ số phóng"
        )
        self.assertEqual(
            [choice[0] for choice in upscale_scale_field["props"]["choices"]],
            [label for label in studio.HIRES_SCALES if label != studio.HIRES_OFF],
        )
        self.assertEqual(upscale_scale_field["props"]["value"], "2×")
        for dep in config["dependencies"][:2]:  # text, img2img: có hires + detailer
            position = dep["inputs"].index(prompt_field["id"])
            self.assertEqual(
                dep["inputs"][position + 1], negative_field["id"]
            )
            self.assertEqual(dep["inputs"][-6:-4], hires_ids)
            self.assertEqual(dep["inputs"][-4:], detailer_ids)
        upscale_dep = config["dependencies"][2]  # phóng to: hệ số/strength riêng
        self.assertFalse(set(hires_ids) & set(upscale_dep["inputs"]))
        self.assertFalse(set(detailer_ids) & set(upscale_dep["inputs"]))
        position = upscale_dep["inputs"].index(prompt_field["id"])
        self.assertEqual(upscale_dep["inputs"][position + 1], negative_field["id"])
        inpaint_dep = config["dependencies"][3]  # sửa vùng: không hires/detailer
        self.assertFalse(set(hires_ids) & set(inpaint_dep["inputs"]))
        self.assertFalse(set(detailer_ids) & set(inpaint_dep["inputs"]))
        position = inpaint_dep["inputs"].index(prompt_field["id"])
        self.assertEqual(inpaint_dep["inputs"][position + 1], negative_field["id"])
        self.assertTrue(
            all(x["api_visibility"] == "private" for x in config["dependencies"])
        )
        self.assertTrue(demo.studio_css)
        self.assertIsNotNone(demo.studio_theme)

    @unittest.skipIf(
        Image is None or not importlib.util.find_spec("gradio"),
        "Gradio and Pillow needed for the semi-auto tag route test",
    )
    def test_semi_auto_tag_events_suggest_then_replace_then_weight(self):
        """Kiểm tra trọn tuyến semi-auto: gõ → dropdown CSV → chọn tag → nút ±."""
        import asyncio
        from gradio.state_holder import SessionState

        rows = studio.load_csv_tags()
        with patch.object(
            studio,
            "prime_prompt_tag_catalog",
            return_value="✅ CSV kho gợi ý đã sẵn sàng.",
        ):
            demo = studio.build_app(self.runtime)
        config = demo.get_config_file()

        def component(kind, needle, prop):
            return next(
                c for c in config["components"]
                if c["type"] == kind and needle in str(c["props"].get(prop))
            )

        prompt_box = component("textbox", "gửi model", "label")
        dropdown = component("dropdown", "Semi-auto tag", "label")
        weight_up = next(
            c for c in config["components"]
            if c["props"].get("elem_id") == "prompt-weight-up"
        )

        def event_index(component_id, event):
            return next(
                d["id"] for d in config["dependencies"]
                if (component_id, event) in d["targets"]
            )

        state = SessionState(demo)

        async def flow():
            async def run(index, inputs):
                result = await demo.process_api(
                    index, inputs, state=state, explicit_call=True
                )
                return result["data"]

            with patch.object(studio, "load_csv_tags", return_value=rows):
                suggested, note = await run(
                    event_index(prompt_box["id"], "input"), ["1girl, mắt", 16]
                )
            self.assertEqual(suggested["__type__"], "update")
            values = [value for _, value in suggested["choices"]]
            self.assertIn("blue_eyes", values)
            self.assertTrue(all(label.startswith(("[G]", "<")) for label, _ in suggested["choices"]))
            self.assertIn("tag từ CSV", note)
            self.assertEqual(len(suggested["choices"]), 16)

            with patch.object(studio, "load_csv_tags", return_value=rows):
                prompt, applied_note, cleared = await run(
                    event_index(dropdown["id"], "input"), ["1girl, mắt", "blue_eyes"]
                )
            self.assertEqual(prompt, "1girl, blue_eyes")
            self.assertIn("thay `mắt`", applied_note)
            self.assertEqual(cleared["choices"], [])

            # ô "Số gợi ý" nới trần danh sách mà không đổi kết quả tìm
            with patch.object(studio, "load_csv_tags", return_value=rows):
                wider, wider_note = await run(
                    event_index(prompt_box["id"], "input"), ["1girl, mắt", 150]
                )
            self.assertGreater(len(wider["choices"]), 16)
            self.assertIn("tăng **Số gợi ý** để xem tiếp", wider_note)
            far_tag = wider["choices"][-1][1]
            self.assertNotIn(far_tag, [value for _, value in suggested["choices"]])
            # tag ngoài top-16 mặc định vẫn được chấp nhận
            with patch.object(studio, "load_csv_tags", return_value=rows):
                applied_far, far_note, _ = await run(
                    event_index(dropdown["id"], "input"), ["1girl, mắt", far_tag]
                )
            self.assertEqual(applied_far, f"1girl, {far_tag}")
            self.assertIn("thay `mắt`", far_note)

            with patch.object(studio, "load_csv_tags", return_value=rows):
                weighted, weight_note, _ = await run(
                    event_index(weight_up["id"], "click"), ["1girl, blue_eyes"]
                )
            self.assertEqual(weighted, "1girl, (blue_eyes:1.1)")
            self.assertIn("tăng trọng số", weight_note)

        asyncio.run(flow())

    @unittest.skipIf(
        not importlib.util.find_spec("gradio"), "Gradio needed for UI wiring test"
    )
    def test_prompt_library_ui_loads_file_and_selection_fills_prompt(self):
        """Nạp file prompt trong UI Gradio: ra danh sách, chọn một dòng là nạp."""
        import asyncio
        from gradio.state_holder import SessionState

        uploaded = self.root / "50-prompts.txt"
        uploaded.write_text(
            "=== 2 PROMPTS – THƯ VIỆN KIỂM THỬ ===\n"
            "\n"
            "Tên tiếng Việt | Nội dung prompts tiếng Anh\n"
            "\n"
            "PROMPT 01 - Cô gái dưới hoa anh đào\n"
            "1girl, solo, under cherry blossoms, masterpiece, best quality\n"
            "Negative: lowres, bad hands, extra fingers\n"
            "Steps: 30\n"
            "\n"
            "PROMPT 02 - Nam thanh niên trên sân thượng\n"
            "1boy, solo, rooftop at night, neon city, masterpiece, best quality\n",
            encoding="utf-8",
        )
        empty = self.root / "rong.txt"
        empty.write_text(
            "Tên tiếng Việt | Nội dung prompts tiếng Anh", encoding="utf-8"
        )

        demo = studio.build_app(self.runtime)
        config = demo.get_config_file()
        components = config["components"]

        def component(kind, needle, field="label"):
            return next(
                c
                for c in components
                if c["type"] == kind and needle in str(c["props"].get(field, ""))
            )

        upload = component("file", "danh sách prompt")
        self.assertEqual(upload["props"]["file_types"], [".txt", ".md", ".json"])
        choice = component("radio", "Chọn prompt để nạp")
        self.assertEqual(choice["props"]["choices"], [])
        self.assertFalse(choice["props"]["interactive"])
        self.assertIn("studio-prompt-list", choice["props"].get("elem_classes", []))
        paste = component("button", "Đọc danh sách đã dán", "value")
        sample = component("button", "thư viện mẫu", "value")
        state = next(c for c in components if c["type"] == "state")
        library_box = component("markdown", "Chưa nạp thư viện", "value")
        prompt_box = component("textbox", "Prompt gửi model")
        negative_box = component("textbox", "Negative gửi model")
        steps_box = component("slider", "Số bước (steps)")
        cfg_box = component("slider", "CFG / độ bám prompt")
        seed_box = component("number", "Seed (-1 = ngẫu nhiên)")
        sizes = [
            c
            for c in components
            if c["type"] == "dropdown"
            and str(c["props"].get("label", "")).startswith("Kích thước")
        ]
        self.assertEqual(len(sizes), 2)

        library_outputs = [state["id"], choice["id"], library_box["id"]]
        library_events = [
            d for d in config["dependencies"] if d["outputs"] == library_outputs
        ]
        self.assertEqual(
            {tuple(targets) for d in library_events for targets in d["targets"]},
            {
                (upload["id"], "upload"),
                (upload["id"], "clear"),
                (paste["id"], "click"),
                (sample["id"], "click"),
            },
        )
        select_event = next(
            d for d in config["dependencies"] if (choice["id"], "select") in d["targets"]
        )
        parameters = [
            prompt_box["id"],
            negative_box["id"],
            steps_box["id"],
            cfg_box["id"],
            seed_box["id"],
            sizes[0]["id"],
            sizes[1]["id"],
        ]
        self.assertEqual(
            select_event["inputs"], [choice["id"], state["id"], *parameters]
        )
        self.assertEqual(select_event["outputs"], [*parameters, library_box["id"]])
        # Nút nạp dự phòng dùng chung tham số với sự kiện chọn dòng (tiện trên
        # điện thoại khi thao tác chạm vào danh sách không như ý).
        button = component("button", "Nạp prompt đã chọn", "value")
        button_event = next(
            d for d in config["dependencies"] if (button["id"], "click") in d["targets"]
        )
        self.assertEqual(button_event["inputs"], [choice["id"], state["id"], *parameters])
        self.assertEqual(button_event["outputs"], select_event["outputs"])

        session = SessionState(demo)

        def fn_index(trigger):
            return next(
                d["id"] for d in config["dependencies"] if trigger in d["targets"]
            )

        def file_data(path):
            return {
                "path": str(path),
                "meta": {"_type": "gradio.FileData"},
                "orig_name": Path(path).name,
            }

        async def process(trigger, inputs):
            return (
                await demo.process_api(
                    fn_index(trigger), inputs, state=session, explicit_call=True
                )
            )["data"]

        async def flow():
            _, dropdown, status = await process(
                (upload["id"], "upload"), [file_data(uploaded)]
            )
            self.assertEqual(
                [entry[0] for entry in dropdown["choices"]],
                ["01 · Cô gái dưới hoa anh đào", "02 · Nam thanh niên trên sân thượng"],
            )
            self.assertTrue(dropdown["interactive"])
            self.assertIsNone(dropdown["value"])
            self.assertIn("2 prompt đã nạp", status)
            self.assertIn("THƯ VIỆN KIỂM THỬ", status)
            items = session[state["id"]]
            self.assertEqual(len(items), 2)

            # Chọn một dòng: prompt + negative + steps trong file được nạp.
            filled = await process(
                (choice["id"], "select"),
                [
                    items[0]["label"],
                    None,
                    "prompt cũ",
                    "negative cũ",
                    25,
                    6.0,
                    -1,
                    "1024x1024",
                    "1024x1024",
                ],
            )
            self.assertEqual(filled[0], items[0]["prompt"])
            self.assertEqual(filled[1], "lowres, bad hands, extra fingers")
            self.assertEqual(int(filled[2]), 30)
            self.assertEqual(filled[3:7], [6.0, -1, "1024x1024", "1024x1024"])
            self.assertIn("Đã nạp", filled[7])
            self.assertIn("Prompt gửi model", filled[7])

            # Nút nạp dự phòng (dùng trên điện thoại) chạy đúng cùng logic: chọn
            # sẵn một dòng rồi bấm nút là prompt được nạp như khi chạm dòng.
            by_button = await process(
                (button["id"], "click"),
                [
                    items[1]["label"],
                    None,
                    "prompt cũ",
                    "negative cũ",
                    25,
                    6.0,
                    -1,
                    "1024x1024",
                    "1024x1024",
                ],
            )
            self.assertEqual(by_button[0], items[1]["prompt"])
            self.assertIn("Đã nạp", by_button[7])

            # Dán nội dung thay vì tải file lên.
            _, pasted, pasted_status = await process(
                (paste["id"], "click"),
                [
                    "PROMPT 01 - A\n1girl, masterpiece, best quality\n\n"
                    "PROMPT 02 - B\n1boy, masterpiece, best quality"
                ],
            )
            self.assertEqual(len(pasted["choices"]), 2)
            self.assertIn("Danh sách đã dán", pasted_status)

            # Thư viện mẫu đóng kèm để xem định dạng.
            _, sampled, sample_status = await process((sample["id"], "click"), [])
            self.assertEqual(len(sampled["choices"]), 12)
            self.assertIn("12 prompt đã nạp", sample_status)

            # File không có prompt: giữ danh sách cũ, chỉ báo lỗi ở trạng thái.
            skipped, skipped_dropdown, error = await process(
                (upload["id"], "upload"), [file_data(empty)]
            )
            self.assertIsNone(skipped)
            self.assertNotIn("choices", skipped_dropdown)
            self.assertIn("Không tìm thấy prompt nào", error)

            # Gỡ file đã nạp: danh sách rỗng trở lại.
            _, cleared, cleared_status = await process((upload["id"], "clear"), [])
            self.assertEqual(cleared["choices"], [])
            self.assertFalse(cleared["interactive"])
            self.assertEqual(cleared_status, "Đã gỡ thư viện prompt.")

        asyncio.run(flow())

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

        # Tra chỉ số theo nhãn nút thật thay vì số thứ tự cứng: thêm/bớt sự kiện
        # gợi ý tag không làm lệch các phép thử bên dưới.
        config = demo.get_config_file()

        def index_of(needle, event="click"):
            button = next(
                c for c in config["components"]
                if c["type"] == "button" and needle in str(c["props"].get("value"))
            )
            return next(
                d["id"] for d in config["dependencies"] if (button["id"], event) in d["targets"]
            )

        generate_index = index_of("Tạo ảnh từ prompt")
        image_index = index_of("Biến đổi ảnh")
        upscale_index = index_of("Phóng to ảnh")
        inpaint_index = index_of("Sửa vùng đã tô")
        eyes_index = index_of("perfect eyes")
        repair_index = index_of("Thêm gợi ý sửa vùng")
        to_image_index = index_of("→ ◈ Biến đổi")
        to_upscale_index = index_of("→ ⤢ Phóng to")
        to_inpaint_index = index_of("→ ✎ Sửa vùng")
        to_all_index = index_of("Nạp ảnh đã chọn vào cả ba tab")

        async def process(index, inputs):
            return (
                await demo.process_api(index, inputs, state=state, explicit_call=True)
            )["data"]

        def shared(positive, negative, eyes=False):
            return [
                positive,
                negative,
                20,
                6,
                42,
                1,
                False,
                0.55,
                eyes,
                0.45,
                False,
            ]

        # Auto-detailer mặc định Tắt: chỉ bốn tham số cấu hình được truyền vào
        # sự kiện tạo ảnh, không đổi số lượt gọi pipe.
        detailer_args = [
            studio.DETAILER_OFF,
            studio.DETAILER_DEFAULT_STRENGTH,
            studio.DETAILER_DEFAULT_CONF,
            studio.DETAILER_DEFAULT_MAX,
        ]

        async def smoke():
            # Không còn preset phong cách: nút trigger mắt chỉ thêm "perfect eyes"
            # vào đúng ô prompt đang hiển thị, negative giữ nguyên.
            triggered = await process(
                eyes_index, [studio.DEFAULT_PROMPT, studio.DEFAULT_NEGATIVE]
            )
            self.assertIn("perfect eyes", triggered[0])
            self.assertEqual(triggered[1], studio.DEFAULT_NEGATIVE)
            self.assertEqual(await process(eyes_index, triggered), triggered)
            # Nút gợi ý sửa vùng cũng chỉ đổi hai ô đang hiển thị.
            leg_prompts = await process(repair_index, [*triggered, "legs"])
            self.assertIn("natural toes", leg_prompts[0])
            self.assertIn("broken legs", leg_prompts[1])
            self.assertEqual(await process(repair_index, [*leg_prompts, "legs"]), leg_prompts)
            adult_prompts = (
                "1girl, adult woman, nsfw, explicit, portrait",
                "bad hands",
            )
            for index, inputs, expected in (
                (
                    generate_index,
                    [
                        "512x512",
                        *shared(
                            "  my own portrait  ", "  no hidden tags  ", eyes=True
                        ),
                        studio.HIRES_OFF,
                        0.4,
                        *detailer_args,
                    ],
                    ("  my own portrait  ", "  no hidden tags  "),
                ),
                (
                    image_index,
                    [
                        file_data(source),
                        "512x512",
                        0.45,
                        *shared("paint this picture", "my bad quality"),
                        studio.HIRES_OFF,
                        0.4,
                        *detailer_args,
                    ],
                    ("paint this picture", "my bad quality"),
                ),
                (
                    inpaint_index,
                    [
                        editor,
                        None,
                        "hands",
                        0.45,
                        8,
                        *shared("no repair suggestions", "bad hands"),
                    ],
                    ("no repair suggestions", "bad hands"),
                ),
                (
                    inpaint_index,
                    [
                        editor,
                        file_data(mask),
                        "legs",
                        0.45,
                        8,
                        *shared(*leg_prompts),
                    ],
                    tuple(leg_prompts),
                ),
                # Prompt do người dùng tự viết; không còn ô tick xác nhận nào.
                (
                    generate_index,
                    [
                        "512x512",
                        *shared(*adult_prompts),
                        "Tắt",
                        0.4,
                        *detailer_args,
                    ],
                    adult_prompts,
                ),
            ):
                data = await process(index, inputs)
                self.assertIn("✅ Đã tạo 1 ảnh", data[2])
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
            # Hires fix qua đường sự kiện UI: 512×512 → 1,5× = 768×768, 2 lượt pipe.
            calls_before = len(FakePipe.calls)
            data = await process(
                generate_index,
                [
                    "512x512",
                    *shared("hires portrait", "bad"),
                    "1.5×",
                    0.35,
                    *detailer_args,
                ],
            )
            self.assertIn("512×512 → 768×768", data[2])
            self.assertEqual(len(FakePipe.calls), calls_before + 2)
            self.assertEqual(FakePipe.calls[-1][1]["strength"], 0.35)
            with Image.open(data[1][0]["path"]) as result:
                self.assertEqual(result.size, (768, 768))
            del FakePipe.calls[calls_before:]
            # Tab Phóng to ảnh qua sự kiện UI: 512×512 → 1024×1024, một lượt pipe.
            calls_before = len(FakePipe.calls)
            data = await process(
                upscale_index, [file_data(source), "2×", 0.4, *shared("upscale me", "bad")]
            )
            self.assertIn("phóng to 512×512 → 1024×1024", data[2])
            self.assertEqual(len(FakePipe.calls), calls_before + 1)
            with Image.open(data[1][0]["path"]) as result:
                self.assertEqual(result.size, (1024, 1024))
            del FakePipe.calls[calls_before:]
            # Không còn ô tick/không có bộ lọc: prompt tới pipe nguyên văn.
            await process(
                generate_index,
                [
                    "512x512",
                    *shared(*adult_prompts),
                    "Tắt",
                    0.4,
                    *detailer_args,
                ],
            )
            self.assertEqual(
                (
                    FakePipe.calls[-1][1]["prompt"],
                    FakePipe.calls[-1][1]["negative_prompt"],
                ),
                adult_prompts,
            )
            self.assertEqual(len(FakePipe.calls), 6)
            # Ảnh nguồn giờ do người dùng CHỈ ĐỊNH (không còn luôn lấy ảnh mới nhất):
            # một lần nạp phải vào cả ba tab, và phải đúng file đã chọn.
            chosen = data[1][-1]["path"]
            chosen_size = Image.open(chosen).size
            loaded = await process(to_all_index, [chosen])
            # Gradio có thể tự lưu ảnh thành file tạm tên khác, nên thứ phải kiểm tra
            # là đúng ảnh (kích thước) và đúng tên file trong dòng trạng thái.
            for slot in (loaded[0], loaded[1], loaded[2]["background"]):
                path = Path(slot["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(Image.open(path).size, chosen_size)
            self.assertIn("Đã nạp", loaded[3])
            self.assertIn(Path(chosen).name, loaded[3])
            for index in (to_image_index, to_upscale_index):
                value = (await process(index, [chosen]))[0]
                self.assertEqual(Image.open(value["path"]).size, chosen_size)
            editor_value = (await process(to_inpaint_index, [chosen]))[0]
            self.assertEqual(Image.open(editor_value["background"]["path"]).size, chosen_size)
            # Chưa chọn ảnh thì phải báo lỗi rõ, không âm thầm lấy một ảnh khác.
            with self.assertRaises(Exception):
                await process(to_all_index, [None])

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


class SourceImagePickerTests(unittest.TestCase):
    """Ô chọn ảnh nguồn cho tab Sửa vùng / Phóng to / Biến đổi (không cần Gradio, Pillow)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.paths = []
        for index in range(3):
            path = self.root / f"img_{index}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
            os.utime(path, (1000 + 1000 * index, 1000 + 1000 * index))
            self.paths.append(path)

    def test_gallery_item_path_accepts_every_gradio_shape(self):
        self.assertEqual(
            studio.gallery_item_path((str(self.paths[0]), "Seed 42")), (str(self.paths[0]), "Seed 42")
        )
        self.assertEqual(
            studio.gallery_item_path({"image": {"path": str(self.paths[1])}, "caption": "Seed 43"}),
            (str(self.paths[1]), "Seed 43"),
        )
        self.assertEqual(studio.gallery_item_path(str(self.paths[2])), (str(self.paths[2]), ""))

    def test_gallery_entries_drops_images_missing_from_disk(self):
        gallery = [(str(p), "") for p in self.paths] + [(str(self.root / "gone.png"), "x")]
        self.assertEqual(
            [path for path, _ in studio.gallery_entries(gallery)], [str(p) for p in self.paths]
        )

    def test_tapped_index_follows_display_order(self):
        gallery = [(str(p), "") for p in self.paths]
        self.assertEqual(studio.tapped_gallery_path(gallery, [1]), str(self.paths[1]))
        self.assertEqual(studio.tapped_gallery_path(gallery, 2), str(self.paths[2]))
        self.assertIsNone(studio.tapped_gallery_path(gallery, 9))
        self.assertIsNone(studio.tapped_gallery_path([], 0))

    def test_output_dir_entries_is_newest_first(self):
        (self.root / "notes.txt").write_text("không phải ảnh")
        entries = studio.output_dir_entries(self.root)
        self.assertEqual(
            [Path(path).name for path, _ in entries], ["img_2.png", "img_1.png", "img_0.png"]
        )
        self.assertEqual(studio.output_dir_entries(None), [])
        self.assertEqual(studio.output_dir_entries(self.root / "khong-ton-tai"), [])

    def test_source_entries_adds_captions_and_falls_back_to_gallery(self):
        gallery = [(str(p), f"Seed {40 + index}") for index, p in enumerate(self.paths)]
        entries = studio.source_entries(self.root, gallery)
        self.assertEqual(entries[0], (str(self.paths[2]), "Seed 42"))
        # Thư mục không đọc được vẫn còn danh sách ảnh của thư viện.
        self.assertEqual(
            [path for path, _ in studio.source_entries(self.root / "khong-ton-tai", gallery)],
            [str(self.paths[2]), str(self.paths[1]), str(self.paths[0])],
        )

    def test_picker_state_defaults_to_newest_but_honours_selection(self):
        entries = studio.source_entries(self.root, [])
        choices, value = studio.source_picker_state(entries)
        self.assertEqual(value, str(self.paths[2]))
        self.assertEqual(choices[0][0], "1 · img_2.png")
        _, keep = studio.source_picker_state(entries, keep=str(self.paths[0]))
        self.assertEqual(keep, str(self.paths[0]))
        _, picked = studio.source_picker_state(
            entries, keep=str(self.paths[0]), pick=str(self.paths[1])
        )
        self.assertEqual(picked, str(self.paths[1]))
        self.assertEqual(studio.source_picker_state([]), ([], None))

    def test_merge_keeps_loaded_images_when_the_gallery_is_only_the_latest_batch(self):
        remembered = [[str(self.paths[0]), "cũ"], [str(self.paths[1]), ""]]
        fresh = [(str(self.paths[2]), "mới"), (str(self.paths[1]), "caption")]
        merged = studio.merge_source_entries(remembered, fresh)
        self.assertEqual(
            [row[0] for row in merged],
            [str(self.paths[2]), str(self.paths[1]), str(self.paths[0])],
        )
        self.assertEqual(merged[1][1], "caption")
        self.assertEqual(merged[2][1], "cũ")
        self.assertEqual(studio.merge_source_entries(None, []), [])

    def test_ui_no_longer_hardcodes_the_newest_image(self):
        source = (Path(__file__).resolve().parents[1] / "colab" / "studio.py").read_text(encoding="utf-8")
        self.assertNotIn("Dùng ảnh mới nhất", source)
        # Bốn nút nạp ảnh + nút ↻ đều lấy giá trị từ ô chọn, không từ State "latest".
        self.assertEqual(source.count("inputs=source_choice"), 5)
        for loader in ("load_into_all", "load_into_image_tab", "load_into_upscale_tab",
                       "load_into_inpaint_tab"):
            self.assertIn(f"fn={loader},", source)
        self.assertIn("Nạp ảnh đã chọn vào cả ba tab", source)
        self.assertIn("GALLERY_SOURCE_LIMIT = 40", source)


class UiResponsivenessTests(unittest.TestCase):
    """Bấm tab/nút khi đang tạo ảnh không được làm đứng cả trang (báo lỗi Đợt 13).

    Nguyên nhân gốc: Gradio chạy handler có ``queue=False`` ngay trên event loop, nên
    một lượt quét catalog 349k dòng (~0,4-1,5 s CPU, lâu hơn nhiều khi GPU đang tải)
    hoặc một lượt glob/stat trên ``/content`` chặn toàn bộ HTTP/SSE — mọi thành phần
    trên trang kẹt ở trạng thái chờ. Các test dưới đây giữ cho phần nặng chạy trong
    queue và giữ cho đường bấm chuột chỉ đọc bộ nhớ.
    """

    source = (ROOT / "colab" / "studio.py").read_text(encoding="utf-8")

    def block(self, anchor):
        """Toàn bộ khai báo sự kiện Gradio tính từ ``anchor`` tới dấu đóng ngoặc cuối."""
        lines = self.source[self.source.index(anchor):].splitlines()
        out = []
        for line in lines:
            out.append(line)
            if line.strip() == ")":  # dấu đóng ngoặc của chính khai báo sự kiện
                break
        return "\n".join(out)

    def test_catalog_search_events_run_in_the_queue(self):
        for anchor in (
            "fn=update_keyword_tag_suggestions,",
            "fn=apply_keyword_tag_suggestion_ui,",
            "fn=apply_csv_tags,",
            "fn=run_tag_check,",
        ):
            self.assertIn(anchor, self.source)
            block = self.block(anchor)
            self.assertLess(len(block), 1200, "cắt khai báo sự kiện quá xa")
            self.assertNotIn(
                "queue=False", block,
                f"{anchor} phải chạy trong queue, không chạy trên event loop",
            )

    def test_queue_leaves_room_for_ui_events_while_generating(self):
        match = re.search(r"demo\.queue\(max_size=(\d+), default_concurrency_limit=(\d+)", self.source)
        self.assertIsNotNone(match, "phải cấu hình hàng đợi rõ ràng")
        max_size, default_limit = (int(value) for value in match.groups())
        # Một lượt tạo ảnh giữ slot của nó nhiều phút; hàng đợi nhỏ thì thao tác khác
        # bị từ chối (HTTP 429) và giao diện kẹt luôn ở trạng thái chờ.
        self.assertGreaterEqual(max_size, 16)
        self.assertGreaterEqual(default_limit, 1)
        # Job GPU vẫn phải chạy một lượt một — tránh tràn VRAM trên Colab.
        self.assertIn('concurrency_id="wai_gpu"', self.source)
        self.assertIn("concurrency_limit=1,", self.source)

    def function_body(self, name):
        """Thân hàm lồng trong build_app, cắt ở def cùng cấp kế tiếp."""
        marker = f"def {name}("
        start = self.source.index(marker)
        start = self.source.rfind("\n", 0, start) + 1
        lines = self.source[start:].splitlines()
        indent = len(lines[0]) - len(lines[0].lstrip())
        out = [lines[0]]
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped and (len(line) - len(stripped)) <= indent and stripped.startswith(
                ("def ", "class ")
            ):
                break
            out.append(line)
        return "\n".join(out)

    def test_tapping_a_gallery_image_never_scans_the_output_dir(self):
        # Handler nằm ngoài khai báo .select( vì lambda không gắn được hint SelectData.
        # Ý định giữ nguyên: bấm/đổi thư viện chỉ đọc bộ nhớ, không glob thư mục xuất.
        for name in ("on_gallery_change", "on_gallery_select"):
            body = self.function_body(name)
            self.assertIn("source_entries(None,", body)
            self.assertIn("verify=False", body)
            self.assertNotIn("picker_entries(", body)
            self.assertNotIn("output_dir_entries(", body)
            self.assertNotIn(".glob(", body)
        select_body = self.function_body("on_gallery_select")
        self.assertIn("data: gr.SelectData", select_body)
        self.assertIn("merge_source_entries(", select_body)
        self.assertNotIn("lambda value, data", self.source)
        for anchor, handler in (
            ("gallery.change(", "on_gallery_change"),
            ("gallery.select(", "on_gallery_select"),
        ):
            block = self.block(anchor)
            self.assertIn(f"fn={handler},", block)
            self.assertIn("queue=False", block)
            self.assertNotIn("picker_entries(", block)
            self.assertNotIn("output_dir_entries(", block)
        # Chỉ nút ↻ và lần mở trang mới đọc thư mục xuất, bỏ cache, và phải qua queue.
        for name in ("refresh_source_picker", "load_source_picker"):
            body = self.function_body(name)
            self.assertIn("picker_entries(ttl=0)", body)
        for anchor in ("source_refresh.click(", "demo.load("):
            block = self.block(anchor)
            self.assertNotIn("queue=False", block)

    def test_source_dropdown_accepts_the_value_the_app_sets(self):
        # Gradio đối chiếu giá trị dropdown với `choices`; nếu choices chưa kịp cập
        # nhật thì server trả "not in the list of choices" thay vì nạp ảnh.
        block = self.block("source_choice = gr.Dropdown(")
        self.assertIn("allow_custom_value=True", block)

    def test_output_listing_sorts_by_embedded_time_without_stat(self):
        root = Path(tempfile.mkdtemp())
        names = [
            "wai_t2i_20260102_000000_000000_1.png",
            "wai_upscale_20260103_000000_000000_2.png",
            "wai_inpaint_20260104_000000_000000_3.png",
        ]
        for index, name in enumerate(names):
            path = root / name
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            # mtime ngược hẳn thứ tự thời gian trong tên: chỉ được dùng tên.
            os.utime(path, (1_700_000_000 - index, 1_700_000_000 - index))

        stats = {"count": 0}
        real_stat = Path.stat

        def counting(self, *args, **kwargs):
            stats["count"] += 1
            return real_stat(self, *args, **kwargs)

        def no_is_file(self, *args, **kwargs):
            raise AssertionError("không được is_file() từng file khi liệt kê thư mục xuất")

        with patch.object(Path, "stat", counting), patch.object(Path, "is_file", no_is_file):
            entries = studio.output_dir_entries(root, ttl=0)
        # Path.glob chỉ stat chính thư mục; không có stat nào cho từng ảnh.
        self.assertLessEqual(stats["count"], 1)
        self.assertEqual([Path(path).name for path, _ in entries], list(reversed(names)))

    def test_output_listing_can_reuse_a_recent_scan(self):
        root = Path(tempfile.mkdtemp())
        (root / "wai_t2i_20260102_000000_000000_1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        first = studio.output_dir_entries(root, ttl=300)
        (root / "wai_t2i_20260103_000000_000000_2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(studio.output_dir_entries(root, ttl=300)), len(first))
        self.assertEqual(len(studio.output_dir_entries(root, ttl=0)), 2)

    def test_valid_tag_set_follows_the_loaded_catalog(self):
        try:
            first = [("solo", "0", 900, "một mình", (), "một mình")]
            second = [("city", "0", 800, "thành phố", (), "thành phố")]
            with patch.object(studio, "load_csv_tags", return_value=first):
                self.assertIn("solo", studio.csv_tag_names())
                self.assertNotIn("city", studio.csv_tag_names())
            with patch.object(studio, "load_csv_tags", return_value=second):
                self.assertIn("city", studio.csv_tag_names())
                self.assertNotIn("solo", studio.csv_tag_names())
        finally:
            studio._CSV_TAG_NAMES = None
            studio._CSV_TAG_ROWS = None

    def test_superseded_catalog_search_stops_instead_of_finishing(self):
        rows = [(f"tag_{index}", "0", index, "", (), "") for index in range(30_000)]
        stale_token = studio.prompt_tag_request_begin()
        studio.prompt_tag_request_begin()  # người dùng gõ tiếp
        self.assertFalse(studio.prompt_tag_search_alive(stale_token))
        with self.assertRaises(studio.PromptTagSearchSuperseded):
            studio._prompt_tag_suggestion_results(rows, "*ag", token=stale_token)
        current = studio.prompt_tag_request_begin()
        found, _ = studio._prompt_tag_suggestion_results(
            [("cat_eared", "0", 5, "", (), "")], "*eared", token=current
        )
        self.assertEqual([row[0] for row in found], ["cat_eared"])

    def test_weight_shortcut_js_does_not_observe_the_whole_document(self):
        # Một MutationObserver trên document.body chạy theo MỌI thay đổi DOM — tức là
        # mỗi lần chuyển tab — nên phải ngắt ngay khi gắn được phím tắt.
        js = studio.PROMPT_TAG_WEIGHT_SHORTCUT_JS
        self.assertIn("observer.disconnect()", js)
        self.assertNotIn("observe(document.body", js)
        self.assertIn('querySelector("#studio-prompt")', js)
        self.assertIn("ArrowUp", js)
        self.assertIn("ArrowDown", js)

    def test_header_shows_the_build_stamp(self):
        # Mã UI nằm trong ô 7: chỉ chạy lại ô 8/9 thì trình duyệt vẫn dùng bản cũ.
        # Con dấu "Bản dựng …" trên header là cách nhanh nhất để xác nhận phiên nào
        # đang thật sự chạy.
        self.assertIn('STUDIO_BUILD = "', self.source)
        self.assertIn("id='studio-build-chip'", self.source)
        self.assertIn("Bản dựng {STUDIO_BUILD}", self.source)
        # Attribute của chip phải dùng nháy đơn: nháy kép sẽ nuốt các thẻ HTML sau đó.
        at = self.source.index("id='studio-build-chip'")
        chip = self.source[at:at + 420]
        self.assertIn("title='", chip)
        self.assertNotIn('title="', chip)

    def test_offline_watchdog_is_installed_and_only_pings_when_useful(self):
        js = studio.STUDIO_OFFLINE_WATCHDOG_JS
        for needle in ("__waiOfflineWatchdog", "setInterval(ping, 20000)", "/config",
                       "studio-offline-banner", "location.reload()"):
            self.assertIn(needle, js)
        # Không tự tải lại khi người dùng còn đang nhìn tab khác (tránh mất trạng thái).
        self.assertIn('visibilityState !== "visible"', js)
        self.assertIn("demo.load(fn=None, js=STUDIO_OFFLINE_WATCHDOG_JS)", self.source)
        # Cú bấm chậm không được mở dải "Mất kết nối" (dải đó có nút tải lại trang).
        note = js.split("const note =", 1)[1].split("let pending", 1)[0]
        self.assertIn("blockBanner(", note)
        self.assertIn("Không phải mất kết nối", note)
        self.assertNotIn("banner()", note)
        self.assertNotIn("studio-offline-banner", note)
        self.assertNotIn("location.reload", note)
        self.assertIn("studio-block-banner", js)

    def test_editor_value_shares_the_loaded_file_and_skips_the_composite(self):
        # composite do trình duyệt tự vẽ; sao chép/convert thêm ở server chỉ để đưa
        # cho tab ✎ một file PNG thứ hai là nguyên nhân khựng khi mở tab đó.
        if Image is None:
            self.skipTest("Pillow required")
        root = Path(tempfile.mkdtemp())
        path = root / "wai_t2i_20260102_000000_000000_1.png"
        Image.new("RGB", (64, 48), "white").save(path)
        image = studio.selected_source_image(str(path))
        value = studio.editor_value_for(image)
        self.assertIsNone(value["composite"])
        self.assertEqual(value["layers"], [])
        self.assertIs(value["background"], image)

    def test_selected_source_image_reads_a_real_png_and_reports_bad_files(self):
        if Image is None or not importlib.util.find_spec("gradio"):
            self.skipTest("Pillow and Gradio are required to load an image")
        import gradio as gr

        root = Path(tempfile.mkdtemp())
        good = root / "wai_t2i_20260102_000000_000000_1.png"
        Image.new("RGB", (24, 16), "white").save(good)
        self.assertEqual(studio.selected_source_image(str(good)).size, (24, 16))
        broken = root / "wai_t2i_20260102_000001_000000_2.png"
        broken.write_text("không phải ảnh")
        with self.assertRaises(gr.Error) as bad:
            studio.selected_source_image(str(broken))
        self.assertIn("Không đọc được ảnh", str(bad.exception))
        with self.assertRaises(gr.Error) as missing:
            studio.selected_source_image(str(root / "khong-co-file.png"))
        self.assertIn("không còn trên đĩa", str(missing.exception))


class UiEventInstrumentationTests(unittest.TestCase):
    """Nhật ký sự kiện UI chậm — bằng chứng để biết "đơ" là do đâu, không phải đoán."""

    class FakeBlockFn:
        def __init__(self, fn):
            self.fn = fn
            self.targets = [(7, "click")]

    class FakeDemo:
        def __init__(self, fns, blocks=None):
            self.fns = fns
            self.blocks = blocks or {}

    def test_slow_handler_is_logged_and_result_is_untouched(self):
        def handler(text):
            return text.upper()

        demo = self.FakeDemo({3: self.FakeBlockFn(handler)})
        studio.instrument_ui_events(demo)
        wrapped = demo.fns[3].fn
        self.assertTrue(getattr(wrapped, "_wai_timed", False))
        self.assertEqual(wrapped("ok"), "OK")
        # Nhẹ thì không in gì; chậm thì phải có dòng ⏱ kèm nhãn sự kiện.
        with contextlib.redirect_stdout(io.StringIO()) as quiet:
            wrapped("ok")
        self.assertEqual(quiet.getvalue().strip(), "")

        def slow(text):
            import time as _time
            _time.sleep(studio.SLOW_EVENT_SECONDS + 0.05)
            return text

        demo2 = self.FakeDemo({9: self.FakeBlockFn(slow)})
        studio.instrument_ui_events(demo2)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(demo2.fns[9].fn("x"), "x")
        self.assertIn("#9", log.getvalue())
        self.assertIn("click", log.getvalue())

    def test_generator_handlers_still_stream_and_are_logged_at_the_end(self):
        def stream():
            yield 1
            yield 2
            yield 3

        demo = self.FakeDemo({1: self.FakeBlockFn(stream)})
        studio.instrument_ui_events(demo)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(list(demo.fns[1].fn()), [1, 2, 3])
        self.assertNotIn("⏱", log.getvalue())

    def test_wrapping_is_idempotent_and_errors_stay_visible(self):
        def boom():
            raise ValueError("sai")

        demo = self.FakeDemo({2: self.FakeBlockFn(boom)})
        studio.instrument_ui_events(demo)
        first = demo.fns[2].fn
        studio.instrument_ui_events(demo)
        self.assertIs(demo.fns[2].fn, first, "không bọc hai lần")
        with contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaises(ValueError):
                first()
        self.assertIn("⚠️", log.getvalue())

    def test_timing_wrapper_keeps_the_selectdata_signature(self):
        def handler(value, data: int):
            return data

        demo = self.FakeDemo({4: self.FakeBlockFn(handler)})
        studio.instrument_ui_events(demo)
        signature = inspect.signature(demo.fns[4].fn)
        self.assertIn("data", signature.parameters)
        self.assertIs(signature.parameters["data"].annotation, int)
        self.assertEqual(demo.fns[4].fn("ảnh", 3), 3)

    def test_notebook_launch_disables_ssr_and_keeps_upload_limit(self):
        # ssr_mode mặc định tắt (GRADIO_SSR_MODE); còn max_file_size chỉ áp cho UPLOAD,
        # không chặn tải ảnh lớn — nên không được nâng lên vô hạn.
        notebook_cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        launch = "".join(notebook_cells[8]["source"])
        self.assertNotIn("ssr_mode=True", launch)
        self.assertIn('max_file_size="12mb"', launch)
