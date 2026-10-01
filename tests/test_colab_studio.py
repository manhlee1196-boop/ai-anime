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
        # Auto-detailer: cài riêng, lỗi chỉ cảnh báo để không kéo sập cả Studio.
        self.assertIn('-q install "ultralytics==8.4.170"', install)
        self.assertIn('optional = {"ultralytics": ">=8.4,<9"}', install)
        required = install[
            install.index("required = {") : install.index(
                "for package, constraint in required"
            )
        ]
        self.assertNotIn("ultralytics", required)
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

    def test_every_code_cell_collapses_into_a_compact_form(self):
        """Cả hai notebook phải gọn: mỗi ô code là một form tiêu đề + nút Run."""
        for path in (NOTEBOOK, BASE):
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
            # ultralytics (auto-detailer) là tuỳ chọn: thiếu hoặc lệch phiên bản chỉ
            # được cảnh báo, không chặn Studio như nhóm gói bắt buộc.
            self.assertIn("Không có ultralytics", output.getvalue())
            installed["ultralytics"] = "7.0.0"
            with contextlib.redirect_stdout(io.StringIO()) as warn:
                exec(check, {})
            self.assertIn("ultralytics đang là 7.0.0", warn.getvalue())
            installed.pop("ultralytics")
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
                    {"title": "Chân dung", "prompt": "1girl, portrait, soft light"},
                    {"name": "Phong cảnh", "en": "1girl, landscape, wide sky"},
                ]
            )
        )
        self.assertEqual(
            [i["title"] for i in parsed["items"]], ["Chân dung", "Phong cảnh"]
        )

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
        self.assertEqual(
            len(library["items"][0]["prompt"]), studio.PROMPT_LIBRARY_LIMIT
        )
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
            studio.prompt_library_dropdown(library["items"])["interactive"], True
        )
        self.assertEqual(studio.prompt_library_dropdown(())["choices"], ())
        self.assertEqual(studio.prompt_library_reset()[0], ())
        # Kích thước trong file mẫu phải là preset hợp lệ của giao diện.
        sized = next(item for item in library["items"] if item["size"])
        self.assertIn(sized["size"], studio.SIZE_PRESETS)
        # Prompt mẫu phải qua được bộ kiểm tra độ dài của runtime.
        for item in library["items"]:
            self.assertLessEqual(len(item["prompt"]), 2200)


class ProfessionalPromptTests(unittest.TestCase):
    """Negative tối ưu, thứ tự thẻ chuẩn và bộ kiểm tra prompt/thông số."""

    def test_negative_presets_are_short_purpose_built_and_unique(self):
        ids = [preset["id"] for preset in studio.NEGATIVE_PRESETS]
        self.assertEqual(len(ids), len(set(ids)), "id preset phải duy nhất")
        # Đúng bộ negative mà nhà phát hành WAI-illustrious v17 công bố.
        self.assertEqual(
            studio.find_negative_preset("publisher")["tags"],
            ("bad quality", "worst quality", "worst detail", "sketch", "censor"),
        )
        for preset in studio.NEGATIVE_PRESETS:
            self.assertLessEqual(len(preset["tags"]), 20, preset["id"])
            self.assertTrue(preset["label"].strip(), preset["id"])
            self.assertTrue(preset["when"].strip(), preset["id"])
            lowered = [tag.casefold() for tag in preset["tags"]]
            self.assertEqual(len(lowered), len(set(lowered)), preset["id"])
            for tag in preset["tags"]:
                self.assertEqual(tag, tag.strip(), preset["id"])
                self.assertNotIn(",", tag, preset["id"])
        choices = studio.negative_preset_choices()
        self.assertEqual([choice[1] for choice in choices], ids)
        self.assertTrue(all(choice[0] for choice in choices))
        with self.assertRaisesRegex(ValueError, "Chọn một bộ negative"):
            studio.find_negative_preset("khong-ton-tai")

    def test_apply_negative_preset_replaces_or_appends_once(self):
        replaced, note = studio.apply_negative_preset(
            "anatomy", studio.DEFAULT_NEGATIVE
        )
        self.assertEqual(
            replaced, ", ".join(studio.find_negative_preset("anatomy")["tags"])
        )
        self.assertIn("Sửa tay", note)
        appended, _ = studio.apply_negative_preset(
            "anatomy", "bad anatomy, extra digit", studio.NEGATIVE_APPEND
        )
        self.assertTrue(appended.startswith("bad anatomy, extra digit, "))
        self.assertEqual(appended.count("bad anatomy"), 1)
        self.assertEqual(appended.count("extra digit"), 1)
        # Không preset nào đẩy negative vượt giới hạn 1700 ký tự của runtime.
        for preset in studio.NEGATIVE_PRESETS:
            for mode in studio.NEGATIVE_MODES:
                value, _ = studio.apply_negative_preset(
                    preset["id"], studio.DEFAULT_NEGATIVE, mode
                )
                self.assertLessEqual(len(value), 1700, preset["id"])
        with self.assertRaisesRegex(ValueError, "Chọn cách áp dụng"):
            studio.apply_negative_preset("core", "", "xoa-het")

    def test_structure_prompt_orders_tags_and_adds_missing_anchors(self):
        messy = (
            "cherry blossoms, 1girl, (blue eyes:1.4), detailed eyes, standing, "
            "soft sunlight, long hair, 1girl, cel shading, watercolor"
        )
        result, note = studio.structure_prompt(messy, "character")
        tags = studio.split_tags(result)
        self.assertEqual(tags.count("1girl"), 1, "thẻ trùng phải bị bỏ")
        self.assertIn("(blue eyes:1.4)", tags, "giữ nguyên cú pháp nhấn mạnh")
        self.assertEqual(tags[:3], list(studio.QUALITY_HEAD))
        self.assertEqual(tags[-1], "absurdres")
        self.assertLess(tags.index("1girl"), tags.index("cherry blossoms"))
        self.assertLess(tags.index("cel shading"), tags.index("absurdres"))
        self.assertIn("Nhân vật", note)
        self.assertIn("không có thẻ nào được thêm ngầm", note)
        # Sắp xếp lại lần nữa không đổi gì (idempotent).
        again, _ = studio.structure_prompt(result, "character")
        self.assertEqual(again, result)
        # Khung phong cảnh thêm "no humans" khi chưa khai báo chủ thể.
        scene, scene_note = studio.structure_prompt(
            "city street at night, wide shot", "scene"
        )
        self.assertIn("no humans", studio.split_tags(scene))
        self.assertIn("no humans", scene_note)

    def test_structure_prompt_rejects_unknown_kind_and_oversized_result(self):
        with self.assertRaisesRegex(ValueError, "Chọn một khung prompt"):
            studio.structure_prompt("1girl, solo", "khong-co")
        huge = ", ".join(f"tag number {index}" for index in range(400))
        with self.assertRaisesRegex(ValueError, "2200 ký tự"):
            studio.structure_prompt(huge, "character")
        self.assertEqual(studio.split_tags(""), [])

    def test_estimate_tokens_is_a_documented_heuristic(self):
        self.assertEqual(studio.estimate_tokens(""), 0)
        self.assertEqual(studio.estimate_tokens("1girl"), 2)
        self.assertEqual(studio.estimate_tokens("masterpiece, best quality"), 4)
        long_prompt = ", ".join(f"very descriptive tag {index}" for index in range(40))
        self.assertGreater(studio.estimate_tokens(long_prompt), studio.SDXL_TOKEN_CHUNK)
        self.assertLessEqual(
            studio.estimate_tokens(studio.DEFAULT_PROMPT), studio.SDXL_TOKEN_CHUNK
        )

    def test_classify_tag_separates_composition_from_appearance(self):
        """'upper body' là bố cục, không phải ngoại hình dù có chữ 'body'."""
        cases = {
            "upper body": "composition",
            "full body": "composition",
            "close-up": "composition",
            "long hair": "appearance",
            "looking at viewer": "pose",
            "white shirt": "outfit",
            "cherry blossoms": "background",
            "soft rim light": "lighting",
            "cel shading": "style",
            "absurdres": "tail",
            "1girl": "subject",
            "2boys": "subject",
            "female focus": "subject",
            "sharp focus": "composition",
            "thighhighs": "outfit",
            "pantyhose": "outfit",
            "glass wall": "background",
            "marble floor": "background",
            "wind": "background",
            "masterpiece": "quality",
            "w_arknights": "extra",
        }
        for tag, section in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(studio.classify_tag(tag), section)
        # Cú pháp nhấn mạnh được bóc trước khi xếp nhóm.
        self.assertEqual(studio.classify_tag("(blue eyes:1.4)"), "appearance")
        self.assertEqual(
            set(studio._SECTION_MATCH_ORDER), set(studio._SECTION_RULE_MAP)
        )

    def test_analyze_prompt_flags_risky_choices(self):
        prompt = (
            "masterpiece, best quality, amazing quality, highest quality, "
            "score_9, 1girl, 1girl, detailed eyes, text, (blue eyes:1.4), long_hair"
        )
        findings, stats = studio.analyze_prompt(
            prompt,
            "masterpiece, 1girl",
            steps=40,
            cfg=9,
            size="512x512",
            hires_scale="1.5×",
            hires_strength=0.65,
        )
        grouped = {"warn": [], "info": [], "ok": []}
        for level, message in findings:
            grouped[level].append(message)
        joined = {level: " ".join(items) for level, items in grouped.items()}
        self.assertEqual(stats["tags"], 11)
        self.assertIn("thẻ chất lượng", joined["warn"])
        self.assertIn("Thẻ trùng lặp", joined["warn"])
        self.assertIn("negative", joined["warn"])
        self.assertIn("Pony", joined["warn"])
        self.assertIn("Trọng số", joined["warn"])
        self.assertIn("detailed eyes", joined["info"])
        self.assertIn("gạch dưới", joined["info"])
        self.assertIn("prompt dương", joined["info"])
        self.assertIn("Steps = 40", joined["info"])
        self.assertIn("CFG = 9", joined["info"])
        self.assertIn("512×512", joined["info"])
        self.assertIn("Hires strength = 0.65", joined["info"])
        report = studio.format_prompt_report(findings, stats)
        self.assertIn("🩺", report)
        self.assertIn("⚠️", report)
        self.assertIn("15–30", report)

    def test_analyze_prompt_accepts_a_clean_professional_prompt(self):
        findings, stats = studio.analyze_prompt(
            studio.DEFAULT_PROMPT,
            "bad quality, worst quality, worst detail, sketch, censor",
            steps=28,
            cfg=6,
            size="832x1216",
        )
        self.assertEqual([level for level, _ in findings].count("warn"), 0)
        messages = " ".join(message for _, message in findings)
        self.assertIn("vừa trong một khối 75 token", messages)
        self.assertIn("Đã khai báo chủ thể", messages)
        self.assertIn("Đã có thẻ phong cách", messages)
        self.assertEqual(stats["tags"], 15)
        self.assertNotIn("⚠️", studio.format_prompt_report(findings, stats))
        # Prompt rỗng được báo rõ thay vì im lặng.
        empty, _ = studio.analyze_prompt("", "")
        self.assertEqual(
            empty, [("warn", "Prompt trống: model sẽ tạo ảnh gần như ngẫu nhiên.")]
        )
        # Negative quá dài bị cảnh báo theo khuyến nghị của nhà phát hành.
        long_negative = ", ".join(f"bad thing {index}" for index in range(60))
        warned, _ = studio.analyze_prompt(
            "1girl, solo, anime illustration", long_negative
        )
        self.assertTrue(
            any(
                level == "warn" and "negative" in message.casefold()
                for level, message in warned
            )
        )


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
        # Gợi ý sửa vùng cũng là hành động tường minh của người dùng.
        repaired, repaired_neg = studio.apply_repair_hints(
            "portrait", "bad hands", "eyes"
        )
        self.assertIn("perfect eyes", repaired)
        self.assertIn("misaligned eyes", repaired_neg)
        with self.assertRaisesRegex(ValueError, "Chọn vùng sửa"):
            studio.apply_repair_hints("portrait", "", "unknown")

    def test_content_guards_reject_underage_and_gate_adult_prompts(self):
        # Từ khóa trẻ em/vị thành niên bị từ chối với MỌI prompt, không cần preset.
        for prompt in (
            "underage character",
            "school girl portrait",
            "teenage",
            "17-year-old",
            "vị thành niên",
        ):
            with (
                self.subTest(prompt=prompt),
                self.assertRaisesRegex(ValueError, "phải trưởng thành"),
            ):
                self.params(prompt=prompt, adult_confirmed=True)
            with self.assertRaisesRegex(ValueError, "phải trưởng thành"):
                self.params(prompt=prompt)
        # Nội dung người lớn do người dùng tự viết thì phải tick xác nhận 18+.
        adult = "1girl, adult woman, nsfw, explicit, detailed anatomy"
        with self.assertRaisesRegex(ValueError, "18 tuổi trở lên"):
            self.params(prompt=adult, adult_confirmed=False)
        self.assertEqual(self.params(prompt=adult, adult_confirmed=True)[0], adult)
        # Prompt không có từ khóa người lớn thì không cần tick.
        safe = "1girl, adult woman, office worker, portrait, masterpiece"
        self.assertEqual(self.params(prompt=safe)[0], safe)
        # Từ khóa vị thành niên trong *negative* không kích hoạt hàng rào prompt dương.
        self.assertEqual(
            self.params(
                prompt="adult", negative="underage, child", adult_confirmed=True
            )[1],
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

    def test_hires_size_is_multiple_of_8_and_clamped_to_pixel_budget(self):
        self.assertIsNone(studio._hires_size(1024, 1024, studio.HIRES_OFF))
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
        self.assertEqual(
            (refine["prompt"], refine["negative_prompt"]), ("anime cat", "bad paws")
        )
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
            {"scale": "2×", "base_width": 512, "base_height": 512, "strength": 0.3},
        )
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
            self.assertIn("upscale", Path(paths[0]).name)
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
            # Mọi vùng (trừ "custom" không có thẻ) đều phải thêm gợi ý và idempotent.
            for target in [key for key in studio.REPAIR_HINTS if key != "custom"]:
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
        eyes_button = next(
            c
            for c in config["components"]
            if c["type"] == "button" and "perfect eyes" in c["props"].get("value", "")
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
            if set(d["outputs"]) == field_ids
            and len(d["targets"]) == 1
            and len(d["inputs"]) == 3
        )
        self.assertEqual(
            repair_event["inputs"][:2],
            [prompt_field["id"], negative_field["id"]],
        )
        # 4 nút tạo ảnh + 3 nút dùng ảnh mới nhất + trigger mắt + gợi ý sửa vùng
        # + 5 sự kiện của thư viện prompt + 3 nút quy trình (sắp xếp prompt, nạp
        # negative, kiểm tra prompt) + nút chi tiết mắt/móng.
        self.assertEqual(len(config["dependencies"]), 18)
        hires_fields = [
            c
            for c in config["components"]
            if "hires" in str(c["props"].get("label") or "").lower()
        ]
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
        detailer_fields = [
            c
            for c in config["components"]
            if str(c["props"].get("label") or "").startswith(
                (
                    "Tự sửa mặt/tay",
                    "Detailer strength",
                    "Ngưỡng phát hiện",
                    "Số vùng tối đa",
                )
            )
        ]
        self.assertEqual(
            [c["type"] for c in detailer_fields],
            ["dropdown", "slider", "slider", "slider"],
        )
        detailer_ids = [c["id"] for c in detailer_fields]
        self.assertEqual(
            [choice[0] for choice in detailer_fields[0]["props"]["choices"]],
            list(studio.DETAILER_TARGETS),
        )
        # Mặc định Tắt: không ai bị sửa ảnh khi chưa chủ động bật.
        self.assertEqual(detailer_fields[0]["props"]["value"], studio.DETAILER_OFF)
        for dep in config["dependencies"][:2]:  # text, img2img: có hires + detailer
            self.assertEqual(dep["inputs"][-6:-4], hires_ids)
            self.assertEqual(dep["inputs"][-4:], detailer_ids)
            self.assertEqual(
                dep["inputs"][-18:-16], [prompt_field["id"], negative_field["id"]]
            )
        upscale_dep = config["dependencies"][2]  # phóng to: hệ số/strength riêng
        self.assertFalse(set(hires_ids) & set(upscale_dep["inputs"]))
        self.assertFalse(set(detailer_ids) & set(upscale_dep["inputs"]))
        self.assertEqual(
            upscale_dep["inputs"][-12:-10], [prompt_field["id"], negative_field["id"]]
        )
        inpaint_dep = config["dependencies"][3]  # sửa vùng: không hires, không detailer
        self.assertFalse(set(hires_ids) & set(inpaint_dep["inputs"]))
        self.assertFalse(set(detailer_ids) & set(inpaint_dep["inputs"]))
        self.assertEqual(
            inpaint_dep["inputs"][-12:-10], [prompt_field["id"], negative_field["id"]]
        )
        self.assertTrue(
            all(x["api_visibility"] == "private" for x in config["dependencies"])
        )
        self.assertTrue(demo.studio_css)
        self.assertIsNotNone(demo.studio_theme)

        # Chi tiết mắt & móng: 4 dropdown mặc định "Không thêm" + 1 nút chỉ ghi
        # vào hai ô prompt đang hiển thị.
        look_dropdowns = [
            c
            for c in config["components"]
            if c["props"].get("label") in studio.LOOK_LABELS.values()
        ]
        self.assertEqual([c["type"] for c in look_dropdowns], ["dropdown"] * 4)
        self.assertTrue(
            all(c["props"]["value"] == studio.LOOK_OFF for c in look_dropdowns)
        )
        look_button = next(
            c
            for c in config["components"]
            if c["type"] == "button" and "mắt/móng" in str(c["props"].get("value", ""))
        )
        look_event = next(
            d
            for d in config["dependencies"]
            if (look_button["id"], "click") in d["targets"]
        )
        self.assertEqual(set(look_event["outputs"]), field_ids)
        self.assertEqual(
            look_event["inputs"][:2], [prompt_field["id"], negative_field["id"]]
        )
        self.assertEqual(
            set(look_event["inputs"][2:]), {c["id"] for c in look_dropdowns}
        )
        self.assertFalse(look_event["queue"])
        # Vùng sửa: nhãn tiếng Việt cho người dùng, giá trị vẫn là khóa tiếng Anh
        # mà `_generate` kiểm tra nên hành vi sửa vùng cũ không đổi.
        repair_dropdown = next(
            c
            for c in config["components"]
            if c["props"].get("label") == "Chi tiết cần sửa"
        )
        self.assertEqual(
            [value for _, value in repair_dropdown["props"]["choices"]],
            list(studio.REPAIR_HINTS),
        )
        self.assertEqual(repair_dropdown["props"]["value"], "hands")

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
        choice = component("dropdown", "Chọn prompt để nạp")
        self.assertEqual(choice["props"]["choices"], [])
        self.assertFalse(choice["props"]["interactive"])
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
            d
            for d in config["dependencies"]
            if (choice["id"], "select") in d["targets"]
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
        "Gradio and Pillow needed for workflow tools UI test",
    )
    def test_workflow_tools_only_fill_visible_fields(self):
        """Ba nút quy trình chỉ ghi vào ô prompt/negative/báo cáo đang hiển thị."""
        import asyncio
        from gradio.state_holder import SessionState

        demo = studio.build_app(self.runtime)
        config = demo.get_config_file()
        components = config["components"]

        def component(kind, needle, field="label"):
            return next(
                c
                for c in components
                if c["type"] == kind and needle in str(c["props"].get(field, ""))
            )

        prompt_box = component("textbox", "Prompt gửi model")
        negative_box = component("textbox", "Negative gửi model")
        gallery = component("gallery", "Kết quả")
        scaffold_kind = component("dropdown", "Khung prompt theo loại ảnh")
        self.assertEqual(
            [choice[1] for choice in scaffold_kind["props"]["choices"]],
            list(studio.PROMPT_SCAFFOLDS),
        )
        negative_choice = component("dropdown", "Negative tối ưu theo mục đích")
        self.assertEqual(
            [choice[1] for choice in negative_choice["props"]["choices"]],
            [preset["id"] for preset in studio.NEGATIVE_PRESETS],
        )
        self.assertIsNone(negative_choice["props"].get("value"))
        negative_mode = component("radio", "Cách áp dụng")
        self.assertEqual(negative_mode["props"]["value"], studio.NEGATIVE_REPLACE)
        scaffold_button = component(
            "button", "Sắp xếp prompt theo thứ tự chuẩn", "value"
        )
        negative_button = component("button", "Nạp negative đã chọn", "value")
        check_button = component("button", "Kiểm tra prompt", "value")

        def event(button):
            return next(
                d
                for d in config["dependencies"]
                if (button["id"], "click") in d["targets"]
            )

        structure_event = event(scaffold_button)
        self.assertEqual(
            structure_event["inputs"], [prompt_box["id"], scaffold_kind["id"]]
        )
        self.assertEqual(structure_event["outputs"][0], prompt_box["id"])
        negative_event = event(negative_button)
        self.assertEqual(
            negative_event["inputs"],
            [negative_choice["id"], negative_box["id"], negative_mode["id"]],
        )
        self.assertEqual(negative_event["outputs"][0], negative_box["id"])
        check_event = event(check_button)
        self.assertEqual(len(check_event["outputs"]), 1)
        self.assertNotEqual(check_event["outputs"][0], prompt_box["id"])
        report_id = check_event["outputs"][0]
        self.assertEqual(
            next(c for c in components if c["id"] == report_id)["type"], "markdown"
        )
        for dependency in (structure_event, negative_event, check_event):
            # Không nút nào tạo ảnh hay đổi gallery.
            self.assertFalse(dependency["queue"])
            self.assertEqual(dependency["api_visibility"], "private")
            self.assertNotIn(gallery["id"], dependency["outputs"])

        session = SessionState(demo)

        def fn_index(trigger):
            return next(
                d["id"] for d in config["dependencies"] if trigger in d["targets"]
            )

        async def process(trigger, inputs):
            return (
                await demo.process_api(
                    fn_index(trigger), inputs, state=session, explicit_call=True
                )
            )["data"]

        async def flow():
            structured = await process(
                (scaffold_button["id"], "click"),
                ["1girl, standing, cherry blossoms, soft sunlight", "character"],
            )
            self.assertIn("masterpiece", structured[0])
            self.assertTrue(structured[0].endswith("absurdres"))
            self.assertIn("Nhân vật", structured[1])
            loaded = await process(
                (negative_button["id"], "click"),
                ["core", studio.DEFAULT_NEGATIVE, studio.NEGATIVE_REPLACE],
            )
            self.assertEqual(
                loaded[0], ", ".join(studio.find_negative_preset("core")["tags"])
            )
            self.assertIn("Illustrious chuẩn", loaded[1])
            appended = await process(
                (negative_button["id"], "click"),
                ["core", "bad hands", studio.NEGATIVE_APPEND],
            )
            self.assertTrue(appended[0].startswith("bad hands, "))
            self.assertEqual(appended[0].count("bad hands"), 1)
            report = await process(
                (check_button["id"], "click"),
                [
                    studio.DEFAULT_PROMPT,
                    studio.DEFAULT_NEGATIVE,
                    25,
                    6.0,
                    "1024x1024",
                    studio.HIRES_OFF,
                    studio.HIRES_DEFAULT_STRENGTH,
                ],
            )
            self.assertIn("🩺", report[0])
            self.assertIn("15–30", report[0])

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

        async def process(index, inputs):
            return (
                await demo.process_api(index, inputs, state=state, explicit_call=True)
            )["data"]

        def shared(positive, negative, eyes=False, adult=False):
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
                adult,
            ]

        detailer_off = [
            studio.DETAILER_OFF,
            studio.DETAILER_DEFAULT_STRENGTH,
            studio.DETAILER_DEFAULT_CONF,
            studio.DETAILER_DEFAULT_MAX,
        ]

        async def smoke():
            # Không còn preset phong cách: nút trigger mắt chỉ thêm "perfect eyes"
            # vào đúng ô prompt đang hiển thị, negative giữ nguyên.
            triggered = await process(
                7, [studio.DEFAULT_PROMPT, studio.DEFAULT_NEGATIVE]
            )
            self.assertIn("perfect eyes", triggered[0])
            self.assertEqual(triggered[1], studio.DEFAULT_NEGATIVE)
            self.assertEqual(await process(7, triggered), triggered)
            # Nút gợi ý sửa vùng cũng chỉ đổi hai ô đang hiển thị.
            leg_prompts = await process(8, [*triggered, "legs"])
            self.assertIn("natural toes", leg_prompts[0])
            self.assertIn("broken legs", leg_prompts[1])
            self.assertEqual(await process(8, [*leg_prompts, "legs"]), leg_prompts)
            adult_prompts = (
                "1girl, adult woman, nsfw, explicit, portrait",
                "bad hands",
            )
            for index, inputs, expected in (
                (
                    0,
                    [
                        "512x512",
                        *shared("  my own portrait  ", "  no hidden tags  ", eyes=True),
                        studio.HIRES_OFF,
                        0.4,
                        *detailer_off,
                    ],
                    ("  my own portrait  ", "  no hidden tags  "),
                ),
                (
                    1,
                    [
                        file_data(source),
                        "512x512",
                        0.45,
                        *shared("paint this picture", "my bad quality"),
                        studio.HIRES_OFF,
                        0.4,
                        *detailer_off,
                    ],
                    ("paint this picture", "my bad quality"),
                ),
                (
                    3,
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
                    3,
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
                # Prompt người lớn do người dùng tự viết, đã tick xác nhận 18+.
                (
                    0,
                    [
                        "512x512",
                        *shared(*adult_prompts, adult=True),
                        "Tắt",
                        0.4,
                        *detailer_off,
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
                0,
                [
                    "512x512",
                    *shared("hires portrait", "bad"),
                    "1.5×",
                    0.35,
                    *detailer_off,
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
                2, [file_data(source), "2×", 0.4, *shared("upscale me", "bad")]
            )
            self.assertIn("phóng to 512×512 → 1024×1024", data[2])
            self.assertEqual(len(FakePipe.calls), calls_before + 1)
            with Image.open(data[1][0]["path"]) as result:
                self.assertEqual(result.size, (1024, 1024))
            del FakePipe.calls[calls_before:]
            # Chưa tick xác nhận 18+ thì prompt người lớn bị chặn, không tới pipe.
            with self.assertRaises(Exception) as blocked:
                await process(
                    0,
                    ["512x512", *shared(*adult_prompts), "Tắt", 0.4, *detailer_off],
                )
            self.assertIn("18 tuổi trở lên", str(blocked.exception))
            # Từ khóa vị thành niên bị chặn kể cả khi đã tick xác nhận.
            with self.assertRaises(Exception) as underage:
                await process(
                    0,
                    [
                        "512x512",
                        *shared("school girl portrait", "", adult=True),
                        "Tắt",
                        0.4,
                        *detailer_off,
                    ],
                )
            self.assertIn("phải trưởng thành", str(underage.exception))
            self.assertEqual(len(FakePipe.calls), 5)
            self.assertTrue(Path((await process(4, [None]))[0]["path"]).is_file())
            self.assertTrue(
                Path((await process(6, [None]))[0]["background"]["path"]).is_file()
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


class FakeDetector:
    """YOLO giả: cùng giao diện `predict()` mà `detect_detail_regions` dùng."""

    def __init__(self, boxes, calls):
        self.boxes = boxes
        self.calls = calls

    def predict(self, source, conf, verbose):
        self.calls.append((source.size, conf, verbose))
        hits = [
            types.SimpleNamespace(xyxy=[[x1, y1, x2, y2]], conf=[score])
            for x1, y1, x2, y2, score in self.boxes
            if score >= conf
        ]
        return [types.SimpleNamespace(boxes=hits)]


class AutoDetailerTests(unittest.TestCase):
    """Auto-detailer mặt/tay: phát hiện → inpaint vùng cắt → dán lại viền mềm."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        root_patch = patch.object(studio, "CONTENT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.ck = self.root / "verified.safetensors"
        self.ck.touch()
        self.detections = {"face": [], "hands": []}
        self.detector_calls = []
        self.runtime = studio.StudioRuntime(
            torch=FakeTorch,
            pipe=FakePipe(),
            create_pipeline=lambda offload: FakePipe(),
            checkpoint=self.ck,
            lora_paths={"anatomy": self.ck},
            lora_manifest={"anatomy": {"weight": 0.55, "version": "v1", "sha256": "h"}},
            vram_mode="auto",
            use_offload=False,
            output_dir=self.root / "output",
            backup_dir=self.root / "backup",
            detailer_cache=str(self.root / "detailer"),
            detailer_detector=self.detector,
        )
        FakePipe.calls = []

    def detector(self, kind, cache_dir):
        assert cache_dir == self.runtime.detailer_cache
        return FakeDetector(self.detections[kind], self.detector_calls)

    def test_detailer_config_targets_and_ranges(self):
        self.assertEqual(studio.detailer_targets("Tắt"), ())
        self.assertEqual(studio.detailer_targets("Mặt"), ("face",))
        self.assertEqual(studio.detailer_targets("Tay"), ("hands",))
        self.assertEqual(studio.detailer_targets("Mặt + tay"), ("face", "hands"))
        with self.assertRaisesRegex(ValueError, "auto-detailer"):
            studio.detailer_targets("Cả người")
        self.assertIsNone(studio.validate_detailer("Tắt", 0.9, 0.9, 9))
        spec = studio.validate_detailer("Mặt + tay", 0.4, 0.3, 2)
        self.assertEqual(spec["targets"], ("face", "hands"))
        self.assertEqual(
            (spec["strength"], spec["conf"], spec["max_regions"]), (0.4, 0.3, 2)
        )
        with self.assertRaisesRegex(ValueError, "Detailer strength"):
            studio.validate_detailer("Mặt", 0.9, 0.3, 2)
        with self.assertRaisesRegex(ValueError, "Ngưỡng phát hiện"):
            studio.validate_detailer("Mặt", 0.4, 0.05, 2)
        with self.assertRaisesRegex(ValueError, "Số vùng tối đa"):
            studio.validate_detailer("Mặt", 0.4, 0.3, 9)

    def test_pad_box_grows_snaps_to_8_and_stays_inside_image(self):
        box = studio.pad_box((400, 300, 500, 380), (1024, 1024))
        self.assertEqual(tuple(value % 8 for value in box), (0, 0, 0, 0))
        self.assertLess(box[0], 400)  # có nới viền để đủ ngữ cảnh
        self.assertGreater(box[2], 500)
        self.assertGreater(box[3], 380)
        corner = studio.pad_box((0, 0, 30, 30), (512, 512))
        self.assertEqual(corner[:2], (0, 0))
        self.assertGreaterEqual(corner[2] - corner[0], 64)  # đủ lớn cho VAE
        self.assertGreaterEqual(corner[3] - corner[1], 64)
        outside = studio.pad_box((2000, 2000, 3000, 3000), (512, 512))
        self.assertTrue(all(0 <= value <= 512 for value in outside))

    @unittest.skipIf(Image is None, "Pillow needed for detection tests")
    def test_regions_sort_by_confidence_and_respect_limit(self):
        image = Image.new("RGB", (1024, 1024), "gray")
        self.detections = {
            "face": [(100, 100, 200, 200, 0.35), (300, 300, 400, 400, 0.9)],
            "hands": [(500, 500, 560, 560, 0.6)],
        }
        regions = studio.detect_detail_regions(
            image,
            ("face", "hands"),
            0.3,
            2,
            detector=self.detector,
            cache_dir=self.runtime.detailer_cache,
        )
        self.assertEqual([kind for _, kind, _ in regions], ["face", "hands"])
        self.assertEqual([round(score, 2) for _, _, score in regions], [0.9, 0.6])
        self.assertEqual(len(self.detector_calls), 2)  # mỗi loại weight một lượt dò
        self.assertEqual(self.detector_calls[0][1:], (0.3, False))
        self.assertTrue(all(region[0][2] <= 1024 for region in regions))
        # Ngưỡng cao: không còn vùng nào để Studio báo rõ thay vì âm thầm bỏ qua.
        self.assertEqual(
            studio.detect_detail_regions(
                image,
                ("face",),
                0.95,
                2,
                detector=self.detector,
                cache_dir=self.runtime.detailer_cache,
            ),
            [],
        )

    def test_cached_weight_is_hash_pinned_like_checkpoint(self):
        payload = b"yolov8n test weights"
        cache = self.root / "cache"
        cache.mkdir()
        target = cache / "face_yolov8n.pt"
        target.write_bytes(payload)
        spec = {
            "face": {
                "file": target.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "version": "test",
            }
        }
        with patch.dict(studio.DETAILER_MODELS, spec):
            self.assertEqual(studio.ensure_detailer_weight("face", str(cache)), target)
            target.write_bytes(payload + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "sai kích thước/SHA-256"):
                studio.ensure_detailer_weight("face", str(cache))
        with self.assertRaisesRegex(ValueError, "Detector không tồn tại"):
            studio.ensure_detailer_weight("person", str(cache))

    def test_downloaded_weight_is_verified_then_kept_or_deleted(self):
        payload = b"downloaded yolov8n weights"
        spec = {
            "face": {
                "file": "face_yolov8n.pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "version": "test",
            }
        }
        state = {"payload": payload}

        def fake_download(**kwargs):
            self.assertEqual(kwargs["repo_id"], studio.DETAILER_REPO)
            self.assertEqual(kwargs["revision"], studio.DETAILER_REVISION)
            self.assertIs(kwargs["token"], False)  # không cần tài khoản/token
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(state["payload"])
            return target

        hub = types.SimpleNamespace(hf_hub_download=fake_download)
        with (
            patch.dict(studio.DETAILER_MODELS, spec),
            patch.dict(sys.modules, {"huggingface_hub": hub}),
        ):
            good = self.root / "good"
            self.assertEqual(
                studio.ensure_detailer_weight("face", str(good)),
                good / "face_yolov8n.pt",
            )
            state["payload"] = payload + b"corrupt"
            bad = self.root / "bad"
            with self.assertRaisesRegex(RuntimeError, "không khớp SHA-256"):
                studio.ensure_detailer_weight("face", str(bad))
            self.assertFalse((bad / "face_yolov8n.pt").exists())  # file sai bị xóa

    def test_missing_ultralytics_raises_vietnamese_instruction(self):
        with patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaisesRegex(RuntimeError, "ultralytics"):
                studio.load_yolo_model(self.ck)

    @unittest.skipIf(Image is None, "Pillow needed for detailer smoke test")
    def test_text_to_image_repaints_detected_face_before_hires(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived

        class PatchPipe(FakePipe):
            """Lượt inpaint trả ảnh đỏ để thấy đúng vùng được dán lại."""

            def __call__(self, **kwargs):
                if "mask_image" in kwargs:
                    FakePipe.calls.append((self, kwargs))
                    return types.SimpleNamespace(
                        images=[
                            Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")
                        ]
                    )
                return super().__call__(**kwargs)

        self.runtime.pipe = PatchPipe()
        self.detections["face"] = [(200, 150, 320, 300, 0.82)]
        args = (
            "anime portrait",
            "bad hands",
            20,
            5.5,
            11,
            1,
            True,
            0.55,
            False,
            0.45,
            True,
        )
        detailer = {
            "detailer": "Mặt",
            "detailer_strength": 0.4,
            "detailer_conf": 0.3,
            "detailer_max": 1,
        }
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            _, paths, status, _ = self.runtime.text_to_image(
                "512x512", *args, **detailer
            )
            self.assertEqual(len(FakePipe.calls), 2)  # tạo nền + inpaint vùng mặt
            base, refine = (call[1] for call in FakePipe.calls)
            self.assertNotIn("image", base)
            self.assertEqual((base["width"], base["height"]), (512, 512))
            self.assertEqual(refine["strength"], 0.4)
            self.assertEqual(refine["padding_mask_crop"], 32)
            self.assertEqual(
                (refine["prompt"], refine["negative_prompt"]),
                ("anime portrait", "bad hands"),  # đúng hai ô đang hiển thị
            )
            self.assertEqual(refine["generator"].seed, 11)  # seed gốc, tái lập được
            self.assertEqual(refine["mask_image"].size, refine["image"].size)
            self.assertTrue(all(value % 8 == 0 for value in refine["image"].size))
            self.assertGreaterEqual(max(refine["image"].size), studio.DETAILER_MIN_CROP)
            # Chạy trước hires fix: vùng cắt ở độ phân giải gốc, hires phóng sau.
            FakePipe.calls = []
            _, paths2, status2, _ = self.runtime.text_to_image(
                "512x512", *args, hires_scale="2×", hires_strength=0.35, **detailer
            )
        self.assertEqual(len(FakePipe.calls), 3)
        base, patch_call, hires_call = (call[1] for call in FakePipe.calls)
        self.assertEqual((base["width"], base["height"]), (512, 512))
        self.assertLessEqual(max(patch_call["image"].size), 512)
        self.assertEqual((hires_call["width"], hires_call["height"]), (1024, 1024))
        self.assertIn("auto-detailer đã sửa mặt", status)
        self.assertIn("tin cậy 0.82", status)
        self.assertIn("1024×1024", status2)
        with Image.open(paths[0]) as result:
            self.assertEqual(result.size, (512, 512))
            self.assertEqual(result.convert("RGB").getpixel((260, 228)), (255, 0, 0))
            self.assertEqual(result.convert("RGB").getpixel((10, 10)), (255, 255, 255))
            metadata = json.loads(result.info["parameters"])
        self.assertEqual(
            metadata["detailer"],
            {
                "targets": ["face"],
                "strength": 0.4,
                "conf": 0.3,
                "max_regions": 1,
                "weights": ["Bingsu/adetailer@c310c21"],
            },
        )
        self.assertEqual(metadata["prompt"], "anime portrait")
        self.assertTrue(Path(paths2[0]).is_file())

    @unittest.skipIf(Image is None, "Pillow needed for detailer smoke test")
    def test_detailer_reports_no_detection_and_rejects_inpaint_mode(self):
        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.AutoPipelineForImage2Image = FakeDerived
        mock_diffusers.AutoPipelineForInpainting = FakeDerived
        args = (
            "anime portrait",
            "bad hands",
            20,
            5.5,
            11,
            1,
            True,
            0.55,
            False,
            0.45,
            True,
        )
        with patch.dict(sys.modules, {"diffusers": mock_diffusers}):
            # Không dò thấy gì: báo rõ thay vì im lặng, ảnh vẫn được tạo.
            _, _, status, _ = self.runtime.text_to_image(
                "512x512", *args, detailer="Mặt + tay"
            )
            self.assertEqual(len(FakePipe.calls), 1)
            self.assertIn("không thấy mặt/tay", status)
            FakePipe.calls = []
            # Cấu hình sai bị chặn trước khi tốn một lượt tạo ảnh nào.
            with self.assertRaisesRegex(ValueError, "Detailer strength"):
                self.runtime.text_to_image(
                    "512x512", *args, detailer="Mặt", detailer_strength=0.95
                )
            self.assertEqual(FakePipe.calls, [])
            with self.assertRaisesRegex(ValueError, "auto-detailer"):
                self.runtime.text_to_image("512x512", *args, detailer="Cả người")
            with self.assertRaisesRegex(ValueError, "Tắt"):
                self.runtime._generate(
                    "inpaint",
                    None,
                    None,
                    "hands",
                    8,
                    0.45,
                    "512x512",
                    "anime portrait",
                    "bad hands",
                    20,
                    6,
                    5,
                    1,
                    True,
                    0.55,
                    False,
                    0.45,
                    False,
                    detailer="Mặt",
                )


class RepairAndLookPromptTests(unittest.TestCase):
    """Gợi ý sửa vùng (tay/móng/chân/mắt/mặt/răng/tóc/da) + chi tiết mắt & móng."""

    def test_repair_regions_all_have_labels_and_stay_idempotent(self):
        self.assertEqual(set(studio.REPAIR_LABELS), set(studio.REPAIR_HINTS))
        for target in studio.REPAIR_HINTS:
            with self.subTest(target=target):
                if target == "custom":
                    self.assertEqual(
                        studio.apply_repair_hints("as-is", "bad", target),
                        ("as-is", "bad"),
                    )
                    continue
                filled = studio.apply_repair_hints(
                    "natural portrait", "bad anatomy", target
                )
                self.assertNotEqual(filled, ("natural portrait", "bad anatomy"))
                self.assertEqual(studio.apply_repair_hints(*filled, target), filled)
                self.assertLessEqual(len(filled[0]), 2200)  # giới hạn _parameters
                self.assertLessEqual(len(filled[1]), 1700)
        # Thông báo lỗi phải kể tên các vùng mới để người dùng biết có gì để chọn.
        with self.assertRaisesRegex(ValueError, "móng tay / móng chân"):
            studio.apply_repair_hints("a", "", "unknown")

    def test_new_repair_regions_cover_nails_face_teeth_hair_skin(self):
        self.assertEqual(studio.REPAIR_LABELS["nails"], "Móng tay / móng chân")
        for target, positive_tag, negative_tag in (
            ("nails", "well-shaped fingernails", "deformed nails"),
            ("face", "symmetrical face", "poorly drawn face"),
            ("teeth", "straight teeth", "crooked teeth"),
            ("hair", "natural hair strands", "messy hairline"),
            ("skin", "even skin tone", "skin blemishes"),
        ):
            with self.subTest(target=target):
                filled = studio.apply_repair_hints("portrait", "bad quality", target)
                self.assertIn(positive_tag, filled[0])
                self.assertIn(negative_tag, filled[1])
        # Khóa gửi runtime vẫn là tiếng Anh: _generate("inpaint", target=...) không đổi.
        self.assertIn("nails", studio.REPAIR_HINTS)

    def test_look_defaults_to_nothing_and_rejects_unknown_choice(self):
        self.assertEqual(studio.look_tags(), ())
        with self.assertRaisesRegex(ValueError, "Chưa chọn chi tiết nào"):
            studio.apply_look_tags("1girl", "bad hands")
        with self.assertRaisesRegex(ValueError, "màu mắt"):
            studio.apply_look_tags("1girl", "", "đỏ rực")
        with self.assertRaisesRegex(ValueError, "màu sơn móng chân"):
            studio.apply_look_tags(
                "1girl",
                "",
                studio.LOOK_OFF,
                studio.LOOK_OFF,
                studio.LOOK_OFF,
                "xanh neon",
            )

    def test_look_choices_write_visible_danbooru_tags(self):
        filled, negative = studio.apply_look_tags(
            "1girl, solo",
            "bad hands",
            "Hai màu (heterochromia)",
            "Hạnh nhân (almond)",
            "Đỏ",
            "Đen",
        )
        self.assertEqual(
            filled,
            "1girl, solo, heterochromia, multicolored eyes, almond-shaped nails, "
            "red nails, nail polish, painted toenails, black nail polish",
        )
        self.assertEqual(negative, "bad hands")  # negative không bị đụng tới
        # Bấm lại không nhân đôi thẻ; thẻ nằm trong ô hiển thị nên xóa được.
        again, _ = studio.apply_look_tags(
            filled,
            negative,
            "Hai màu (heterochromia)",
            "Hạnh nhân (almond)",
            "Đỏ",
            "Đen",
        )
        self.assertEqual(again, filled)
        self.assertEqual(again.count("red nails"), 1)
        # Chỉ chọn móng chân vẫn hoạt động độc lập.
        toes, _ = studio.apply_look_tags(
            "", "", studio.LOOK_OFF, studio.LOOK_OFF, studio.LOOK_OFF, "Không sơn"
        )
        self.assertEqual(toes, "natural toenails")

    def test_every_look_option_is_classified_as_appearance_and_short(self):
        tags = set()
        for field, label, choices in studio.LOOK_FIELDS:
            with self.subTest(field=field):
                self.assertIn(studio.LOOK_OFF, choices)
                self.assertTrue(len(choices) > 2, label)
            for option in choices.values():
                tags.update(option)
        self.assertTrue(tags)
        for tag in sorted(tags):
            with self.subTest(tag=tag):
                self.assertEqual(studio.classify_tag(tag), "appearance")
        everything = studio.apply_look_tags(
            "", "", *[list(choices)[-1] for _, _, choices in studio.LOOK_FIELDS]
        )[0]
        self.assertLessEqual(len(everything), 2200)

    def test_structure_prompt_keeps_look_tags_in_appearance_group(self):
        prompt, _ = studio.structure_prompt(
            "1girl, solo, red nails, almond-shaped nails, blue eyes, upper body, "
            "masterpiece, best quality",
            "character",
        )
        groups = studio.sort_prompt_tags(studio.split_tags(prompt))[2]
        for tag in ("red nails", "almond-shaped nails", "blue eyes"):
            self.assertIn(tag, groups["appearance"])
        # Nhóm Ngoại hình phải đứng trước Bố cục như thứ tự chuẩn.
        self.assertLess(prompt.index("red nails"), prompt.index("upper body"), prompt)


if __name__ == "__main__":
    unittest.main()
