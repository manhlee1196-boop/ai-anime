"""CPU-only smoke tests for the self-contained Colab notebook.

Run: python -m unittest discover -s tests -v
Real GPU inference/download of the 6.94 GB checkpoint must be tried in Colab.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

NOTEBOOK = Path(__file__).resolve().parents[1] / "WAI_Illustrious_Colab.ipynb"


def cell_source(cell_id):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(
        next(cell for cell in notebook["cells"] if cell["metadata"]["id"] == cell_id)[
            "source"
        ]
    )


def set_form_fields(source, **values):
    for name, value in values.items():
        source, count = re.subn(
            rf"^{re.escape(name)} = .*?(?= # @param)",
            f"{name} = {value!r}",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Colab form field missing: {name}")
    return source


def replace_constants(source, **replacements):
    """Shrink fixture sizes/hashes only in tests, never in the real notebook."""
    for name, value in replacements.items():
        source, count = re.subn(
            rf"^{name} = .*?$",
            f"{name} = {value!r}",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Missing constant: {name}")
    return source


def substitute_test_checkpoint(source, content):
    return replace_constants(
        source,
        HF_MODEL_BYTES=len(content),
        HF_SHA256=hashlib.sha256(content).hexdigest(),
        MIN_CHECKPOINT_BYTES=1,
        DISK_RESERVE_BYTES=1,
    )


def substitute_test_loras(source, anatomy=b"anatomy mock", eyes=b"eyes mock"):
    return replace_constants(
        source,
        ANATOMY_LORA_BYTES=len(anatomy),
        ANATOMY_LORA_SHA256=hashlib.sha256(anatomy).hexdigest(),
        EYE_LORA_BYTES=len(eyes),
        EYE_LORA_SHA256=hashlib.sha256(eyes).hexdigest(),
        LORA_DISK_RESERVE_BYTES=1,
    )


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeImage:
    def __init__(self):
        self.metadata = None

    def save(self, path, *, format, pnginfo):
        if format != "PNG":
            raise AssertionError("PNG format was not requested")
        Path(path).write_bytes(b"PNG smoke test")
        self.metadata = pnginfo.values


class FakePipeline:
    last_instance = None
    instances = []
    fail_gpu_to_once = False
    fail_gpu_inference_once = False
    calls = []

    def __init__(self):
        self.scheduler = types.SimpleNamespace(config={"scheduler": "original"})
        self.vae = types.SimpleNamespace(
            enable_slicing=self.slicing, enable_tiling=self.tiling
        )
        self.sliced = False
        self.tiled = False
        self.mode = None
        self.image = FakeImage()
        self.loras = []
        self.adapter_settings = None
        self.hooks_removed = 0
        FakePipeline.last_instance = self
        FakePipeline.instances.append(self)

    @classmethod
    def from_single_file(cls, path, **kwargs):
        if not Path(path).is_file():
            raise AssertionError("Model path was not prepared")
        if kwargs["torch_dtype"] != "float16" or not kwargs["use_safetensors"]:
            raise AssertionError("Model should load as FP16 safetensors")
        return cls()

    def slicing(self):
        self.sliced = True

    def tiling(self):
        self.tiled = True

    def enable_model_cpu_offload(self):
        self.mode = "offload"

    def remove_all_hooks(self):
        self.hooks_removed += 1

    def load_lora_weights(self, path, **kwargs):
        if not kwargs["use_safetensors"] or not kwargs["local_files_only"]:
            raise AssertionError("Must load a verified local safetensors file only")
        if not (Path(path) / kwargs["weight_name"]).is_file():
            raise AssertionError("LoRA path missing")
        self.loras.append((path, kwargs))

    def set_adapters(self, names, adapter_weights):
        self.adapter_settings = (names, adapter_weights)

    def to(self, device):
        if FakePipeline.fail_gpu_to_once:
            FakePipeline.fail_gpu_to_once = False
            raise FakeOutOfMemoryError("mock GPU load OOM")
        self.mode = device

    def __call__(self, **kwargs):
        FakePipeline.calls.append((self.mode, kwargs))
        if self.mode == "cuda" and FakePipeline.fail_gpu_inference_once:
            FakePipeline.fail_gpu_inference_once = False
            raise FakeOutOfMemoryError("mock generation OOM")
        return types.SimpleNamespace(images=[self.image])


class FakePngInfo:
    def __init__(self):
        self.values = {}

    def add_text(self, key, value):
        self.values[key] = value


class FakePILImage:
    """Tiny symbolic image stub: boxes, sizes and mask behavior without Pillow/GPU."""

    def __init__(self, size=(1024, 1024), boxes=None):
        self.size = size
        self.boxes = boxes or []
        self.metadata = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def convert(self, mode):
        return self

    def point(self, function):
        assert (function(0), function(255)) == (0, 255)
        return self

    def getbbox(self):
        if not self.boxes:
            return None
        return (
            min(box[0] for box in self.boxes),
            min(box[1] for box in self.boxes),
            max(box[2] for box in self.boxes),
            max(box[3] for box in self.boxes),
        )

    def getextrema(self):
        return (
            (255, 255)
            if self.boxes == [(0, 0, *self.size)]
            else (0, 255) if self.boxes else (0, 0)
        )

    def filter(self, blur):
        return self

    def save(self, path, *, format, pnginfo):
        assert format == "PNG"
        Path(path).write_bytes(b"repaired PNG smoke test")
        self.metadata = pnginfo.values


class FakeInpaintPipeline:
    calls = []
    fail_gpu_once = False

    def __init__(self, base):
        self.base = base
        self.mode = base.mode
        self.hooks_removed = 0

    def enable_model_cpu_offload(self):
        self.mode = "offload"

    def remove_all_hooks(self):
        self.hooks_removed += 1

    def __call__(self, **kwargs):
        FakeInpaintPipeline.calls.append((self.mode, kwargs))
        if self.mode == "cuda" and FakeInpaintPipeline.fail_gpu_once:
            FakeInpaintPipeline.fail_gpu_once = False
            raise FakeOutOfMemoryError("mock inpaint OOM")
        return types.SimpleNamespace(images=[FakePILImage(kwargs["image"].size)])


def enable_fake_inpainting(modules):
    image = types.ModuleType("PIL.Image")
    image.open = lambda path: FakePILImage(
        boxes=[(50, 50, 110, 120)] if "mask" in str(path) else None
    )
    image.new = lambda mode, size, color: FakePILImage(size)
    image.composite = lambda repaired, original, mask: FakePILImage(original.size)
    draw = types.ModuleType("PIL.ImageDraw")

    class Drawer:
        def __init__(self, mask):
            self.mask = mask

        def rectangle(self, box, fill):
            assert fill == 255
            self.mask.boxes.append((box[0], box[1], box[2] + 1, box[3] + 1))

    draw.Draw = Drawer
    filter_module = types.ModuleType("PIL.ImageFilter")
    filter_module.GaussianBlur = lambda radius: radius
    ops = types.ModuleType("PIL.ImageOps")
    ops.exif_transpose = lambda raw: raw
    chops = types.ModuleType("PIL.ImageChops")
    chops.multiply = lambda binary, blurred: binary
    modules.update(
        {
            "PIL.Image": image,
            "PIL.ImageChops": chops,
            "PIL.ImageDraw": draw,
            "PIL.ImageFilter": filter_module,
            "PIL.ImageOps": ops,
        }
    )
    modules["diffusers"].AutoPipelineForInpainting = types.SimpleNamespace(
        from_pipe=lambda pipeline: FakeInpaintPipeline(pipeline)
    )
    return modules


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def fake_modules(free_gib=15, gpu=True, total_gib=16):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: gpu,
        get_device_properties=lambda index: types.SimpleNamespace(name="Fake GPU"),
        mem_get_info=lambda: (free_gib * 2**30, total_gib * 2**30),
        empty_cache=lambda: None,
        OutOfMemoryError=FakeOutOfMemoryError,
    )
    torch.__version__ = "mock"
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch.Generator = FakeGenerator
    torch.inference_mode = contextlib.nullcontext

    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusionXLPipeline = FakePipeline
    diffusers.EulerAncestralDiscreteScheduler = types.SimpleNamespace(
        from_config=lambda config: types.SimpleNamespace(config=config)
    )

    class Header:
        def __init__(self, path):
            self.path = str(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            if "anatomy" in self.path.lower() or "eyes" in self.path.lower():
                return ["lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight"]
            return [
                "model.diffusion_model.input_blocks.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            ]

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda path, **kwargs: Header(path)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("Unexpected model download")
    )
    psutil = types.ModuleType("psutil")
    psutil.virtual_memory = lambda: types.SimpleNamespace(available=16 * 2**30)
    ipython = types.ModuleType("IPython")
    ipython.__path__ = []
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda image: None
    pil = types.ModuleType("PIL")
    pil.__path__ = []
    pil_png = types.ModuleType("PIL.PngImagePlugin")
    pil_png.PngInfo = FakePngInfo
    return {
        "torch": torch,
        "diffusers": diffusers,
        "safetensors": safetensors,
        "huggingface_hub": hub,
        "psutil": psutil,
        "IPython": ipython,
        "IPython.display": ipython_display,
        "PIL": pil,
        "PIL.PngImagePlugin": pil_png,
    }


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.last_instance = None
        FakePipeline.instances = []
        FakePipeline.calls = []
        FakePipeline.fail_gpu_to_once = False
        FakePipeline.fail_gpu_inference_once = False
        FakeInpaintPipeline.calls = []
        FakeInpaintPipeline.fail_gpu_once = False

    def test_notebook_is_valid_and_clean(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        ids = [cell["metadata"]["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                if cell["metadata"]["id"] != "install-packages":
                    compile(source, cell["metadata"]["id"], "exec")
        self.assertIn("%pip", cell_source("install-packages"))
        self.assertNotIn('"torch==', cell_source("install-packages"))
        executable = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for removed in (
            "drive.mount(",
            "MOUNT_DRIVE",
            "CACHE_MODEL_LOCAL",
            "PERSIST_MODEL_TO_DRIVE",
            "PERSIST_LORAS_TO_DRIVE",
            "copy_atomic(",
        ):
            self.assertNotIn(removed, executable)
        prepare = cell_source("prepare-model")
        self.assertIn("local_dir=str(local_cache_root)", prepare)
        self.assertNotIn("cache_dir=", prepare)
        self.assertIn("token=False", prepare)
        self.assertIn("HF_MODEL_BYTES = 6_938_040_682", prepare)
        self.assertIn(
            'HF_SHA256 = "f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04"',
            prepare,
        )
        install = cell_source("install-packages")
        self.assertIn('"peft==0.17.1"', install)
        lora = cell_source("prepare-loras")
        for name, expected in {
            "ANATOMY_LORA_REV": "bed49d45df95c0695aedad3b2aa6aff389fb3777",
            "ANATOMY_LORA_SHA256": "bf6a950036b7599212a2c68d65f3ba07b28689067e167915d2a0ecb2018c26ca",
            "ANATOMY_LORA_BYTES": 228_473_940,
            "EYE_LORA_REV": "1abbc862f53f5101962ebf1c337513aff91bd206",
            "EYE_LORA_SHA256": "97c1a083ffe6b4d45c545196eabd01c754936b996ade0c9db6d072f3bd340c55",
            "EYE_LORA_BYTES": 228_457_660,
        }.items():
            literal = f'"{expected}"' if isinstance(expected, str) else f"{expected:_}"
            self.assertIn(f"{name} = {literal}", lora)

    def configure(self, namespace, model_path, output_path, **extra_fields):
        fields = {"USE_ANATOMY_LORA": False, "USE_EYE_LORA": False, **extra_fields}
        config = set_form_fields(
            cell_source("configure-paths"),
            MODEL_PATH=str(model_path),
            OUTPUT_DIR=str(output_path),
            **fields,
        )
        root = Path(output_path).parent
        config = config.replace(
            'content_root = Path("/content")', f"content_root = Path({str(root)!r})"
        )
        exec(config, namespace)

    def test_existing_checkpoint_skips_download_and_generates_png(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "custom-WAI-illustrious.safetensors"
            model_path.write_bytes(b"user-provided-checkpoint")
            output = directory / "outputs"
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, model_path, output)
                exec(
                    substitute_test_checkpoint(
                        cell_source("prepare-model"), b"other bytes"
                    ),
                    namespace,
                )
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=42), namespace)
            self.assertEqual(namespace["checkpoint"], model_path)
            self.assertEqual(FakePipeline.last_instance.mode, "cuda")
            self.assertTrue(FakePipeline.last_instance.sliced)
            self.assertTrue(FakePipeline.last_instance.tiled)
            images = list(output.glob("wai_*.png"))
            self.assertEqual(len(images), 1)
            self.assertFalse(list(output.glob("*.partial")))
            metadata = json.loads(
                FakePipeline.last_instance.image.metadata["parameters"]
            )
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["model"], model_path.name)

    def test_auto_download_verifies_sha_and_runs_from_single_cache_file(self):
        payload = b"v17 mock full checkpoint (use real SHA only in production)"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output = directory / "outputs"
            calls = []
            modules = fake_modules()

            def download(**kwargs):
                calls.append(kwargs)
                downloaded = Path(kwargs["local_dir"]) / kwargs["filename"]
                downloaded.write_bytes(payload)
                return str(downloaded)

            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, directory / "missing.safetensors", output)
                prepare = substitute_test_checkpoint(
                    cell_source("prepare-model"), payload
                )
                with patch(
                    "shutil.copyfile",
                    side_effect=AssertionError("No duplicate model copy"),
                ):
                    exec(prepare, namespace)
                    exec(
                        prepare, namespace
                    )  # reuse local file even if MODEL_PATH is missing
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=99), namespace)
            downloaded = (
                namespace["local_cache_root"] / "waiIllustriousSDXL_v170.safetensors"
            )
            self.assertEqual(namespace["checkpoint"], downloaded)
            self.assertEqual(downloaded.read_bytes(), payload)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0]["revision"], "32be7bfdcd406db70df663b9cee3313957deb68f"
            )
            self.assertEqual(calls[0]["token"], False)
            self.assertEqual(calls[0]["local_dir"], str(namespace["local_cache_root"]))
            self.assertNotIn("cache_dir", calls[0])
            self.assertEqual(len(list(output.glob("wai_*.png"))), 1)

    def test_bad_hash_never_loads_checkpoint(self):
        payload = b"wrong-data"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modules = fake_modules()

            def download(**kwargs):
                downloaded = Path(kwargs["local_dir"]) / kwargs["filename"]
                downloaded.write_bytes(payload)
                return str(downloaded)

            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                namespace = {}
                self.configure(
                    namespace, directory / "missing.safetensors", directory / "outputs"
                )
                prepare = substitute_test_checkpoint(
                    cell_source("prepare-model"), b"right-data"
                )
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(prepare, namespace)
                # Do not silently reuse a same-sized but corrupted cached file.
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(prepare, namespace)
            self.assertIsNone(FakePipeline.last_instance)

    def test_missing_model_without_auto_download_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                self.configure(
                    namespace,
                    directory / "missing.safetensors",
                    directory / "outputs",
                    AUTO_DOWNLOAD=False,
                )
                with self.assertRaisesRegex(FileNotFoundError, "Bật AUTO_DOWNLOAD"):
                    exec(
                        substitute_test_checkpoint(
                            cell_source("prepare-model"), b"fixture"
                        ),
                        namespace,
                    )

    def test_sha_progress_and_full_digest_do_not_modify_file(self):
        with tempfile.TemporaryDirectory() as directory:
            content = b"verified fixture" * 64
            saved = Path(directory) / "checkpoint.safetensors"
            saved.write_bytes(content)
            source = cell_source("prepare-model")
            definitions = source[
                : source.index(
                    "if source_model.exists() and not source_model.is_file():"
                )
            ]
            with patch.dict(sys.modules, fake_modules()):
                namespace = {"Path": Path}
                exec(definitions, namespace)
                namespace["PROGRESS_MIN_BYTES"] = 1
                namespace["PROGRESS_INTERVAL_SECONDS"] = 0
                progress = io.StringIO()
                with contextlib.redirect_stdout(progress):
                    checksum = namespace["sha256_file"](saved)
                self.assertEqual(checksum, hashlib.sha256(content).hexdigest())
                self.assertEqual(saved.read_bytes(), content)
                self.assertIn("Đã kiểm SHA-256: 100%", progress.getvalue())

    def test_config_rejects_outside_drive_mount_and_symlinked_local_cache(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            output = root / "outputs"
            with self.assertRaisesRegex(ValueError, "/content"):
                self.configure({}, Path(other) / "model.safetensors", output)
            with self.assertRaisesRegex(ValueError, "Drive"):
                self.configure(
                    {}, root / "drive" / "MyDrive" / "model.safetensors", output
                )
            (root / "drive" / "MyDrive").mkdir(parents=True)
            (root / "wai_lora_cache").symlink_to(
                root / "drive" / "MyDrive", target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "Drive"):
                self.configure({}, root / "model.safetensors", output)

    def test_low_disk_stops_local_model_download_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = fake_modules()
            with patch.dict(sys.modules, modules):
                ns = {}
                self.configure(ns, root / "missing.safetensors", root / "outputs")
                with patch(
                    "shutil.disk_usage", return_value=types.SimpleNamespace(free=0)
                ):
                    with self.assertRaisesRegex(OSError, "Thiếu đĩa /content"):
                        exec(
                            substitute_test_checkpoint(
                                cell_source("prepare-model"), b"fixture"
                            ),
                            ns,
                        )
            self.assertIsNone(FakePipeline.last_instance)

    def test_output_falls_back_when_requested_folder_is_not_writable(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.safetensors"
            model.write_bytes(b"existing-model")
            blocked = directory / "blocked-output"
            blocked.write_bytes(b"not a directory")
            backup = directory / "backup-output"
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, model, blocked)
                exec(
                    substitute_test_checkpoint(
                        cell_source("prepare-model"), b"fixture"
                    ),
                    namespace,
                )
                exec(cell_source("load-pipeline"), namespace)
                generate = set_form_fields(cell_source("generate-image"), SEED=101)
                generate = generate.replace(
                    'backup_dir = Path("/content/wai_outputs")',
                    f"backup_dir = Path({str(backup)!r})",
                )
                exec(generate, namespace)
            self.assertEqual(len(list(backup.glob("wai_*.png"))), 1)
            self.assertTrue(blocked.is_file())

    def test_auto_mode_offloads_on_low_vram(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.safetensors"
            model.write_bytes(b"existing-model")
            with patch.dict(sys.modules, fake_modules(free_gib=12)):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, model, directory / "outputs")
                exec(
                    substitute_test_checkpoint(
                        cell_source("prepare-model"), b"fixture"
                    ),
                    namespace,
                )
                exec(cell_source("load-pipeline"), namespace)
            self.assertEqual(FakePipeline.last_instance.mode, "offload")
            self.assertTrue(FakePipeline.last_instance.tiled)

    def test_t4_auto_prefers_gpu_with_both_verified_loras(self):
        anatomy = b"anatomy correct bytes"
        eyes = b"eyes correct bytes"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            anatomy_path = directory / "anatomy.safetensors"
            eyes_path = directory / "eyes.safetensors"
            anatomy_path.write_bytes(anatomy)
            eyes_path.write_bytes(eyes)
            # Representative free VRAM on a 15 GiB T4: the old 14.8 GiB
            # threshold would offload unnecessarily with two default LoRAs.
            with patch.dict(sys.modules, fake_modules(free_gib=14, total_gib=15)):
                namespace = {}
                self.prepare_existing_for_loras(
                    namespace,
                    directory,
                    USE_ANATOMY_LORA=True,
                    USE_EYE_LORA=True,
                    ANATOMY_LORA_PATH=str(anatomy_path),
                    EYE_LORA_PATH=str(eyes_path),
                )
                exec(
                    substitute_test_loras(cell_source("prepare-loras"), anatomy, eyes),
                    namespace,
                )
                with patch("builtins.print") as prints:
                    exec(cell_source("load-pipeline"), namespace)
            self.assertEqual(namespace["auto_min_vram"] / 2**30, 13.3)
            self.assertFalse(namespace["use_offload"])
            self.assertEqual(len(FakePipeline.instances), 1)
            self.assertEqual(FakePipeline.last_instance.mode, "cuda")
            self.assertTrue(FakePipeline.last_instance.tiled)
            self.assertEqual(len(FakePipeline.last_instance.loras), 2)
            self.assertIn(
                "VRAM trống trước khi nạp: 14.0/15.0 GiB", str(prints.call_args_list)
            )
            self.assertIn("Chế độ sau khi nạp:", str(prints.call_args_list))

    def test_gpu_oom_during_generation_retries_with_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.safetensors"
            model.write_bytes(b"existing-model")
            FakePipeline.fail_gpu_inference_once = True
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, model, directory / "outputs")
                exec(
                    substitute_test_checkpoint(
                        cell_source("prepare-model"), b"fixture"
                    ),
                    namespace,
                )
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=42), namespace)
            self.assertEqual(
                [mode for mode, _ in FakePipeline.calls], ["cuda", "offload"]
            )
            self.assertEqual(
                [args["generator"].seed for _, args in FakePipeline.calls], [42, 42]
            )
            self.assertTrue(namespace["use_offload"])
            self.assertEqual(len(list((directory / "outputs").glob("wai_*.png"))), 1)

    def test_gpu_oom_during_load_retries_with_offload(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.safetensors"
            model.write_bytes(b"existing-model")
            FakePipeline.fail_gpu_to_once = True
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, model, directory / "outputs")
                exec(
                    substitute_test_checkpoint(
                        cell_source("prepare-model"), b"fixture"
                    ),
                    namespace,
                )
                exec(cell_source("load-pipeline"), namespace)
            self.assertTrue(namespace["use_offload"])
            self.assertEqual(FakePipeline.last_instance.mode, "offload")

    def prepare_existing_for_loras(self, namespace, directory, **fields):
        if "torch" not in namespace:
            exec(cell_source("check-gpu"), namespace)
        model = directory / "wai-model.safetensors"
        model.write_bytes(b"mock pre-existing checkpoint")
        self.configure(namespace, model, directory / "outputs", **fields)
        exec(
            substitute_test_checkpoint(
                cell_source("prepare-model"), b"other mock bytes"
            ),
            namespace,
        )

    def test_lora_downloads_pins_hashes_and_loads_both_with_retry(self):
        anatomy = b"anatomy correct bytes"
        eyes = b"eyes correct bytes"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            files = {
                "ench100/bodyandface": anatomy,
                "Muapi/eyes-for-illustrious-lora-perfect-anime-eyes": eyes,
            }
            calls = []
            modules = fake_modules(free_gib=15)

            def download(**kwargs):
                calls.append(kwargs)
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.write_bytes(files[kwargs["repo_id"]])
                return str(path)

            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                ns = {}
                exec(cell_source("check-gpu"), ns)
                self.prepare_existing_for_loras(
                    ns, directory, USE_ANATOMY_LORA=True, USE_EYE_LORA=True
                )
                exec(
                    substitute_test_loras(cell_source("prepare-loras"), anatomy, eyes),
                    ns,
                )
                self.assertEqual(set(ns["lora_paths"]), {"anatomy", "eyes"})
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    [c["revision"] for c in calls],
                    [
                        "bed49d45df95c0695aedad3b2aa6aff389fb3777",
                        "1abbc862f53f5101962ebf1c337513aff91bd206",
                    ],
                )
                self.assertTrue(
                    all(
                        c["token"] is False
                        and c["local_dir"] == str(ns["local_lora_cache"])
                        for c in calls
                    )
                )
                exec(cell_source("load-pipeline"), ns)
                self.assertEqual(
                    FakePipeline.last_instance.adapter_settings,
                    (
                        ["anatomy", "eyes"],
                        [0.55, 0.45],
                    ),
                )
                self.assertEqual(len(FakePipeline.last_instance.loras), 2)
                FakePipeline.fail_gpu_inference_once = True
                exec(set_form_fields(cell_source("generate-image"), SEED=123), ns)
            self.assertEqual(
                [mode for mode, _ in FakePipeline.calls], ["cuda", "offload"]
            )
            self.assertEqual(len(FakePipeline.instances), 2)
            self.assertTrue(
                all(len(pipe.loras) == 2 for pipe in FakePipeline.instances)
            )
            self.assertEqual(
                [kw["generator"].seed for _, kw in FakePipeline.calls], [123, 123]
            )
            self.assertTrue(
                all("perfect eyes" in kw["prompt"] for _, kw in FakePipeline.calls)
            )
            self.assertEqual(
                json.loads(FakePipeline.last_instance.image.metadata["parameters"])[
                    "loras"
                ][0]["name"],
                "anatomy",
            )

    def test_downloaded_lora_bad_hash_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modules = fake_modules()

            def download(**kwargs):
                file = Path(kwargs["local_dir"]) / kwargs["filename"]
                file.write_bytes(b"wrong-20-byte-payload")
                return str(file)

            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                ns = {}
                self.prepare_existing_for_loras(ns, directory, USE_ANATOMY_LORA=True)
                with self.assertRaisesRegex(
                    RuntimeError, "tải/xác minh thất bại"
                ) as error:
                    exec(
                        substitute_test_loras(
                            cell_source("prepare-loras"), b"right-20-byte-payload"
                        ),
                        ns,
                    )
                self.assertIsInstance(error.exception.__cause__, ValueError)
                self.assertIn("SHA-256", str(error.exception.__cause__))
            self.assertIsNone(FakePipeline.last_instance)

    def test_manual_lora_must_match_official_hash_even_if_filename_differs(self):
        anatomy = b"the correct anatomy file"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            local = directory / "anatomy_helper.safetensors"
            local.write_bytes(anatomy)
            modules = fake_modules()
            with patch.dict(sys.modules, modules):
                ns = {}
                self.prepare_existing_for_loras(
                    ns,
                    directory,
                    USE_ANATOMY_LORA=True,
                    ANATOMY_LORA_PATH=str(local),
                    EYE_LORA_PATH="",
                )
                exec(substitute_test_loras(cell_source("prepare-loras"), anatomy), ns)
                self.assertEqual(ns["lora_paths"]["anatomy"], local)
                exec(cell_source("load-pipeline"), ns)
                self.assertEqual(
                    [
                        entry[1]["adapter_name"]
                        for entry in FakePipeline.last_instance.loras
                    ],
                    ["anatomy"],
                )
                local.write_bytes(b"bad checksum - same size")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(
                        substitute_test_loras(cell_source("prepare-loras"), anatomy), ns
                    )

    def test_verified_local_lora_is_reused_and_corruption_is_not_overwritten(self):
        anatomy = b"verified local anatomy"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(sys.modules, fake_modules()):
                ns = {}
                self.prepare_existing_for_loras(ns, root, USE_ANATOMY_LORA=True)
                saved = ns["local_lora_cache"] / "anatomy_helper.safetensors"
                saved.parent.mkdir(parents=True)
                saved.write_bytes(anatomy)
                prepare = substitute_test_loras(cell_source("prepare-loras"), anatomy)
                exec(prepare, ns)
                self.assertEqual(ns["lora_paths"]["anatomy"], saved)
                saved.write_bytes(b"z" * len(anatomy))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(prepare, ns)
                self.assertEqual(saved.read_bytes(), b"z" * len(anatomy))

    def test_low_disk_stops_local_lora_download_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(sys.modules, fake_modules()):
                ns = {}
                self.prepare_existing_for_loras(ns, root, USE_EYE_LORA=True)
                with patch(
                    "shutil.disk_usage", return_value=types.SimpleNamespace(free=0)
                ):
                    with self.assertRaisesRegex(OSError, "Thiếu đĩa /content"):
                        exec(substitute_test_loras(cell_source("prepare-loras")), ns)
            self.assertIsNone(FakePipeline.last_instance)

    def test_lora_config_changed_requires_reverification(self):
        anatomy = b"verified anatomy"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            local = directory / "anatomy_helper.safetensors"
            local.write_bytes(anatomy)
            with patch.dict(sys.modules, fake_modules()):
                ns = {}
                self.prepare_existing_for_loras(
                    ns, directory, USE_ANATOMY_LORA=True, ANATOMY_LORA_PATH=str(local)
                )
                exec(substitute_test_loras(cell_source("prepare-loras"), anatomy), ns)
                ns["ANATOMY_WEIGHT"] = 0.4
                with self.assertRaisesRegex(RuntimeError, "Cấu hình LoRA đã đổi"):
                    exec(cell_source("load-pipeline"), ns)
                ns["ANATOMY_WEIGHT"] = 0.55
                ns["USE_ANATOMY_LORA"] = False
                with self.assertRaisesRegex(RuntimeError, "Cấu hình LoRA đã đổi"):
                    exec(cell_source("load-pipeline"), ns)

    def test_existing_model_claims_v17_only_when_full_hash_matches(self):
        contents = b"v17 correct bytes"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "wai-model.safetensors"
            with patch.dict(sys.modules, fake_modules()):
                ns = {}
                self.configure(ns, model, directory / "outputs")
                source = substitute_test_checkpoint(
                    cell_source("prepare-model"), contents
                )
                model.write_bytes(contents)
                with patch("builtins.print") as prints:
                    exec(source, ns)
                self.assertTrue(
                    any(
                        "đã xác minh là WAI-illustrious v17" in str(c)
                        for c in prints.call_args_list
                    )
                )
                model.write_bytes(b"wrong same length")
                with patch("builtins.print") as prints:
                    exec(source, ns)
                self.assertTrue(
                    any(
                        "KHÔNG xác thực là WAI v17" in str(c)
                        for c in prints.call_args_list
                    )
                )

    def test_lora_mutated_after_verification_is_rejected_before_model_load(self):
        anatomy = b"verified anatomy"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            local = directory / "anatomy_helper.safetensors"
            local.write_bytes(anatomy)
            with patch.dict(sys.modules, fake_modules()):
                ns = {}
                self.prepare_existing_for_loras(
                    ns, directory, USE_ANATOMY_LORA=True, ANATOMY_LORA_PATH=str(local)
                )
                exec(substitute_test_loras(cell_source("prepare-loras"), anatomy), ns)
                local.write_bytes(b"different payload")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(cell_source("load-pipeline"), ns)
                self.assertEqual(FakePipeline.last_instance.loras, [])

    def test_inpainting_uses_masked_boxes_shares_pipeline_and_saves_new_png(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modules = enable_fake_inpainting(fake_modules(free_gib=12))
            with patch.dict(sys.modules, modules):
                ns = {}
                exec(cell_source("check-gpu"), ns)
                self.prepare_existing_for_loras(ns, directory)
                exec(cell_source("load-pipeline"), ns)
                exec(set_form_fields(cell_source("generate-image"), SEED=42), ns)
                source_file = ns["output_path"]
                exec(
                    set_form_fields(
                        cell_source("refine-image"),
                        BOXES="100,300,240,490;700,400,850,590",
                        TARGET="hands",
                        REFINE_SEED=777,
                    ),
                    ns,
                )
            self.assertTrue(source_file.exists())
            self.assertTrue(ns["repaired_path"].exists())
            self.assertNotEqual(source_file, ns["repaired_path"])
            self.assertEqual(ns["mask_binary"].getbbox(), (100, 300, 850, 590))
            self.assertEqual(
                ns["mask_binary"].boxes, [(100, 300, 240, 490), (700, 400, 850, 590)]
            )
            self.assertEqual(len(FakeInpaintPipeline.calls), 1)
            mode, args = FakeInpaintPipeline.calls[0]
            self.assertEqual(mode, "offload")
            self.assertIs(args["mask_image"], ns["mask_binary"])
            self.assertEqual(args["generator"].seed, 777)
            self.assertEqual(args["padding_mask_crop"], 32)
            self.assertTrue(FakePipeline.last_instance.hooks_removed)
            metadata = json.loads(ns["repaired"].metadata["parameters"])
            self.assertEqual(metadata["source"], str(source_file))
            self.assertEqual(metadata["target"], "hands")

    def test_inpainting_rejects_mask_box_errors_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modules = enable_fake_inpainting(fake_modules())
            with patch.dict(sys.modules, modules):
                ns = {}
                self.prepare_existing_for_loras(ns, directory)
                exec(cell_source("load-pipeline"), ns)
                image_file = directory / "sample.png"
                image_file.write_bytes(b"an existing PNG in the fixture")
                for fields, error in [
                    (
                        dict(SOURCE_IMAGE=str(image_file), BOXES="1,2,1100,500"),
                        "ngoài ảnh",
                    ),
                    (
                        dict(SOURCE_IMAGE=str(image_file), BOXES="0,0,1024,1024"),
                        "không phủ toàn bộ ảnh",
                    ),
                    (
                        dict(SOURCE_IMAGE=str(image_file), BOXES="not four numbers"),
                        "BOXES phải",
                    ),
                    (dict(SOURCE_IMAGE=str(image_file), BOXES=""), "Chọn đúng MỘT"),
                    (
                        dict(
                            SOURCE_IMAGE=str(image_file),
                            MASK_PATH="/tmp/mask.png",
                            BOXES="1,2,3,4",
                        ),
                        "Chọn đúng MỘT",
                    ),
                ]:
                    with self.subTest(fields=fields):
                        with self.assertRaisesRegex(ValueError, error):
                            exec(
                                set_form_fields(cell_source("refine-image"), **fields),
                                ns,
                            )
                self.assertEqual(FakeInpaintPipeline.calls, [])

    def test_inpainting_accepts_explicit_black_white_mask_without_generation_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modules = enable_fake_inpainting(fake_modules())
            with patch.dict(sys.modules, modules):
                ns = {}
                self.prepare_existing_for_loras(ns, directory)
                exec(cell_source("load-pipeline"), ns)
                original = directory / "portrait.png"
                mask = directory / "eyes_mask.png"
                original.write_bytes(b"photo fixture")
                mask.write_bytes(b"black and white fixture")
                exec(
                    set_form_fields(
                        cell_source("refine-image"),
                        SOURCE_IMAGE=str(original),
                        MASK_PATH=str(mask),
                        TARGET="eyes",
                        REFINE_PROMPT="anime girl",
                        REFINE_SEED=10,
                    ),
                    ns,
                )
            self.assertEqual(ns["mask_binary"].getbbox(), (50, 50, 110, 120))
            self.assertIn("perfect eyes", FakeInpaintPipeline.calls[0][1]["prompt"])
            self.assertTrue(ns["repaired_path"].exists())

    def test_inpaint_oom_reloads_verified_loras_with_offload_same_seed(self):
        anatomy = b"verified anatomy"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            local = directory / "anatomy_helper.safetensors"
            local.write_bytes(anatomy)
            modules = enable_fake_inpainting(fake_modules(free_gib=15))
            with patch.dict(sys.modules, modules):
                ns = {}
                self.prepare_existing_for_loras(
                    ns,
                    directory,
                    USE_ANATOMY_LORA=True,
                    ANATOMY_LORA_PATH=str(local),
                )
                exec(substitute_test_loras(cell_source("prepare-loras"), anatomy), ns)
                exec(cell_source("load-pipeline"), ns)
                source_file = directory / "source.png"
                source_file.write_bytes(b"mock image")
                FakeInpaintPipeline.fail_gpu_once = True
                exec(
                    set_form_fields(
                        cell_source("refine-image"),
                        SOURCE_IMAGE=str(source_file),
                        BOXES="20,20,120,120",
                        REFINE_PROMPT="anime character",
                        REFINE_SEED=456,
                    ),
                    ns,
                )
            self.assertEqual(
                [mode for mode, _ in FakeInpaintPipeline.calls], ["cuda", "offload"]
            )
            self.assertEqual(
                [args["generator"].seed for _, args in FakeInpaintPipeline.calls],
                [456, 456],
            )
            self.assertEqual(len(FakePipeline.instances), 2)
            self.assertTrue(
                all(len(pipe.loras) == 1 for pipe in FakePipeline.instances)
            )
            self.assertTrue(ns["repaired_path"].exists())

    def test_no_gpu_fails_before_download(self):
        with patch.dict(sys.modules, fake_modules(gpu=False)):
            with self.assertRaisesRegex(RuntimeError, "Chưa có GPU"):
                exec(cell_source("check-gpu"), {})


if __name__ == "__main__":
    unittest.main()
