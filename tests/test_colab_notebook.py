"""CPU-only smoke tests for the self-contained Colab notebook.

Run: python -m unittest discover -s tests -v
Real GPU inference/download of the 6.94 GB checkpoint must be tried in Colab.
"""

import contextlib
import hashlib
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
    return "".join(next(cell for cell in notebook["cells"] if cell["metadata"]["id"] == cell_id)["source"])


def set_form_fields(source, **values):
    for name, value in values.items():
        source, count = re.subn(
            rf"^{re.escape(name)} = .*?(?= # @param)",
            f"{name} = {value!r}", source, count=1, flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Colab form field missing: {name}")
    return source


def substitute_test_checkpoint(source, content):
    """Shrink expected download size/hash only in tests, never in the real notebook."""
    replacements = {
        "HF_MODEL_BYTES": len(content),
        "HF_SHA256": hashlib.sha256(content).hexdigest(),
        "MIN_CHECKPOINT_BYTES": 1,
        "DISK_RESERVE_BYTES": 1,
    }
    for name, value in replacements.items():
        source, count = re.subn(
            rf"^{name} = .*?$", f"{name} = {value!r}",
            source, count=1, flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Missing constant: {name}")
    return source


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
    fail_gpu_to_once = False
    fail_gpu_inference_once = False
    calls = []

    def __init__(self):
        self.scheduler = types.SimpleNamespace(config={"scheduler": "original"})
        self.vae = types.SimpleNamespace(enable_slicing=self.slicing, enable_tiling=self.tiling)
        self.sliced = False
        self.tiled = False
        self.mode = None
        self.image = FakeImage()
        FakePipeline.last_instance = self

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


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def fake_modules(free_gib=15, gpu=True):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: gpu,
        get_device_properties=lambda index: types.SimpleNamespace(name="Fake GPU"),
        mem_get_info=lambda: (free_gib * 2**30, 16 * 2**30),
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
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return [
                "model.diffusion_model.input_blocks.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            ]

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *args, **kwargs: Header()
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected model download"))
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
        "torch": torch, "diffusers": diffusers, "safetensors": safetensors,
        "huggingface_hub": hub, "psutil": psutil,
        "IPython": ipython, "IPython.display": ipython_display,
        "PIL": pil, "PIL.PngImagePlugin": pil_png,
    }


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.last_instance = None
        FakePipeline.calls = []
        FakePipeline.fail_gpu_to_once = False
        FakePipeline.fail_gpu_inference_once = False

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
        self.assertNotIn("\"torch==", cell_source("install-packages"))
        prepare = cell_source("prepare-model")
        self.assertIn("token=False", prepare)
        self.assertIn('HF_MODEL_BYTES = 6_938_040_682', prepare)
        self.assertIn('HF_SHA256 = "f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04"', prepare)

    def configure(self, namespace, model_path, output_path, **extra_fields):
        config = set_form_fields(
            cell_source("configure-paths"),
            MOUNT_DRIVE=False, MODEL_PATH=str(model_path), OUTPUT_DIR=str(output_path),
            **extra_fields,
        )
        exec(config, namespace)
        namespace["local_cache_root"] = output_path.parent / "model_cache"

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
                exec(substitute_test_checkpoint(cell_source("prepare-model"), b"other bytes"), namespace)
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=42), namespace)
            self.assertEqual(namespace["checkpoint"], model_path)
            self.assertEqual(FakePipeline.last_instance.mode, "cuda")
            self.assertTrue(FakePipeline.last_instance.sliced)
            self.assertFalse(FakePipeline.last_instance.tiled)
            images = list(output.glob("wai_*.png"))
            self.assertEqual(len(images), 1)
            self.assertFalse(list(output.glob("*.partial")))
            metadata = json.loads(FakePipeline.last_instance.image.metadata["parameters"])
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["model"], model_path.name)

    def test_auto_download_verifies_sha_and_runs_from_single_cache_file(self):
        payload = b"v17 mock full checkpoint (use real SHA only in production)"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            downloaded = directory / "waiIllustriousSDXL_v170.safetensors"
            downloaded.write_bytes(payload)
            output = directory / "outputs"
            calls = []
            modules = fake_modules()
            def download(**kwargs):
                calls.append(kwargs)
                return str(downloaded)
            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                namespace = {}
                exec(cell_source("check-gpu"), namespace)
                self.configure(namespace, directory / "missing.safetensors", output, PERSIST_MODEL_TO_DRIVE=False)
                exec(substitute_test_checkpoint(cell_source("prepare-model"), payload), namespace)
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=99), namespace)
            self.assertEqual(namespace["checkpoint"], downloaded)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], "32be7bfdcd406db70df663b9cee3313957deb68f")
            self.assertEqual(calls[0]["token"], False)
            self.assertIn("cache_dir", calls[0])
            self.assertEqual(len(list(output.glob("wai_*.png"))), 1)

    def test_bad_hash_never_loads_checkpoint(self):
        payload = b"wrong-data"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            downloaded = directory / "waiIllustriousSDXL_v170.safetensors"
            downloaded.write_bytes(payload)
            modules = fake_modules()
            modules["huggingface_hub"].hf_hub_download = lambda **kwargs: str(downloaded)
            with patch.dict(sys.modules, modules):
                namespace = {}
                self.configure(namespace, directory / "missing.safetensors", directory / "outputs")
                # Same byte length, different SHA-256: the hash check must reject it.
                prepare = substitute_test_checkpoint(cell_source("prepare-model"), b"right-data")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    exec(prepare, namespace)
            self.assertIsNone(FakePipeline.last_instance)

    def test_missing_model_without_auto_download_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                self.configure(
                    namespace, directory / "missing.safetensors", directory / "outputs",
                    AUTO_DOWNLOAD=False,
                )
                with self.assertRaisesRegex(FileNotFoundError, "Bật AUTO_DOWNLOAD"):
                    exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)

    def test_download_persists_verified_model_without_extra_local_copy(self):
        payload = b"verified-v17-mock"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_drive = directory / "drive"
            model_path = fake_drive / "MyDrive" / "models" / "WAI-illustrious.safetensors"
            model_path.parent.mkdir(parents=True)
            downloaded = directory / "waiIllustriousSDXL_v170.safetensors"
            downloaded.write_bytes(payload)
            modules = fake_modules()
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return str(downloaded)
            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                namespace = {}
                self.configure(namespace, model_path, directory / "outputs")
                namespace["drive_root"] = fake_drive
                exec(substitute_test_checkpoint(cell_source("prepare-model"), payload), namespace)
            self.assertEqual(namespace["checkpoint"], downloaded)
            self.assertEqual(model_path.read_bytes(), payload)
            self.assertEqual(len(calls), 1)
            self.assertIn("cache_dir", calls[0])
            self.assertFalse(list(model_path.parent.glob("*.partial")))

    def test_drive_write_failure_still_runs_from_verified_local_cache(self):
        payload = b"verified-v17-mock"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_drive = directory / "drive"
            model_path = fake_drive / "MyDrive" / "models" / "WAI-illustrious.safetensors"
            model_path.parent.mkdir(parents=True)
            downloaded = directory / "waiIllustriousSDXL_v170.safetensors"
            downloaded.write_bytes(payload)
            modules = fake_modules()
            modules["huggingface_hub"].hf_hub_download = lambda **kwargs: str(downloaded)
            with patch.dict(sys.modules, modules):
                namespace = {}
                self.configure(namespace, model_path, directory / "outputs")
                namespace["drive_root"] = fake_drive
                with patch("shutil.copyfile", side_effect=OSError("Drive quota reached")):
                    exec(substitute_test_checkpoint(cell_source("prepare-model"), payload), namespace)
            self.assertEqual(namespace["checkpoint"], downloaded)
            self.assertFalse(model_path.exists())

    def test_existing_drive_checkpoint_uses_direct_path_on_low_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_drive = directory / "drive"
            model_path = fake_drive / "MyDrive" / "models" / "existing.safetensors"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"existing checkpoint")
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                self.configure(namespace, model_path, directory / "outputs")
                namespace["drive_root"] = fake_drive
                with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=0)):
                    exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)
            self.assertEqual(namespace["checkpoint"], model_path)

    def test_low_local_disk_downloads_directly_to_drive(self):
        payload = b"verified-v17-mock"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_drive = directory / "drive"
            model_path = fake_drive / "MyDrive" / "models" / "WAI-illustrious.safetensors"
            model_path.parent.mkdir(parents=True)
            modules = fake_modules()
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                remote_file = Path(kwargs["local_dir"]) / "waiIllustriousSDXL_v170.safetensors"
                remote_file.write_bytes(payload)
                return str(remote_file)
            modules["huggingface_hub"].hf_hub_download = download
            with patch.dict(sys.modules, modules):
                namespace = {}
                self.configure(namespace, model_path, directory / "outputs")
                namespace["drive_root"] = fake_drive
                with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=0)):
                    exec(substitute_test_checkpoint(cell_source("prepare-model"), payload), namespace)
            self.assertEqual(namespace["checkpoint"], model_path)
            self.assertEqual(model_path.read_bytes(), payload)
            self.assertEqual(len(calls), 1)
            self.assertIn("local_dir", calls[0])
            self.assertNotIn("cache_dir", calls[0])

    def test_existing_drive_checkpoint_is_cached_once_with_preserved_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_drive = directory / "drive"
            model_path = fake_drive / "MyDrive" / "models" / "existing.safetensors"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"existing checkpoint")
            with patch.dict(sys.modules, fake_modules()):
                namespace = {}
                self.configure(namespace, model_path, directory / "outputs")
                namespace["drive_root"] = fake_drive
                prepare = substitute_test_checkpoint(cell_source("prepare-model"), b"fixture")
                with patch("shutil.copy2", wraps=__import__("shutil").copy2) as copy2:
                    exec(prepare, namespace)
                    cached = namespace["checkpoint"]
                    self.assertEqual(cached.read_bytes(), b"existing checkpoint")
                    self.assertEqual(cached.stat().st_mtime_ns, model_path.stat().st_mtime_ns)
                    exec(prepare, namespace)
                    self.assertEqual(copy2.call_count, 1)

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
                exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)
                exec(cell_source("load-pipeline"), namespace)
                generate = set_form_fields(cell_source("generate-image"), SEED=101)
                generate = generate.replace(
                    'backup_dir = Path("/content/wai_outputs")', f"backup_dir = Path({str(backup)!r})"
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
                exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)
                exec(cell_source("load-pipeline"), namespace)
            self.assertEqual(FakePipeline.last_instance.mode, "offload")
            self.assertTrue(FakePipeline.last_instance.tiled)

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
                exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)
                exec(cell_source("load-pipeline"), namespace)
                exec(set_form_fields(cell_source("generate-image"), SEED=42), namespace)
            self.assertEqual([mode for mode, _ in FakePipeline.calls], ["cuda", "offload"])
            self.assertEqual([args["generator"].seed for _, args in FakePipeline.calls], [42, 42])
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
                exec(substitute_test_checkpoint(cell_source("prepare-model"), b"fixture"), namespace)
                exec(cell_source("load-pipeline"), namespace)
            self.assertTrue(namespace["use_offload"])
            self.assertEqual(FakePipeline.last_instance.mode, "offload")

    def test_no_gpu_fails_before_download(self):
        with patch.dict(sys.modules, fake_modules(gpu=False)):
            with self.assertRaisesRegex(RuntimeError, "Chưa có GPU"):
                exec(cell_source("check-gpu"), {})


if __name__ == "__main__":
    unittest.main()
