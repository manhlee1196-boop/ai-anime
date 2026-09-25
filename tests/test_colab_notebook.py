"""CPU-only smoke tests for the self-contained Colab notebook.

Run with: python -m unittest discover -s tests
Real GPU inference still requires Colab and an actual WAI checkpoint.
"""

import contextlib
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
            f"{name} = {value!r}",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Colab form field missing: {name}")
    return source


class FakeImage:
    def save(self, path, pnginfo):
        Path(path).write_bytes(b"PNG smoke test")
        self.metadata = pnginfo.values


class FakePipeline:
    last_instance = None

    def __init__(self):
        self.scheduler = types.SimpleNamespace(config={"scheduler": "original"})
        self.vae = types.SimpleNamespace(enable_slicing=self.slicing, enable_tiling=self.tiling)
        self.sliced = False
        self.tiled = False
        self.mode = None
        self.image = FakeImage()
        self.arguments = None
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
        self.mode = device

    def __call__(self, **kwargs):
        self.arguments = kwargs
        return types.SimpleNamespace(images=[self.image])


class FakePngInfo:
    def __init__(self):
        self.values = {}

    def add_text(self, key, value):
        self.values[key] = value


def fake_modules(free_gib=15, gpu=True):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: gpu,
        get_device_properties=lambda index: types.SimpleNamespace(name="Fake GPU"),
        mem_get_info=lambda: (free_gib * 2**30, 16 * 2**30),
        empty_cache=lambda: None,
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
    )
    torch.__version__ = "mock"
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda seed: seed)
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
        "IPython": ipython,
        "IPython.display": ipython_display,
        "PIL": pil,
        "PIL.PngImagePlugin": pil_png,
    }


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_has_clean_code_cells(self):
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

    def run_mock_notebook(self, model_path, output_path, free_gib=15):
        namespace = {}
        with patch.dict(sys.modules, fake_modules(free_gib=free_gib)):
            exec(cell_source("check-gpu"), namespace)
            config = set_form_fields(
                cell_source("configure-paths"),
                MOUNT_DRIVE=False,
                MODEL_PATH=str(model_path),
                CACHE_MODEL_LOCAL=False,
                OUTPUT_DIR=str(output_path),
            )
            exec(config, namespace)
            exec(cell_source("load-pipeline"), namespace)
            generate = set_form_fields(cell_source("generate-image"), SEED=42)
            exec(generate, namespace)
        return FakePipeline.last_instance, list(output_path.glob("wai_*.png"))

    def test_generates_saves_and_embeds_metadata_with_enough_vram(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "WAI-illustrious.safetensors"
            with model_path.open("wb") as model:
                model.truncate(101 * 2**20)  # sparse test file, never ships in the repository
            pipeline, images = self.run_mock_notebook(model_path, directory / "outputs")
            self.assertEqual(pipeline.mode, "cuda")
            self.assertTrue(pipeline.sliced)
            self.assertFalse(pipeline.tiled)
            self.assertEqual(pipeline.arguments["num_images_per_prompt"], 1)
            self.assertEqual(len(images), 1)
            metadata = json.loads(pipeline.image.metadata["parameters"])
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["model"], "WAI-illustrious.safetensors")

    def test_low_vram_uses_model_offload_and_vae_tiling(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "WAI-illustrious.safetensors"
            with model_path.open("wb") as model:
                model.truncate(101 * 2**20)
            pipeline, images = self.run_mock_notebook(model_path, directory / "outputs", free_gib=12)
            self.assertEqual(pipeline.mode, "offload")
            self.assertTrue(pipeline.tiled)
            self.assertEqual(len(images), 1)

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.dict(sys.modules, fake_modules()):
                config = set_form_fields(
                    cell_source("configure-paths"),
                    MOUNT_DRIVE=False,
                    MODEL_PATH=str(path / "missing.safetensors"),
                    CACHE_MODEL_LOCAL=False,
                    OUTPUT_DIR=str(path / "outputs"),
                )
                with self.assertRaisesRegex(FileNotFoundError, "Không tìm thấy model"):
                    exec(config, {})

    def test_rejects_checkpoint_without_text_encoders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model_path = path / "unet-only.safetensors"
            with model_path.open("wb") as model:
                model.truncate(101 * 2**20)
            modules = fake_modules()

            class IncompleteHeader:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def keys(self):
                    return ["model.diffusion_model.input_blocks.0.0.weight"]

            modules["safetensors"].safe_open = lambda *args, **kwargs: IncompleteHeader()
            with patch.dict(sys.modules, modules):
                config = set_form_fields(
                    cell_source("configure-paths"),
                    MOUNT_DRIVE=False,
                    MODEL_PATH=str(model_path),
                    CACHE_MODEL_LOCAL=False,
                    OUTPUT_DIR=str(path / "outputs"),
                )
                with self.assertRaisesRegex(ValueError, "UNet và text encoder"):
                    exec(config, {})

    def test_no_gpu_fails_before_installing_dependencies(self):
        with patch.dict(sys.modules, fake_modules(gpu=False)):
            with self.assertRaisesRegex(RuntimeError, "Chưa có GPU"):
                exec(cell_source("check-gpu"), {})


if __name__ == "__main__":
    unittest.main()
