import importlib
import platform

import torch
import torchaudio


def check_import(package_name: str):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {package_name}: {version}")
    except Exception as error:
        print(f"[FAIL] {package_name}: {error}")


print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

packages = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "tqdm",
    "yaml",
    "soundfile",
    "librosa",
    "einops",
    "rich",
    "torchmetrics",
    "pesq",
    "pystoi",
    "mir_eval",
    "psutil",
    "sklearn",
]

for package in packages:
    check_import(package)
