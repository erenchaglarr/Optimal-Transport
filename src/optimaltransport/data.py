from pathlib import Path
from torchvision import datasets

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    raise RuntimeError("Project root not found")

PROJECT_ROOT = find_project_root(Path.cwd())
DATA_DIR = PROJECT_ROOT / "data"

training_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
)

test_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
)

print("Data downloaded")