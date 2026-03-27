from pathlib import Path # Imported because we need to find where the project is located so the data is downloaded to the right place
from torchvision import datasets, transforms # We will use torchvisions dataset class for importing the data and also for any transform.
from torch.utils.data import DataLoader, Subset 
import numpy as np


## Following function find pyproject.toml or .git to download data in to the same location as them

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    raise RuntimeError("Project root not found")

PROJECT_ROOT = find_project_root(Path.cwd())
DATA_DIR = PROJECT_ROOT / "data"

## This function transforms the data to torch tensor using torchvision transforms when its downloaded
def get_transform():
    return transforms.Compose([transforms.ToTensor()])

## Following function downloads the data 
def get_mnist_dataset(data_root=DATA_DIR, train=True, download=True):
    return datasets.MNIST(
        root=data_root,
        train=train,
        download=download,
        transform=get_transform(),
    )
    
## Following function is used to get the input shape for the autoencoder.
## It does it by taking first object from the dataset and finding the shape
def get_input_shape(dataset):
    x, _ = dataset[0]
    return tuple(x.shape)

def get_labels(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        if hasattr(targets, "cpu"):
            return targets.cpu().numpy()
        return np.asarray(targets)

    return np.array([dataset[i][1] for i in range(len(dataset))])


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def make_fold_loaders(dataset, train_idx, val_idx, batch_size, num_workers=0):
    if hasattr(train_idx, "tolist"):
        train_idx = train_idx.tolist()
    if hasattr(val_idx, "tolist"):
        val_idx = val_idx.tolist()

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = make_loader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader



