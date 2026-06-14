"""
Standalone CIFAR-10 Equinox/JAX Convolutional Autoencoder Trainer

This file is independent of your previous code.

It trains a nonlinear convolutional neural network autoencoder on CIFAR-10 and saves:

    1. The trained autoencoder as .eqx
    2. The training latent space as .eqx
    3. The validation latent space as .eqx
    4. MSE logs as .csv
    5. Metadata as .json

Example:
    python train_cifar10_eqx_autoencoder.py --latent-dims 2 5 10

Example test run:
    python train_cifar10_eqx_autoencoder.py --latent-dims 2 --epochs 3

The model is nonlinear and convolutional:

    image 3x32x32
        -> Conv encoder
        -> latent vector
        -> Conv decoder
        -> reconstructed image 3x32x32

Important:
    .eqx files are Equinox PyTree serialization files.
    To reload them, you need to instantiate the same model skeleton first.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_root: str = "./data"
    output_dir: str = "./cifar10_eqx_autoencoder_runs"

    latent_dims: tuple[int, ...] = (2, 5, 10)

    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    val_fraction: float = 0.1

    save_latent_spaces: bool = True


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def torch_batch_to_jax(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.detach().cpu().numpy(), dtype=jnp.float32)


def labels_to_jax(y: torch.Tensor) -> jax.Array:
    return jnp.asarray(y.detach().cpu().numpy(), dtype=jnp.int32)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def get_cifar10_dataset(data_root: str, train: bool = True, download: bool = True):
    transform = transforms.Compose([transforms.ToTensor()])

    return datasets.CIFAR10(
        root=data_root,
        train=train,
        download=download,
        transform=transform,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def make_train_val_split(dataset, val_fraction: float, seed: int):
    n_total = len(dataset)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        lengths=[n_train, n_val],
        generator=generator,
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------
# Equinox model
# ---------------------------------------------------------------------

class Encoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, latent_dim: int, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k1,
        )

        self.conv2 = eqx.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k2,
        )

        self.conv3 = eqx.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k3,
        )

        self.fc1 = eqx.nn.Linear(128 * 4 * 4, 256, key=k4)
        self.fc2 = eqx.nn.Linear(256, latent_dim, key=k5)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (3, 32, 32)
        x = jax.nn.relu(self.conv1(x))   # (32, 16, 16)
        x = jax.nn.relu(self.conv2(x))   # (64, 8, 8)
        x = jax.nn.relu(self.conv3(x))   # (128, 4, 4)

        x = jnp.ravel(x)
        x = jax.nn.relu(self.fc1(x))
        z = self.fc2(x)

        return z


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(self, latent_dim: int, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.fc1 = eqx.nn.Linear(latent_dim, 256, key=k1)
        self.fc2 = eqx.nn.Linear(256, 128 * 4 * 4, key=k2)

        # Decoder uses nearest-neighbor upsampling by jnp.repeat,
        # followed by convolution. This avoids relying on ConvTranspose2d.
        self.conv1 = eqx.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            key=k3,
        )

        self.conv2 = eqx.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            key=k4,
        )

        self.conv3 = eqx.nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            key=k5,
        )

    def upsample2x(self, x: jax.Array) -> jax.Array:
        # x shape: (channels, height, width)
        x = jnp.repeat(x, repeats=2, axis=1)
        x = jnp.repeat(x, repeats=2, axis=2)
        return x

    def __call__(self, z: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.fc1(z))
        x = jax.nn.relu(self.fc2(x))

        x = jnp.reshape(x, (128, 4, 4))

        x = self.upsample2x(x)             # (128, 8, 8)
        x = jax.nn.relu(self.conv1(x))     # (64, 8, 8)

        x = self.upsample2x(x)             # (64, 16, 16)
        x = jax.nn.relu(self.conv2(x))     # (32, 16, 16)

        x = self.upsample2x(x)             # (32, 32, 32)
        x = jax.nn.sigmoid(self.conv3(x))  # (3, 32, 32)

        return x


class ConvAutoEncoder(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)

        self.encoder = Encoder(latent_dim=latent_dim, key=k1)
        self.decoder = Decoder(latent_dim=latent_dim, key=k2)
        self.latent_dim = int(latent_dim)

    def encode(self, x: jax.Array) -> jax.Array:
        return self.encoder(x)

    def decode(self, z: jax.Array) -> jax.Array:
        return self.decoder(z)

    def __call__(self, x: jax.Array) -> jax.Array:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class LatentSpace(eqx.Module):
    """
    Container for saving latent vectors as an .eqx file.

    z:
        shape (n_images, latent_dim)

    y:
        shape (n_images,)
    """

    z: jax.Array
    y: jax.Array
    latent_dim: int = eqx.field(static=True)
    split: str = eqx.field(static=True)

    def __init__(
        self,
        z: jax.Array,
        y: jax.Array,
        latent_dim: int,
        split: str,
    ):
        self.z = z
        self.y = y
        self.latent_dim = int(latent_dim)
        self.split = str(split)


# ---------------------------------------------------------------------
# Loss and train steps
# ---------------------------------------------------------------------

def reconstruction_mse_loss(model: ConvAutoEncoder, x_batch: jax.Array) -> jax.Array:
    x_hat_batch = jax.vmap(model)(x_batch)
    return jnp.mean((x_hat_batch - x_batch) ** 2)


def make_train_step(optimizer):
    @eqx.filter_jit
    def train_step(model, opt_state, x_batch):
        loss, grads = eqx.filter_value_and_grad(reconstruction_mse_loss)(
            model,
            x_batch,
        )

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )

        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    return train_step


@eqx.filter_jit
def eval_step(model, x_batch):
    return reconstruction_mse_loss(model, x_batch)


@eqx.filter_jit
def encode_batch(model, x_batch):
    return jax.vmap(model.encode)(x_batch)


# ---------------------------------------------------------------------
# Saving and loading helpers
# ---------------------------------------------------------------------

def save_model_eqx(path: str | Path, model: ConvAutoEncoder, metadata: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    eqx.tree_serialise_leaves(path, model)

    metadata_path = path.with_suffix(".json")
    save_json(metadata_path, metadata)


def load_model_eqx(path: str | Path, latent_dim: int, key: jax.Array) -> ConvAutoEncoder:
    """
    Reload helper.

    Example:
        key = jax.random.PRNGKey(0)
        model = load_model_eqx(
            "cifar10_eqx_autoencoder_runs/checkpoints/autoencoder_latent2_final.eqx",
            latent_dim=2,
            key=key,
        )
    """
    skeleton = ConvAutoEncoder(latent_dim=latent_dim, key=key)
    model = eqx.tree_deserialise_leaves(path, skeleton)
    return model


def save_latent_space_eqx(
    path: str | Path,
    latent_space: LatentSpace,
    metadata: dict,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    eqx.tree_serialise_leaves(path, latent_space)

    metadata_path = path.with_suffix(".json")
    save_json(metadata_path, metadata)


def load_latent_space_eqx(
    path: str | Path,
    n_images: int,
    latent_dim: int,
    split: str,
) -> LatentSpace:
    """
    Reload helper.

    You need n_images and latent_dim to build the skeleton.

    Example:
        latent_space = load_latent_space_eqx(
            "cifar10_eqx_autoencoder_runs/latent_spaces/latent_space_train_latent2.eqx",
            n_images=45000,
            latent_dim=2,
            split="train",
        )

        z = latent_space.z
        y = latent_space.y
    """
    skeleton = LatentSpace(
        z=jnp.zeros((n_images, latent_dim), dtype=jnp.float32),
        y=jnp.zeros((n_images,), dtype=jnp.int32),
        latent_dim=latent_dim,
        split=split,
    )

    latent_space = eqx.tree_deserialise_leaves(path, skeleton)

    return latent_space


# ---------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------

def compute_latent_space(
    model: ConvAutoEncoder,
    dataset,
    config: TrainConfig,
    split: str,
) -> LatentSpace:
    loader = make_loader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    z_batches = []
    y_batches = []

    for x_torch, y_torch in loader:
        x_batch = torch_batch_to_jax(x_torch)
        y_batch = labels_to_jax(y_torch)

        z_batch = encode_batch(model, x_batch)

        z_batches.append(np.asarray(z_batch))
        y_batches.append(np.asarray(y_batch))

    z = jnp.asarray(np.concatenate(z_batches, axis=0), dtype=jnp.float32)
    y = jnp.asarray(np.concatenate(y_batches, axis=0), dtype=jnp.int32)

    return LatentSpace(
        z=z,
        y=y,
        latent_dim=model.latent_dim,
        split=split,
    )


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_latent_dim(
    config: TrainConfig,
    latent_dim: int,
    dataset,
    run_id: str,
    key: jax.Array,
) -> dict:
    print("\n" + "=" * 80)
    print(f"Training Equinox CIFAR-10 convolutional autoencoder | latent_dim={latent_dim}")
    print("=" * 80)

    output_dir = ensure_dir(config.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    latent_dir = ensure_dir(output_dir / "latent_spaces")

    train_dataset, val_dataset = make_train_val_split(
        dataset=dataset,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ConvAutoEncoder(latent_dim=latent_dim, key=key)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_step = make_train_step(optimizer)

    epoch_log_path = logs_dir / f"epoch_mse_latent{latent_dim}_{run_id}.csv"

    train_mse_history = []
    val_mse_history = []

    best_val_mse = float("inf")
    best_epoch = 0

    best_model_path = checkpoint_dir / f"autoencoder_latent{latent_dim}_best.eqx"
    final_model_path = checkpoint_dir / f"autoencoder_latent{latent_dim}_final.eqx"

    for epoch in range(1, config.epochs + 1):
        train_losses = []

        for x_torch, _ in train_loader:
            x_batch = torch_batch_to_jax(x_torch)

            model, opt_state, loss = train_step(
                model,
                opt_state,
                x_batch,
            )

            train_losses.append(float(loss))

        val_losses = []

        for x_torch, _ in val_loader:
            x_batch = torch_batch_to_jax(x_torch)
            val_loss = eval_step(model, x_batch)
            val_losses.append(float(val_loss))

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        print(
            f"latent={latent_dim:>3} | "
            f"epoch={epoch:03d}/{config.epochs:03d} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val_mse:.6f}"
        )

        append_csv_row(
            epoch_log_path,
            {
                "run_id": run_id,
                "latent_dim": latent_dim,
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
            },
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

            save_model_eqx(
                path=best_model_path,
                model=model,
                metadata={
                    "run_id": run_id,
                    "latent_dim": latent_dim,
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_mse,
                    "config": asdict(config),
                },
            )

    final_train_mse = train_mse_history[-1]
    final_val_mse = val_mse_history[-1]

    save_model_eqx(
        path=final_model_path,
        model=model,
        metadata={
            "run_id": run_id,
            "latent_dim": latent_dim,
            "final_train_mse": final_train_mse,
            "final_val_mse": final_val_mse,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "config": asdict(config),
        },
    )

    train_latent_path = None
    val_latent_path = None

    if config.save_latent_spaces:
        print("Computing and saving latent spaces as .eqx files...")

        train_latent_space = compute_latent_space(
            model=model,
            dataset=train_dataset,
            config=config,
            split="train",
        )

        val_latent_space = compute_latent_space(
            model=model,
            dataset=val_dataset,
            config=config,
            split="val",
        )

        train_latent_path = latent_dir / f"latent_space_train_latent{latent_dim}.eqx"
        val_latent_path = latent_dir / f"latent_space_val_latent{latent_dim}.eqx"

        save_latent_space_eqx(
            path=train_latent_path,
            latent_space=train_latent_space,
            metadata={
                "run_id": run_id,
                "latent_dim": latent_dim,
                "split": "train",
                "n_images": int(train_latent_space.z.shape[0]),
                "z_shape": list(train_latent_space.z.shape),
                "y_shape": list(train_latent_space.y.shape),
                "source_model": str(final_model_path),
            },
        )

        save_latent_space_eqx(
            path=val_latent_path,
            latent_space=val_latent_space,
            metadata={
                "run_id": run_id,
                "latent_dim": latent_dim,
                "split": "val",
                "n_images": int(val_latent_space.z.shape[0]),
                "z_shape": list(val_latent_space.z.shape),
                "y_shape": list(val_latent_space.y.shape),
                "source_model": str(final_model_path),
            },
        )

    summary_path = logs_dir / "mse_summary.csv"

    append_csv_row(
        summary_path,
        {
            "run_id": run_id,
            "latent_dim": latent_dim,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "final_train_mse": final_train_mse,
            "final_val_mse": final_val_mse,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
            "train_latent_path": str(train_latent_path),
            "val_latent_path": str(val_latent_path),
        },
    )

    result = {
        "run_id": run_id,
        "latent_dim": latent_dim,
        "final_train_mse": final_train_mse,
        "final_val_mse": final_val_mse,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "train_latent_path": str(train_latent_path),
        "val_latent_path": str(val_latent_path),
        "epoch_log_path": str(epoch_log_path),
    }

    save_json(
        logs_dir / f"results_latent{latent_dim}_{run_id}.json",
        result,
    )

    print("\nSaved files:")
    print(f"Best model:          {best_model_path}")
    print(f"Final model:         {final_model_path}")

    if train_latent_path is not None:
        print(f"Train latent space:  {train_latent_path}")
        print(f"Val latent space:    {val_latent_path}")

    print(f"MSE summary:         {summary_path}")

    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Standalone Equinox/JAX CIFAR-10 convolutional autoencoder."
    )

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./cifar10_eqx_autoencoder_runs")

    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 5, 10])

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument(
        "--no-latent-spaces",
        action="store_true",
        help="Do not save encoded latent spaces.",
    )

    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        latent_dims=tuple(args.latent_dims),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        save_latent_spaces=not args.no_latent_spaces,
    )


def main() -> None:
    config = parse_args()

    set_seed(config.seed)

    run_id = timestamp()

    output_dir = ensure_dir(config.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")

    save_json(
        logs_dir / f"config_{run_id}.json",
        asdict(config),
    )

    print("Run ID:", run_id)
    print("Output directory:", output_dir)
    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())

    dataset = get_cifar10_dataset(
        data_root=config.data_root,
        train=True,
        download=True,
    )

    print("CIFAR-10 train images:", len(dataset))
    print("Image shape:", tuple(dataset[0][0].shape))

    key = jax.random.PRNGKey(config.seed)

    all_results = []

    for latent_dim in config.latent_dims:
        key, subkey = jax.random.split(key)

        result = train_latent_dim(
            config=config,
            latent_dim=latent_dim,
            dataset=dataset,
            run_id=run_id,
            key=subkey,
        )

        all_results.append(result)

    save_json(
        logs_dir / f"all_results_{run_id}.json",
        {
            "run_id": run_id,
            "config": asdict(config),
            "results": all_results,
        },
    )

    print("\n" + "=" * 80)
    print("Finished all latent dimensions.")
    print("=" * 80)
    print(f"Main MSE summary: {logs_dir / 'mse_summary.csv'}")
    print(f"Full JSON results: {logs_dir / f'all_results_{run_id}.json'}")


if __name__ == "__main__":
    main()
