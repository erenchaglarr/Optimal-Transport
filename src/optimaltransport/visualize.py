from __future__ import annotations

from pathlib import Path

import jax
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .save import load_checkpoint
from .data import get_mnist_dataset, make_loader
from .lossfn import torch_batch_to_jax

def plot_latent_space_with_images(model, loader, max_points=200, zoom=0.5, title="Latent Space with Images"):
    all_z = []
    all_x = []
    all_y = []

    for x_batch_torch, y_batch_torch in loader:
        x_batch = torch_batch_to_jax(x_batch_torch)
        z_batch = jax.vmap(model.encoder)(x_batch)

        all_z.append(np.array(z_batch))
        all_x.append(np.array(x_batch))
        all_y.append(np.array(y_batch_torch))

    all_z = np.concatenate(all_z, axis=0)
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # only plot a subset so it stays readable
    n = min(max_points, len(all_z))
    idx = np.random.choice(len(all_z), size=n, replace=False)

    z_subset = all_z[idx]
    x_subset = all_x[idx]
    y_subset = all_y[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(z_subset[:, 0], z_subset[:, 1], c=y_subset, cmap="tab10", s=10, alpha=0.3)

    for i in range(n):
        img = x_subset[i].squeeze()   # (28, 28)
        imagebox = OffsetImage(img, cmap="gray", zoom=zoom)
        ab = AnnotationBbox(imagebox, (z_subset[i, 0], z_subset[i, 1]), frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    plt.show()

def plot_latent_space(model, loader, title="2D Latent Space"):
    all_z = []
    all_y = []

    for x_batch_torch, y_batch_torch in loader:
        x_batch = torch_batch_to_jax(x_batch_torch)
        z_batch = jax.vmap(model.encoder)(x_batch)

        all_z.append(np.array(z_batch))
        all_y.append(np.array(y_batch_torch))

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    plt.figure(figsize=(7, 7))
    plt.scatter(all_z[:, 0], all_z[:, 1], c=all_y, cmap="tab10", s=8)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_latent_fortnite(model, loader, title="2D Latent Space fortnit"):
    all_z = []
    all_y = []

    for x_batch_torch, y_batch_torch, in loader:
        x_batch = torch_batch_to_jax(x_batch_torch)
        y_batch = torch_batch_to_jax(y_batch_torch)
        filter = jnp.logical_or((y_batch == 1), (y_batch == 2))
        x_batch = x_batch[filter]
        y_batch = y_batch[filter]
        z_batch = jax.vmap(model.encoder)(x_batch)

        all_z.append(np.array(z_batch))
        all_y.append(np.array(y_batch))

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    plt.figure(figsize=(7, 7))
    plt.scatter(all_z[:, 0], all_z[:, 1], c=all_y, cmap="tab10", s=8)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_reconstructions(model, loader, n_examples=5):
    x_batch_torch, y_batch = next(iter(loader))
    x_batch = torch_batch_to_jax(x_batch_torch)
    x_hat_batch = jax.vmap(model)(x_batch)

    fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 4))

    for i in range(n_examples):
        axes[0, i].imshow(np.array(x_batch[i].squeeze()), cmap="gray")
        axes[0, i].set_title(f"Orig: {int(y_batch[i])}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.array(x_hat_batch[i].squeeze()), cmap="gray")
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_checkpoint(config, checkpoint_path=None, split="train"):
    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.model_dir) / config.paths.final_model_name

    model, _ = load_checkpoint(checkpoint_path)

    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )

    loader = make_loader(
        dataset,
        batch_size=int(config.hyperparameters.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
    )

    plot_latent_fortnite(model, loader, title="2D Latent Space fortnit")

    plot_latent_space(model, loader, title=f"2D Latent Space ({split} split)")
#     plot_reconstructions(
#         model,
#         loader,
#         n_examples=int(config.visualization.num_examples),
        
#     )
#     plot_latent_space_with_images(
#     model,
#     loader,
#     max_points=1000,
#     zoom=0.4,
#     title=f"Latent Space with Images ({split} split)",
# )

def plot_knn_decision_regions(
    model,
    loader,
    k=10,
    standardize=True,
    grid_resolution=300,
    title="KNN decision regions in latent space",
):
    all_z = []
    all_y = []

    for x_batch_torch, y_batch_torch in loader:
        x_batch = torch_batch_to_jax(x_batch_torch)
        z_batch = jax.vmap(model.encoder)(x_batch)

        all_z.append(np.asarray(z_batch))
        all_y.append(np.asarray(y_batch_torch))

    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    if all_z.shape[1] != 2:
        raise ValueError(
            f"KNN decision-region visualization requires latent_dim=2, "
            f"but got latent_dim={all_z.shape[1]}"
        )

    if standardize:
        knn = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(k),
                metric="euclidean",
            ),
        )
    else:
        knn = KNeighborsClassifier(
            n_neighbors=int(k),
            metric="euclidean",
        )

    knn.fit(all_z, all_y)

    z1_min, z1_max = all_z[:, 0].min(), all_z[:, 0].max()
    z2_min, z2_max = all_z[:, 1].min(), all_z[:, 1].max()

    padding_z1 = 0.1 * (z1_max - z1_min)
    padding_z2 = 0.1 * (z2_max - z2_min)

    z1_min -= padding_z1
    z1_max += padding_z1
    z2_min -= padding_z2
    z2_max += padding_z2

    z1_grid, z2_grid = np.meshgrid(
        np.linspace(z1_min, z1_max, grid_resolution),
        np.linspace(z2_min, z2_max, grid_resolution),
    )

    grid_points = np.c_[z1_grid.ravel(), z2_grid.ravel()]
    grid_predictions = knn.predict(grid_points)
    grid_predictions = grid_predictions.reshape(z1_grid.shape)

    plt.figure(figsize=(9, 8))

    plt.contourf(
        z1_grid,
        z2_grid,
        grid_predictions,
        levels=np.arange(11) - 0.5,
        cmap="tab10",
        alpha=0.25,
    )

    scatter = plt.scatter(
        all_z[:, 0],
        all_z[:, 1],
        c=all_y,
        cmap="tab10",
        s=8,
        alpha=0.8,
    )

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(f"{title}, k={k}")
    plt.colorbar(scatter, ticks=np.arange(10), label="Digit label")
    plt.tight_layout()
    plt.show()
    

def visualize_checkpoint(config, checkpoint_path=None, split="train"):
    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.model_dir) / config.paths.final_model_name

    model, _ = load_checkpoint(checkpoint_path)

    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )

    loader = make_loader(
        dataset,
        batch_size=int(config.hyperparameters.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
    )

    plot_latent_fortnite(
        model,
        loader,
        title="2D Latent Space, digits 1 and 2",
    )

    plot_latent_space(
        model,
        loader,
        title=f"2D Latent Space ({split} split)",
    )

    plot_knn_decision_regions(
        model,
        loader,
        k=10,
        standardize=True,
        title=f"KNN decision regions ({split} split)",
    )