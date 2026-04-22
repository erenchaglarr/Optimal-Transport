from __future__ import annotations

from pathlib import Path

import jax
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

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

#     plot_latent_space(model, loader, title=f"2D Latent Space ({split} split)")
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

    

