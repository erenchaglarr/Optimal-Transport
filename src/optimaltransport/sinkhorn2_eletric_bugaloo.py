from __future__ import annotations

from pathlib import Path

import jax
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from .data import get_mnist_dataset, get_labels
from .save import load_checkpoint
from .lossfn import torch_batch_to_jax
from .sinkhorn import sinkhorn

def cost_matrix(config,  checkpoint_path=None, split="train"):
    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )
    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.model_dir) / config.paths.final_model_name

    model, _ = load_checkpoint(checkpoint_path)
    He = jnp.array(dataset.data.numpy())
    z = jax.vmap(model.encoder)(He)
    y = get_labels(dataset)
    filter_a = y == 1
    filter_b = y == 2
    za = z[filter_a]
    zb = z[filter_b]
    diff = za[:, None, :] - zb[None, :, :]
    cost = jnp.sum(diff**2, axis=-1)
    C = jnp.sqrt(cost)

    a_n = len(za)
    b_n = len(zb)
    a = jnp.ones(a_n) 
    b = jnp.ones(b_n) * (b_n/a_n)
    
    _,_, P = jax.jit(sinkhorn)(a,b,C)

    row_mass = jnp.sum(P, axis=1, keepdims=True)
    row_mass = jnp.maximum(row_mass, 1e-8)  # avoid divide-by-zero

    za_moved = (P @ zb) / row_mass

    i = 11
    z_old = za[i]
    z_new = za_moved[i]

    t = 0.5
    z_halfway = (1 - t) * z_old + t * z_new

    print("z_old:", z_old)
    print("z_new:", z_new)
    print("z_halfway:", z_halfway)

    za_np = np.array(za)
    zb_np = np.array(zb)
    za_moved_np = np.array(za_moved)

    plt.figure(figsize=(7, 7))

    # Plot first two latent dimensions
    plt.scatter(za_np[:, 0], za_np[:, 1], s=8, alpha=0.4, label="digit 1 latent points")
    plt.scatter(zb_np[:, 0], zb_np[:, 1], s=8, alpha=0.4, label="digit 2 latent points")

    n_arrows = min(50, len(za_np))

    for i in range(n_arrows):
        plt.arrow(
            za_np[i, 0],
            za_np[i, 1],
            za_moved_np[i, 0] - za_np[i, 0],
            za_moved_np[i, 1] - za_np[i, 1],
            length_includes_head=True,
            head_width=0.03,
            alpha=0.7,
        )

    # Highlight one example point and its transported version
    plt.scatter(z_old[0], z_old[1], s=80, marker="x", label="chosen source point")
    plt.scatter(z_new[0], z_new[1], s=80, marker="*", label="transported point")
    plt.scatter(z_halfway[0], z_halfway[1], s=80, marker="o", label="halfway point")

    plt.xlabel("latent dimension 1")
    plt.ylabel("latent dimension 2")
    plt.legend()
    plt.title("Sinkhorn transport: digit 1 moved toward digit 2")
    plt.axis("equal")
    plt.show()


    x_old = model.decoder(z_old)
    x_halfway = model.decoder(z_halfway)
    x_new = model.decoder(z_new)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(np.array(x_old).squeeze(), cmap="gray")
    axes[0].set_title("Original digit 1")

    axes[1].imshow(np.array(x_halfway).squeeze(), cmap="gray")
    axes[1].set_title("Halfway moved")

    axes[2].imshow(np.array(x_new).squeeze(), cmap="gray")
    axes[2].set_title("Moved toward digit 2")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return P, za, zb, za_moved

