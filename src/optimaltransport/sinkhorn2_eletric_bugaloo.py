from __future__ import annotations

from pathlib import Path
from functools import partial

import jax
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from .data import get_mnist_dataset, get_labels
from .save import load_checkpoint
from .lossfn import torch_batch_to_jax
from .sinkhorn import sinkhorn

def gaussian_kernel(sigma, x, y):
    """This function computes the gaussian kernel for points x and y

    Args:
        x (vector)
        y (vector)
        sigma (int)

    Returns:
        Matrix
    """
    d2 = np.linalg.norm(x,y)
    return jax.numpy.exp(-d2/2.0*sigma**2)

def mmd(X, Y, kernel):
    """
    This function computes Maximum Mean Discrepancy 
    for given probability distributions X and Y a kernel and sigma
    """
    
    Kxx = kernel(X,X)
    Kyy = kernel(Y,Y)
    Kxy = kernel(X,Y)
    
    return jnp.mean(Kxx) + jnp.mean(Kyy) - 2.0*jnp.mean(Kxy)

def gen_cost_matrix(za, zb):
    diff = za[:, None, :] - zb[None, :, :]
    cost = jnp.sum(diff**2, axis=-1)
    a_n = len(za)
    b_n = len(zb)
    a = jnp.ones(a_n) 
    b = jnp.ones(b_n) * (b_n/a_n)
    C = jnp.sqrt(cost)
    return a, b, C

def viz_interp(model, n_images, z, z_moved):
    fig, axes = plt.subplots(n_images, 3, figsize=(9, 3))
    for i in range(n_images):
        old = model.decoder(z[i])
        halfway = model.decoder(z_moved[i] - (0.5) * z[i])
        new = model.decoder(z_moved[i])
        axes[i, 0].imshow(np.array(old).squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original digit")
        axes[i, 1].imshow(np.array(halfway).squeeze(), cmap="gray")
        axes[i, 1].set_title("Halfway moved")
        axes[i, 2].imshow(np.array(new).squeeze(), cmap="gray")
        axes[i, 2].set_title("Transported")
    plt.tight_layout()
    plt.show()

def arrow_plot(n_arrows, za, za_moved, zb):
    za_np = np.array(za)
    zb_np = np.array(zb)
    za_moved_np = np.array(za_moved)
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
    plt.figure(figsize=(7, 7))
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
    
def project_barycentric(z, P):
    row_mass = jnp.sum(P, axis=1, keepdims=True)
    row_mass = jnp.maximum(row_mass, 1e-8)  # avoid divide-by-zero
    z_moved = (P @ z) / row_mass
    return z_moved

def embed_and_run_sinkhorn(config,  checkpoint_path=None, split="train"):
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
    a, b, C = gen_cost_matrix(za, zb)
    _,_, P = jax.jit(sinkhorn)(a,b,C)
    za_moved = project_barycentric(zb, P)

    viz_interp(model, 3, za, za_moved)
    
    mmd(za_moved, zb, partial(gaussian_kernel, 1))
        
    return P, za, zb, za_moved,

