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
from .sinkhorn import sinkhorn, gen_cost_matrix, wasserstein, sinkhorn_log
from .eval_pot_perf import pot_sinkhorn

@jax.jit
def pairwise_sq_dists(x, y):
    x_norm = jnp.sum(x ** 2, axis=1, keepdims=True)
    y_norm = jnp.sum(y ** 2, axis=1, keepdims=True).T

    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return jnp.maximum(d2, 0.0)

@jax.jit
def gaussian_kernel_matrix(x, y, sigma):
    d2 = pairwise_sq_dists(x, y)
    return jnp.exp(-d2 / (2.0 * sigma ** 2))

@jax.jit
def mmd2_rbf(x, y, sigma):
    Kxx = gaussian_kernel_matrix(x, x, sigma)
    Kyy = gaussian_kernel_matrix(y, y, sigma)
    Kxy = gaussian_kernel_matrix(x, y, sigma)

    return jnp.mean(Kxx) + jnp.mean(Kyy) - 2.0 * jnp.mean(Kxy)

def median_heuristic(x, y, eps=1e-8):
    z = jnp.concatenate([x, y], axis=0)
    d2 = pairwise_sq_dists(z, z)

    d2_np = np.array(d2)
    d2_nonzero = d2_np[d2_np > eps]

    sigma = np.sqrt(0.5 * np.median(d2_nonzero) + eps)

    return jnp.array(sigma)


def evaluate_transport_mmd(zb, za_moved, sigma=None):
    """
    Compares source-target MMD before and after transport.
    """
    if sigma is None:
        sigma = median_heuristic(za_moved, zb)
    mmd = mmd2_rbf(za_moved, zb, sigma)

    return {
        "sigma": sigma,
        "mmd": mmd,
    }
def pca_project_2d(*arrays):
    """
    Projects multiple latent arrays into the same 2D PCA space.

    If the latent space is 1D, the second plotted coordinate is set to zero.
    """
    arrays_np = [np.asarray(a) for a in arrays]

    X = np.concatenate(arrays_np, axis=0)

    # Handle 1D latent space
    if X.shape[1] == 1:
        X_centered = X - X.mean(axis=0, keepdims=True)
        X_2d = np.concatenate([X_centered, np.zeros_like(X_centered)], axis=1)

        projected = []
        start = 0
        for a in arrays_np:
            n = len(a)
            projected.append(X_2d[start:start + n])
            start += n

        explained = np.array([1.0, 0.0])
        return (*projected, explained)

    # Normal PCA case for dimension >= 2
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:2]
    X_pca = X_centered @ components.T

    variances = (S ** 2) / (len(X) - 1)
    explained = variances[:2] / variances.sum()

    projected = []
    start = 0
    for a in arrays_np:
        n = len(a)
        projected.append(X_pca[start:start + n])
        start += n

    return (*projected, explained)

def viz_interp(model, n_images, z, z_moved, key=jax.random.PRNGKey(0)):
    n = len(z)
    idx = jax.random.choice(key, n, shape=(n_images,), replace=False)

    fig, axes = plt.subplots(n_images, 3, figsize=(9, 3 * n_images))

    for row, i in enumerate(idx):
        old = model.decoder(z[i])

        halfway_z = z[i] + 0.5 * (z_moved[i] - z[i])
        halfway = model.decoder(halfway_z)

        new = model.decoder(z_moved[i])

        axes[row, 0].imshow(np.array(old).squeeze(), cmap="gray")
        axes[row, 0].set_title("Original digit")

        axes[row, 1].imshow(np.array(halfway).squeeze(), cmap="gray")
        axes[row, 1].set_title("Halfway moved")

        axes[row, 2].imshow(np.array(new).squeeze(), cmap="gray")
        axes[row, 2].set_title("Transported")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def distribution_plot_pca(
    za,
    za_moved,
    zb,
    source_class=None,
    target_class=None,
):
    za_pca, za_moved_pca, zb_pca, explained = pca_project_2d(
        za,
        za_moved,
        zb,
    )

    plt.figure(figsize=(7, 7))

    plt.scatter(
        za_pca[:, 0],
        za_pca[:, 1],
        s=8,
        alpha=0.5,
        label="source distribution",
    )

    plt.scatter(
        zb_pca[:, 0],
        zb_pca[:, 1],
        s=8,
        alpha=0.5,
        label="target distribution",
    )

    plt.scatter(
        za_moved_pca[:, 0],
        za_moved_pca[:, 1],
        s=8,
        alpha=0.5,
        label="transported source distribution",
    )

    plt.xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")

    if source_class is not None and target_class is not None:
        plt.title(
            f"PCA projection: digit {source_class} transported toward digit {target_class}"
        )
    else:
        plt.title("PCA projection of latent distributions")

    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    
    
def plot_transport_outputs_across_dims(
    config,
    checkpoint_paths,
    source_class,
    target_class,
    split="test",
    max_points=1000,
    visual_rank=0,
    show_halfway=True,
    save_path=None,
):

    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )

    He = jnp.array(dataset.data.numpy())
    y = get_labels(dataset)

    idx_a_all = np.where(np.array(y) == source_class)[0]
    idx_b_all = np.where(np.array(y) == target_class)[0]

    if max_points is not None:
        idx_a = idx_a_all[:max_points]
        idx_b = idx_b_all[:max_points]
    else:
        idx_a = idx_a_all
        idx_b = idx_b_all

    if visual_rank >= len(idx_a):
        raise ValueError(
            f"visual_rank={visual_rank} is too large. "
            f"Only {len(idx_a)} source points available."
        )

    # This is the same actual input image for all latent dimensions
    source_idx = idx_a[visual_rank]
    raw_input = np.array(dataset.data[source_idx])

    dims = sorted(checkpoint_paths.keys())

    if show_halfway:
        n_rows = 2
        row_labels = ["Halfway", "Transported"]
    else:
        n_rows = 1
        row_labels = ["Transported"]

    n_cols = len(dims) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.35 * n_cols, 1.35 * n_rows),
        dpi=300,
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )

    if n_rows == 1:
        axes = axes[None, :]

    # First column: show the same input digit
    for row in range(n_rows):
        axes[row, 0].imshow(raw_input, cmap="gray")
        axes[row, 0].axis("off")

    axes[0, 0].set_title("Input", fontsize=9)

    if n_rows == 2:
        axes[0, 0].set_ylabel("Halfway", fontsize=9)
        axes[1, 0].set_ylabel("Transported", fontsize=9)
    else:
        axes[0, 0].set_ylabel("Transported", fontsize=9)

    for col, dim in enumerate(dims, start=1):
        checkpoint_path = Path(checkpoint_paths[dim])
        model, _ = load_checkpoint(checkpoint_path)

        z = jax.vmap(model.encoder)(He)

        za = z[idx_a]
        zb = z[idx_b]

        a, b, C = gen_cost_matrix(za, zb)
        _, _, P = jax.jit(sinkhorn)(a, b, C)

        za_moved = project_barycentric(zb, P)

        z_old = za[visual_rank]
        z_new = za_moved[visual_rank]
        z_half = z_old + 0.5 * (z_new - z_old)

        transported_img = model.decoder(z_new)

        if show_halfway:
            halfway_img = model.decoder(z_half)

            imgs = [halfway_img, transported_img]

            for row in range(2):
                axes[row, col].imshow(
                    np.array(imgs[row]).squeeze(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[row, col].axis("off")
        else:
            axes[0, col].imshow(
                np.array(transported_img).squeeze(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[0, col].axis("off")

        axes[0, col].set_title(f"{dim}D", fontsize=9)

    plt.subplots_adjust(
        left=0.03,
        right=0.99,
        top=0.88,
        bottom=0.03,
        wspace=0.05,
        hspace=0.12,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    plt.show()
    
def project_barycentric(z, P):
    row_mass = jnp.sum(P, axis=1, keepdims=True)
    row_mass = jnp.maximum(row_mass, 1e-8)  # avoid divide-by-zero
    z_moved = (P @ z) / row_mass
    return z_moved

def gen_bad_plan(za, zb):# used as a baseline for the real sinkhorn-generated plan
    n_a = len(za)
    n_b = len(zb)
    entry = n_b / n_a
    return jnp.ones((n_a, n_b)) * entry

def heatmap_distance(z, y, dist):
    distances = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            filter_a = y == i
            filter_b = y == j
            za = z[filter_a]
            zb = z[filter_b]
            a, b, C = gen_cost_matrix(za, zb)
            _,_, P = jax.jit(sinkhorn)(a,b,C)
            za_moved = project_barycentric(zb, P)
            discrepancy = dist(zb, za_moved) 
            distances[i, j] = discrepancy
            print(i, j, distances[i, j])
    return distances

def mmd_target_target(z, y, class_label, sigma=0.28, key=jax.random.PRNGKey(0)):
    filt = y == class_label
    z_class = z[filt]

    n = len(z_class)
    if n < 2:
        raise ValueError(f"Class {class_label} has fewer than 2 samples.")

    perm = jax.random.permutation(key, n)
    z_class = z_class[perm]

    n_half = n // 2
    z1 = z_class[:n_half]
    z2 = z_class[n_half:2 * n_half]

    return evaluate_transport_mmd(z1, z2, sigma=sigma)["mmd"]

def heat(d, filename="a.png", cmap="viridis"):
    plt.figure(figsize=(8, 5))
    plt.imshow(d, cmap=cmap, aspect="auto")
    plt.colorbar(label="P")
    plt.title("P")
    plt.xlabel("Target image")
    plt.ylabel("Source image")
    plt.savefig(filename)
    
def embed_and_run_sinkhorn(
    config,
    checkpoint_path=None,
    split="train",
    source_class=5,
    target_class=9,
    max_points=None,
):
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

    idx_a = np.where(np.array(y) == source_class)[0]
    idx_b = np.where(np.array(y) == target_class)[0]

    if max_points is not None:
        idx_a = idx_a[:max_points]
        idx_b = idx_b[:max_points]

    za = z[idx_a]
    zb = z[idx_b]

    a, b, C = gen_cost_matrix(za, zb)
    _, _, P = jax.jit(sinkhorn)(a, b, C)

    za_moved = project_barycentric(zb, P)

    viz_interp(model, 3, za, za_moved)
    checkpoint_paths = {
    1: "models/latent1.eqx",
    2: "models/latent2.eqx",
    3: "models/latent3.eqx",
    5: "models/latent5.eqx",
    10: "models/latent10.eqx",
    20: "models/latent20.eqx",
    30: "models/latent30.eqx",
    50: "models/latent50.eqx",
}

    plot_transport_outputs_across_dims(
    config=config,
    checkpoint_paths=checkpoint_paths,
    source_class=0,
    target_class=7,
    split="test",
    visual_rank=0,
    show_halfway=False,
    save_path="transport_1_to_5_outputs_only.png",
)

    distribution_plot_pca(
        za=za,
        za_moved=za_moved,
        zb=zb,
        source_class=source_class,
        target_class=target_class,
    )

    return P, za, zb, za_moved
