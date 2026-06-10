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
from .sinkhorn import sinkhorn, gen_cost_matrix, wasserstein

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
    was = jnp.array([[0.           , 2037.23657227, 2243.05371094, 1986.56921387, 1743.63513184, 2376.46923828, 1875.74255371, 1978.67871094, 1602.74572754, 1354.85595703], 
                    [2089.22241211, 0.           , 2702.7253418 , 2249.33007812, 2137.18920898, 3084.06518555, 2253.28076172, 2223.27246094, 2065.79614258, 1768.02563477],
                    [1656.54016113, 1943.671875  , 0.           , 1978.17333984, 1765.7434082 , 2390.82910156, 1929.07653809, 1987.14611816, 1617.79931641, 1370.67236328],
                    [1766.81323242, 2066.74633789, 2324.69848633, 0.           , 1892.93518066, 2497.55078125, 2051.59570312, 2065.34912109, 1718.94604492, 1443.56018066],
                    [1602.35058594, 1900.74853516, 2213.64624023, 1947.33996582,    0.        , 2370.56347656, 1831.69030762, 1871.44189453, 1557.21630859, 1319.85375977],
                    [1377.12463379, 1681.03186035, 2062.10473633, 1619.32385254, 1442.90930176, 0.           , 1571.10461426, 1627.63317871, 1343.7755127 , 1186.22155762],
                    [1627.43737793, 2003.16650391, 2243.81054688, 1986.77380371, 1728.61560059, 2385.90917969, 0.           , 1983.14245605, 1598.2779541 , 1353.3104248 ],
                    [1858.16699219, 2081.3190918 , 2394.41162109, 2101.80883789, 1947.69580078, 2629.109375  , 2086.03271484,    0.        , 1815.8482666 , 1502.015625  ],
                    [1599.60375977, 1959.40759277, 2209.02124023, 1925.38574219, 1695.89831543, 2346.62475586, 1853.43811035, 1921.99230957,    0.        , 1325.28283691],
                    [1668.87145996, 2042.28637695, 2263.36401367, 2028.1862793 , 1747.00256348, 2448.61010742, 1931.74108887, 1968.60546875, 1613.84069824,    0.        ]])
    lens = []
    for i in range(10):
        lens.append(len(z[y == i]))
    lens = np.array(lens)
    print(lens[:, None])
    print(was / lens[:, None])
    # print("wasserstein distances:")
    # print(wasserstein_heatmap)
    # mmd_heatmap = heatmap_distance(z, y, (lambda zb, za_moved: evaluate_transport_mmd(zb, za_moved, sigma=0.28)["mmd"]))
    # print("mmd distances:")
    # print(mmd_heatmap)
    
    # print(evaluate_transport_mmd(zb, za_moved))
    # print(evaluate_transport_mmd(zb, project_barycentric(zb, gen_bad_plan(za, zb))))
    return P, za, zb, za_moved,
