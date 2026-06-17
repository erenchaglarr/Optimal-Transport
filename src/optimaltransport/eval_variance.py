from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .data import get_mnist_dataset, get_labels
from .save import load_checkpoint
from .sinkhorn import sinkhorn, gen_cost_matrix
from .sinkhorn2_eletric_bugaloo import project_barycentric


def latent_variance_stats(z_target, z_moved, eps: float = 1e-8) -> dict[str, Any]:
    """
    Compare the spread of the true target latent distribution and the
    barycentrically transported source distribution.

    Main diagnostic:
        total_variance_ratio = transported_total_variance / target_total_variance

    Interpretation:
        ratio ≈ 1: transported distribution has similar spread to target
        ratio < 1: transported distribution is more collapsed than target
        ratio > 1: transported distribution is more spread out than target
    """
    z_target_np = np.asarray(z_target)
    z_moved_np = np.asarray(z_moved)

    if z_target_np.ndim != 2 or z_moved_np.ndim != 2:
        raise ValueError(
            "Expected z_target and z_moved to be 2D arrays of shape "
            "(n_points, latent_dim)."
        )

    if z_target_np.shape[1] != z_moved_np.shape[1]:
        raise ValueError(
            f"Latent dimensions do not match: "
            f"z_target has dim {z_target_np.shape[1]}, "
            f"z_moved has dim {z_moved_np.shape[1]}."
        )

    # Coordinate-wise variances.
    target_var_per_dim = np.var(z_target_np, axis=0)
    moved_var_per_dim = np.var(z_moved_np, axis=0)

    # Total variance is the trace of the covariance matrix.
    # This is equivalent to the sum of coordinate-wise variances.
    target_total_var = float(np.sum(target_var_per_dim))
    moved_total_var = float(np.sum(moved_var_per_dim))

    target_mean_var = float(np.mean(target_var_per_dim))
    moved_mean_var = float(np.mean(moved_var_per_dim))

    total_var_ratio = moved_total_var / (target_total_var + eps)
    mean_var_ratio = moved_mean_var / (target_mean_var + eps)

    # Mean squared distance to centroid gives another intuitive spread measure.
    target_center = np.mean(z_target_np, axis=0, keepdims=True)
    moved_center = np.mean(z_moved_np, axis=0, keepdims=True)

    target_mean_sq_radius = float(
        np.mean(np.sum((z_target_np - target_center) ** 2, axis=1))
    )
    moved_mean_sq_radius = float(
        np.mean(np.sum((z_moved_np - moved_center) ** 2, axis=1))
    )

    mean_sq_radius_ratio = moved_mean_sq_radius / (target_mean_sq_radius + eps)

    return {
        "latent_dim": int(z_target_np.shape[1]),
        "target_total_variance": target_total_var,
        "transported_total_variance": moved_total_var,
        "total_variance_ratio": float(total_var_ratio),
        "target_mean_variance": target_mean_var,
        "transported_mean_variance": moved_mean_var,
        "mean_variance_ratio": float(mean_var_ratio),
        "target_mean_sq_radius": target_mean_sq_radius,
        "transported_mean_sq_radius": moved_mean_sq_radius,
        "mean_sq_radius_ratio": float(mean_sq_radius_ratio),
        "target_var_per_dim": target_var_per_dim.tolist(),
        "transported_var_per_dim": moved_var_per_dim.tolist(),
    }


def _write_results_csv(results: list[dict[str, Any]], export_csv_path: str | Path) -> None:
    if len(results) == 0:
        raise ValueError("No variance results to write.")

    export_csv_path = Path(export_csv_path)
    export_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with export_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            csv_row = dict(row)
            csv_row["target_var_per_dim"] = json.dumps(csv_row["target_var_per_dim"])
            csv_row["transported_var_per_dim"] = json.dumps(
                csv_row["transported_var_per_dim"]
            )
            writer.writerow(csv_row)


def evaluate_transport_variance_across_dims(
    config,
    checkpoint_paths: dict[int, str | Path],
    split: str = "test",
    max_points: int | None = None,
    n_classes: int = 10,
    export_csv_path: str | Path | None = "transport_variance_across_dims.csv",
) -> list[dict[str, Any]]:
    """
    Evaluate target-vs-transported variance for every source-target pair
    and every latent dimension.

    checkpoint_paths example:
        {
            1: "models/latent1.eqx",
            2: "models/latent2.eqx",
            3: "models/latent3.eqx",
            5: "models/latent5.eqx",
            10: "models/latent10.eqx",
            20: "models/latent20.eqx",
            30: "models/latent30.eqx",
            50: "models/latent50.eqx",
        }

    For each pair i -> j, this computes:
        - variance of target digit j in latent space
        - variance of transported digit i after barycentric projection
        - transported / target variance ratio
    """
    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )

    He = jnp.array(dataset.data.numpy())
    y = np.array(get_labels(dataset))

    results: list[dict[str, Any]] = []

    print("\n========== Transport variance diagnostics ==========")
    print(f"Split: {split}")
    print(f"Max points per class: {max_points}")
    print(f"Latent dimensions: {list(sorted(checkpoint_paths.keys()))}")

    for latent_dim, checkpoint_path in sorted(checkpoint_paths.items()):
        checkpoint_path = Path(checkpoint_path)
        model, _ = load_checkpoint(checkpoint_path)

        print(f"\n========== Latent dimension {latent_dim} ==========")
        print(f"Checkpoint: {checkpoint_path}")

        z = jax.vmap(model.encoder)(He)

        variance_ratio_matrix = np.full((n_classes, n_classes), np.nan)

        for source_class in range(n_classes):
            for target_class in range(n_classes):
                if source_class == target_class:
                    continue

                idx_a = np.where(y == source_class)[0]
                idx_b = np.where(y == target_class)[0]

                if max_points is not None:
                    idx_a = idx_a[:max_points]
                    idx_b = idx_b[:max_points]

                if len(idx_a) == 0 or len(idx_b) == 0:
                    print(
                        f"Skipping {source_class} -> {target_class}: "
                        "empty source or target class."
                    )
                    continue

                za = z[idx_a]
                zb = z[idx_b]

                a, b, C = gen_cost_matrix(za, zb)
                _, _, P = jax.jit(sinkhorn)(a, b, C)

                za_moved = project_barycentric(zb, P)

                stats = latent_variance_stats(
                    z_target=zb,
                    z_moved=za_moved,
                )

                variance_ratio_matrix[source_class, target_class] = stats[
                    "total_variance_ratio"
                ]

                row = {
                    "checkpoint": str(checkpoint_path),
                    "source_class": int(source_class),
                    "target_class": int(target_class),
                    "n_source": int(len(za)),
                    "n_target": int(len(zb)),
                    **stats,
                }

                results.append(row)

                print(
                    f"{source_class} -> {target_class}: "
                    f"target var = {stats['target_total_variance']:.6f}, "
                    f"transported var = {stats['transported_total_variance']:.6f}, "
                    f"ratio = {stats['total_variance_ratio']:.4f}"
                )

        mean_ratio = float(np.nanmean(variance_ratio_matrix))
        std_ratio = float(np.nanstd(variance_ratio_matrix))

        print(f"\nMean transported/target variance ratio for {latent_dim}D:")
        print(f"{mean_ratio:.4f} ± {std_ratio:.4f}")

    if export_csv_path is not None:
        _write_results_csv(results, export_csv_path)
        print(f"\nSaved variance results to: {export_csv_path}")

    return results


def summarize_variance_results_by_dim(
    results: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """
    Print and return summary statistics grouped by latent dimension.
    """
    dims = sorted(set(int(row["latent_dim"]) for row in results))
    summary: dict[int, dict[str, float]] = {}

    print("\n========== Variance summary by latent dimension ==========")

    for dim in dims:
        rows = [r for r in results if int(r["latent_dim"]) == dim]

        ratios = np.array([r["total_variance_ratio"] for r in rows], dtype=float)
        target_vars = np.array([r["target_total_variance"] for r in rows], dtype=float)
        moved_vars = np.array(
            [r["transported_total_variance"] for r in rows],
            dtype=float,
        )
        radius_ratios = np.array(
            [r["mean_sq_radius_ratio"] for r in rows],
            dtype=float,
        )

        dim_summary = {
            "mean_total_variance_ratio": float(np.mean(ratios)),
            "std_total_variance_ratio": float(np.std(ratios)),
            "min_total_variance_ratio": float(np.min(ratios)),
            "max_total_variance_ratio": float(np.max(ratios)),
            "mean_target_total_variance": float(np.mean(target_vars)),
            "mean_transported_total_variance": float(np.mean(moved_vars)),
            "mean_sq_radius_ratio": float(np.mean(radius_ratios)),
        }

        summary[dim] = dim_summary

        print(
            f"{dim}D: "
            f"mean ratio = {dim_summary['mean_total_variance_ratio']:.4f}, "
            f"std = {dim_summary['std_total_variance_ratio']:.4f}, "
            f"min = {dim_summary['min_total_variance_ratio']:.4f}, "
            f"max = {dim_summary['max_total_variance_ratio']:.4f}, "
            f"mean target var = {dim_summary['mean_target_total_variance']:.6f}, "
            f"mean transported var = {dim_summary['mean_transported_total_variance']:.6f}"
        )

    return summary


def variance_ratio_matrix_from_results(
    results: list[dict[str, Any]],
    latent_dim: int,
    n_classes: int = 10,
    value_key: str = "total_variance_ratio",
) -> np.ndarray:
    """
    Convert list-of-dicts results into a source-target matrix.

    matrix[i, j] = value for transport i -> j
    diagonal is NaN.
    """
    matrix = np.full((n_classes, n_classes), np.nan)

    for row in results:
        if int(row["latent_dim"]) != int(latent_dim):
            continue

        i = int(row["source_class"])
        j = int(row["target_class"])
        matrix[i, j] = float(row[value_key])

    return matrix


def plot_variance_ratio_heatmap(
    results: list[dict[str, Any]],
    latent_dim: int,
    n_classes: int = 10,
    value_key: str = "total_variance_ratio",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Plot a heatmap of transported/target variance ratios for one latent dimension.
    """
    matrix = variance_ratio_matrix_from_results(
        results=results,
        latent_dim=latent_dim,
        n_classes=n_classes,
        value_key=value_key,
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Transported / target variance ratio")
    plt.xlabel("Target digit")
    plt.ylabel("Source digit")
    plt.title(f"Variance ratio heatmap, {latent_dim}D latent space")
    plt.xticks(range(n_classes))
    plt.yticks(range(n_classes))
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return matrix


def plot_mean_variance_ratio_by_dim(
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the average transported/target variance ratio as a function of latent dimension.
    """
    summary = summarize_variance_results_by_dim(results)

    dims = sorted(summary.keys())
    means = [summary[d]["mean_total_variance_ratio"] for d in dims]
    stds = [summary[d]["std_total_variance_ratio"] for d in dims]

    plt.figure(figsize=(6, 4))
    plt.errorbar(dims, means, yerr=stds, marker="o", capsize=4)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Latent dimension")
    plt.ylabel("Mean transported / target variance ratio")
    plt.title("Variance ratio across latent dimensions")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
