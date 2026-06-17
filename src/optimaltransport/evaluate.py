from __future__ import annotations

from pathlib import Path

import equinox as eqx
import numpy as np

from .data import get_mnist_dataset, make_loader, get_labels
from .lossfn import reconstruction_mse_only, torch_batch_to_jax
from .KNN_classifier import evaluate_knn_on_eqx_checkpoints
from .eval_perf import eval_perf, print_report
import jax
import jax.numpy as jnp

from .sinkhorn import gen_cost_matrix, sinkhorn
from .save import load_checkpoint, load_classifier_checkpoint
from .sinkhorn2_eletric_bugaloo import project_barycentric

@eqx.filter_jit
def eval_step(model, x_batch):
    return reconstruction_mse_only(model, x_batch)

def evaluate_model(model, loader):
    losses = []

    for x_batch_torch, _ in loader:
        x_batch = torch_batch_to_jax(x_batch_torch)
        loss = eval_step(model, x_batch)
        losses.append(float(loss))

    return {"reconstruction_mse": float(np.mean(losses))}

def evaluate_checkpoint(config, checkpoint_path=None, split="test"):
    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.model_dir) / config.paths.final_model_name
    else:
        checkpoint_path = Path(checkpoint_path)

    model, _ = load_checkpoint(checkpoint_path)

    use_train_split = split == "train"

    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=use_train_split,
        download=bool(config.data.download),
    )

    loader = make_loader(
        dataset,
        batch_size=int(config.hyperparameters.batch_size),
        shuffle=False,
        num_workers=int(config.training.num_workers),
    )

    metrics = evaluate_model(model, loader)

    print(f"\nEvaluation on {split} split")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")

    perf_report = eval_perf(model, dataset, 1, 2)
    print_report(perf_report)

    return { # not used
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        **metrics,
    }


def evaluate_knn_checkpoints(
    config,
    checkpoint_path=None,
    k_values=(1, 3, 5, 10, 20),
    standardize=True,
    export_name="knn_eqx_checkpoint_results.csv",
):
    if checkpoint_path is None:
        checkpoint_paths = None
    else:
        checkpoint_paths = [Path(checkpoint_path)]

    export_path = Path(config.paths.model_dir) / export_name

    knn_table = evaluate_knn_on_eqx_checkpoints(
        config=config,
        checkpoint_paths=checkpoint_paths,
        checkpoint_glob="*.eqx",
        k_values=k_values,
        standardize=standardize,
        export_path=export_path,
    )

    return {
        "knn_table": knn_table,
        "knn_table_path": str(export_path),
    }


def run_evaluation_pipeline(config):
    """
    Runs both reconstruction evaluation and KNN latent-space evaluation.
    """

    checkpoint_path = Path(config.paths.model_dir) / config.paths.final_model_name

    reconstruction_results = evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        split="test",
    )

    knn_results = evaluate_knn_on_eqx_checkpoints(
        config=config,
        checkpoint_path=checkpoint_path,
        k_values=(1, 3, 5, 10, 20),
        standardize=True,
        export_name="knn_latent_results.csv",
    )

    return {
        "reconstruction_results": reconstruction_results,
        "knn_results": knn_results,
    }

def _class_distribution(preds, n_classes=10):
    counts = np.bincount(np.array(preds), minlength=n_classes)
    total = counts.sum()

    return {
        int(i): {
            "count": int(counts[i]),
            "fraction": float(counts[i] / total),
        }
        for i in range(n_classes)
    }

def evaluate_transportplans_with_classifier(
    config,
    classifier_checkpoint_path,
    autoencoder_checkpoint_path=None,
    split="test",
    max_points=None,
    n_classes=10,
):
    if autoencoder_checkpoint_path is None:
        autoencoder_checkpoint_path = (
            Path(config.paths.model_dir) / config.paths.final_model_name
        )
    else:
        autoencoder_checkpoint_path = Path(autoencoder_checkpoint_path)

    classifier_checkpoint_path = Path(classifier_checkpoint_path)

    autoencoder, _ = load_checkpoint(autoencoder_checkpoint_path)
    classifier, _ = load_classifier_checkpoint(classifier_checkpoint_path)

    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=(split == "train"),
        download=bool(config.data.download),
    )

    He = jnp.array(dataset.data.numpy())
    z = jax.vmap(autoencoder.encoder)(He)
    y = np.array(get_labels(dataset))

    accuracy_matrix = np.full((n_classes, n_classes), np.nan)
    n_images_matrix = np.zeros((n_classes, n_classes), dtype=int)

    all_target_rates = []

    print("\n========== Average transport evaluation with classifier ==========")
    print(f"Autoencoder checkpoint: {autoencoder_checkpoint_path}")
    print(f"Classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Split: {split}")

    for source_class in range(n_classes):
        for target_class in range(n_classes):
            if source_class == target_class:
                continue

            idx_a = np.where(y == source_class)[0]
            idx_b = np.where(y == target_class)[0]

            if max_points is not None:
                idx_a = idx_a[:max_points]
                idx_b = idx_b[:max_points]

            za = z[idx_a]
            zb = z[idx_b]

            a, b, C = gen_cost_matrix(za, zb)
            _, _, P = jax.jit(sinkhorn)(a, b, C)

            za_moved = project_barycentric(zb, P)

            transported_images = jax.vmap(autoencoder.decoder)(za_moved)

            classifier_outputs = jax.vmap(classifier)(transported_images)
            preds = jnp.argmax(classifier_outputs, axis=1)

            target_hits = preds == int(target_class)
            target_rate = float(jnp.mean(target_hits))

            accuracy_matrix[source_class, target_class] = target_rate
            n_images_matrix[source_class, target_class] = len(preds)
            all_target_rates.append(target_rate)

            print(
                f"{source_class} -> {target_class}: "
                f"{target_rate:.4f} "
                f"({int(jnp.sum(target_hits))}/{len(preds)})"
            )

    mean_accuracy = float(np.mean(all_target_rates))

    per_source_accuracy = np.nanmean(accuracy_matrix, axis=1)
    per_target_accuracy = np.nanmean(accuracy_matrix, axis=0)

    print("\nAccuracy matrix:")
    print(accuracy_matrix)

    print("\nAverage target accuracy over all source-target pairs:")
    print(f"{mean_accuracy:.4f}")

    print("\nAverage accuracy by source class:")
    for cls, acc in enumerate(per_source_accuracy):
        print(f"source class {cls}: {acc:.4f}")

    print("\nAverage accuracy by target class:")
    for cls, acc in enumerate(per_target_accuracy):
        print(f"target class {cls}: {acc:.4f}")

    return {
        "autoencoder_checkpoint_path": str(autoencoder_checkpoint_path),
        "classifier_checkpoint_path": str(classifier_checkpoint_path),
        "split": split,
        "mean_accuracy": mean_accuracy,
        "accuracy_matrix": accuracy_matrix,
        "n_images_matrix": n_images_matrix,
        "per_source_accuracy": per_source_accuracy,
        "per_target_accuracy": per_target_accuracy,
    }
    
