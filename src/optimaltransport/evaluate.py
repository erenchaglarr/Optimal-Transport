from __future__ import annotations

from pathlib import Path

import equinox as eqx
import numpy as np

from .save import load_checkpoint
from .data import get_mnist_dataset, make_loader
from .lossfn import reconstruction_mse_loss, torch_batch_to_jax
from .KNN_classifier import evaluate_knn_on_eqx_checkpoints


@eqx.filter_jit
def eval_step(model, x_batch):
    return reconstruction_mse_loss(model, x_batch)


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

    return {
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