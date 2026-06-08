from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import get_mnist_dataset, make_loader
from .lossfn import torch_batch_to_jax
from .save import load_checkpoint


def encode_loader(model, loader):
    z_batches = []
    y_batches = []

    for x_torch, y_torch in loader:
        x_batch = torch_batch_to_jax(x_torch)

        z_batch = jax.vmap(model.encoder)(x_batch)

        z_batches.append(np.asarray(z_batch))
        y_batches.append(y_torch.detach().cpu().numpy())

    z_all = np.concatenate(z_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)

    return z_all, y_all


def evaluate_knn_on_eqx_checkpoints(
    config,
    checkpoint_paths=None,
    checkpoint_glob="*.eqx",
    k_values=(1, 3, 5, 10, 20),
    standardize=True,
    export_path=None,
):
    """
    Evaluate KNN cluster separability for one or more trained .eqx autoencoder checkpoints.

    For each checkpoint:
        .eqx model -> encoder -> latent train/test vectors -> KNN -> metrics table
    """

    model_dir = Path(config.paths.model_dir)

    if checkpoint_paths is None:
        checkpoint_paths = sorted(model_dir.glob(checkpoint_glob))
    else:
        checkpoint_paths = [Path(p) for p in checkpoint_paths]

    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir} with pattern {checkpoint_glob}"
        )

    batch_size = int(config.hyperparameters.batch_size)

    train_dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=True,
        download=bool(config.data.download),
    )

    test_dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=False,
        download=bool(config.data.download),
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.training.num_workers),
    )

    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.training.num_workers),
    )

    rows = []

    print("\n========== KNN evaluation over .eqx checkpoints ==========")

    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading checkpoint: {checkpoint_path}")

        model, hparams = load_checkpoint(checkpoint_path)

        print("Encoding train split...")
        z_train, y_train = encode_loader(model, train_loader)

        print("Encoding test split...")
        z_test, y_test = encode_loader(model, test_loader)

        latent_dim = z_train.shape[1]

        print(f"Latent dimension: {latent_dim}")
        print(f"Train latent shape: {z_train.shape}")
        print(f"Test latent shape:  {z_test.shape}")

        for k in k_values:
            if standardize:
                clf = make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(
                        n_neighbors=int(k),
                        metric="euclidean",
                    ),
                )
            else:
                clf = KNeighborsClassifier(
                    n_neighbors=int(k),
                    metric="euclidean",
                )

            clf.fit(z_train, y_train)
            y_pred = clf.predict(z_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")

            row = {
                "checkpoint": checkpoint_path.name,
                "latent_dim": int(latent_dim),
                "k": int(k),
                "standardized": bool(standardize),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
            }

            rows.append(row)

            print(
                f"k={k:>2} | "
                f"accuracy={accuracy:.4f} | "
                f"macro_f1={macro_f1:.4f} | "
                f"weighted_f1={weighted_f1:.4f}"
            )

    results_df = pd.DataFrame(rows)

    if export_path is None:
        export_path = model_dir / "knn_eqx_checkpoint_results.csv"
    else:
        export_path = Path(export_path)

    export_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(export_path, index=False)

    print("\n========== Final KNN table ==========")
    print(results_df.to_string(index=False))
    print(f"\nSaved KNN table to: {export_path}")

    return results_df