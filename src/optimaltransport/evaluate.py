from __future__ import annotations

from pathlib import Path

import equinox as eqx
import numpy as np

from .save import load_checkpoint
from .data import get_mnist_dataset, make_loader, get_labels
from .lossfn import reconstruction_mse_loss, torch_batch_to_jax


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
        checkpoint_path = Path(config.paths.checkpoint_dir) / config.paths.best_checkpoint_name

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

def run_sinkhorn(config, class_a, class_b):
    model, _ = load_checkpoint(checkpoint_path)
    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=False,
        download=bool(config.data.download),
    )
    labels = get_labels(dataset)
    dataset_tensor = dataset.data
    class_a_x = dataset_tensor[labels == class_a]
    class_b_x = dataset_tensor[labels == class_b]
    class_a_z = jax.vmap(model.encoder)(class_a_x)
    class_b_z = jax.vmap(model.encoder)(class_b_x)
    a_n, b_n  = len(class_a_z), len(class_b_z)
    a = jnp.ones(a_n)
    b = jnp.ones(b_n) * (b_n / a_n)

    def c(az, bz):
        return sqrt(sum((azi-bzi)**2 for (azi, bzi) in zip(az, bz)))

    C = jnp.asarray([[c(az[i], bz[j])
                           for i in range(a_n)]
                          for j in range(b_n)])

    P, u, v = sinkhorn(a, b, C)

    xhat = (dataset_tensor[0])

