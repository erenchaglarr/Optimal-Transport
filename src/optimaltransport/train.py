from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import optax
from sklearn.model_selection import KFold

from .save import save_checkpoint
from .data import (
    get_input_shape,
    get_mnist_dataset,
    make_fold_loaders,
    make_loader,
)
from .lossfn import reconstruction_mse_loss, torch_batch_to_jax
from .model import make_model


def build_hparams(config, input_shape):
    return {
        "input_shape": list(input_shape),
        "hidden_dim": int(config.hyperparameters.hidden_dim),
        "latent_dim": int(config.hyperparameters.latent_dim),
    }


def make_optimizer(config):
    return optax.adam(float(config.hyperparameters.learning_rate))


def make_model_and_state(config, input_shape, key, optimizer):
    model = make_model(
        input_shape=input_shape,
        hidden_dim=int(config.hyperparameters.hidden_dim),
        latent_dim=int(config.hyperparameters.latent_dim),
        key=key,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return model, opt_state


def make_train_step(optimizer):
    @eqx.filter_jit
    def train_step(model, opt_state, x_batch):
        loss, grads = eqx.filter_value_and_grad(reconstruction_mse_loss)(model, x_batch)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step


@eqx.filter_jit
def eval_step(model, x_batch):
    return reconstruction_mse_loss(model, x_batch)


def cross_validate(config):
    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=True,
        download=bool(config.data.download),
    )
    input_shape = get_input_shape(dataset)

    optimizer = make_optimizer(config)
    train_step = make_train_step(optimizer)

    splitter = KFold(
        n_splits=int(config.folds.num_folds),
        shuffle=True,
        random_state=int(config.training.seed),
    )

    fold_summaries = []
    all_fold_histories = []

    master_key = jax.random.PRNGKey(int(config.training.seed))

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.arange(len(dataset))),
        start=1,
    ):
        print(f"\n========== Fold {fold}/{int(config.folds.num_folds)} ==========")

        train_loader, val_loader = make_fold_loaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=int(config.hyperparameters.batch_size),
            num_workers=int(config.training.num_workers),
        )

        master_key, fold_key = jax.random.split(master_key)
        model, opt_state = make_model_and_state(
            config=config,
            input_shape=input_shape,
            key=fold_key,
            optimizer=optimizer,
        )

        train_loss_history = []
        val_loss_history = []

        for epoch in range(int(config.hyperparameters.num_epochs)):
            train_losses = []
            for x_train, _ in train_loader:
                x_batch = torch_batch_to_jax(x_train)
                model, opt_state, loss = train_step(model, opt_state, x_batch)
                train_losses.append(float(loss))

            val_losses = []
            for x_val, _ in val_loader:
                x_batch = torch_batch_to_jax(x_val)
                val_loss = eval_step(model, x_batch)
                val_losses.append(float(val_loss))

            mean_train_loss = float(np.mean(train_losses))
            mean_val_loss = float(np.mean(val_losses))

            train_loss_history.append(mean_train_loss)
            val_loss_history.append(mean_val_loss)

            print(
                f"Fold {fold:02d} | "
                f"Epoch {epoch + 1:02d}/{int(config.hyperparameters.num_epochs)} | "
                f"Train: {mean_train_loss:.6f} | "
                f"Val: {mean_val_loss:.6f}"
            )

        final_train_loss = train_loss_history[-1]
        final_val_loss = val_loss_history[-1]
        best_val_loss = float(np.min(val_loss_history))

        generalization_gap = final_val_loss - final_train_loss
        relative_generalization_gap = generalization_gap / (final_train_loss + 1e-12)

        fold_summaries.append(
            {
                "fold": fold,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "best_val_loss": best_val_loss,
                "generalization_gap": generalization_gap,
                "relative_generalization_gap": relative_generalization_gap,
            }
        )

        all_fold_histories.append(
            {
                "fold": fold,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }
        )

    final_train_losses = [d["final_train_loss"] for d in fold_summaries]
    final_val_losses = [d["final_val_loss"] for d in fold_summaries]
    best_val_losses = [d["best_val_loss"] for d in fold_summaries]
    generalization_gaps = [d["generalization_gap"] for d in fold_summaries]
    relative_generalization_gaps = [d["relative_generalization_gap"] for d in fold_summaries]

    print("\n========== Cross-validation summary ==========")
    for d in fold_summaries:
        print(
            f"Fold {d['fold']:02d}: "
            f"final train = {d['final_train_loss']:.6f}, "
            f"final val = {d['final_val_loss']:.6f}, "
            f"best val = {d['best_val_loss']:.6f}, "
            f"gap = {d['generalization_gap']:.6f}, "
            f"rel_gap = {100 * d['relative_generalization_gap']:.2f}%"
        )

    mean_train_loss = np.mean(final_train_losses)
    mean_val_loss = np.mean(final_val_losses)
    mean_gap = np.mean(generalization_gaps)
    std_gap = np.std(generalization_gaps, ddof=1)
    sem_gap = std_gap / np.sqrt(len(generalization_gaps))
    mean_relative_gap = np.mean(relative_generalization_gaps)

    t_critical = 2.262  # for 10 folds
    ci_low = mean_gap - t_critical * sem_gap
    ci_high = mean_gap + t_critical * sem_gap

    print(f"\nMean final train loss over {config.folds.num_folds} folds: {mean_train_loss:.6f}")
    print(f"Mean final val loss over {config.folds.num_folds} folds: {mean_val_loss:.6f}")
    print(f"Std final val loss over {config.folds.num_folds} folds: {np.std(final_val_losses, ddof=1):.6f}")
    print(f"Mean best val loss over {config.folds.num_folds} folds: {np.mean(best_val_losses):.6f}")

    print(f"\nMean generalization gap: {mean_gap:.6f}")
    print(f"Std generalization gap: {std_gap:.6f}")
    print(f"Mean relative generalization gap: {100 * mean_relative_gap:.2f}%")
    print(f"95% CI for generalization gap: [{ci_low:.6f}, {ci_high:.6f}]")

    return {
        "fold_summaries": fold_summaries,
        "histories": all_fold_histories,
        "mean_train_loss": float(mean_train_loss),
        "mean_val_loss": float(mean_val_loss),
        "mean_generalization_gap": float(mean_gap),
        "std_generalization_gap": float(std_gap),
        "ci_generalization_gap": [float(ci_low), float(ci_high)],
    }


def train_full_model(config):
    dataset = get_mnist_dataset(
        data_root=config.data.root,
        train=True,
        download=bool(config.data.download),
    )
    input_shape = get_input_shape(dataset)

    loader = make_loader(
        dataset,
        batch_size=int(config.hyperparameters.batch_size),
        shuffle=True,
        num_workers=int(config.training.num_workers),
    )

    optimizer = make_optimizer(config)
    train_step = make_train_step(optimizer)

    key = jax.random.PRNGKey(int(config.training.seed))
    model, opt_state = make_model_and_state(
        config=config,
        input_shape=input_shape,
        key=key,
        optimizer=optimizer,
    )

    train_loss_history = []

    print("\n========== Final training on full training set ==========")
    for epoch in range(int(config.hyperparameters.num_epochs)):
        epoch_losses = []

        for x_batch_torch, _ in loader:
            x_batch = torch_batch_to_jax(x_batch_torch)
            model, opt_state, loss = train_step(model, opt_state, x_batch)
            epoch_losses.append(float(loss))

        mean_epoch_loss = float(np.mean(epoch_losses))
        train_loss_history.append(mean_epoch_loss)

        print(
            f"Full Train | "
            f"Epoch {epoch + 1:02d}/{int(config.hyperparameters.num_epochs)} | "
            f"Train: {mean_epoch_loss:.6f}"
        )

    checkpoint_dir = Path(config.paths.model_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / config.paths.final_model_name

    hparams = build_hparams(config, input_shape)
    save_checkpoint(final_checkpoint_path, model, hparams)

    print(f"\nFinal full-data checkpoint saved to: {final_checkpoint_path}")

    return {
        "model": model,
        "train_loss_history": train_loss_history,
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_train_loss": float(train_loss_history[-1]),
    }


def run_training_pipeline(config):
    cv_results = cross_validate(config)
    final_results = train_full_model(config)

    return {
        "cv_results": cv_results,
        "final_results": final_results,
        "best_checkpoint_path": final_results["final_checkpoint_path"],
    }