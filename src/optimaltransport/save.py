from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax

from .model import make_model



## This function saves the trained model 
def save_checkpoint(path, model, hparams):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write((json.dumps(hparams) + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(f, model)

## This function is used for loading a trained model.
def load_checkpoint(path):
    path = Path(path)

    with path.open("rb") as f:
        hparams = json.loads(f.readline().decode("utf-8"))
        skeleton = make_model(
            input_shape=tuple(hparams["input_shape"]),
            hidden_dim=int(hparams["hidden_dim"]),
            latent_dim=int(hparams["latent_dim"]),
            key=jax.random.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(f, skeleton)

    return model, hparams