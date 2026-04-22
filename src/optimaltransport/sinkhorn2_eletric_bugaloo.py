from __future__ import annotations

from pathlib import Path

import jax
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from .data import get_mnist_dataset, get_labels
from .save import load_checkpoint
from .lossfn import torch_batch_to_jax
from .sinkhorn import sinkhorn

def cost_matrix(config,  checkpoint_path=None, split="train"):
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
    diff = za[:, None, :] - zb[None, :, :]
    cost = jnp.sum(diff**2, axis=-1)
    C = jnp.sqrt(cost)

    a_n = len(za)
    b_n = len(zb)
    a = jnp.ones(a_n)
    b = jnp.ones(b_n) * (b_n/a_n)
    
    print(jax.jit(sinkhorn)(a,b,C))



