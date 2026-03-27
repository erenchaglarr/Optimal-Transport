from __future__ import annotations

import jax
import jax.numpy as jnp

## This function changes torch dataloader batches to datatype jax can work with.
def torch_batch_to_jax(x):
    return jnp.asarray(x.detach().cpu().numpy(), dtype=jnp.float32)

## This is the MSE loss function.
def reconstruction_mse_loss(model, x_batch):
    x_hat_batch = jax.vmap(model)(x_batch)
    return jnp.mean((x_hat_batch - x_batch) ** 2)