from __future__ import annotations

import jax
import jax.numpy as jnp

## This function changes torch dataloader batches to datatype jax can work with.
def torch_batch_to_jax(x):
    return jnp.asarray(x.detach().cpu().numpy(), dtype=jnp.float32)

## This is the MSE loss function.
#def reconstruction_mse_loss(model, x_batch):
    #x_hat_batch = jax.vmap(model)(x_batch)
    #return jnp.mean((x_hat_batch - x_batch) ** 2)

def reconstruction_mse_loss(model, x_batch, latent_l2_weight=1e-3):
    z_batch = jax.vmap(model.encoder)(x_batch)
    x_hat_batch = jax.vmap(model.decoder)(z_batch)

    rec_loss = jnp.mean((x_hat_batch - x_batch) ** 2)
    latent_l2 = jnp.mean(jnp.sum(z_batch ** 2, axis=-1))

    total_loss = rec_loss + latent_l2_weight * latent_l2

    return total_loss

def reconstruction_mse_only(model, x_batch):
    z_batch = jax.vmap(model.encoder)(x_batch)
    x_hat_batch = jax.vmap(model.decoder)(z_batch)

    return jnp.mean((x_hat_batch - x_batch) ** 2)