from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp


class Encoder(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear

    def __init__(self, input_shape, hidden_dim, latent_dim, key):
        in_features = math.prod(input_shape)
        k1, k2, k3 = jax.random.split(key, 3)
        self.l1 = eqx.nn.Linear(in_features, hidden_dim, key=k1)
        self.l2 = eqx.nn.Linear(hidden_dim, hidden_dim // 2, key=k2)
        self.l3 = eqx.nn.Linear(hidden_dim // 2, latent_dim, key=k3)

    def __call__(self, x):
        x = jnp.ravel(x)
        x = jax.nn.sigmoid(self.l1(x))
        x = jax.nn.sigmoid(self.l2(x))
        x = self.l3(x)
        return x


class Decoder(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear
    output_shape: tuple

    def __init__(self, input_shape, hidden_dim, latent_dim, key):
        out_features = math.prod(input_shape)
        k1, k2, k3 = jax.random.split(key, 3)

        self.l1 = eqx.nn.Linear(latent_dim, hidden_dim // 2, key=k1)
        self.l2 = eqx.nn.Linear(hidden_dim // 2, hidden_dim, key=k2)
        self.l3 = eqx.nn.Linear(hidden_dim, out_features, key=k3)
        self.output_shape = tuple(input_shape)

    def __call__(self, z):
        x = jax.nn.sigmoid(self.l1(z))
        x = jax.nn.sigmoid(self.l2(x))
        x = jax.nn.sigmoid(self.l3(x))
        x = jnp.reshape(x, self.output_shape)
        return x


class AutoEncoder(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, input_shape, hidden_dim, latent_dim, key):
        k1, k2 = jax.random.split(key, 2)
        self.encoder = Encoder(input_shape, hidden_dim, latent_dim, k1)
        self.decoder = Decoder(input_shape, hidden_dim, latent_dim, k2)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def make_model(input_shape, hidden_dim, latent_dim, key):
    return AutoEncoder(
        input_shape=input_shape,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        key=key,
    )