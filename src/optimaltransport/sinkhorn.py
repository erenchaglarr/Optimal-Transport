#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx



def sinkhorn(a, b, C, eps=0.1, min_error=0.1, max_iters=200):
    n = a.shape[0]
    m = b.shape[0]
    u = jnp.ones((n,))
    v = jnp.ones((m,))
    K = jnp.exp(-C / eps)
    iters = 0
    def do_iteration(i, uv):
        (u, v) = uv
        u = a/((K @ v)+1e-6)
        v = b/((K.T @ u)+1e-6)
        return (u, v)

    (u, v) = jax.lax.fori_loop(0, max_iters, do_iteration, (u, v))

    P = jnp.diag(u) @ K @ jnp.diag(v)
    return u,v

