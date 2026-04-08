#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx

def sinkhorn(a, b, C, eps=0.1, min_error=0.1, max_iters=200):
    n = a.shape[0]
    if not (a.shape == (n,) and b.shape == (n,)):
        raise Exception("a and b must be vectors of equal length")
    u = jnp.ones((n,))
    v = jnp.ones((n,))
    K = (-C / eps)
    iters = 0
    def do_iteration(i, uv):
        (u, v) = uv
        u = a/((K @ v)+1e-6)
        v = b/((K.T @ u)+1e-6)
        return (u, v)

    (u, v) = jax.lax.fori_loop(0, max_iters, do_iteration, (u, v))

    P = jnp.diag(u) @ K @ jnp.diag(v)
    return P

ex_a = jnp.array([1, 0])
ex_b = jnp.array([0, 1])
ex_C = jnp.array([[0, 1], [1, 0]])

c = jax.jit(sinkhorn).lower(ex_a, ex_b, ex_C).compile()
print(c.as_text())
print(dir(c.runtime_executable()))
       # .execute(ex_a, ex_b, ex_C)))
