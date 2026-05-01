#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx

def sinkhorn(a, b, C, eps=0.1, min_error=0.1, max_iters=20000):
    n = a.shape[0]
    m = b.shape[0]
    log_a = jnp.log(a)
    log_b = jnp.log(b)
    u = jnp.ones((n,))
    v = jnp.ones((m,))
    K = -C / eps

    def do_iteration(i, uv):
        (u, v) = uv
        u = log_a - (K   @ v)
        v = log_b - (K.T @ u)
        return (u, v)

    (u, v) = jax.lax.fori_loop(0, max_iters, do_iteration, (u, v))

    plain_u = jnp.exp(u)
    plain_v = jnp.exp(v)
    P = jnp.diag(plain_u) @ jnp.exp(k) @ jnp.diag(plain_v)
    return (plain_u, plain_v, P)

ex_a = jnp.array([1, 1, 0])
ex_b = jnp.array([0, 1, 1])
ex_C = jnp.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

print(jax.jit(sinkhorn)(ex_a, ex_b, ex_C))

# c = jax.jit(sinkhorn).lower(ex_a, ex_b, ex_C).compile()
# print(c.as_text())
# print(dir(c.runtime_executable()))
       # .execute(ex_a, ex_b, ex_C)))


ddsa
