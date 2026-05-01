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
        u = log_a - (K   @ v)
        v = log_b - (K.T @ u)
        return (u, v)

    (u, v) = jax.lax.fori_loop(0, max_iters, do_iteration, (u, v))

    P = jnp.diag(u) @ K @ jnp.diag(v)
    return u,v , P 


ex_a = jnp.array([1, 1, 0])
ex_b = jnp.array([0, 1, 1])
ex_C = jnp.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

#print(jax.jit(sinkhorn)(ex_a, ex_b, ex_C))
#print(jax.jit(sinkhorn)(ex_a, ex_b, ex_C))

# c = jax.jit(sinkhorn).lower(ex_a, ex_b, ex_C).compile()
# print(c.as_text())
# print(dir(c.runtime_executable()))
       # .execute(ex_a, ex_b, ex_C)))

# c = jax.jit(sinkhorn).lower(ex_a, ex_b, ex_C).compile()
# print(c.as_text())
# print(dir(c.runtime_executable()))
       # .execute(ex_a, ex_b, ex_C)))
