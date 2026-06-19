#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx

def sinkhorn(a, b, C, eps=0.1, min_error=0.1, max_iters=100):
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
    return u, v, P

def sinkhorn_log(a, b, C, eps=0.1, max_iters=100, tol=1e-12): 
    log_a = jnp.log(a + 1e-300) 
    log_b = jnp.log(b + 1e-300) 
    logK = -C / eps 
    log_u = jnp.zeros_like(a) 
    log_v = jnp.zeros_like(b) 
    def do_iteration(i, logulogv):
        (log_u, log_v) = logulogv
        log_u = log_a - jax.scipy.special.logsumexp(logK   + log_v[None, :], axis=1) 
        log_v = log_b - jax.scipy.special.logsumexp(logK.T + log_u[None, :], axis=1) 
        return (log_u, log_v)
    (log_u, log_v) = jax.lax.fori_loop(0, max_iters, do_iteration, (log_u, log_v))
    logP = log_u[:, None] + logK + log_v[None, :] 
    P = jnp.exp(logP) 
    # P /= P.sum() 
    return jnp.exp(log_u),jnp.exp(log_v), P

def gen_cost_matrix(za, zb):
    diff = za[:, None, :] - zb[None, :, :]
    cost = jnp.sum(diff**2, axis=-1)
    a_n = len(za)
    b_n = len(zb)
    a = jnp.ones(a_n) 
    b = jnp.ones(b_n) * (b_n/a_n)
    C = jnp.sqrt(cost)
    return a, b, C

def normalized_wasserstein(za, zb):
    a, b, C = gen_cost_matrix(za, zb)
    na = len(za)
    a, b = a/na, b/na
    u, v, P = jax.jit(sinkhorn_log)(a, b, C)
    return jnp.vdot(P, C) 

def wasserstein(za, zb):
    a, b, C = gen_cost_matrix(za, zb)
    u, v, P = jax.jit(sinkhorn_log)(a, b, C)
    return jnp.vdot(P, C) 
