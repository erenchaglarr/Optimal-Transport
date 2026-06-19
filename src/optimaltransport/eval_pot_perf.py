#!/usr/bin/env python3

import ot
import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter_ns
from .sinkhorn import sinkhorn_log

def pot_sinkhorn(a, b, C, eps=0.1):
    our_sinkhorn = jax.jit(sinkhorn)
    a = np.array(a)
    b = np.array(b)
    C = np.array(C)
    a = a / np.sum(a)
    b = b / np.sum(b)
    start = perf_counter_ns()
    P_POT = ot.sinkhorn(a, b, C, reg=eps) 
    pot_end = perf_counter_ns()
    our_start = perf_counter_ns()
    P_our = our_sinkhorn(a, b, C, eps=eps)
    our_end = perf_counter_ns()
    return P_POT, POT_end - POT_start, P_our, our_end - our_start
