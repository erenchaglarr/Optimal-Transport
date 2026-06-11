#!/usr/bin/env python3

import ot
import jax.numpy as jnp
import numpy as np
from time import perf_counter_ns

def pot_sinkhorn(a, b, C, eps=0.1):
    a = np.array(a)
    b = np.array(b)
    C = np.array(C)
    a = a / np.sum(a)
    b = b / np.sum(b)
    start = perf_counter_ns()
    P = ot.sinkhorn(a, b, C, reg=eps) 
    end = perf_counter_ns()
    return P, end - start
