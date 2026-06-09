#!/usr/bin/env python3

from .sinkhorn import sinkhorn, sinkhorn_log
from equinox import filter_jit
from itertools import product
from time import perf_counter_ns
from .data import get_mnist_dataset, make_loader, get_labels
from .lossfn import torch_batch_to_jax
import jax
import jax.numpy as jnp
from math import sqrt

def jax_aot(f, *args): # this function takes the actual data, like a jit function but does not use it,
                         # only the shape of the data
    return jax.jit(f).trace(*args).lower().compile()

def jax_devices():
    cpus = [device for device in jax.devices() if device.platform == "cpu"]
    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    return ((platform_name, platform_devices[0])
            for platform_name, platform_devices
            in [("cpu", cpus), ("gpu", gpus)]
            if len(platform_devices) > 0)

def gen_cost_matrix(model, dataset, class_a, class_b, n=None, m=None):
    labels = get_labels(dataset)
    dataset_tensor = dataset.data
    class_a_x = dataset_tensor[labels == class_a]
    class_b_x = dataset_tensor[labels == class_b]
    if n is None:
        n = class_a_x.shape[0]
    if m is None:
        m = class_b_x.shape[0]
    class_a_x = class_a_x[:n, ...]
    class_b_x = class_b_x[:m, ...]
    print("embedding test split ...")
    class_a_z = jax.vmap(model.encoder)(torch_batch_to_jax(class_a_x))
    class_b_z = jax.vmap(model.encoder)(torch_batch_to_jax(class_b_x))
    a_n, b_n  = len(class_a_z), len(class_b_z)
    a = jnp.ones(a_n)
    b = jnp.ones(b_n) * (a_n / b_n)

    def c(az, bz):
        return sqrt(sum((azi-bzi)**2 for (azi, bzi) in zip(az, bz)))

    print("generating cost matrix ... ")
    C = jnp.asarray([[c(class_a_z[i], class_b_z[j])
                           for i in range(a_n)]
                          for j in range(b_n)])

    return a, b, C

def eval_sinkhorn_func(comp_n, f_n, device_n, a, b, C):
    f_name, f = f_n
    comp_name, comp_f, run_f = comp_n
    device_name, device = device_n
    print(f"evaluating {f_name} sinkhorn {comp_name} compiled on {device} ...")
    start = perf_counter_ns()
    compiled_f = comp_f(f, a, b, C)
    comptime_end = perf_counter_ns()
    trace_filename = f"/tmp/jax-trace-{f_name}-{comp_name}-{device_name}"
    with jax.profiler.trace(trace_filename):
        P, u, v = [pt.block_until_ready() for pt in run_f(compiled_f, a, b, C)] # necessary to avoid lazy evaluation in jax and fully materialize P, u and v
    end = perf_counter_ns()
    return (comptime_end - start), (end - comptime_end), trace_filename

def eval_perf(model, dataset, class_a, class_b):
    a, b, C = gen_cost_matrix(model, dataset, class_a, class_b, 100, 100)
    compilation_functions = (("jit", lambda f, *_: f,                    lambda f, *args: filter_jit(f)(*args)),
                             ("aot", lambda f, *args: jax_aot(f, *args), lambda f, *args: f(*args)))
    devices = jax_devices() 
    sinkhorn_functions = (("linear", sinkhorn), ("log", sinkhorn_log))
    report =  [(comp_f[0], sinkhorn_f[0], device[0], eval_sinkhorn_func(comp_f, sinkhorn_f, device, a, b, C))
            for (comp_f, sinkhorn_f, device)
            in product(compilation_functions, sinkhorn_functions, devices)]
    return report 

def print_report(report):
    print(f"|{'compilation':20}|{'impl':20}|{'device':20}|{'comptime(ns)':20}|{'runtime(ns)':20}|{'total(ns)':20}|{'xprof file':20}")
    print(("+" + ("-" * 20)) * 7)
    for compilation, impl, device, perf in report:
        print(f"|{compilation:20}|{impl:20}|{device:20}|{perf[0]:20}|{perf[1]:20}|{perf[1] + perf[0]:20}|{perf[2]:20}")
