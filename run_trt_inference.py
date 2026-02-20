#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def compute_stats(ms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(ms.mean()),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
    }


def _get_io_names(engine, trt):
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name
    if input_name is None or output_name is None:
        raise RuntimeError("Failed to resolve input/output tensor names")
    return input_name, output_name


def benchmark_engine(engine_path: str, obs_dim: int = 11, warmup: int = 200, iters: int = 2000) -> dict[str, dict[str, float]]:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    path = Path(engine_path)
    if not path.exists():
        raise FileNotFoundError(f"Engine not found: {path}")

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)

    with path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    input_name, output_name = _get_io_names(engine, trt)

    in_shape = tuple(engine.get_tensor_shape(input_name))
    if any(d < 0 for d in in_shape):
        in_shape = (1, obs_dim)
        context.set_input_shape(input_name, in_shape)

    out_shape = tuple(context.get_tensor_shape(output_name))
    if any(d < 0 for d in out_shape):
        context.set_input_shape(input_name, in_shape)
        out_shape = tuple(context.get_tensor_shape(output_name))

    in_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    out_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    host_in = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=in_dtype)
    host_out = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype)
    host_in[:] = np.random.randn(host_in.size).astype(in_dtype)

    dev_in = cuda.mem_alloc(host_in.nbytes)
    dev_out = cuda.mem_alloc(host_out.nbytes)

    context.set_tensor_address(input_name, int(dev_in))
    context.set_tensor_address(output_name, int(dev_out))

    stream = cuda.Stream()

    # Warmup (full path)
    for _ in range(warmup):
        cuda.memcpy_htod_async(dev_in, host_in, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    # Kernel-only benchmark using CUDA events
    kernel_ms = np.empty(iters, dtype=np.float64)
    start_ev = cuda.Event()
    end_ev = cuda.Event()
    for i in range(iters):
        start_ev.record(stream)
        context.execute_async_v3(stream_handle=stream.handle)
        end_ev.record(stream)
        end_ev.synchronize()
        kernel_ms[i] = start_ev.time_till(end_ev)

    # End-to-end benchmark (H2D + execute + D2H)
    e2e_ms = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(dev_in, host_in, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
        stream.synchronize()
        e2e_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "kernel_ms": compute_stats(kernel_ms),
        "e2e_ms": compute_stats(e2e_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="ppo_hopper_fp16.engine")
    parser.add_argument("--obs-dim", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    stats = benchmark_engine(args.engine, obs_dim=args.obs_dim, warmup=args.warmup, iters=args.iters)
    k = stats["kernel_ms"]
    e = stats["e2e_ms"]
    print(f"TensorRT kernel-only ms: mean={k['mean']:.6f} p50={k['p50']:.6f} p95={k['p95']:.6f} p99={k['p99']:.6f}")
    print(f"TensorRT e2e ms:        mean={e['mean']:.6f} p50={e['p50']:.6f} p95={e['p95']:.6f} p99={e['p99']:.6f}")


if __name__ == "__main__":
    main()
