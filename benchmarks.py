#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO


def compute_stats(ms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(ms.mean()),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
    }


def make_sess_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Low-latency single-stream behavior for tiny MLP inference.
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def bench_sb3_cpu(model_path: str, warmup: int, iters: int) -> dict[str, float]:
    model = PPO.load(model_path, device="cpu")
    obs = np.zeros((11,), dtype=np.float32)

    for _ in range(warmup):
        model.predict(obs, deterministic=True)

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        model.predict(obs, deterministic=True)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return compute_stats(times)


def bench_onnx_provider(onnx_path: str, provider: str, warmup: int, iters: int) -> dict[str, object]:
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=make_sess_options(), providers=[provider])
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    actual = sess.get_providers()
    if provider not in actual:
        return {"ok": False, "error": f"Requested {provider}, got {actual}"}

    inp = {"obs": np.zeros((1, 11), dtype=np.float32)}
    for _ in range(warmup):
        sess.run(None, inp)

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        sess.run(None, inp)
        times[i] = (time.perf_counter() - t0) * 1000.0

    out: dict[str, object] = {"ok": True, "providers": actual}
    out.update(compute_stats(times))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_hopper.zip")
    parser.add_argument("--onnx", default="ppo_hopper.onnx")
    parser.add_argument("--engine", default="ppo_hopper_fp16.engine")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    out: dict[str, object] = {
        "warmup": args.warmup,
        "iters": args.iters,
        "providers_available": ort.get_available_providers(),
    }

    out["sb3_cpu_ms"] = bench_sb3_cpu(args.model, args.warmup, args.iters)
    out["onnx_cpu_ms"] = bench_onnx_provider(args.onnx, "CPUExecutionProvider", args.warmup, args.iters)
    out["onnx_cuda_ms"] = bench_onnx_provider(args.onnx, "CUDAExecutionProvider", args.warmup, args.iters)
    out["onnx_trt_ep_ms"] = bench_onnx_provider(args.onnx, "TensorrtExecutionProvider", args.warmup, args.iters)

    engine_path = Path(args.engine)
    if engine_path.exists():
        try:
            from run_trt_inference import benchmark_engine

            out["trt_engine_ms"] = benchmark_engine(
                str(engine_path),
                obs_dim=11,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:
            out["trt_engine_ms"] = {"ok": False, "error": str(exc)}
    else:
        out["trt_engine_ms"] = {"ok": False, "error": f"Engine not found: {engine_path}"}

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
