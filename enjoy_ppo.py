#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def compute_stats(ms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(ms.mean()),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
    }


def run_visual(model_path: str, env_id: str, fps: float) -> None:
    env = gym.make(env_id, render_mode="human")
    model = PPO.load(model_path, device="cpu")
    sleep_s = 0.0 if fps <= 0 else 1.0 / fps

    obs, _ = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            if sleep_s > 0:
                time.sleep(sleep_s)
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Stopped visual mode")
    finally:
        env.close()


def run_benchmark(model_path: str, env_id: str, warmup: int, iters: int) -> None:
    env = gym.make(env_id)
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    try:
        for _ in range(warmup):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        times_ms = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            times_ms[i] = (time.perf_counter() - t0) * 1000.0
            if terminated or truncated:
                obs, _ = env.reset()

        s = compute_stats(times_ms)
        print(f"enjoy benchmark ({env_id}) predict+step ms: mean={s['mean']:.6f} p50={s['p50']:.6f} p95={s['p95']:.6f} p99={s['p99']:.6f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["visual", "benchmark"], default="visual")
    parser.add_argument("--model", default="ppo_hopper.zip")
    parser.add_argument("--env-id", default="Hopper-v4")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    if args.mode == "visual":
        run_visual(args.model, args.env_id, args.fps)
    else:
        run_benchmark(args.model, args.env_id, args.warmup, args.iters)


if __name__ == "__main__":
    main()
