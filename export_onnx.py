#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import gymnasium as gym
import torch
from stable_baselines3 import PPO


class DeterministicPolicyWrapper(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.policy(obs, deterministic=True)
        return actions


def get_obs_shape():
    for env_id in ("Hopper-v5", "Hopper-v4"):
        try:
            env = gym.make(env_id)
            shape = env.observation_space.shape
            env.close()
            return shape, env_id
        except Exception:
            continue
    raise RuntimeError("Could not create Hopper-v5 or Hopper-v4 to infer observation shape.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_hopper.zip", help="Path to SB3 PPO zip model")
    parser.add_argument("--out", default="ppo_hopper.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=15)")
    args = parser.parse_args()

    if args.opset < 15:
        raise ValueError("Please use opset >= 15.")

    model_path = Path(args.model)
    out_path = Path(args.out)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    obs_shape, env_id = get_obs_shape()
    model = PPO.load(str(model_path), device="cpu")
    model.policy.eval()

    wrapper = DeterministicPolicyWrapper(model.policy).cpu().eval()
    dummy_obs = torch.zeros((1, *obs_shape), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export failed: {out_path}")

    print(f"Exported ONNX ({env_id}) -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
