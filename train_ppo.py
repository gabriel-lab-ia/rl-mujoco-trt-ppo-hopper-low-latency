#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--env-id", default="Hopper-v4")
    args = parser.parse_args()

    env = gym.make(args.env_id)
    log_dir = Path("runs")
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        tensorboard_log=str(log_dir),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
    )

    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training {args.env_id} on CPU for {args.timesteps} steps")
    model.learn(total_timesteps=args.timesteps, tb_log_name=f"ppo_hopper_{run_tag}")
    model.save("ppo_hopper")
    env.close()
    print("Saved: ppo_hopper.zip")


if __name__ == "__main__":
    main()
