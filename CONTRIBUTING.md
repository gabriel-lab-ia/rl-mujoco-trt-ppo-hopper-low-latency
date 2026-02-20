# Contributing

## Development Environment
1. `cd ~/rl-mujoco-trt/ppo_hopper`
2. `source ../rl-mujoco-trt/bin/activate`
3. Run checks before PR:
   - `python -m py_compile train_ppo.py enjoy_ppo.py export_onnx.py run_trt_inference.py benchmarks.py`
   - `bash -n build_trt_engine.sh`
   - `python benchmarks.py --warmup 200 --iters 2000`

## Pull Requests
- Keep changes minimal and focused.
- Update `README.md` when behavior changes.
- Include benchmark deltas when touching inference/training performance.

## Safety
- Do not add secrets, tokens, private keys or credentials.
- Do not commit generated artifacts (`.onnx`, `.zip`, `.engine`, logs, TensorBoard runs).
