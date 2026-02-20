# PPO Hopper Low-Latency Pipeline

Isolated project path: `~/rl-mujoco-trt/ppo_hopper`

## Scope
- Train PPO (`Hopper-v4`) with stable CPU path.
- Export ONNX.
- Build TensorRT FP16 engine.
- Benchmark: SB3 CPU, ONNX CPU/CUDA/TRT EP, and TensorRT engine (kernel-only + e2e).

## Setup
```bash
cd ~/rl-mujoco-trt/ppo_hopper
source ../rl-mujoco-trt/bin/activate
```

## 1) Train PPO
```bash
python train_ppo.py --timesteps 200000 --env-id Hopper-v4
```
Artifact: `ppo_hopper.zip`

## 2) Export ONNX
```bash
python export_onnx.py --model ppo_hopper.zip --out ppo_hopper.onnx --opset 17
```
Artifact: `ppo_hopper.onnx`

## 3) Build TensorRT FP16 Engine
```bash
./build_trt_engine.sh ppo_hopper.onnx ppo_hopper_fp16.engine trtexec_ppo_hopper_fp16.log
```
Artifacts: `ppo_hopper_fp16.engine`, `trtexec_ppo_hopper_fp16.log`

If `trtexec` is not in `/usr/bin/trtexec`, define:
```bash
export TRTEXEC_BIN=/full/path/to/trtexec
```

## 4) Visualize / Quick Benchmark
Visual mode:
```bash
python enjoy_ppo.py --mode visual --env-id Hopper-v4 --fps 60
```
Benchmark mode (predict+step):
```bash
python enjoy_ppo.py --mode benchmark --env-id Hopper-v4 --warmup 200 --iters 2000
```

## 5) Unified Benchmarks
```bash
python benchmarks.py --warmup 200 --iters 2000
```

## 6) TensorRT Engine Benchmarks
```bash
python run_trt_inference.py --engine ppo_hopper_fp16.engine --warmup 200 --iters 2000
```

## Runtime Notes (Current Host)
- ONNX TensorRT EP can be listed by ORT but still fallback to CPU if TensorRT libs are not resolvable at runtime.
- `run_trt_inference.py` requires Python bindings `tensorrt` and `pycuda` inside the venv.
- For tiny MLP policy with batch=1, ONNX CPU may be lower-latency than GPU e2e due to transfer overhead.

## Publish Checklist
- [ ] `python benchmarks.py --warmup 200 --iters 2000`
- [ ] `./build_trt_engine.sh ...` (if `trtexec` available)
- [ ] `python run_trt_inference.py ...` (if `tensorrt` python bindings available)
- [ ] Commit and push
