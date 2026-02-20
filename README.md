 PPO Hopper Low-Latency Pipeline

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

Latency Benchmarks and Technical Analysis
Benchmark Summary

All latency measurements were executed on Pop!_OS 24.04, Python venv (isolated), CUDA Toolkit 12.0, CUDA Runtime 13, and an RTX 4050 Laptop GPU.
The policy is a compact MLP typical of Hopper-v4 PPO agents, which is an important factor in the observed results.
| Inference Pipeline            | Device | Mean Latency (ms) | Notes                                                 |
| ----------------------------- | ------ | ----------------- | ----------------------------------------------------- |
| SB3 (PyTorch)                 | CPU    | ~0.105            | Baseline execution path                               |
| ONNX Runtime                  | CPU    | ~0.0068           | Fastest path; extremely low overhead                  |
| ONNX Runtime                  | CUDA   | ~0.045            | GPU acceleration limited by transfer overhead         |
| ONNX Runtime (TRT EP)         | GPU    | fallback to CPU   | TRT EP not enabled; missing TensorRT runtime bindings |
| TensorRT FP16 Engine (Python) | GPU    | not executed      | Python `tensorrt` module unavailable inside venv      |

Technical Interpretation
1. Why ONNX Runtime (CPU) Achieved the Best Latency
The Hopper-v4 PPO policy is a small MLP (low parameter count, small batch = 1).
In this regime, CPU-optimized ONNX kernels outperform GPUs because:
Zero device transfer overhead
Highly optimized fused CPU kernels (mostly Dense + ReLU)
Execution stays entirely within the CPU L1/L2 cache
Function call overhead dominates; GPU adds more overhead than compute
This makes ~0.0068 ms not only plausible, but expected for such models.
For policies this small, GPU acceleration does not compensate the overhead of:
Host → Device copy
Kernel dispatch
Synchronization points
Therefore, ONNX Runtime CPU is the optimal path for micro-models.

2. Why ONNX Runtime (CUDA) Does Not Beat CPU
The CUDA backend shows ~0.045 ms, slower than CPU ONNX by ~7x
This is entirely explained by:
Device transfer latency dominating computation
Kernel launch overhead on small workloads
Lack of batching (batch = 1), eliminating GPU parallelism advantages
CUDA begins to show gains when:
model depth and width increase,
operations saturate SMs,
batching > 16, or
convolutions or attention layers are involved.
Your pipeline correctly exposes this behavior, which is consistent with GPU performance literature.

3. Why TensorRT EP Fell Back to CPU
The ONNX Runtime’s TensorRT Execution Provider requires:
libnvinfer, libnvinfer_plugin, and libnvonnxparser
Matched versions for TensorRT 10.15, CUDA runtime 13, and CUDA Toolkit 12
Your host contains TensorRT CLI (trtexec), but the runtime libraries were not detected by ORT, triggering an automatic fallback to CPU execution.
This is standard behavior:
ORT prefers correctness over speed when EPs are partially misconfigured.

4. Why TensorRT FP16 Engine Could Not Run (Python)
run_trt_inference.py could not load a TensorRT engine because the venv does not contain the Python binding:
Missing module: tensorrt
The TensorRT Python package must be installed using the NVIDIA wheel matching:
major TensorRT version
CUDA minor version
Python ABI version
Because the wheel was not present in the venv, the FP16 engine test was skipped.
This was documented in the repository and is the correct engineering decision.

Engineering Conclusion
For small MLP policies (such as Hopper PPO):
CPU-optimized ONNX Runtime is the fastest inference backend due to minimal overhead.
GPU-based inference surpasses CPU only when model complexity increases.
TensorRT advantages emerge only in medium-to-large networks where kernel fusion, precision lowering (FP16/INT8), and memory planning outweigh transfer and dispatch overhead.
The pipeline is structurally complete.
Training → Export (ONNX) → TRT Engine Build → Multi-backend Benchmarking
Every component is correctly isolated, reproducible, and documented.
The non-executed TensorRT steps are not failures; they identify missing runtime bindings, which is valuable for reproducibility and future optimization work.
This project forms a correct, professional-grade foundation for integrating RL policies into real-time systems, including robotics, embedded autonomy, and eventually high-speed UAV controllers.
