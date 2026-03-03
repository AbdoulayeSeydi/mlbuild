# MLBuild

<div align="center">

**Performance CI/CD for Production ML Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/mlbuild.svg)](https://pypi.org/project/MLBuild/)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://github.com/AbdoulayeSeydi/mlbuild)

MLBuild is the missing performance layer for ML CI/CD. While MLflow, DVC, and W&B track training experiments, MLBuild enforces production SLAs — automatically benchmarking inference performance, validating against thresholds, blocking regressions in CI, and generating deployment-ready reports.

[Installation](#installation) · [Quick Start](#quick-start) · [Documentation](#documentation) · [Roadmap](#roadmap)

</div>

---

## Current Status

| Feature | Status |
|---------|--------|
| Input formats | ONNX, TFLite |
| Backends | CoreML, TFLite, ONNX Runtime |
| Storage | Local + S3-compatible (AWS S3, R2, B2) |
| Targets | Apple Silicon, A-series, Android (arm64) |
| Platform | macOS |

---

## The Problem

```bash
# Your CI passes
pytest              ✓
black --check       ✓
mypy                ✓

# But in production
Latency:  8ms  --> 15ms   (88% slower)
Memory:   50MB --> 120MB  (140% more)
Size:     6MB  --> 10MB   (67% larger)

# Nobody caught it until users complained
```

**The gap:** Existing tools don't validate production performance in CI.

---

## The Solution

```bash
# Add one step to your CI pipeline
mlbuild build --model model.onnx --target apple_m1
mlbuild ci-check $BASELINE_ID $CANDIDATE_ID

# Output:
# ⚠ REGRESSION DETECTED
#   • latency +12.3% > 10.0% threshold
#   • size +8.1% > 5.0% threshold
# Exit code: 1 — PR blocked
```

Catch latency AND size regressions before they reach production.

---

## What MLBuild Does

| Feature | MLflow / W&B / DVC | MLBuild |
|---------|-------------------|---------|
| Track training experiments | Yes | No (use MLflow) |
| Automated p50/p95/p99 benchmarking | Manual | Built-in |
| CI fails on latency regression | Not native | `mlbuild ci-check` |
| CI fails on model size regression | Not native | `--size-threshold` |
| Quantization tradeoff analysis | No | `mlbuild compare-quantization` |
| Performance reports | No | `mlbuild report` |
| S3-compatible remote storage | No | Built-in |
| TFLite benchmarking | No | Built-in |

MLBuild complements your existing stack — it doesn't replace it.

---

## Installation

```bash
pip install mlbuild
mlbuild doctor
```

For TFLite support:
```bash
pip install "mlbuild[tflite]"
```

For S3 remote storage:
```bash
pip install "mlbuild[s3]"
```

---

## Quick Start

```bash
# 1. Build and convert model
mlbuild build --model model.onnx --target apple_m1 --quantize fp16

# 2. Benchmark (automatic p50/p95/p99)
mlbuild benchmark <build-id>

# 3. Validate SLAs
mlbuild validate <build-id> --max-latency 10 --max-p95 15 --ci

# 4. Compare vs baseline (latency + size)
mlbuild compare baseline candidate --threshold 5 --size-threshold 5 --ci

# 5. Analyze quantization tradeoffs
mlbuild compare-quantization fp32-build int8-build

# 6. Generate performance report
mlbuild report <build-id> --open

# 7. Tag for production
mlbuild tag create <build-id> production
```

### GitHub Actions Integration

```yaml
- name: Performance Validation
  run: |
    pip install "mlbuild[tflite,s3]"

    # Build model
    mlbuild build --model models/model.onnx --target android_arm64
    BUILD_ID=$(mlbuild log --limit 1 --json | jq -r '.[0].build_id')

    # Gate: block PR if latency or size regresses
    mlbuild ci-check $BASELINE_ID $BUILD_ID \
      --latency-threshold 10 \
      --size-threshold 5 \
      --json
```

See `.github/workflows/examples/` for complete examples.

---

## Documentation

### Core Commands

#### Build and Convert

```bash
mlbuild build --model model.onnx --target apple_m1 --quantize fp16 --name "v2.0"
mlbuild build --model model.onnx --target android_arm64 --quantize int8
```

#### Benchmark

```bash
mlbuild benchmark <build-id> --runs 100 --warmup 20 --json
```

#### Validate SLAs

```bash
mlbuild validate <build-id> \
  --max-latency 10 \
  --max-p95 15 \
  --max-memory 100 \
  --ci
```

#### Compare and Detect Regressions

```bash
# Compare with independent latency + size thresholds
mlbuild compare baseline candidate \
  --threshold 5 \
  --size-threshold 10 \
  --metric p95 \
  --ci

# Dedicated CI gate (cleaner defaults for pipelines)
mlbuild ci-check baseline candidate
mlbuild ci-check baseline candidate --latency-threshold 10 --size-threshold 5
mlbuild ci-check baseline candidate --strict   # any positive delta fails
mlbuild ci-check baseline candidate --json     # machine-readable output
```

**Exit codes:**
- `0` — no regression (safe to ship)
- `1` — regression detected (block the PR)
- `2` — error (infra failure, check logs)

#### Quantization Tradeoff Analysis

```bash
# Analyze the accuracy/size/latency tradeoff between two quantization levels
mlbuild compare-quantization fp32-build int8-build

# Options
mlbuild compare-quantization fp32-build int8-build --accuracy-samples 100
mlbuild compare-quantization fp32-build int8-build --skip-accuracy  # faster
mlbuild compare-quantization fp32-build int8-build --json
```

**Output includes:**
- Size reduction %
- Latency improvement %
- Accuracy loss % (relative error on synthetic inputs)
- Cosine similarity (output ranking preservation)
- Tradeoff score and deployment recommendation

#### Performance Report

```bash
# Generate self-contained HTML report
mlbuild report <build-id>
mlbuild report <build-id> --open              # open in browser immediately
mlbuild report <build-id> --output report.html
mlbuild report <build-id> --format pdf        # requires: pip install weasyprint
```

**Report includes:**
- Latest benchmark metrics (p50/p95/p99, throughput, memory)
- Full benchmark history
- Build metadata and quantization details
- Related builds with size comparison
- Deployment recommendations

#### Deep Profiling

```bash
# TFLite: full 6-feature deep profile (no device required)
mlbuild profile <build-id> --deep

# CoreML: cold start decomposition (all formats)
#         per-layer timing, memory flow, bottleneck classification,
#         and fusion detection (NeuralNetwork format only)
mlbuild profile <build-id> --deep

# Options
mlbuild profile <build-id> --deep --top 20           # show top 20 layers/ops
mlbuild profile <build-id> --deep --runs 100         # more runs for stability
mlbuild profile <build-id> --deep --int8-build <id>  # TFLite: quant sensitivity
```

**TFLite deep profiling features (`--deep`):**

| # | Feature | Description |
|---|---------|-------------|
| ① | Per-op timing | Real hardware timing via TFLite's built-in op profiler |
| ② | Memory flow | Activation memory at each layer boundary, peak flagged |
| ③ | Bottleneck classification | COMPUTE vs MEMORY bound per op (arithmetic intensity) |
| ④ | Cold start decomposition | Load → first inference → stable, with warmup sparkline |
| ⑤ | Quantization sensitivity | Per-layer fp32 vs int8 divergence (requires `--int8-build`) |
| ⑥ | Fusion detection | Fused kernels identified + missed fusion opportunities flagged |

```bash
# TFLite example
mlbuild profile <build-id> --deep

# 1 · Per-Op Timing
# ┌────┬──────────────────────────┬──────────────────┬───────────┬─────────┬───────┐
# │ #  │ Op Name                  │ Type             │ Time (ms) │ % Total │ Fused │
# ├────┼──────────────────────────┼──────────────────┼───────────┼─────────┼───────┤
# │  0 │ conv2d/Conv2D            │ CONV_2D          │     2.141 │  38.2%  │   ✓   │
# │  1 │ depthwise_conv/...       │ DEPTHWISE_CONV   │     1.872 │  33.4%  │   ✓   │
# │  2 │ MobilenetV1/Predictions  │ FULLY_CONNECTED  │     0.431 │   7.7%  │   —   │
# └────┴──────────────────────────┴──────────────────┴───────────┴─────────┴───────┘

# With int8 quant sensitivity:
mlbuild profile <build-id> --deep --int8-build <int8-build-id>

# 5 · Quantization Sensitivity Map
# ┌──────────────────────┬──────────┬──────────┬────────────┬────────────┬─────────────┐
# │ Layer                │ MSE      │ MAE      │ Max Error  │ Cosine Sim │ Sensitivity │
# ├──────────────────────┼──────────┼──────────┼────────────┼────────────┼─────────────┤
# │ conv2d/BiasAdd       │ 0.000021 │ 0.003412 │ 0.021443   │ 0.9991     │ LOW         │
# │ predictions/Softmax  │ 0.000847 │ 0.018234 │ 0.094312   │ 0.9743     │ HIGH        │
# └──────────────────────┴──────────┴──────────┴────────────┴────────────┴─────────────┘
```

**CoreML deep profiling features (`--deep`):**

| # | Feature | NeuralNetwork | MLProgram |
|---|---------|---------------|-----------|
| ① | Per-layer timing | ✓ Sliced subgraph benchmarking | ✗ Requires Xcode |
| ② | Memory flow | ✓ Estimated from weight dims | ✗ Requires Xcode |
| ③ | Bottleneck classification | ✓ COMPUTE vs MEMORY bound | ✗ Requires Xcode |
| ④ | Cold start decomposition | ✓ | ✓ |
| ⑤ | Fusion detection | ✓ Conv+Relu, BN+Scale, etc. | ✗ Requires Xcode |

> **Note:** coremltools 7+ converts all models to MLProgram format regardless of deployment target. MLProgram models are compiled by the ANE toolchain at load time — per-layer slicing is not possible without Xcode Instruments. Cold start decomposition (④) is available for all CoreML builds.

#### Profile Layers (Standard)

```bash
mlbuild profile <build-id> --top 15 --analyze-warmup
```

#### Version Management

```bash
mlbuild log --limit 20
mlbuild diff build-a build-b
mlbuild tag create <build-id> v1.0.0
```

#### Experiment Tracking

```bash
mlbuild experiment create "quantization-search"
mlbuild run start --experiment "quantization-search"
mlbuild run log-param quantization int8
mlbuild run log-metric latency_p50 5.6
mlbuild run end
```

#### Remote Storage

```bash
# Set up S3-compatible remote (one-time)
mlbuild remote add prod \
  --backend s3 \
  --bucket your-bucket \
  --region us-east-1

# Push/pull/sync builds
mlbuild push <build-id>
mlbuild pull <build-id>
mlbuild sync
```

**Supported backends:**
- AWS S3
- Cloudflare R2 (recommended — free 10 GB)
- Backblaze B2
- Any S3-compatible storage

---

## Quantization Workflow

A common workflow for mobile deployment:

```bash
# 1. Build FP32 baseline
mlbuild build --model model.onnx --target android_arm64 --name mobilenet-fp32

# 2. Build INT8 variant
mlbuild build --model model.onnx --target android_arm64 --quantize int8 --name mobilenet-int8

# 3. Analyze tradeoffs
mlbuild compare-quantization mobilenet-fp32 mobilenet-int8

# Output:
# Size:     6.69 MB → 3.82 MB  (-42.9%)
# Latency:  15.3 ms → 5.6 ms   (-63.7%)
# Cosine similarity: 0.9718
# Recommendation: INT8 promising — validate on real data

# 4. Generate report with all findings
mlbuild report mobilenet-int8 --open
```

---

## CI/CD Regression Gate

```bash
# In your CI pipeline:
mlbuild ci-check $BASELINE_ID $CANDIDATE_ID
echo "Exit: $?"   # 0 = pass, 1 = regression, 2 = error

# With JSON output for log parsing:
mlbuild ci-check $BASELINE_ID $CANDIDATE_ID --json
# {
#   "regression_detected": false,
#   "change": { "p50": -63.75, "size": -42.86 },
#   "thresholds": { "latency_pct": 10.0, "size_pct": 5.0 }
# }
```

---

## Architecture

```
Training Phase
├── Experiment Tracking:   MLflow / W&B / Neptune
└── Data Versioning:       DVC

              ↓

Production Phase
├── Model Building:         MLBuild build
├── Performance Validation: MLBuild ci-check     ← regression gate
├── Quantization Analysis:  MLBuild compare-quantization
├── Reporting:              MLBuild report
└── Deployment:             GitHub Actions / K8s
```

---

## How It Works

### 1. Deterministic Builds

```python
# Content-addressed storage (Git-style)
build_id = sha256(source_hash + config_hash + env_fingerprint)
# Same inputs = Same output (byte-for-byte)
```

### 2. Automated Benchmarking

```python
# Runs N iterations with warmup
# Calculates p50, p95, p99, mean, std
# Measures memory RSS delta, throughput
# Outlier trimming (top/bottom 5%)
```

### 3. Dual Regression Detection

```python
# Independent thresholds for latency and size
latency_regression = latency_change_pct > latency_threshold
size_regression    = size_change_pct    > size_threshold
regression_detected = latency_regression or size_regression
```

### 4. Quantization Tradeoff Scoring

```python
score = (size_reduction% + latency_improvement%) / 2 - accuracy_loss% * 2
# accuracy penalized 2x — impacts users directly
```

---

## Features

### Build and Convert
- ONNX → CoreML conversion (Apple Silicon, A-series)
- ONNX → TFLite conversion (Android arm64)
- Quantization: FP32 / FP16 / INT8
- Deterministic builds (content-addressed)

### Performance Validation
- Automated p50/p95/p99 benchmarking
- SLA enforcement (`--max-latency`, `--max-memory`)
- Latency regression detection
- Model size regression detection
- Statistical significance testing (Welch's t-test)

### Deep Profiling (`--deep`)
- **TFLite:** Per-op timing (real hardware), tensor memory flow, COMPUTE/MEMORY bottleneck classification, cold start decomposition, per-layer quantization sensitivity (fp32 vs int8), op fusion detection
- **CoreML:** Cold start decomposition (all formats); per-layer timing, memory flow, bottleneck classification, and fusion detection (NeuralNetwork format only)

### Quantization Analysis
- Accuracy estimation on synthetic inputs
- Cosine similarity (output ranking preservation)
- Tradeoff scoring with deployment recommendation
- Support for TFLite int8/fp16 pairs

### Performance Reports
- Self-contained HTML (no external dependencies)
- Benchmark history table
- Related builds comparison
- Deployment recommendations
- Optional PDF export (requires weasyprint)

### Remote Storage
- S3-compatible backends (AWS, R2, B2)
- Git-style push/pull/sync
- Prefix resolution
- Integrity verification (SHA-256)

### CI/CD Integration
- `mlbuild ci-check` — dedicated regression gate
- Exit codes: 0 (pass) / 1 (regression) / 2 (error)
- JSON output for log parsing
- GitHub Actions examples

---

## Project Structure

```
mlbuild/
├── src/mlbuild/
│   ├── cli/
│   │   ├── commands/
│   │   │   ├── benchmark.py           # mlbuild benchmark
│   │   │   ├── compare.py             # mlbuild compare + ci-check
│   │   │   ├── compare_quantization.py # mlbuild compare-quantization
│   │   │   ├── report.py              # mlbuild report
│   │   │   ├── profile.py             # mlbuild profile
│   │   │   ├── validate.py            # mlbuild validate
│   │   │   ├── push.py / pull.py      # remote storage
│   │   │   └── ...
│   │   └── main.py                    # CLI entry point
│   ├── backends/
│   │   ├── coreml/                    # CoreML exporter + deep profiler
│   │   ├── tflite/                    # TFLite backend + deep profiler
│   │   └── onnxruntime/               # ONNX Runtime backend
│   ├── benchmark/                     # Benchmark runner + stats
│   ├── profiling/                     # Layer-by-layer profiling + cold start
│   ├── registry/                      # SQLite registry (WAL mode)
│   ├── storage/                       # S3-compatible remote storage
│   └── experiments/                   # Experiment + run tracking
├── tests/
├── pyproject.toml
└── README.md
```

---

## vs. Existing Tools

| | MLflow | DVC | W&B | **MLBuild** |
|---|---|---|---|---|
| Training experiments | ✓ | — | ✓ | — |
| Data versioning | — | ✓ | — | — |
| Inference benchmarking | Manual | No | No | **Automated** |
| CI regression gate | No | No | No | **Built-in** |
| Size regression detection | No | No | No | **Built-in** |
| Quantization analysis | No | No | No | **Built-in** |
| Per-layer deep profiling | No | No | No | **Built-in** |
| Performance reports | No | No | Dashboard | **HTML/PDF** |

Use MLflow/W&B for training. Use MLBuild for inference.

---

## Development

```bash
git clone https://github.com/AbdoulayeSeydi/mlbuild.git
cd mlbuild
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

```bash
pytest tests/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR process.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Roadmap

### Phase 1 — Device-Connected Benchmarking *(next)*
- Android ADB bridge — benchmark on connected Android devices without Android Studio
- Xcode Instruments integration — real iPhone hardware profiling

### Phase 2 — More Backends
- TensorRT — NVIDIA GPU inference
- Qualcomm QNN — Snapdragon NPU

### Phase 3 — Cloud Benchmarking
- Remote benchmark execution on cloud hardware

---

<div align="center">
Built by <a href="https://github.com/AbdoulayeSeydi/mlbuild">Abdoulaye Seydi</a>
</div>