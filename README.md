# MLBuild

<div align="center">

<img src="assets/mlbuild_logo.png" alt="MLBuild Logo" width="120" /><br/><br/>

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
| Input formats | ONNX, TFLite, CoreML |
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
| Task-aware synthetic inputs | No | **Auto-detected** |
| NLP multi-seq-len benchmarking | No | **Built-in** |
| Quantization tradeoff analysis | No | `mlbuild compare-quantization` |
| Performance reports | No | `mlbuild report` |
| S3-compatible remote storage | No | Built-in |
| TFLite benchmarking | No | Built-in |
| Import pre-built models | No | `mlbuild import` |

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

# 1b. Or import a pre-built model
mlbuild import --model model.tflite --target android_arm64
mlbuild import --model model.mlpackage --target apple_m1 --quantize fp16

# 2. Benchmark (automatic p50/p95/p99, task auto-detected)
mlbuild benchmark <build-id>

# 2b. Override task detection explicitly
mlbuild benchmark <build-id> --task nlp

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

#### Import Pre-built Models

Register an existing TFLite or CoreML model directly — no conversion required. Once imported, all MLBuild commands (benchmark, profile, compare, report, ci-check) work on it immediately.

```bash
# Import a TFLite model
mlbuild import --model model.tflite --target android_arm64

# Import a CoreML model
mlbuild import --model model.mlpackage --target apple_m1

# Import with metadata
mlbuild import --model model.tflite --target android_arm64 \
  --quantize int8 \
  --name "vendor-v2" \
  --notes "Supplied by vendor, int8 quantized"

# JSON output (for CI pipelines)
mlbuild import --model model.tflite --target android_arm64 --json
```

**Supported formats:**
- `.tflite` — validated via FlatBuffer magic bytes (TFL3/TFL2)
- `.mlpackage` — validated via Manifest.json + Data/ structure
- `.mlmodel` — legacy CoreML flat file

**Format/target compatibility:**

| Format | Valid Targets |
|--------|--------------|
| `tflite` | `android_arm64`, `android_arm32`, `android_x86`, `raspberry_pi`, `coral_tpu`, `generic_linux` |
| `coreml` | `apple_m1`, `apple_m2`, `apple_m3`, `apple_a15`, `apple_a16`, `apple_a17` |

Imported builds are marked `[imported]` in `mlbuild log` output and tracked with `"imported": true` in their metadata, distinguishing them from builds produced by `mlbuild build`.

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

## Task-Aware Benchmarking

MLBuild automatically detects what kind of model you're benchmarking — vision, NLP, or audio — and generates semantically correct synthetic inputs for it. No dummy zero arrays, no manual shape specification.

### Automatic Task Detection

Detection runs through three tiers in order of confidence:

| Tier | Method | Formats | Confidence | CLI Behavior |
|------|--------|---------|------------|--------------|
| **Graph** | Op/layer analysis (`Conv`, `Attention`, `STFT`, etc.) | ONNX, CoreML NN | High | Silent |
| **Name** | Tensor name heuristics (`input_ids`, `pixel_values`, `mel`) | All | Medium | Warning |
| **Shape** | Dtype + rank heuristics (rank-4 float = vision, rank-2 int = NLP) | All | Low | Warning + zeros fallback |

```bash
# High confidence — silent, correct inputs generated automatically
mlbuild benchmark <build-id>

# Medium confidence — warning printed, benchmark proceeds
# ⚠  Task auto-detected as 'nlp' (medium confidence)
#    If incorrect, re-run with: --task vision|nlp|audio
mlbuild benchmark <build-id>

# Low confidence or unknown — zeros used as safe fallback
# ⚠  Task could not be detected — running with zero tensors
#    Specify task explicitly: --task vision|nlp|audio
mlbuild benchmark <build-id>
```

### Override with `--task`

If auto-detection gets it wrong, pass `--task` explicitly. This bypasses all detection logic entirely and is the recommended approach for audio models, which are always considered low-confidence by the shape heuristic.

```bash
mlbuild benchmark <build-id> --task vision
mlbuild benchmark <build-id> --task nlp
mlbuild benchmark <build-id> --task audio

mlbuild profile  <build-id> --task nlp
mlbuild validate <build-id> --task vision --strict-output
```

### Task-Specific Synthetic Inputs

Each task gets semantically correct inputs rather than zeros.

| Task | Inputs Generated |
|------|-----------------|
| **Vision** | Float32 image tensor, NCHW layout, spatial dims resolved to 224×224 |
| **NLP** | `int64` token IDs (random vocab up to 30k), `int64` attention mask (all ones), token type IDs |
| **Audio** | Float32 waveform `[-1, 1]` or log-mel spectrogram — role inferred from tensor name/shape |
| **Unknown** | Zero tensors — safe fallback that never blocks CI |

Dynamic dimensions (`-1`, `None`) are resolved using task-appropriate defaults. Concrete dimensions are always preserved as-is.

### NLP Multi-Sequence Benchmarking

NLP models are benchmarked across a sequence length ladder by default, giving you a full latency curve rather than a single point. This is essential for understanding how your model scales with context length.

```bash
# Default ladder: [16, 64, 128, 256]
mlbuild benchmark <build-id> --task nlp

# seq_len=16   p50=1.2ms  p95=1.4ms
# seq_len=64   p50=2.1ms  p95=2.4ms
# seq_len=128  p50=3.8ms  p95=4.2ms
# seq_len=256  p50=7.1ms  p95=8.0ms

# Clip to your model's actual max sequence length
mlbuild benchmark <build-id> --task nlp --seq-len 128
# Ladder: [16, 64, 128]

# Model with non-standard max (appended to ladder as extra benchmark point)
mlbuild benchmark <build-id> --task nlp --seq-len 200
# Ladder: [16, 64, 128, 200]
```

### Strict Output Validation

After inference, MLBuild validates model outputs against expectations for the detected task. In soft mode (default) issues are reported as warnings without blocking CI. Use `--strict-output` to fail on any anomaly.

```bash
# Soft mode (default) — warns but proceeds
mlbuild benchmark <build-id> --task nlp

# Strict mode — exits non-zero on output anomaly
mlbuild benchmark <build-id> --task nlp --strict-output
mlbuild validate  <build-id> --task vision --strict-output

# Global strict mode — applies to all commands in the invocation
mlbuild --strict-output benchmark <build-id> --task nlp
```

**Checks run on outputs:**

| Check | Tasks | Strict mode promotes to FAIL |
|-------|-------|------------------------------|
| NaN / Inf detection | All | Always fails regardless of mode |
| All-zeros output | All | Yes |
| Zero logit variance | Vision, NLP | Yes |
| Bounding box shape `[B, N, 4+]` | Vision | Yes |
| CTC output shape `[B, T, vocab]` | Audio | Yes |
| Box / score count alignment | Vision | Yes |

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
├── Model Importing:        MLBuild import          ← pre-built TFLite / CoreML
├── Task Detection:         MLBuild (automatic)     ← vision / nlp / audio
├── Performance Validation: MLBuild ci-check        ← regression gate
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

### 3. Task-Aware Input Generation

```python
# Three-tier detection: graph ops → tensor names → shapes
# Task-specific synthetic inputs (never zeros for known tasks)
# NLP: multi-seq-len ladder [16, 64, 128, 256]
# Post-inference output validation with configurable strictness
```

### 4. Dual Regression Detection

```python
# Independent thresholds for latency and size
latency_regression = latency_change_pct > latency_threshold
size_regression    = size_change_pct    > size_threshold
regression_detected = latency_regression or size_regression
```

### 5. Quantization Tradeoff Scoring

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

### Import Pre-built Models
- Import existing `.tflite`, `.mlmodel`, `.mlpackage` files directly
- Format validation via magic bytes (TFLite) and structure checks (CoreML)
- Format/target compatibility enforcement
- Imported builds tracked with `[imported]` badge in `mlbuild log`
- Full MLBuild toolchain available immediately after import

### Task-Aware Benchmarking
- Three-tier automatic task detection (graph ops → tensor names → shapes)
- Task-specific synthetic inputs: real image tensors, token IDs + attention masks, waveforms/spectrograms
- NLP multi-sequence-length benchmarking ladder `[16, 64, 128, 256]`
- Configurable `--task` override for explicit control
- Post-inference output validation with soft/strict modes (`--strict-output`)

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
│   │   │   ├── benchmark.py              # mlbuild benchmark
│   │   │   ├── build.py                  # mlbuild build
│   │   │   ├── compare.py                # mlbuild compare + ci-check
│   │   │   ├── compare_compute_units.py  # mlbuild compare-compute-units
│   │   │   ├── compare_quantization.py   # mlbuild compare-quantization
│   │   │   ├── diff.py                   # mlbuild diff
│   │   │   ├── doctor.py                 # mlbuild doctor
│   │   │   ├── experiment.py             # mlbuild experiment
│   │   │   ├── import_cmd.py             # mlbuild import
│   │   │   ├── log.py                    # mlbuild log
│   │   │   ├── profile.py                # mlbuild profile
│   │   │   ├── pull.py                   # mlbuild pull
│   │   │   ├── push.py                   # mlbuild push
│   │   │   ├── remote.py                 # mlbuild remote
│   │   │   ├── report.py                 # mlbuild report
│   │   │   ├── run.py                    # mlbuild run
│   │   │   ├── sync.py                   # mlbuild sync
│   │   │   ├── tag.py                    # mlbuild tag
│   │   │   └── validate.py               # mlbuild validate
│   │   └── main.py                       # CLI entry point
│   ├── backends/
│   │   ├── base.py                       # Backend base class
│   │   ├── registry.py                   # Backend auto-discovery
│   │   ├── coreml/                       # CoreML exporter + deep profiler
│   │   ├── tflite/                       # TFLite backend + deep profiler
│   │   └── onnxruntime/                  # ONNX Runtime backend
│   ├── benchmark/                        # Benchmark runner + stats
│   ├── core/
│   │   ├── environment.py                # Environment fingerprinting
│   │   ├── errors.py                     # Error types
│   │   ├── format_detection.py           # Format detection + target validation
│   │   ├── hash.py                       # Deterministic artifact hashing
│   │   ├── task_detection.py             # Three-tier task detection
│   │   ├── task_inputs.py                # Task-aware synthetic input generation
│   │   ├── task_validation.py            # Post-inference output validation
│   │   ├── tasks.py                      # Task types + arbitration + output schemas
│   │   ├── types.py                      # Core data types (Build, Benchmark)
│   │   └── ...
│   ├── experiments/                      # Experiment + run tracking
│   ├── loaders/                          # ONNX model loading
│   ├── profiling/                        # Layer-by-layer profiling + cold start
│   ├── registry/
│   │   ├── local.py                      # SQLite registry (WAL mode)
│   │   └── schema.py                     # Schema + migrations (v6)
│   ├── storage/                          # S3-compatible remote storage
│   └── visualization/                    # Chart generation
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
| Task-aware synthetic inputs | No | No | No | **Auto-detected** |
| CI regression gate | No | No | No | **Built-in** |
| Size regression detection | No | No | No | **Built-in** |
| Quantization analysis | No | No | No | **Built-in** |
| Per-layer deep profiling | No | No | No | **Built-in** |
| Import pre-built models | No | No | No | **Built-in** |
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