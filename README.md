# MLBuild

<div align="center">

**Performance CI/CD for Production ML Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/mlbuild.svg)](https://pypi.org/project/MLBuild/)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://github.com/AbdoulayeSeydi/mlbuild)

MLBuild is the missing performance layer for ML CI/CD. While MLflow, DVC, and W&B track training experiments, MLBuild enforces production SLAs -- automatically benchmarking inference performance, validating against thresholds, and blocking regressions in CI.

[Installation](#installation) - [Quick Start](#quick-start) - [Documentation](#documentation) - [Roadmap](#roadmap)

</div>

---

## Current Status

| Feature | Status |
|---------|--------|
| Input formats | ONNX only |
| Backends | CoreML, ONNX Runtime |
| Storage | Local only |
| Targets | Apple Silicon, A-series chips |
| Platform | macOS only |

> Cloud storage, Android, TFLite, TensorRT, and Qualcomm QNN support are on the [roadmap](#roadmap).

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
P95:      12ms --> 25ms   (108% worse)

# Nobody caught it until users complained
```

**The gap:** Existing tools don't validate production performance in CI.

---

## The Solution

```bash
# Add one step to your CI pipeline
mlbuild build --model model.onnx --target apple_m1
mlbuild compare baseline candidate --threshold 5 --ci

# Output:
# REGRESSION DETECTED: 12% slower (threshold: 5%)
# Build FAILED
# PR blocked
```

Catch performance regressions before they reach production.

---

## What MLBuild Does

| Feature | MLflow / W&B / DVC | MLBuild |
|---------|-------------------|---------|
| Track training experiments | Yes | Yes |
| Automated p50/p95/p99 benchmarking | Manual | Built-in |
| CI fails if >5% slower | Not native | One command |
| SLA enforcement (max latency 10ms) | DIY scripts | `--max-latency 10` |
| Layer-by-layer profiling | No | `--analyze-warmup` |

MLBuild complements your existing stack -- it doesn't replace it.

---

## Installation

```bash
pip install mlbuild
mlbuild doctor
```

---

## Quick Start

```bash
# 1. Build model
mlbuild build --model model.onnx --target apple_m1 --quantize fp16

# 2. Benchmark (automatic p50/p95/p99)
mlbuild benchmark <build-id>

# 3. Validate SLAs
mlbuild validate <build-id> --max-latency 10 --max-p95 15 --ci

# 4. Compare vs baseline
mlbuild compare baseline candidate --threshold 5 --ci

# 5. Tag for production
mlbuild tag create <build-id> production
```

### GitHub Actions Integration

Add to `.github/workflows/model-ci.yml`:

```yaml
- name: Performance Validation
  run: |
    pip install mlbuild

    # Build model
    mlbuild build --model models/model.onnx --target apple_m1
    BUILD_ID=$(mlbuild log --limit 1 --json | jq -r '.[0].build_id')

    # Validate SLAs
    mlbuild validate $BUILD_ID --max-p95 10 --ci

    # Compare vs production
    mlbuild compare production $BUILD_ID --threshold 5 --ci
```

See `.github/workflows/examples/` for complete examples.

---

## Documentation

### Core Commands

#### Build and Convert

```bash
mlbuild build --model model.onnx --target apple_m1 --quantize fp16 --name "v2.0"
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
mlbuild compare baseline candidate \
  --threshold 5 \
  --metric p95 \
  --ci
```

#### Profile Layers

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
mlbuild experiment create "hyperparameter-search"
mlbuild run start --experiment "hyperparameter-search"
mlbuild run log-param learning_rate 0.001
mlbuild run log-metric accuracy 0.95
mlbuild run end
```

#### Remote Storage

```bash
mlbuild remote add prod --backend local --path /storage --default
mlbuild push v1.0.0
mlbuild pull v1.0.0
mlbuild sync
```

---

## Use Cases

### Mobile ML Teams

**Problem:** Ship models to billions of devices. 5% slower = millions in battery and compute costs.

```bash
mlbuild build --model model.onnx --target apple_a17
mlbuild validate <build-id> --max-latency 8 --ci
# Blocks PR if too slow for iPhone 15
```

### Edge AI (Autonomous Vehicles, Drones)

**Problem:** Can't afford production slowdowns in safety-critical systems.

```bash
mlbuild compare baseline candidate --threshold 2 --ci
# Fails if >2% regression
```

### SLA-Critical APIs

**Problem:** Sub-10ms inference requirements for real-time applications.

```bash
mlbuild validate <build-id> --max-p99 10 --ci
# Enforces 99th percentile < 10ms
```

---

## Architecture

MLBuild fits into your existing ML stack:

```
Training Phase
├── Experiment Tracking:  MLflow / W&B / Neptune
└── Data Versioning:      DVC

              |
              v

Production Phase
├── Model Building:        MLBuild
├── Performance Validation: MLBuild  <-- new layer
└── Deployment:            GitHub Actions / K8s
```

MLBuild works WITH your existing tools, not against them.

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
# Runs 100 iterations with 20 warmup
# Calculates p50, p95, p99, mean, std
# Measures memory, latency, throughput
```

### 3. SLA Enforcement

```python
if p95_latency > max_p95:
    fail_build()  # Exit code 1

if regression > threshold:
    block_pr()    # Prevents merge
```

### 4. Regression Detection

```python
baseline_p95  = 8.2ms
candidate_p95 = 9.5ms
change = ((9.5 - 8.2) / 8.2) * 100  # 15.8%

if change > threshold:  # 5%
    fail_build()
```

---

## vs. Existing Tools

### vs. MLflow
- **MLflow:** Tracks training experiments, logs metrics manually
- **MLBuild:** Automates inference benchmarking, enforces SLAs in CI

Use both: Track experiments in MLflow, validate performance in MLBuild.

### vs. DVC
- **DVC:** Versions data and models, Git-native pipelines
- **MLBuild:** Benchmarks models, enforces performance thresholds

Use both: Version with DVC, validate with MLBuild.

### vs. Weights and Biases
- **W&B:** Beautiful dashboards, experiment collaboration
- **MLBuild:** Automated performance CI/CD, regression detection

Use both: Visualize in W&B, enforce in MLBuild.

---

## Features

### Build and Convert
- ONNX to CoreML conversion
- Quantization (FP32 / FP16 / INT8)
- Multi-target support (Apple Silicon, A-series)
- Deterministic builds (content-addressed)

### Performance Validation
- Automated p50/p95/p99 benchmarking
- SLA enforcement (`--max-latency`, `--max-memory`)
- Regression detection (`--threshold 5`)
- Layer-by-layer profiling

### Experiment Tracking
- MLflow-style run tracking
- Parameter and metric logging
- Build-to-experiment linkage

### Version Control
- Docker-style tags (v1.0.0, production)
- Build history and diffing
- Immutable artifacts

### Remote Storage
- Git-style push/pull
- Prefix resolution (1ddf07c1)
- Bidirectional sync
- Local filesystem (S3 on roadmap)

### CI/CD Integration
- GitHub Actions examples
- Exit codes for CI failure
- JSON output for automation
- Performance reports in PRs

---

## Roadmap

MLBuild is currently macOS-only with ONNX input and CoreML/ONNX Runtime backends. Here is what is coming next:

### Phase 1 -- More Runtimes
- **TFLite backend** -- support TensorFlow Lite models for mobile deployment

### Phase 2 -- Device-Connected Benchmarking
- **Xcode Instruments integration** -- benchmark directly on connected iOS devices with real hardware profiling
- **Android Studio integration** -- benchmark on connected Android devices using Android profiling tools

### Phase 3 -- More Backends
- **TensorRT** -- NVIDIA GPU inference optimization and benchmarking
- **Qualcomm QNN** -- Snapdragon NPU support for on-device AI

### Phase 4 -- Cloud
- **Remote storage** -- S3, GCS, Azure Blob for artifact storage and sync
- **Cloud benchmarking** -- run benchmarks on remote hardware without local setup

---

## Development

### Setup

```bash
git clone https://github.com/AbdoulayeSeydi/mlbuild.git
cd mlbuild
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Project Structure

```
mlbuild/
├── src/mlbuild/
│   ├── cli/                  # CLI commands
│   │   ├── commands/         # Individual command implementations
│   │   └── main.py           # CLI entry point
│   ├── core/                 # Core types and logic
│   │   ├── types.py          # Build, Benchmark, Run types
│   │   ├── hash_utils.py     # Content addressing
│   │   ├── environment.py    # Environment fingerprinting
│   │   └── validation.py     # Build validation
│   ├── backends/             # Model conversion backends
│   │   ├── coreml/           # CoreML exporter
│   │   └── onnxruntime/      # ONNX Runtime backend
│   ├── benchmark/            # Performance measurement
│   │   ├── runner.py         # Benchmark execution
│   │   └── device_runner.py  # Device-specific runners
│   ├── profiling/            # Layer-by-layer profiling
│   │   ├── layer_profiler.py
│   │   └── warmup_analyzer.py
│   ├── registry/             # SQLite registry
│   │   ├── local.py          # Local registry implementation
│   │   └── schema.py         # Database schema
│   ├── storage/              # Remote storage
│   │   ├── backend.py        # Storage backend interfaces
│   │   ├── local.py          # Local filesystem backend
│   │   └── config.py         # Remote configuration
│   ├── experiments/          # Experiment tracking
│   │   ├── experiment.py     # Experiment management
│   │   ├── manager.py        # Run management
│   │   └── active_run.py     # Active run context
│   ├── loaders/              # Model loaders
│   │   └── onnx_loader.py
│   └── visualization/        # Charts and visualization
│       └── charts.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_artifact_hash.py
│   └── conftest.py
├── .github/
│   └── workflows/
│       └── examples/         # CI/CD examples
├── pyproject.toml
└── README.md
```

---

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding standards
- PR process
- Testing requirements

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by:
- [MLflow](https://mlflow.org) - Experiment tracking
- [DVC](https://dvc.org) - Data versioning
- [Git](https://git-scm.com) - Content-addressed storage

---

## Contact

- Issues: [GitHub Issues](https://github.com/AbdoulayeSeydi/mlbuild/issues)
- Email: abdoulayeaseydi@gmail.com

---

<div align="center">
Built by <a href="https://github.com/AbdoulayeSeydi">Abdoulaye Seydi</a>
</div>