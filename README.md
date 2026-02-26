# MLBuild

**Performance CI/CD for Production ML Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

MLBuild is the missing performance layer for ML CI/CD. While MLflow, DVC, and W&B track training experiments, MLBuild enforces production SLAs—automatically benchmarking inference performance, validating against thresholds, and blocking regressions in CI.

---

## The Problem
```bash
# Your CI passes
pytest ✓
black --check ✓  
mypy ✓

# But in production
Latency: 8ms → 15ms (88% slower)
Memory: 50MB → 120MB (140% more)
P95: 12ms → 25ms (108% worse)

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

| Feature | MLflow/W&B/DVC | MLBuild |
|---------|----------------|---------|
| Track training experiments | Yes | Yes |
| Automated p50/p95/p99 benchmarking | Manual | Built-in |
| CI fails if >5% slower | Not native | One command |
| SLA enforcement (max latency 10ms) | DIY scripts | --max-latency 10 |
| Layer-by-layer profiling | No | --analyze-warmup |

MLBuild complements your existing stack—it doesn't replace it.

---

## Quick Start

### Installation
```bash
pip install mlbuild
mlbuild init
mlbuild doctor
```

### Basic Usage
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
    mlbuild init
    
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

**Build & Convert**
```bash
mlbuild build --model model.onnx --target apple_m1 --quantize fp16 --name "v2.0"
```

**Benchmark**
```bash
mlbuild benchmark <build-id> --runs 100 --warmup 20 --json
```

**Validate SLAs**
```bash
mlbuild validate <build-id> \
  --max-latency 10 \
  --max-p95 15 \
  --max-memory 100 \
  --ci  # Fails build if SLA violated
```

**Compare & Detect Regressions**
```bash
mlbuild compare baseline candidate \
  --threshold 5 \
  --metric p95 \
  --ci  # Fails if >5% slower
```

**Profile Layers**
```bash
mlbuild profile <build-id> --top 15 --analyze-warmup
```

**Version Management**
```bash
mlbuild log --limit 20
mlbuild diff build-a build-b
mlbuild tag create <build-id> v1.0.0
```

**Experiment Tracking**
```bash
mlbuild experiment create "hyperparameter-search"
mlbuild run start --experiment "hyperparameter-search"
mlbuild run log-param learning_rate 0.001
mlbuild run log-metric accuracy 0.95
mlbuild run end
```

**Remote Storage**
```bash
mlbuild remote add prod --backend local --path /storage --default
mlbuild push v1.0.0
mlbuild pull v1.0.0
mlbuild sync
```

---

## Use Cases

### 1. Mobile ML Teams
**Problem:** Ship models to billions of devices. 5% slower = millions in battery/compute costs.

**Solution:**
```bash
mlbuild build --model model.onnx --target apple_a17
mlbuild validate <build-id> --max-latency 8 --ci
# Blocks PR if too slow for iPhone 15
```

### 2. Edge AI (Autonomous Vehicles, Drones)
**Problem:** Can't afford production slowdowns in safety-critical systems.

**Solution:**
```bash
mlbuild compare baseline candidate --threshold 2 --ci
# Fails if >2% regression
```

### 3. SLA-Critical APIs
**Problem:** Sub-10ms inference requirements for real-time applications.

**Solution:**
```bash
mlbuild validate <build-id> --max-p99 10 --ci
# Enforces 99th percentile < 10ms
```

---

## Architecture

MLBuild fits into your existing ML stack:
```
Training Phase
├── Experiment Tracking: MLflow / W&B / Neptune
└── Data Versioning: DVC

                ↓

Production Phase
├── Model Building: MLBuild
├── Performance Validation: MLBuild (NEW LAYER)
└── Deployment: GitHub Actions / K8s
```

MLBuild works WITH your existing tools, not against them.

---

## How It Works

### 1. Deterministic Builds
```python
# Content-addressed storage (Git-style)
build_id = sha256(source_hash + config_hash + env_fingerprint)

# Same inputs → Same output (byte-for-byte)
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
baseline_p95 = 8.2ms
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

**Use both:** Track experiments in MLflow, validate performance in MLBuild

### vs. DVC
- **DVC:** Versions data/models, Git-native pipelines
- **MLBuild:** Benchmarks models, enforces performance thresholds

**Use both:** Version with DVC, validate with MLBuild

### vs. Weights & Biases
- **W&B:** Beautiful dashboards, experiment collaboration
- **MLBuild:** Automated performance CI/CD, regression detection

**Use both:** Visualize in W&B, enforce in MLBuild

---

## Features

### Build & Convert
- ONNX to CoreML conversion
- Quantization (FP32/FP16/INT8)
- Multi-target support (Apple Silicon, A-series)
- Deterministic builds (content-addressed)

### Performance Validation
- Automated p50/p95/p99 benchmarking
- SLA enforcement (--max-latency, --max-memory)
- Regression detection (--threshold 5)
- Layer-by-layer profiling

### Experiment Tracking
- MLflow-style run tracking
- Parameter & metric logging
- Build-to-experiment linkage

### Version Control
- Docker-style tags (v1.0.0, production)
- Build history & diffing
- Immutable artifacts

### Remote Storage
- Git-style push/pull
- Prefix resolution (1ddf07c1)
- Bidirectional sync
- S3-ready architecture

### CI/CD Integration
- GitHub Actions examples
- Exit codes for CI failure
- JSON output for automation
- Performance reports in PRs

---

## Development

### Setup
```bash
git clone https://github.com/yourusername/mlbuild.git
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
│   ├── cli/              # CLI commands
│   │   ├── commands/     # Individual command implementations
│   │   └── main.py       # CLI entry point
│   ├── core/             # Core types & logic
│   │   ├── types.py      # Build, Benchmark, Run types
│   │   ├── hash_utils.py # Content addressing
│   │   ├── environment.py # Environment fingerprinting
│   │   └── validation.py # Build validation
│   ├── backends/         # Model conversion backends
│   │   ├── coreml/       # CoreML exporter
│   │   └── onnxruntime/  # ONNX Runtime backend
│   ├── benchmark/        # Performance measurement
│   │   ├── runner.py     # Benchmark execution
│   │   └── device_runner.py # Device-specific runners
│   ├── profiling/        # Layer-by-layer profiling
│   │   ├── layer_profiler.py
│   │   └── warmup_analyzer.py
│   ├── registry/         # SQLite registry
│   │   ├── local.py      # Local registry implementation
│   │   └── schema.py     # Database schema
│   ├── storage/          # Remote storage
│   │   ├── backend.py    # Storage backend interfaces
│   │   ├── local.py      # Local filesystem backend
│   │   └── config.py     # Remote configuration
│   ├── experiments/      # Experiment tracking
│   │   ├── experiment.py # Experiment management
│   │   ├── manager.py    # Run management
│   │   └── active_run.py # Active run context
│   ├── loaders/          # Model loaders
│   │   └── onnx_loader.py
│   └── visualization/    # Charts & visualization
│       └── charts.py
├── tests/                # Test suite
│   ├── test_benchmark.py
│   ├── test_artifact_hash.py
│   └── conftest.py
├── .github/
│   └── workflows/
│       └── examples/     # CI/CD examples
├── pyproject.toml        # Project metadata
└── README.md             # This file
```

---

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for:
- Development setup
- Coding standards
- PR process
- Testing requirements

---

## License

MIT License - see LICENSE for details.

---

## Acknowledgments

Inspired by:
- MLflow - Experiment tracking
- DVC - Data versioning
- Git - Content-addressed storage

---

## Contact

- Documentation: Coming soon
- Issues: GitHub Issues
- Email: abdoulayeaseydi@gmail.com

---

Built by Abdoulaye Seydi
```

---

# Updated LICENSE
```
MIT License

Copyright (c) 2026 Abdoulaye Seydi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.