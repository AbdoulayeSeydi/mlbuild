# MLBuild

<div align="center">

<img src="assets/mlbuild_logo.png" alt="MLBuild Logo" width="120" /><br/><br/>

**Performance CI/CD for On-Device Production ML Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/mlbuild.svg)](https://pypi.org/project/MLBuild/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/AbdoulayeSeydi/mlbuild)

MLBuild is the missing performance layer for on-device ML CI/CD. While MLflow, DVC, and W&B track training experiments, MLBuild enforces production SLAs — automatically benchmarking inference performance, validating against thresholds, blocking regressions in CI, and generating deployment-ready reports.

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
| Platform | macOS, Linux (TFLite) |
| Command history | Local, searchable, filterable by every command |
| Performance budget | Persistent constraints in .mlbuild/budget.toml |
| Baseline management | Reserved tag with clean CLI |
| Workspace status | Quick health snapshot |
| Build inspection | Single-build deep-dive via `mlbuild inspect` |
| Registry pruning | Remove old builds via `mlbuild prune` with dry-run safety |
| Build renaming | Rename builds in-place via `mlbuild rename` |
| Build pinning | Protect builds from pruning via `mlbuild pin` / `mlbuild unpin` |
| Registry search | Fuzzy search with filters via `mlbuild search` |
| Model conversion | PyTorch → ONNX / CoreML / TFLite via `mlbuild convert` |
| Android ADB bridge | USB device-connected benchmarking (Android arm64) |

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
# Tag your main branch baseline once
mlbuild tag create <build_id> main-mobilenet

# Add one step to your CI pipeline
mlbuild ci --model model.onnx --baseline main-mobilenet

# Output:
# MLBuild CI Report
# ──────────────────────────────────────────────────
# Model:     mobilenet
# Baseline:  3f36810e (main-mobilenet)
# Candidate: b8aa1ef6 (fp16)
#
#                      Baseline     Candidate       Delta
# Latency (p50)         2.49 ms       0.74 ms     -70.27%
# Size                 13.39 MB       6.74 MB     -49.64%
#
# Result: ✓ PASS
# Exit code: 0

# Or use the low-level gate directly
mlbuild ci-check $BASELINE_ID $CANDIDATE_ID --latency-threshold 10
# Exit code: 1 — PR blocked on regression
```

Catch latency AND size regressions before they reach production.

---

## Where MLBuild Fits

MLBuild is the missing on-device performance layer in your ML CI/CD stack.
```
┌─────────────────────────────────────────────────────────────────┐
│  ML Training                                                    │
│  ├── Experiment Tracking ──────────────── MLflow / W&B         │
│  └── Data Versioning ──────────────────── DVC                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  On-Device Optimization              MLBuild                    │
│  ├── Model Packaging ──────────────── mlbuild build             │
│  ├── Model Import ─────────────────── mlbuild import      
│  ├── Model Conversion ──────────────── mlbuild convert          │
│  ├── Task Detection ───────────────── automatic                 │
│  ├── Performance Validation ───────── mlbuild benchmark         │
│  ├── Quantization Benchmarking ────── mlbuild compare-quant     │
│  └── Reporting ────────────────────── mlbuild report            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Regression Gate                     MLBuild CI                 │
│  ✕  Bad performance → blocks deployment                        │
│  ├── CI Performance Gate ─────────── mlbuild ci-check          │
│  └── Full CI Orchestration ───────── mlbuild ci               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Deployment                                                     │
│  └── Release & Ship ───────────────── GitHub Actions / K8s     │
└─────────────────────────────────────────────────────────────────┘
```



| Feature | MLflow / W&B / DVC | MLBuild |
|---------|-------------------|---------|
| Track training experiments | Yes | No (use MLflow) |
| Automated p50/p95/p99 benchmarking | Manual | Built-in |
| CI fails on latency regression | Not native | `mlbuild ci-check` |
| CI fails on model size regression | Not native | `--size-threshold` |
| Task-aware synthetic inputs | No | **Auto-detected** |
| NLP multi-seq-len benchmarking | No | **Built-in** |
| Optimization sweep (fp16 + int8) | No | `mlbuild explore` |
| Static INT8 with calibration data | No | `--calibration-data` |
| Magnitude pruning (ONNX + CoreML) | No | `mlbuild optimize --pass prune` |
| Output divergence checking | No | `mlbuild accuracy` |
| Optimization chain visualization | No | `mlbuild log --tree` |
| Quantization tradeoff analysis | No | `mlbuild compare-quantization` |
| Performance reports | No | `mlbuild report` |
| S3-compatible remote storage | No | Built-in |
| TFLite benchmarking | No | Built-in |
| Import pre-built models | No | `mlbuild import` |
| Build export | JSON and CSV export via `mlbuild export` with multi-table directory mode |
| Self-upgrade | `mlbuild upgrade` — updates from PyPI with version comparison |

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

For macOS (CoreML + TFLite full stack):
```bash
pip install "mlbuild[macos]"
```

For Linux / CI (TFLite only, no CoreML):
```bash
pip install "mlbuild[linux]"
```

---

## Quick Start
```bash
# 1. Build and convert model
mlbuild build --model model.onnx --target apple_m1 --quantize fp16

# 1b. Or import a pre-built model
mlbuild import --model model.tflite --target android_arm64
mlbuild import --model model.mlpackage --target apple_m1 --quantize fp16

# 1c. Or benchmark directly on a connected Android device
mlbuild build --model model.onnx --target device-connected --backend tflite
mlbuild benchmark <build-id>   # auto-routes to ADB pipeline

# 2. Benchmark (automatic p50/p95/p99, task auto-detected)
mlbuild benchmark <build-id>

# 3. Sweep all optimization variants automatically
mlbuild explore model.onnx --target apple_m1

# 4. Check output divergence between variants
mlbuild accuracy <baseline-id> <candidate-id>

# 5. Validate SLAs (performance + accuracy in one command)
mlbuild validate <build-id> --max-latency 10 --dataset ./imagenet-mini/

# 6. Run full CI check against registered baseline
mlbuild ci --model model.onnx --baseline main-mobilenet

# 6b. Or use low-level compare
mlbuild compare baseline candidate --threshold 5 --check-accuracy --ci

# 7. View full optimization lineage
mlbuild log --source model.onnx --tree

# 8. Generate performance report
mlbuild report <build-id> --open

# 9. Tag for production
mlbuild tag create <build-id> production
```

### GitHub Actions Integration
```yaml
- name: MLBuild CI
  run: |
    pip install mlbuild

    # Full CI check — explore, compare, report in one command
    mlbuild ci \
      --model models/mobilenet.onnx \
      --baseline main-mobilenet \
      --latency-regression 15 \
      --size-regression 10

- name: Upload CI report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: mlbuild-report
    path: .mlbuild/ci_report.json
```

See `.github/workflows/mlbuild.yml` for a complete example with PR comment posting.

---

## Documentation

### Core Commands

#### Build and Convert
```bash
mlbuild build --model model.onnx --target apple_m1 --quantize fp16 --name "v2.0"
mlbuild build --model model.onnx --target android_arm64 --quantize int8
```

---

#### Convert Models

Convert raw PyTorch, ONNX, or TF SavedModel files to MLBuild-compatible formats. Once converted, the output is registered in the local registry and available to all MLBuild commands immediately.
```bash
# PyTorch → ONNX
mlbuild convert --model ./model.pt --to onnx

# PyTorch → CoreML (--target required)
mlbuild convert --model ./model.pt --to coreml --target apple_m2

# PyTorch → TFLite (chained via ONNX automatically)
mlbuild convert --model ./model.pt --to tflite

# TF SavedModel → TFLite
mlbuild convert --model ./saved_model/ --to tflite

# With quantization
mlbuild convert --model ./model.pt --to coreml --target apple_a17 --quantize fp16

# Preview conversion plan without executing
mlbuild convert --model ./model.pt --to tflite --dry-run

# Keep intermediate artifacts (e.g. ONNX on the way to TFLite)
mlbuild convert --model ./model.pt --to tflite --keep-intermediate
```

**Supported conversion paths:**

| Input | Output | Via |
|---|---|---|
| `.pt` / `.pth` | `onnx` | `torch.onnx.export` |
| `.pt` / `.pth` | `coreml` | `torch.jit.trace` → `coremltools` |
| `.pt` / `.pth` | `tflite` | ONNX intermediate → `onnx2tf` |
| `.onnx` | `coreml` | `coremltools` |
| `.onnx` | `tflite` | `onnx2tf` |
| `saved_model/` | `tflite` | `tf.lite.TFLiteConverter` |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--to` | required | Output format: `onnx`, `coreml`, `tflite` |
| `--target` | required for CoreML | `apple_m1` \| `apple_m2` \| `apple_m3` \| `apple_a15` \| `apple_a16` \| `apple_a17` \| `apple_a18` |
> **Targets specify a minimum OS deployment version, not a specific chip.** Multiple Apple chips share the same target. Models using newer features (e.g. int4 quantization) require higher OS targets regardless of chip selection — MLBuild will error early if your model and target are incompatible.
| `--input-shape` | `1,3,224,224` | Comma-separated input shape for tracing |
| `--quantize` | `fp32` | `fp32` or `fp16` — int8 requires a representative dataset and is not yet supported |
| `--load-mode` | `auto` | `auto`, `jit`, `eager` — controls how PyTorch model is loaded |
| `--opset` | auto | ONNX opset version override (default tries 17→16→15→14→13→12) |
| `--keep-intermediate` | off | Register intermediate artifacts (e.g. ONNX) alongside final output |
| `--no-register` | off | Convert without adding to MLBuild registry |
| `--dry-run` | off | Print conversion plan without executing |
| `--debug` | off | Preserve temp files even on success |
| `--timeout` | `300` | Per-step timeout in seconds |

> **PyTorch model format requirement:** `mlbuild convert` requires PyTorch models to be saved as TorchScript. Plain `torch.save(model, path)` on a class defined interactively will fail with a pickle error when MLBuild tries to load it in a different context.
>
> Save your model as TorchScript before converting:
> ```python
> # Recommended — trace-based (works for most vision models)
> traced = torch.jit.trace(model, torch.zeros(1, 3, 224, 224))
> traced.save("model.pt")
>
> # Alternative — script-based (required for models with control flow)
> scripted = torch.jit.script(model)
> scripted.save("model.pt")
> ```
> Models saved from a proper `.py` file (not an interactive session) can also be loaded with `--load-mode eager`.

---


#### Import Pre-built Models

Register an existing TFLite or CoreML model directly — no conversion required. Once imported, all MLBuild commands (benchmark, profile, compare, report, ci-check) work on it immediately.
```bash
# Import a TFLite model
mlbuild import --model model.tflite --target android_arm64

# Import a CoreML model
mlbuild import --model model.mlpackage --target apple_m1

# Import an ONNX model (benchmarked via ONNX Runtime)
mlbuild import --model model.onnx --target onnxruntime_cpu
mlbuild import --model model.onnx --target onnxruntime_gpu

# Import with metadata
mlbuild import --model model.tflite --target android_arm64 \
  --quantize int8 \
  --name "vendor-v2" \
  --notes "Supplied by vendor, int8 quantized"

# JSON output (for CI pipelines)
mlbuild import --model model.tflite --target android_arm64 --json
```

**Supported formats:**
- `.onnx` — validated via protobuf check, runs via ONNX Runtime
- `.tflite` — validated via FlatBuffer magic bytes (TFL3/TFL2)
- `.mlpackage` — validated via Manifest.json + Data/ structure
- `.mlmodel` — legacy CoreML flat file

**Format/target compatibility:**

**Format/target compatibility:**

| Format | Valid Targets |
|--------|--------------|
| `onnx` | `onnxruntime_cpu`, `onnxruntime_gpu`, `onnxruntime_ane` |
| `tflite` | `android_arm64`, `android_arm32`, `android_x86`, `raspberry_pi`, `coral_tpu`, `generic_linux`, `device-connected` |
| `coreml` | `apple_m1`, `apple_m2`, `apple_m3`, `apple_a15`, `apple_a16`, `apple_a17`, `apple_a18`, `device-connected` |

Imported builds are marked `[imported]` in `mlbuild log` output and tracked with `"imported": true` in their metadata.

---

#### Optimize

Generate optimized variants of a registered build. Supports quantization and magnitude pruning. All variants are registered as children of the source build with full lineage tracking.

##### Quantization
```bash
# FP16 — recompiles from ONNX graph (lower precision weights)
mlbuild optimize <build_id> --pass quantize --method fp16

# Dynamic range INT8 — weight-only, no calibration data needed
mlbuild optimize <build_id> --pass quantize --method int8

# Static INT8 — quantizes weights + activations using calibration data
mlbuild optimize <build_id> --pass quantize --method int8 \
  --calibration-data ./imagenet-mini/
```

**Calibration data formats for static INT8:**
- Directory of images (`.jpg`, `.png`, `.bmp`, `.webp`) — auto-resized to model input shape, normalized to [0, 1]
- Directory of `.npy` files — one array per sample
- Single `.npz` file — named array, first axis = samples

Static and dynamic INT8 are stored as distinct builds (`int8` vs `int8_static`) — both can coexist in the registry with different build IDs.

> **Note:** Full static INT8 (weight + activation quantization) requires coremltools 9.1+. On 9.0, MLBuild automatically falls back to dynamic range INT8 with a clear warning — no crash, no silent misbehavior.

##### Pruning

Magnitude-based unstructured weight pruning. Zeros out the smallest weights by absolute value up to a target sparsity level. No retraining required.
```bash
# 50% sparsity
mlbuild optimize <build_id> --pass prune --sparsity 0.5

# 75% sparsity
mlbuild optimize <build_id> --pass prune --sparsity 0.75
```

**Routing logic:**
- `has_graph=True` → ONNX magnitude pruning → re-convert via existing build pipeline (works for CoreML **and** TFLite)
- `has_graph=False + coreml` → CT9 `OpMagnitudePrunerConfig` post-hoc on compiled `.mlpackage`
- `has_graph=False + tflite` → Error with actionable message (`Re-register using 'mlbuild build' or 'mlbuild import --graph model.onnx'`)

Pruning skips bias, batch norm, and small tensors (< 256 params) automatically. Sparsity level is baked into the method name (`prune_0.50`), so each level gets a distinct build ID.

##### Method chaining

Pruning and quantization can be chained arbitrarily:
```bash
# Prune first, then quantize
mlbuild optimize <build_id> --pass prune --sparsity 0.5
mlbuild optimize <pruned_build_id> --pass quantize --method int8
```

---

#### Explore

Sweeps all optimization variants for a model in one command. Builds the fp32 baseline, generates fp16 and int8 variants, benchmarks all of them, and assigns verdicts.
```bash
# Full sweep (fp16 + int8, all backends)
mlbuild explore model.onnx --target apple_m1

# Fast mode (fp16 only, 20 benchmark runs)
mlbuild explore model.onnx --target apple_m1 --fast

# Specific backends
mlbuild explore model.onnx --backends coreml
mlbuild explore model.onnx --backends coreml,tflite

# With static INT8 calibration data
mlbuild explore model.onnx --calibration-data ./imagenet-mini/

# With output divergence check per variant
mlbuild explore model.onnx --check-accuracy --cosine-threshold 0.99

# JSON output
mlbuild explore model.onnx --output-json
```

**Verdict logic (score-based):**
```
score = 0.6 × (baseline_latency / variant_latency)
      + 0.4 × (baseline_size / variant_size)

score > 1.0  → candidate for recommended or aggressive
score ≤ 1.0  → skip (strictly worse on both axes)
```

- `recommended` — highest composite score (best balance of speed + size)
- `aggressive` — smallest size among remaining candidates
- `skip` — no improvement, or accuracy check failed
- `baseline` — fp32 reference
```
COREML
  Verdict       Method         Size      p50 Latency   vs Baseline    Accuracy
  baseline      fp32           13.39 MB  3.29ms        —              —
  aggressive    fp16            6.74 MB  3.29ms        ↑0% lat        —
                                                       ↓50% size
  recommended   int8(static)    3.58 MB  2.81ms        ↓14% lat       ✓ 0.9999
                                                       ↓73% size
```

---

#### Accuracy

Standalone output divergence check between two builds. Runs inference on both with synthetic inputs and computes similarity metrics.
```bash
mlbuild accuracy <baseline_id> <candidate_id>
mlbuild accuracy <baseline_id> <candidate_id> --samples 64 --seed 42
mlbuild accuracy <baseline_id> <candidate_id> \
  --cosine-threshold 0.99 \
  --top1-threshold 0.99
```

**Metrics:**
- **Cosine similarity** — angle between output vectors (1.0 = identical direction)
- **Mean absolute error** — average per-element absolute difference
- **Max absolute error** — worst-case per-element difference
- **Top-1 agreement** — fraction of samples where both models pick the same top class (classifiers only)

Results are persisted to the registry's `accuracy_checks` table.

**Example results on MobileNet:**
```
fp32 → fp16:  cosine=0.9999  top1=1.00   passed=True
fp32 → int8:  cosine=0.9983  top1=0.97   passed=False (< 0.99 threshold)
```

---

#### Benchmark
```bash
mlbuild benchmark <build-id> --runs 100 --warmup 20 --json
mlbuild benchmark <build-id> --compute-unit CPU_ONLY
```

---

#### Validate SLAs

Validates a build against performance and accuracy constraints. All checks compose in a single command.
```bash
# Performance constraints only
mlbuild validate <build_id> --max-latency 10 --max-size 8

# Accuracy constraint with dataset
mlbuild validate <build_id> \
  --dataset ./imagenet-mini/ \
  --cosine-threshold 0.99 \
  --top1-threshold 0.99

# All checks composed
mlbuild validate <build_id> \
  --max-latency 10 \
  --max-size 8 \
  --dataset ./imagenet-mini/ \
  --cosine-threshold 0.99

# CI mode (suppress output, exit codes only)
mlbuild validate <build_id> --max-latency 5 --ci
```

**Options:**
- `--max-latency` — maximum p50 latency in ms
- `--max-p95` — maximum p95 latency in ms
- `--max-memory` — maximum peak memory in MB
- `--max-size` — maximum model size in MB
- `--dataset` — calibration data for accuracy check (images dir, `.npy` dir, or `.npz`)
- `--baseline-id` — reference build for accuracy comparison (default: root build)
- `--cosine-threshold` — minimum cosine similarity (default: 0.99)
- `--top1-threshold` — minimum top-1 agreement (default: 0.99)
- `--accuracy-samples` — max calibration samples (default: 200)

If `--dataset` is provided but the build is the root (no baseline to compare against), accuracy check is skipped with a message rather than erroring.

Exit codes: `0` = all constraints passed, `1` = one or more violations.

---

#### Compare and Detect Regressions
```bash
# Compare with independent latency + size thresholds
mlbuild compare baseline candidate \
  --threshold 5 \
  --size-threshold 10 \
  --metric p95 \
  --ci

# Use cached benchmark results (skip re-benchmarking)
mlbuild compare baseline candidate --use-cached

# Include output divergence check
mlbuild compare baseline candidate --check-accuracy

# Dedicated CI gate
mlbuild ci-check baseline candidate
mlbuild ci-check baseline candidate --latency-threshold 10 --size-threshold 5
mlbuild ci-check baseline candidate --strict   # any positive delta fails
mlbuild ci-check baseline candidate --json
```

**Exit codes:**
- `0` — no regression (safe to ship)
- `1` — regression detected (block the PR)
- `2` — error (infra failure, check logs)

---

#### CI Orchestration

Full CI check in one command — resolves baseline, explores variants, compares, enforces thresholds, and writes a structured report.
```bash
# Run full CI check against a tagged baseline
mlbuild ci --model mobilenet.onnx --baseline main-mobilenet

# Use an existing build (skips explore — useful when builds happen earlier in pipeline)
mlbuild ci --build <build_id> --baseline main-mobilenet

# With absolute budgets (independent of baseline)
mlbuild ci --model mobilenet.onnx --baseline main-mobilenet \
  --latency-budget 3.0 \
  --size-budget 10.0

# With accuracy gate
mlbuild ci --model mobilenet.onnx --baseline main-mobilenet \
  --dataset ./imagenet-mini/ \
  --cosine-threshold 0.99

# JSON output (for dashboards and GitHub bots)
mlbuild ci --build <build_id> --baseline main-mobilenet --json

# Fail if baseline tag not found (strict CI)
mlbuild ci --model mobilenet.onnx --baseline main-mobilenet --fail-on-missing-baseline
```

**Tagging baselines:**
```bash
# Tag a build as the main branch baseline
mlbuild tag create <build_id> main-mobilenet

# --baseline accepts tag names or build ID prefixes
mlbuild ci --baseline main-mobilenet   # tag lookup
mlbuild ci --baseline 3f36810e         # build ID prefix
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | ONNX model path (runs explore) | — |
| `--build` | Existing build ID (skips explore) | — |
| `--baseline` | Tag name or build ID | required |
| `--target` | Device target for explore | auto |
| `--latency-regression` | Max latency regression % | 10.0 |
| `--size-regression` | Max size regression % | 5.0 |
| `--latency-budget` | Hard latency cap in ms | none |
| `--size-budget` | Hard size cap in MB | none |
| `--dataset` | Calibration data for accuracy check | none |
| `--cosine-threshold` | Min cosine similarity | 0.99 |
| `--top1-threshold` | Min top-1 agreement | 0.99 |
| `--fail-on-missing-baseline` | Exit 1 if baseline not found | false |
| `--json` | Print JSON report to stdout | false |

**CI Report** — always written to `.mlbuild/ci_report.json`:
```json
{
  "model": "mobilenet.onnx",
  "baseline": {
    "tag": "main-mobilenet",
    "build_id": "3f36810e...",
    "latency_ms": 2.49,
    "size_mb": 13.39
  },
  "candidate": {
    "build_id": "b8aa1ef6...",
    "variant": "fp16",
    "parent_build_id": "3f36810e...",
    "latency_ms": 0.74,
    "size_mb": 6.74
  },
  "delta": { "latency_pct": -70.27, "size_pct": -49.64 },
  "thresholds": {
    "latency_regression_pct": 10.0,
    "size_regression_pct": 5.0,
    "latency_budget_ms": null,
    "size_budget_mb": null
  },
  "result": "pass",
  "violations": []
}
```

The report always stores `baseline.build_id` — even if the tag is later repointed, the report preserves exactly what was compared.

**Exit codes:** `0` = pass or skipped, `1` = regression/failure, `2` = error.

**Configuration** via `.mlbuild/config.toml`:
```toml
[ci]
latency_regression_pct = 10
size_regression_pct = 5
latency_budget_ms = 3.0
size_budget_mb = 10.0

[ci.accuracy]
cosine_threshold = 0.99
top1_threshold = 0.99
```

---

#### Quantization Tradeoff Analysis
```bash
mlbuild compare-quantization fp32-build int8-build
mlbuild compare-quantization fp32-build int8-build --accuracy-samples 100
mlbuild compare-quantization fp32-build int8-build --json
```

---

#### Performance Report
```bash
mlbuild report <build-id>
mlbuild report <build-id> --open
mlbuild report <build-id> --output report.html
mlbuild report <build-id> --format pdf        # requires: pip install weasyprint
```

---

#### Deep Profiling
```bash
# TFLite: full 6-feature deep profile (no device required)
mlbuild profile <build-id> --deep

# CoreML: cold start decomposition (all formats)
mlbuild profile <build-id> --deep

# Options
mlbuild profile <build-id> --deep --top 20
mlbuild profile <build-id> --deep --runs 100
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

---

#### Build History
```bash
# All builds
mlbuild log

# Specific build detail
mlbuild log <build_id>

# Filter by source model filename (substring match)
mlbuild log --source mobilenet.onnx

# Full optimization lineage tree (recursive parent-child)
mlbuild log --source mobilenet.onnx --tree

# Other filters
mlbuild log --name mobilenet
mlbuild log --format coreml
mlbuild log --task vision
mlbuild log --roots-only
mlbuild log --target apple_m1

# Export
mlbuild log --json
mlbuild log --csv builds.csv
```

The `--tree` flag renders the full optimization DAG using actual parent-child lineage. Method chaining (e.g. prune → int8) shows as nested children, not flat siblings — causality is preserved:
```
3f36810e  mobilenet  coreml  fp32  13.39 MB  2.49ms
├── b8aa1ef6  coreml  fp16  6.74 MB  0.74ms
├── 2921f0fa  coreml  int8  3.58 MB  3.15ms
├── 9df061cb  coreml  int8(static)  3.58 MB  2.81ms
├── 329f3b78  coreml  prune(0.50)  13.39 MB  3.89ms
│   └── 3fa93712  coreml  int8  3.58 MB  2.61ms
└── 0a17ce03  coreml  prune(0.75)  13.39 MB  2.94ms
```

Method labels are human-readable: `prune(0.50)`, `int8(static)` instead of raw internal strings.

---

#### Build Inspection

Full deep-dive on a single build. Shows metadata, artifacts, benchmarks, accuracy records, tags, and notes in one structured view.
```bash
# Full view
mlbuild inspect <build-id>

# Build + primary artifact + latest benchmark per compute unit
mlbuild inspect <build-id> --short

# JSON output (deterministic, sorted keys)
mlbuild inspect <build-id> --json
```

Output sections: Build → Task → Artifacts → Benchmarks → Accuracy → Tags → Notes. All sections always rendered — empty sections show `no records` rather than being omitted.

---

#### Registry Pruning

Remove old builds from the local registry. Supports soft-delete (hidden from queries) and hard delete with artifact file removal.
```bash
# Preview what would be pruned — never touches anything
mlbuild prune --keep-last 20 --dry-run
mlbuild prune --older-than 30d --dry-run

# Soft-delete (hidden from queries, files kept)
mlbuild prune --keep-last 20
mlbuild prune --older-than 30d
mlbuild prune --older-than 30d --tag experiment

# Hard delete rows + artifact files (irreversible)
mlbuild prune --older-than 30d --purge --force
mlbuild prune --keep-last 10 --purge --force

# Flags compose
mlbuild prune --older-than 30d --keep-last 5 --dry-run
```

Protected builds (`mlbuild-baseline`, `main-*`, `production-*` tags) are always skipped — even if explicitly targeted. `--keep-last` is enforced globally before any filters apply, guaranteeing your N newest builds always survive.

---

#### Build Renaming

Rename a build in-place without affecting benchmarks, tags, or history.
```bash
mlbuild rename <build_id> <new_name>
mlbuild rename 3fa93712 mobilenet-int8-prod        # prompts for confirmation
mlbuild rename 3fa93712 mobilenet-int8-prod --yes  # skip prompt
```

Names are mutable — build ID, artifact hash, and all linked data are unaffected.

---

#### Build Pinning

Pin a build to protect it from `mlbuild prune`. Pinned builds are always skipped, even with `--force` or `--purge`.
```bash
mlbuild pin <build_id>    # protect from pruning
mlbuild unpin <build_id>  # remove protection
```

Pin uses the reserved tag `mlbuild-pinned` under the hood. Pinned builds appear in `mlbuild prune --dry-run` output under "Skipped (protected)".

---

#### Registry Search

Search builds by name, format, target, task, or tag. Faster than `mlbuild log` for large registries — fuzzy name matching in one query.
```bash
# Fuzzy search by name
mlbuild search mobilenet

# Filter by format
mlbuild search mobilenet --format coreml

# Filter by target
mlbuild search mobilenet --target apple_m1

# Filter by task
mlbuild search --task vision

# Filter by tag
mlbuild search --tag mlbuild-baseline
mlbuild search --tag mlbuild-pinned

# Date range
mlbuild search mobilenet --date-from 2026-03-01 --date-to 2026-03-31

# Limit results
mlbuild search mobilenet --limit 10

# JSON output
mlbuild search mobilenet --json
```

Output includes: build ID, name, format, target, task, optimization method, p50, p95, peak memory, size, and tags per result.

---

#### Build Export

Export a build and all its data for use in external tools, pipelines, and dashboards.
```bash
# JSON to stdout (default)
mlbuild export <build-id>

# JSON to file
mlbuild export <build-id> --output out.json

# Flat CSV to stdout (one row per benchmark)
mlbuild export <build-id> --format csv

# Flat CSV to file
mlbuild export <build-id> --output out.csv

# Full multi-table export to directory
mlbuild export <build-id> --output out/
mlbuild export <build-id> --output out/ --force  # overwrite existing
```

JSON output is versioned (`"version": "1"`) and stable — suitable for piping into CI pipelines, dashboards, and the future cloud API. Directory mode writes five files: `build.csv`, `artifacts.csv`, `benchmarks.csv`, `accuracy.csv`, `tags.csv` — every table joinable by `build_id`. All exports are deterministic: sorted, UTF-8, ISO 8601 timestamps.

---

#### Command History

A permanent log of every MLBuild command ever run. Searchable, filterable, deletable.
```bash
# Show all recent commands
mlbuild history

# Filter by command type — every command is filterable
mlbuild history --filter build
mlbuild history --filter benchmark
mlbuild history --filter validate
mlbuild history --filter baseline
mlbuild history --filter budget
mlbuild history --filter status
mlbuild history --filter import
mlbuild history --filter compare
mlbuild history --filter failed
# ...and all other commands (accuracy, ci, diff, explore, optimize, profile, etc.)

# Filter by time
mlbuild history --since yesterday
mlbuild history --since "7 days ago"
mlbuild history --since "2024-01-01"

# Filter by build ID — everything that touched a specific build
mlbuild history --build-id a3f91c2

# Limit results
mlbuild history --limit 100

# Delete one entry by ID (min 4 chars)
mlbuild history delete d58cc62f

# Clear all history (prompts for confirmation)
mlbuild history clear
```

History is an audit log of CLI actions — separate from build and benchmark data. Deleting a history entry never touches builds or benchmarks.

---

#### Performance Budget

Persistent performance constraints committed to git. Set once, enforced automatically by `mlbuild validate` and `mlbuild ci`. Explicit flags always override budget values.
```bash
# Set constraints once
mlbuild budget set --max-latency 10 --max-p95 15 --max-size 8

# Show current budget
mlbuild budget show

# Preview what would apply to a build without benchmarking
mlbuild budget validate <build_id>

# Update one constraint without touching others
mlbuild budget set --max-latency 5

# Remove one constraint
mlbuild budget clear --constraint max-latency

# Remove all constraints (prompts for confirmation)
mlbuild budget clear

# After budget is set, validate uses it automatically
mlbuild validate <build_id>           ← uses budget
mlbuild validate <build_id> --max-latency 3  ← overrides latency, budget for rest
```

Budget is stored in `.mlbuild/budget.toml` — commit it so your whole team enforces the same constraints automatically.

**Merge priority:** explicit CLI flag > budget file > no constraint

**Violation output** shows the source of each constraint:
```
┃ Constraint     ┃   Limit ┃  Actual ┃     Violation       ┃        Source ┃
│ max_latency_ms │ 1.00 ms │ 2.66 ms │ +1.66 (166% over)   │ explicit flag │
│ max_size_mb    │ 8.00 MB │ 9.10 MB │ +1.10 (13.8% over)  │ budget        │
```

---

#### Baseline Management

Clean UX wrapper around `mlbuild tag`. Uses the reserved tag `mlbuild-baseline` so `mlbuild ci` resolves it automatically — zero CI changes required.
```bash
# Set a build as the performance baseline
mlbuild baseline set <build_id>

# Show current baseline
mlbuild baseline
mlbuild baseline show

# Show all baseline-style tags (mlbuild-baseline, main-*, production-*)
mlbuild baseline history

# Remove baseline (prompts for confirmation)
mlbuild baseline unset
```

The baseline integrates directly with `mlbuild ci`:
```bash
mlbuild ci --model model.onnx --baseline mlbuild-baseline
```

Prompts before overwriting an existing baseline. Use `--force` to skip the prompt.

---

#### Workspace Status

Quick health check of the current workspace. Reads from existing data — no new storage.
```bash
mlbuild status
mlbuild status --json
```

Output:
```
MLBuild Status  Abdoulayes-MacBook-Air.local

  ✓ Workspace    .mlbuild/
  ✓ Registry     26 builds  |  18 benchmarks
  Last build:  mobilenet (coreml, 3.58 MB) — 2h ago
  Last bench:  p50=2.61 ms — 2h ago

  ✓ Baseline     3fa9371209e6  mobilenet  2.61 ms  3.58 MB
  Last validate: PASSED — 52m ago

  ✓ Budget       .mlbuild/budget.toml
    Max latency (p50)    10.0 ms
    Max size             8.0 MB
```
---

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

**Supported backends:** AWS S3, Cloudflare R2 (recommended — free 10 GB), Backblaze B2, any S3-compatible storage.

---

## Task-Aware Benchmarking

MLBuild automatically detects what kind of model you're benchmarking — vision, NLP, or audio — and generates semantically correct synthetic inputs for it. No dummy zero arrays, no manual shape specification.

### Supported Model Subtypes

MLBuild works across the full production model landscape — not just image classifiers.

| Domain | Subtypes |
|--------|----------|
| **Vision** | Image classification, object detection (with/without NMS in graph), segmentation, vision embedding (CLIP-style) |
| **NLP** | Text classification, token classification, embedding, question answering, seq2seq |
| **Audio** | Speech recognition, audio classification, speaker embedding, waveform and spectrogram inputs |
| **Multimodal** | Dual-encoder (image + text), asymmetric inputs, vision-language hybrids |
| **Timeseries** | LSTM/GRU, TCN, PatchTST-style dynamic sequence, stateful models with explicit h0/c0 |
| **Recommendation** | Neural matrix factorization, two-tower, large Gather (vocab 100k+) |
| **Generative** | Autoregressive (GPT-style), KV-cache stateful, single-pass approximation |

MLBuild handles the real production edge cases:
- Dynamic input shapes (batch, sequence length, spatial dims)
- Multi-input/output graphs
- Pre/post-processing baked into the graph
- FP32, FP16, and INT8 quantized variants
- Large models (100MB+)
- Stateful models (h0/c0, KV-cache, rolling context)
- NMS inside vs. outside the graph

Use `--force-domain`, `--force-subtype`, and `--force-execution` to override auto-detection when needed.

### Automatic Task Detection

Detection runs through three tiers in order of confidence:

| Tier | Method | Formats | Confidence | CLI Behavior |
|------|--------|---------|------------|--------------|
| **Graph** | Op/layer analysis (`Conv`, `Attention`, `STFT`, etc.) | ONNX, TFLite, CoreML | High | Silent |
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
mlbuild benchmark <build-id>
```

### Override with `--task`
```bash
mlbuild benchmark <build-id> --task vision
mlbuild benchmark <build-id> --task nlp
mlbuild benchmark <build-id> --task audio

mlbuild profile  <build-id> --task nlp
mlbuild validate <build-id> --task vision --strict-output
```

### Task-Specific Synthetic Inputs

| Task | Inputs Generated |
|------|-----------------|
| **Vision** | Float32 image tensor, NCHW layout, spatial dims resolved to 224×224 |
| **NLP** | `int64` token IDs (random vocab up to 30k), `int64` attention mask (all ones), token type IDs |
| **Audio** | Float32 waveform `[-1, 1]` or log-mel spectrogram — role inferred from tensor name/shape |
| **Unknown** | Zero tensors — safe fallback that never blocks CI |

### NLP Multi-Sequence Benchmarking

NLP models are benchmarked across a sequence length ladder by default:
```bash
# Default ladder: [16, 64, 128, 256]
mlbuild benchmark <build-id> --task nlp

# seq_len=16   p50=1.2ms  p95=1.4ms
# seq_len=64   p50=2.1ms  p95=2.4ms
# seq_len=128  p50=3.8ms  p95=4.2ms
# seq_len=256  p50=7.1ms  p95=8.0ms

# Clip to model's actual max sequence length
mlbuild benchmark <build-id> --task nlp --seq-len 128
```

### Strict Output Validation
```bash
# Soft mode (default) — warns but proceeds
mlbuild benchmark <build-id> --task nlp

# Strict mode — exits non-zero on output anomaly
mlbuild benchmark <build-id> --task nlp --strict-output
mlbuild validate  <build-id> --task vision --strict-output

# Global strict mode — applies to all commands
mlbuild --strict-output benchmark <build-id> --task nlp
```

---

## Optimization Workflow

A complete optimization workflow from ONNX to deployment-ready model:
```bash
# 1. Build FP32 baseline
mlbuild build --model mobilenet.onnx --target apple_m1 --name mobilenet

# 2. Sweep all variants automatically
mlbuild explore mobilenet.onnx --target apple_m1 --check-accuracy

# 3. Prune best variant and quantize the result
mlbuild optimize <fp32_id> --pass prune --sparsity 0.5
mlbuild optimize <pruned_id> --pass quantize --method int8

# 4. Validate final model against SLAs
mlbuild validate <final_id> \
  --max-latency 5 \
  --max-size 6 \
  --dataset ./imagenet-mini/

# 5. View full lineage
mlbuild log --source mobilenet.onnx --tree

# 6. Tag for production
mlbuild tag create <final_id> production-v2
```

---

## CI/CD Regression Gate
```bash
# Full CI orchestration (recommended)
mlbuild ci --model mobilenet.onnx --baseline main-mobilenet
echo "Exit: $?"   # 0 = pass, 1 = fail, 2 = error

# Low-level build-to-build comparison
mlbuild ci-check $BASELINE_ID $CANDIDATE_ID
echo "Exit: $?"   # 0 = pass, 1 = regression, 2 = error

# JSON output for dashboards and PR bots
mlbuild ci --build $BUILD_ID --baseline main-mobilenet --json
# {
#   "result": "pass",
#   "baseline": { "tag": "main-mobilenet", "build_id": "3f36810e...", "latency_ms": 2.49 },
#   "candidate": { "build_id": "b8aa1ef6...", "variant": "fp16", "latency_ms": 0.74 },
#   "delta": { "latency_pct": -70.27, "size_pct": -49.64 },
#   "violations": []
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
├── Optimization Sweep:     MLBuild explore         ← fp16 + int8 + pruning
├── Accuracy Validation:    MLBuild accuracy        ← output divergence
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

### 2. Build Lineage Tracking

Every variant stores its full ancestry:
```python
build.parent_build_id      # direct parent
build.root_build_id        # original source in the chain
build.optimization_method  # "fp16", "int8", "int8_static", "prune_0.50"
```

Identical optimization chains always produce the same build ID — deduplication is automatic.

### 3. Automated Benchmarking
```python
# Runs N iterations with warmup
# Calculates p50, p95, p99, mean, std
# Measures memory RSS delta, throughput
# Outlier trimming (top/bottom 5%)
```

### 4. Task-Aware Input Generation
```python
# Three-tier detection: graph ops → tensor names → shapes
# Task-specific synthetic inputs (never zeros for known tasks)
# NLP: multi-seq-len ladder [16, 64, 128, 256]
# Post-inference output validation with configurable strictness
```

### 5. Output Divergence Checking
```python
# Cosine similarity — output direction preservation
# MAE / max absolute error — per-element differences
# Top-1 agreement — classifier label consistency
# Streaming accumulators — memory-efficient over large batches
# Results persisted to accuracy_checks registry table
```

### 6. Dual Regression Detection
```python
# Independent thresholds for latency and size
latency_regression = latency_change_pct > latency_threshold
size_regression    = size_change_pct    > size_threshold
regression_detected = latency_regression or size_regression
```

### 7. Explore Verdict Scoring
```python
score = 0.6 * (baseline_latency / variant_latency) \
      + 0.4 * (baseline_size / variant_size)
# score > 1.0 → candidate for recommended/aggressive
# score ≤ 1.0 → skip (strictly worse on both axes)
```

---

## Features

### Build and Convert
- ONNX → CoreML conversion (Apple Silicon, A-series)
- ONNX → TFLite conversion (Android arm64)
- Quantization: FP32 / FP16 / INT8
- Deterministic builds (content-addressed)
- ONNX graph storage for re-conversion
- PyTorch (.pt/.pth) → ONNX / CoreML / TFLite via `mlbuild convert`
- TF SavedModel → TFLite via `mlbuild convert`
- Automatic multi-step chaining (e.g. PyTorch → ONNX → TFLite in one command)
- Intermediate artifact tracking with `--keep-intermediate`
- Per-step output validation, caching, and timeout

### Import Pre-built Models
- Import existing `.onnx`, `.tflite`, `.mlmodel`, `.mlpackage` files directly
- ONNX import runs via ONNX Runtime — `onnxruntime_cpu`, `onnxruntime_gpu`, `onnxruntime_ane` targets
- Format validation via protobuf check (ONNX), magic bytes (TFLite), structure checks (CoreML)
- Tier 1 task detection for all import formats — ONNX via graph ops, TFLite via FlatBuffer parsing, CoreML via coremltools spec
- Format/target compatibility enforcement
- Imported builds tracked with `[imported]` badge in `mlbuild log`
- Full MLBuild toolchain available immediately after import

#### Device-Connected Benchmarking

Benchmark directly on a physical Android device or emulator connected via USB-C — no Android Studio required.
```bash
# Build TFLite model for connected device (auto-detects ABI)
mlbuild build --model model.onnx --target device-connected --backend tflite

# Import an existing TFLite model for the connected device
mlbuild import --model model.tflite --target device-connected

# Benchmark — automatically routes to ADB pipeline
mlbuild benchmark <build-id>
```

**Requirements:** `adb` installed and on PATH, USB debugging enabled on the device.

**What it measures:**
- Latency: p50, p95, mean, std, min, max, variance
- Init time (model load)
- Peak memory (MB)
- Thermal drift (pre/post temperature delta in °C)
- Stability score (stable / noisy / unreliable)
- GPU and NNAPI delegate validation

**Format/device compatibility:**

| Format | Device | Result |
|--------|--------|--------|
| TFLite | Android (USB) | ✓ Benchmarks via ADB |
| TFLite | Android emulator | ✓ Works with emulator warning |
| CoreML | Android | ✗ Format mismatch — clear error |
| CoreML | iPhone (IDB) | Coming in Phase 2 |

Emulator runs are detected automatically and flagged with a warning — latency numbers from emulators are not representative of real hardware.

### Format × Device Compatibility Matrix

| Format | Target | Device Connected | Result |
|--------|--------|-----------------|--------|
| TFLite | `device-connected` | Android (USB-C) | ✓ Benchmarks via ADB |
| TFLite | `device-connected` | Android emulator | ✓ Works with emulator warning |
| TFLite | `device-connected` | iOS connected | ✗ Format mismatch — clear error |
| TFLite | `device-connected` | No device | ✗ No device error |
| TFLite | `android_arm64` | Mac host | ✓ Local TFLite runner |
| TFLite | `android_arm32` | Mac host | ✓ Local TFLite runner |
| TFLite | `raspberry_pi` | Mac host | ✓ Local TFLite runner |
| CoreML | `device-connected` | Android connected | ✗ Device mismatch at build time |
| CoreML | `device-connected` | Android connected | ✗ Format mismatch at benchmark |
| CoreML | `device-connected` | iPhone via IDB | ✓ IDB pipeline (Phase 2) |
| CoreML | `device-connected` | No device | ✓ Builds with `apple_a17` default + warning |
| CoreML | `apple_m1/m2/m3` | Mac host | ✓ Local CoreML runner |
| CoreML | `apple_a15`–`a18` | Mac host | ✓ Local CoreML runner |
| ONNX | `onnxruntime_cpu` | Any | ✓ Local ONNX Runtime |
| ONNX | `onnxruntime_gpu` | Any | ✓ Local ONNX Runtime |
| ONNX | `onnxruntime_ane` | macOS only | ✓ Local ONNX Runtime |

**Import is always device-agnostic** — format/device compatibility is enforced at benchmark time only.

### Optimization
- **FP16 quantization** — recompilation from ONNX graph
- **Dynamic range INT8** — weight-only, no calibration data needed
- **Static INT8** — weights + activations quantized using representative calibration data; gracefully falls back to dynamic range on coremltools 9.0
- **Magnitude pruning** — global threshold-based, ONNX path works for both CoreML and TFLite, CoreML post-hoc path for imported models
- **Method chaining** — prune → quantize, any depth
- Distinct build IDs per optimization level (`int8_static` ≠ `int8`, `prune_0.50` ≠ `prune_0.75`)
- Deduplication — identical optimization chains reuse existing builds

### Optimization Sweep
- `mlbuild explore` — single command sweeps fp16 + int8 across all backends
- Score-based verdict assignment (recommended / aggressive / skip / baseline)
- Accuracy check per variant with `--check-accuracy` — failed variants get `skip` verdict
- Calibration data support with `--calibration-data` for static INT8 in sweep
- Fast mode (`--fast`) — fp16 only, 20 benchmark runs

### Accuracy / Output Divergence
- Cosine similarity, MAE, max absolute error, top-1 agreement
- Dtype-aware random input generation
- `precomputed_batch` — inputs generated once, reused across all variants in explore
- Results persisted to `accuracy_checks` registry table
- Standalone `mlbuild accuracy` command
- Integrated into `mlbuild compare --check-accuracy` and `mlbuild explore --check-accuracy`

### Task-Aware Benchmarking
- Three-tier automatic task detection (graph ops → tensor names → shapes)
- Task-specific synthetic inputs: real image tensors, token IDs + attention masks, waveforms/spectrograms
- NLP multi-sequence-length benchmarking ladder `[16, 64, 128, 256]`
- Configurable `--task` override for explicit control
- Post-inference output validation with soft/strict modes (`--strict-output`)

### Performance Validation
- Automated p50/p95/p99 benchmarking
- SLA enforcement (`--max-latency`, `--max-p95`, `--max-memory`, `--max-size`)
- Accuracy validation via `--dataset` (calibration data), composes with performance checks
- Baseline accuracy comparison with `--baseline-id` (defaults to root build)
- Root builds skip accuracy check gracefully rather than erroring

### Deep Profiling (`--deep`)
- **TFLite:** Per-op timing (real hardware), tensor memory flow, COMPUTE/MEMORY bottleneck classification, cold start decomposition, per-layer quantization sensitivity (fp32 vs int8), op fusion detection
- **CoreML:** Cold start decomposition (all formats); per-layer timing, memory flow, bottleneck classification, and fusion detection (NeuralNetwork format only)

### Build History and Lineage
- `mlbuild log --source` — filter builds by source model filename
- `mlbuild log --tree` — recursive parent-child DAG — causality preserved across optimization chains
- Human-readable method labels in tree: `prune(0.50)`, `int8(static)`
- Filter by name, format, task, target, date range, roots-only
- JSON and CSV export

### Command History
- `mlbuild history` — permanent audit log of every CLI command ever run
- Searchable by command type, time window, build ID
- Filterable: build, benchmark, validate, compare, profile, failed
- Delete individual entries or clear all — never affects build or benchmark data
- Machine identity captured on every row — ready for cross-machine team view when cloud login lands

### Performance Budget
- `mlbuild budget set/show/clear/validate` — persistent constraint management
- Stored in `.mlbuild/budget.toml` — commit to git for team-wide enforcement
- Merge logic: explicit CLI flag > budget > no constraint
- Constraint source shown in violation output (`budget` vs `explicit flag`)
- All four constraints: `max_latency_ms`, `max_p95_ms`, `max_memory_mb`, `max_size_mb`
- Applied automatically by `mlbuild validate` and `mlbuild ci`
- `budget validate <build_id>` — dry run, evaluates size immediately, flags latency as pending

### Baseline Management
- `mlbuild baseline set/show/unset/history` — clean UX wrapper around `mlbuild tag`
- Uses reserved tag `mlbuild-baseline` — integrates with `mlbuild ci` automatically
- Prompts before overwriting existing baseline
- `baseline history` — shows all baseline-style tags: `mlbuild-baseline`, `main-*`, `production-*`

### Workspace Status
- `mlbuild status` — instant workspace health snapshot
- Shows build/benchmark counts, last build, last benchmark, last validate result
- Shows current baseline and active budget constraints
- JSON output via `--json` for scripting

### Performance Reports
- Self-contained HTML (no external dependencies)
- Benchmark history table
- Related builds comparison
- Deployment recommendations
- Optional PDF export (requires weasyprint)

### Remote Storage
- S3-compatible backends (AWS, R2, B2)
- Git-style push/pull/sync
- Integrity verification (SHA-256)

### CI/CD Integration
- `mlbuild ci` — full CI orchestration (explore + compare + threshold enforcement + JSON report)
- Tag-based baseline resolution — `mlbuild tag create <id> main-mobilenet`
- Baseline immutability — report stores both tag name and build ID for reproducibility
- Baseline benchmark guard — auto-benchmarks baseline if no cached latency
- Relative regression thresholds (`--latency-regression`, `--size-regression`)
- Absolute budget constraints (`--latency-budget`, `--size-budget`) independent of baseline
- Accuracy gate via `--dataset` — cosine similarity + top-1 agreement
- `--fail-on-missing-baseline` — strict mode for production pipelines
- Structured JSON report at `.mlbuild/ci_report.json` — readable by GitHub bots, dashboards, Slack
- `mlbuild ci-check` — low-level build-to-build regression gate
- Exit codes: 0 (pass/skip) / 1 (regression/fail) / 2 (error)
- GitHub Actions workflow with artifact upload and PR comment posting (`.github/workflows/mlbuild.yml`)

---

## Project Structure
```
mlbuild/
├── src/mlbuild/
│   ├── cli/
│   │   ├── formatters/
│   │   │   ├── export.py                 # JSON + CSV serialization
│   │   │   ├── inspect.py
│   │   │   └── utils.py
│   ├── models/
│   │   └── build_view.py                 # BuildView, Artifact, BenchmarkRow, AccuracyRow
│   │   ├── commands/
│   │   │   ├── accuracy.py               # mlbuild accuracy
│   │   │   ├── baseline.py               # mlbuild baseline
│   │   │   ├── benchmark.py              # mlbuild benchmark
│   │   │   ├── budget.py                 # mlbuild budget
│   │   │   ├── build.py                  # mlbuild build
│   │   │   ├── convert.py                # mlbuild convert
│   │   │   ├── ci.py                     # mlbuild ci + ci-checkheck
│   │   │   ├── compare.py                # mlbuild compare
│   │   │   ├── compare_compute_units.py  # mlbuild compare-compute-units
│   │   │   ├── compare_quantization.py   # mlbuild compare-quantization
│   │   │   ├── diff.py                   # mlbuild diff
│   │   │   ├── doctor.py                 # mlbuild doctor
│   │   │   ├── experiment.py             # mlbuild experiment
│   │   │   ├── explore.py                # mlbuild explore
│   │   │   ├── export.py                 # mlbuild export
│   │   │   ├── history.py                # mlbuild history
│   │   │   ├── inspect.py                # mlbuild inspect
│   │   │   ├── import_cmd.py             # mlbuild import
│   │   │   ├── log.py                    # mlbuild log
│   │   │   ├── optimize.py               # mlbuild optimize
│   │   │   ├── pin.py                    # mlbuild pin / mlbuild unpin
│   │   │   ├── profile.py                # mlbuild profile
│   │   │   ├── prune.py                  # mlbuild prune
│   │   │   ├── pull.py                   # mlbuild pull
│   │   │   ├── push.py                   # mlbuild push
│   │   │   ├── remote.py                 # mlbuild remote
│   │   │   ├── rename.py                 # mlbuild rename
│   │   │   ├── report.py                 # mlbuild report
│   │   │   ├── run.py                    # mlbuild run
│   │   │   ├── search.py                 # mlbuild search
│   │   │   ├── status.py                 # mlbuild status
│   │   │   ├── sync.py                   # mlbuild sync
│   │   │   ├── tag.py                    # mlbuild tag
│   │   │   ├── upgrade.py                # mlbuild upgrade
│   │   │   └── validate.py               # mlbuild validate
│   └── main.py                       # CLI entry point
│   ├── backends/
│   │   ├── base.py                       # Backend base class
│   │   ├── registry.py                   # Backend auto-discovery
│   │   ├── coreml/                       # CoreML exporter + deep profiler
│   │   ├── tflite/                       # TFLite backend + deep profiler
│   │   └── onnxruntime/                  # ONNX Runtime backend
│   ├── convert/
│   │   ├── __init__.py
│   │   ├── cache.py                      # Step-level cache keys with env version awareness
│   │   ├── coreml.py                     # PyTorch/ONNX → CoreML executors + TARGET_OS_MAP
│   │   ├── detector.py                   # Input format detection from path
│   │   ├── feature_compat.py             # Feature → OS compatibility validation (hard/soft)
│   │   ├── graph.py                      # Conversion DAG + BFS path resolution
│   │   ├── pytorch.py                    # PyTorch → ONNX executor
│   │   ├── service.py                    # Pipeline orchestration (cache, timeout, temp dirs)
│   │   ├── tensorflow.py                 # ONNX/SavedModel → TFLite executors
│   │   ├── types.py                      # ConvertContext, ConvertOutput, StepResult, ConvertResult
│   │   └── validator.py                  # Per-format artifact validation
│   ├── core/
│   │   ├── budget.py                     # Budget load/save/merge/validate
│   │   ├── accuracy/
│   │   │   ├── calibration.py            # CalibrationLoader (images/npy/npz)
│   │   │   ├── checker.py                # run_accuracy_check()
│   │   │   ├── config.py                 # AccuracyConfig, AccuracyResult
│   │   │   ├── inputs.py                 # InputSpec, generate_batch
│   │   │   └── metrics.py                # cosine_similarity, MAE, top-1
│   │   ├── ci/
│   │   │   ├── reporter.py               # CIReport + text/JSON/markdown formatters
│   │   │   ├── runner.py                 # CIRunner orchestration
│   │   │   └── thresholds.py             # ThresholdConfig + violation evaluation
│   │   ├── environment.py                # Environment fingerprinting
│   │   ├── errors.py                     # Error types
│   │   ├── format_detection.py           # Format detection + target validation
│   │   ├── hash.py                       # Deterministic artifact hashing
│   │   ├── ir.py                         # ModelIR — format-agnostic model graph
│   │   ├── machine.py                    # Machine identity (UUID + hostname)
│   │   ├── task_detection.py             # Three-tier task detection
│   │   ├── task_inputs.py                # Task-aware synthetic input generation
│   │   ├── task_validation.py            # Post-inference output validation
│   │   ├── tasks.py                      # Task types + arbitration + output schemas
│   │   └── types.py                      # Build, Benchmark, VariantResult dataclasses
│   ├── experiments/                      # Experiment + run tracking
│   ├── explore/
│   │   └── explorer.py                   # explore(), assign_verdicts(), accuracy integration
│   ├── loaders/
│   │   ├── loader.py                     # Model loading entrypoint
│   │   └── onnx_loader.py                # ONNX loader + ModelIR builder
│   ├── optimize/
│   │   ├── optimizer.py                  # optimize() + prune() entrypoints
│   │   ├── passes/
│   │   │   ├── pruning.py                # PruningPass (ONNX + CoreML post-hoc)
│   │   │   └── quantization.py           # QuantizationPass (fp16/int8/int8_static)
│   │   └── backends/
│   │       ├── coreml_backend.py         # compile_from_graph, quantize_weights,
│   │       │                             # quantize_weights_static, prune_weights
│   │       └── tflite_backend.py         # quantize_from_graph
│   ├── platforms/
│   │   ├── android/
│   │   │   ├── __init__.py
│   │   │   ├── adb.py                    # ADB transport layer (push/pull/shell/devices)
│   │   │   ├── baseline.py               # CPU baseline benchmark execution + parsing
│   │   │   ├── benchmark.py              # Delegate benchmark execution (GPU/NNAPI)
│   │   │   ├── delegate.py               # Delegate validation + failure classification
│   │   │   ├── deploy.py                 # Model + binary deployment to device
│   │   │   ├── device.py                 # ADBDevice entry point + BenchmarkConfig
│   │   │   ├── history.py                # Per-device benchmark history persistence
│   │   │   ├── introspect.py             # Device profiling (ABI, API level, chipset)
│   │   │   ├── recomend.py               # Recommendation engine (use_cpu/rerun/delegate)
│   │   │   ├── result.py                 # AndroidBuildView assembly + serialization
│   │   │   ├── stability.py              # Stability scoring + thermal instability detection
│   │   │   ├── thermal.py                # Thermal snapshot (battery temp, CPU freq)
│   │   │   └── binaries/
│   │   │       └── arm64-v8a/
│   │   │           └── benchmark_model   # TFLite benchmark binary (6.8MB, nightly)
│   │   └── ios/                          # IDB bridge — coming in Phase 2
│   ├── profiling/
│   │   ├── cold_start.py                 # Cold start decomposition
│   │   ├── layer_profiler.py             # Per-layer timing
│   │   ├── memory_profiler.py            # Memory tracking
│   │   └── warmup_analyzer.py            # Warmup analysis
│   ├── registry/
│   │   ├── local.py                      # SQLite registry (WAL mode)
│   │   └── schema.py                     # Schema + migrations (v9)
│   ├── storage/
│   │   ├── backend.py                    # Storage backend interface
│   │   ├── config.py                     # Remote config
│   │   ├── local.py                      # Local storage
│   │   └── s3.py                         # S3-compatible storage
│   ├── validation/
│   │   └── accuracy_validator.py         # AccuracyValidator for mlbuild validate
│   └── visualization/
│       └── charts.py                     # Chart generation
├── tests/
├── pyproject.toml
└── README.md
```

---

## vs. Existing Tools

| Feature | Custom Scripts | Profilers | **MLBuild** |
|---------|---------------|-----------|-------------|
| Hardware inference benchmarking | Manual | Partial | **Automated** |
| Performance regression detection | Custom | Manual | **Built-in** |
| CI performance gate | Custom | — | **Built-in** |
| Cross-device testing | Manual | — | **Yes** |
| Performance history & tracking | — | — | **Built-in** |
| CI-automated per-layer profiling | Custom | Manual | **Automated** |
| Quantization performance benchmarking | — | Manual | **Automated** |
| Auto-generated task inputs | — | — | **Auto-detected** |
| Performance reports | — | — | **HTML/PDF** |

Use MLflow/W&B for training experiments. Use MLBuild for on-device inference performance.

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

### Phase 1 — Device-Connected Benchmarking *(complete)*
- ✓ Android ADB bridge — benchmark on connected Android devices via USB-C
- ✓ Emulator detection with clear warning
- ✓ GPU/NNAPI delegate validation and classification
- ✓ Stability scoring, thermal drift, per-run stats
- Xcode IDB integration — real iPhone hardware profiling *(next)*

### Phase 2 — More Backends *(next)*
- TensorRT — NVIDIA GPU inference
- Qualcomm QNN — Snapdragon NPU

### Phase 3 — Cloud Benchmarking
- Remote benchmark execution on cloud hardware

---

<div align="center">
Built by <a href="https://github.com/AbdoulayeSeydi/mlbuild">Abdoulaye Seydi</a>
</div>
