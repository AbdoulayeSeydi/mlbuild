# MLBuild CI Integration

## Overview

`mlbuild ci` runs a full performance gate in CI: explores variants, compares against your registered baseline, enforces thresholds, and writes a structured report.

## Quickstart

### 1. Tag your main branch baseline
```bash
# After building your model on main
mlbuild build --model mobilenet.onnx --target apple_m1 --name mobilenet
mlbuild tag create main-mobilenet <build_id>
```

### 2. Run CI on a PR
```bash
mlbuild ci \
  --model mobilenet.onnx \
  --baseline main-mobilenet \
  --latency-regression 15 \
  --size-regression 10
```

Exit codes: `0` = pass, `1` = regression detected, `2` = error.

### 3. Use an existing build (skip explore)
```bash
mlbuild ci \
  --build <build_id> \
  --baseline main-mobilenet
```

Useful when builds happen earlier in your pipeline.

## Configuration

Add to `.mlbuild/config.toml`:
```toml
[ci]
latency_regression_pct = 10   # fail if >10% slower than baseline
size_regression_pct = 5       # fail if >5% larger than baseline
latency_budget_ms = 3.0       # hard cap regardless of baseline
size_budget_mb = 10.0         # hard cap regardless of baseline

[ci.accuracy]
cosine_threshold = 0.99
top1_threshold = 0.99
```

CLI flags override config.toml values.

## Options

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

## CI Report

The report is always written to `.mlbuild/ci_report.json`:
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
    "latency_ms": 2.81,
    "size_mb": 6.74
  },
  "delta": { "latency_pct": 12.8, "size_pct": -49.7 },
  "thresholds": {
    "latency_regression_pct": 15,
    "size_regression_pct": 10,
    "latency_budget_ms": null,
    "size_budget_mb": null
  },
  "accuracy": null,
  "result": "pass",
  "violations": []
}
```

The report always stores `baseline.build_id` — even if the tag is later repointed, the report preserves exactly what was compared.

## GitHub Actions

See `.github/workflows/mlbuild.yml` for a complete example including artifact upload and PR comment posting.
```yaml
- name: Run MLBuild CI
  run: |
    mlbuild ci \
      --model models/mobilenet.onnx \
      --baseline main-mobilenet \
      --latency-regression 15 \
      --size-regression 10
```

## Baseline resolution

`--baseline` accepts either a tag name or a build ID prefix:
```bash
mlbuild ci --baseline main-mobilenet        # tag lookup
mlbuild ci --baseline 3f36810e              # build ID prefix
```

If the baseline is missing:
- Default: warning printed, exit 0 (CI passes)
- With `--fail-on-missing-baseline`: exit 1

## Candidate selection

When using `--model` (explore mode), the best candidate is selected deterministically:

1. Variant with `verdict="recommended"` from explore
2. Lowest latency variant among remaining
3. First variant (fallback)

The selected `candidate_build_id` is always stored in the report.
