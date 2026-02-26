# MLBuild

**Deterministic build system for CoreML models with real device benchmarking**

MLBuild tracks CoreML model performance and prevents regressions in CI by:
- Building deterministic CoreML artifacts from ONNX models
- Benchmarking on real devices (Mac and iPhone)
- Tracking performance history in a local registry
- Enforcing performance budgets in CI

## Quick Start
```bash
# Install
pip install -e .

# Build a model
mlbuild build --model model.onnx --target apple_a17 --name prod-v1

# Benchmark on local Mac
mlbuild benchmark <build_id> --device local

# View build history
mlbuild log

# Compare two builds
mlbuild diff prod-v1 prod-v2
```

## Status

**In Development** - Currently implementing Phase 1 (Core Primitives)

See [PLAN.md](PLAN.md) for full roadmap.

## License

MIT