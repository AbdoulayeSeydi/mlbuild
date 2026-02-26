"""
Test the benchmark runner on MobileNetV2.
"""

from pathlib import Path
from mlbuild.benchmark.runner import ( 
    CoreMLBenchmarkRunner,
    ComputeUnit,
    detect_apple_chip,
    hardware_fingerprint,
)

def test_device_detection():
    """Test that device detection works."""
    print("\n" + "="*60)
    print("DEVICE DETECTION TEST")
    print("="*60)
    
    chip = detect_apple_chip()
    print(f"✓ Detected chip: {chip}")
    
    hw = hardware_fingerprint()
    print(f"✓ Machine: {hw.machine}")
    print(f"✓ macOS: {hw.macos_version}")
    print(f"✓ CPUs (logical): {hw.cpu_count_logical}")
    print(f"✓ CPUs (physical): {hw.cpu_count_physical}")
    print(f"✓ Memory: {hw.memory_gb} GB")


def test_benchmark_runner():
    """Test benchmarking on actual model."""
    print("\n" + "="*60)
    print("BENCHMARK RUNNER TEST")
    print("="*60)
    
    # Find the FP16 model
    artifact_dir = Path(".mlbuild/artifacts")
    
    # List all artifacts
    artifacts = list(artifact_dir.glob("*/"))
    if not artifacts:
        print("❌ No artifacts found. Run 'mlbuild build' first.")
        return
    
    # Use the first one (or find FP16 if you want)
    model_path = artifacts[0]
    print(f"\nModel: {model_path.name[:16]}...")
    
    # Create runner
    runner = CoreMLBenchmarkRunner(
        model_path=model_path,
        compute_unit=ComputeUnit.CPU_ONLY,  # Start with CPU only
        warmup_runs=5,      # Quick test
        benchmark_runs=20,  # Quick test
        ci_mode=False,
    )
    
    print(f"Input shapes: {runner.inputs}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    result, raw_latencies = runner.run(
        build_id="test-run",
        return_raw=True,
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Device:           {result.chip}")
    print(f"Compute Unit:     {result.compute_unit}")
    print(f"Runs:             {result.num_runs} (failed: {result.failures})")
    print(f"\nLatency:")
    print(f"  p50:            {result.latency_p50:.3f} ms")
    print(f"  p95:            {result.latency_p95:.3f} ms")
    print(f"  p99:            {result.latency_p99:.3f} ms")
    print(f"  mean:           {result.latency_mean:.3f} ms")
    print(f"  std:            {result.latency_std:.3f} ms")
    print(f"\nConfidence Interval (p50):")
    print(f"  95% CI:         [{result.p50_ci_low:.3f}, {result.p50_ci_high:.3f}] ms")
    print(f"\nQuality Metrics:")
    print(f"  Autocorr (lag1): {result.autocorr_lag1:.3f}")
    print(f"  Thermal drift:   {result.thermal_drift_ratio:.3f}")
    print(f"\nMemory:")
    print(f"  Peak:           {result.memory_peak_mb:.2f} MB")
    
    # Export JSON
    output_path = Path("benchmark_test_result.json")
    runner.export_json(result, output_path)
    print(f"\n✓ Exported to {output_path}")
    
    return result, raw_latencies


if __name__ == "__main__":
    # Test 1: Device detection
    test_device_detection()
    
    # Test 2: Benchmark runner
    result, latencies = test_benchmark_runner()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)