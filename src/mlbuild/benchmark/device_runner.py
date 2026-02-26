"""
iOS Physical Device Benchmark Runner.
Real iPhone/iPad benchmarking via XCTest over USB.
No simulator - production metrics only.
"""

import subprocess
import json
import shutil
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class iOSBenchmarkResult:
    """Results from physical iOS device benchmark."""
    device_name: str
    device_model: str
    os_version: str
    is_low_power_mode: bool
    thermal_state: str
    
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    
    memory_peak_mb: float
    num_runs: int
    failures: int
    timestamp: str


class iOSDeviceBenchmarkRunner:
    """
    Physical iPhone/iPad benchmarking via XCTest.
    
    Requirements:
    - Device connected via USB
    - Device unlocked and trusted
    - Developer mode enabled on device
    """
    
    def __init__(
        self,
        xcode_project_path: Path = None,
        device_id: Optional[str] = None,
    ):
        """
        Initialize iOS device benchmark runner.
        
        Args:
            xcode_project_path: Path to MLBuildRunner.xcodeproj
            device_id: Device UDID (auto-detect if None)
        """
        # Default to ios/MLBuildRunner
        if xcode_project_path is None:
            xcode_project_path = Path(__file__).parent.parent.parent.parent / "ios" / "MLBuildRunner"
        
        self.xcode_project_path = Path(xcode_project_path)
        self.xcodeproj = self.xcode_project_path / "MLBuildRunner.xcodeproj"
        self.resources_dir = self.xcode_project_path / "MLBuildRunnerTests" / "Resources"
        self.device_id = device_id or self._detect_device()
        
        if not self.xcodeproj.exists():
            raise RuntimeError(f"Xcode project not found: {self.xcodeproj}")
        
        print(f"📱 Device: {self.device_id}")
    
    def _detect_device(self) -> str:
        """Auto-detect connected iOS device using xcodebuild."""
        result = subprocess.run(
            ["xcodebuild", "-showdestinations", "-project", str(self.xcodeproj), "-scheme", "MLBuildRunner"],
            capture_output=True,
            text=True,
        )
        
        # Parse for physical device
        for line in result.stdout.split('\n'):
            if 'platform:iOS' in line and 'id:' in line and 'name:' in line:
                # Extract ID from line like: { platform:iOS, arch:arm64, id:00008110-..., name:iPhone }
                match = re.search(r'id:([0-9A-F-]+)', line)
                if match:
                    device_id = match.group(1)
                    # Filter out placeholder IDs
                    if len(device_id) == 40 and not 'placeholder' in line.lower():
                        return device_id
        
        raise RuntimeError(
            "No physical iOS device detected.\n"
            "Make sure:\n"
            "  1. Device is connected via USB\n"
            "  2. Device is unlocked\n"
            "  3. You've trusted this computer on the device\n"
            "  4. Developer mode is enabled (Settings → Privacy → Developer Mode)"
        )
    
    def benchmark(
        self,
        mlpackage_path: Path,
        build_id: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        compute_unit: str = "all",
    ) -> iOSBenchmarkResult:
        """
        Run benchmark on physical iOS device.
        
        Args:
            mlpackage_path: Path to .mlpackage model
            build_id: Build identifier
            warmup_runs: Warmup iterations
            benchmark_runs: Benchmark iterations
            compute_unit: "cpu", "gpu", or "all"
            
        Returns:
            iOSBenchmarkResult with real device metrics
        """
        print(f"\n📱 iOS Device Benchmark (Physical Hardware)")
        print(f"Model: {mlpackage_path.name}\n")
        
        # 1. Inject model
        self._inject_model(mlpackage_path)
        
        # 2. Build test bundle
        self._build_for_testing()
        
        # 3. Run on device
        test_output = self._run_tests()
        
        # 4. Parse results
        result = self._parse_results(test_output)
        
        # 5. Cleanup
        self._cleanup_model()
        
        return result
    
    def _inject_model(self, mlpackage_path: Path):
        """Copy .mlpackage with sanitized name to avoid Swift conflicts."""
        print("📦 Injecting model into Xcode project...")
        
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        
        # Use clean name for Swift code generation
        target_path = self.resources_dir / "BenchmarkModel.mlpackage"
        
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(mlpackage_path, target_path)
        
        print(f"✓ Model injected as BenchmarkModel.mlpackage")
    
    def _build_for_testing(self):
        """Build test bundle for device."""
        print("🔨 Building test bundle for device...")
        
        cmd = [
            "xcodebuild",
            "build-for-testing",
            "-project", str(self.xcodeproj),
            "-scheme", "MLBuildRunner",
            "-destination", f"platform=iOS,id={self.device_id}",
            "-derivedDataPath", str(self.xcode_project_path / "build"),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ Build failed:")
            print(result.stderr[-2000:])
            raise RuntimeError("xcodebuild build failed")
        
        print("✓ Test bundle built")
    
    def _run_tests(self) -> str:
        """Run tests on physical device and capture output."""
        print("🧪 Running benchmark on device...")
        print("   (Unlock your iPhone if prompted)\n")
        
        cmd = [
            "xcodebuild",
            "test-without-building",
            "-project", str(self.xcodeproj),
            "-scheme", "MLBuildRunner",
            "-destination", f"platform=iOS,id={self.device_id}",
            "-derivedDataPath", str(self.xcode_project_path / "build"),
        ]
        
        try:
            # Add 2 minute timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120  # 2 minutes max
            )
            
            return result.stdout + result.stderr
            
        except subprocess.TimeoutExpired as e:
            print("\n❌ Test timed out after 2 minutes")
            print("Possible issues:")
            print("  - Device is locked")
            print("  - 'Trust this computer' dialog on device")
            print("  - Test is hanging")
            print("\nPartial output:")
            if e.stdout:
                print(e.stdout.decode()[-1000:])
            if e.stderr:
                print(e.stderr.decode()[-1000:])
            raise RuntimeError("Test execution timed out")
    
    def _parse_results(self, test_output: str) -> iOSBenchmarkResult:
        """Extract JSON results from test output."""
        start_marker = "BENCHMARK_RESULT_JSON_START"
        end_marker = "BENCHMARK_RESULT_JSON_END"
        
        if start_marker not in test_output or end_marker not in test_output:
            print("\n❌ No benchmark results found")
            print("Output excerpt:")
            print(test_output[-1500:])
            raise RuntimeError("Benchmark did not produce JSON results")
        
        start_idx = test_output.index(start_marker) + len(start_marker)
        end_idx = test_output.index(end_marker)
        
        json_str = test_output[start_idx:end_idx].strip()
        data = json.loads(json_str)
        
        return iOSBenchmarkResult(
            device_name=data["deviceName"],
            device_model=data["deviceModel"],
            os_version=data["osVersion"],
            is_low_power_mode=data["isLowPowerMode"],
            thermal_state=data["thermalState"],
            latency_p50_ms=data["latencyP50Ms"],
            latency_p95_ms=data["latencyP95Ms"],
            latency_p99_ms=data["latencyP99Ms"],
            latency_mean_ms=data["latencyMeanMs"],
            latency_std_ms=data["latencyStdMs"],
            memory_peak_mb=data["memoryPeakMB"],
            num_runs=data["numRuns"],
            failures=data["failures"],
            timestamp=data["timestamp"],
        )
    
    def _cleanup_model(self):
        """Remove model after test."""
        model_path = self.resources_dir / "BenchmarkModel.mlpackage"
        if model_path.exists():
            shutil.rmtree(model_path)