"""
MLBuild Doctor Command (Enterprise-Grade)

Performs a comprehensive check of the MLBuild environment:
- Python version
- Required ML packages (coremltools, onnx, torch)
- GPU / NPU availability
- macOS version & optional tools (Xcode)
- .mlbuild directory structure and permissions
- Provides CLI rich output and JSON output for automation

Supports warnings vs errors and soft failures for optional tools.
"""

import sys
import platform
import json
import logging
import os
from pathlib import Path
from typing import Tuple, List

import click
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger("mlbuild.doctor")

# -------------------------------
# Constants
# -------------------------------
REQUIRED_PYTHON = (3, 9)
OPTIONAL_TOOLS = ["xcodebuild"]

ML_FRAMEWORKS = [
    "coremltools",
    "onnx",
    "torch",
    "onnxruntime",
]

# -------------------------------
# Helpers
# -------------------------------
def check_import(pkg_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed and return version."""
    try:
        module = __import__(pkg_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except (ImportError, ValueError, Exception) as e:
        # Catch all exceptions including TensorFlow's ValueError
        logger.debug(f"Failed to import {pkg_name}: {e}")
        return False, None

def check_macos_tools(tools: List[str]) -> dict:
    """Check optional macOS developer tools."""
    import subprocess
    results = {}
    for tool in tools:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                results[tool] = result.stdout.splitlines()[0]
            else:
                results[tool] = "not found"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            results[tool] = "not found"
    return results

def check_ml_frameworks(frameworks: List[str]) -> dict:
    """Check required ML frameworks and GPU/NPU availability."""
    result = {}
    for fw in frameworks:
        installed, version = check_import(fw)
        result[fw] = {"installed": installed, "version": version}
        # Optional: check GPU/NPU availability for frameworks that support it
        if installed and fw == "torch":
            try:
                import torch
                result[fw]["gpu"] = torch.cuda.is_available()
                result[fw]["npu"] = hasattr(torch, "npu") and torch.npu.is_available()
            except Exception as e:
                result[fw]["gpu"] = False
                result[fw]["npu"] = False
                logger.warning(f"Failed to check GPU/NPU for {fw}: {e}")
    return result

def check_mlbuild_directory() -> dict:
    """Check .mlbuild directory structure, permissions, and contents."""
    base_dir = Path.cwd() / ".mlbuild"
    result = {"exists": base_dir.exists(), "readable": False, "writable": False, "artifacts": None, "registry_size_kb": None}

    if base_dir.exists():
        result["readable"] = base_dir.exists() and os.access(base_dir, os.R_OK)
        result["writable"] = base_dir.exists() and os.access(base_dir, os.W_OK)
        artifacts_dir = base_dir / "artifacts"
        registry_db = base_dir / "registry.db"
        if artifacts_dir.exists():
            result["artifacts"] = len(list(artifacts_dir.iterdir()))
        if registry_db.exists():
            result["registry_size_kb"] = registry_db.stat().st_size / 1024
    return result

# -------------------------------
# CLI Command
# -------------------------------
@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output results in JSON format")
@click.option("--soft", is_flag=True, help="Do not exit with error code on missing optional tools")
def doctor(as_json: bool, soft: bool):
    """
    Check the MLBuild environment for reproducibility and readiness.
    """
    results = {"checks": {}, "platform": {}, "mlbuild_dir": {}}
    failures = []

    # Python version
    py_info = sys.version_info
    py_ok = py_info >= REQUIRED_PYTHON
    results["checks"]["python"] = {"version": f"{py_info.major}.{py_info.minor}.{py_info.micro}", "ok": py_ok}
    if not py_ok:
        failures.append(f"Python >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required")

    # ML Frameworks
    framework_results = check_ml_frameworks(ML_FRAMEWORKS)
    results["checks"]["ml_frameworks"] = framework_results
    for fw, data in framework_results.items():
        if not data["installed"]:
            failures.append(f"Missing required framework: {fw}")

    # Platform
    results["platform"] = {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
    }
    # macOS-specific: Xcode version (critical for CoreML compilation)
    if platform.system() == "Darwin":
        # Check macOS version
        mac_version = platform.mac_ver()[0]
        console.print(f"  macOS:        {mac_version}")

        # Check for Xcode
        import subprocess
        try:
            result = subprocess.run(
                ["xcodebuild", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                env=os.environ.copy()  # Use current environment
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                xcode_version = lines[0] if lines else "unknown"
                console.print(f"  {xcode_version}")
            else:
                console.print(f"  [yellow]xcodebuild: not found ⚠️[/yellow]")
        except FileNotFoundError:
            console.print(f"  [yellow]xcodebuild: not found ⚠️[/yellow]")
        except subprocess.TimeoutExpired:
            console.print(f"  [yellow]xcodebuild: timeout[/yellow]")

    # .mlbuild directory
    mlbuild_dir_results = check_mlbuild_directory()
    results["mlbuild_dir"] = mlbuild_dir_results
    if not mlbuild_dir_results["exists"] and not soft:
        failures.append(".mlbuild directory not initialized")

    # -------------------------------
    # Output
    # -------------------------------
    if as_json:
        results["status"] = "fail" if failures else "pass"
        results["failures"] = failures
        console.print(json.dumps(results, indent=2))
    else:
        console.print("\n[bold]MLBuild Environment Check[/bold]\n")
        # Python
        py_status = "[green]✓[/green]" if py_ok else "[red]✗[/red]"
        console.print(f"{py_status} Python {py_info.major}.{py_info.minor}.{py_info.micro}")

        # ML Frameworks
        console.print("\n[bold]ML Frameworks:[/bold]")
        for fw, data in framework_results.items():
            status = "[green]✓[/green]" if data["installed"] else "[red]✗[/red]"
            version = data.get("version", "unknown")
            gpu = data.get("gpu", None)
            gpu_str = f", GPU available: {gpu}" if gpu is not None else ""
            console.print(f"  {status} {fw} {version}{gpu_str}")

        # Platform
        console.print("\n[bold]Platform:[/bold]")
        console.print(f"  OS: {results['platform']['os']} {results['platform']['release']}")
        console.print(f"  Architecture: {results['platform']['arch']}")
        if "mac_version" in results["platform"]:
            console.print(f"  macOS: {results['platform']['mac_version']}")
            for tool, version in results["platform"].get("optional_tools", {}).items():
                tool_status = "[green]✓[/green]" if version != "not found" else "[yellow]⚠️[/yellow]"
                console.print(f"  {tool}: {version} {tool_status}")

        # Backends
        from ...backends.registry import BackendRegistry
        backend_validations = BackendRegistry.list_backends()
        results["backends"] = {
            name: {
                "is_valid": v.is_valid,
                "errors": v.errors,
                "warnings": v.warnings,
                "info": v.info
            }
            for name, v in backend_validations.items()
        }
        console.print("\n[bold]Available Backends:[/bold]")
        for backend_name, validation in backend_validations.items():
            status = "[green]✓[/green]" if validation.is_valid else "[red]✗[/red]"
            console.print(f"  {status} {backend_name}")

            # Show info
            for key, value in validation.info.items():
                console.print(f"      {key}: {value}")

            # Show warnings
            if validation.warnings:
                for warning in validation.warnings:
                    console.print(f"      [yellow]⚠️  {warning}[/yellow]")

            # Show errors
            if validation.errors:
                for error in validation.errors:
                    console.print(f"      [red]✗ {error}[/red]")

        # MLBuild Directory
        console.print("\n[bold].mlbuild Directory:[/bold]")
        if mlbuild_dir_results["exists"]:
            read_status = "[green]✓[/green]" if mlbuild_dir_results["readable"] else "[red]✗[/red]"
            write_status = "[green]✓[/green]" if mlbuild_dir_results["writable"] else "[red]✗[/red]"
            console.print(f"  Exists: {read_status}")
            console.print(f"  Writable: {write_status}")
            console.print(f"  Artifacts: {mlbuild_dir_results.get('artifacts', 0)}")
            console.print(f"  Registry Size: {mlbuild_dir_results.get('registry_size_kb') or 0:.1f} KB")
        else:
            console.print("  [yellow]Not initialized (will be created on first build)[/yellow]")

        # Summary
        console.print()
        if failures:
            console.print("[bold red]✗ Some checks failed[/bold red]")
            for f in failures:
                console.print(f"  {f}")
        else:
            console.print("[bold green]✓ All checks passed![/bold green]")

    # Exit code
    if failures and not soft:
        sys.exit(1)