"""
MLBuild Error System

Design goals:
- Single canonical error code namespace (E#### format only)
- Explicit category per error (not prefix-derived)
- Immutable error_code per class
- Stable exit codes for CI integration
- Machine-safe formatting (no emoji, no decoration)
- Deterministic fingerprinting (not message-based)
- Support for structured details dict
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Exit Codes (Process-Level Contract)
# ---------------------------------------------------------------------

class ExitCode(int, Enum):
    """
    Stable process exit codes.
    These values are part of the public CLI contract.
    """
    OK = 0

    PLATFORM_ERROR = 10
    MODEL_LOAD_ERROR = 11
    CONVERSION_ERROR = 12
    VALIDATION_ERROR = 13
    BENCHMARK_ERROR = 14
    CONVERT_ERROR = 15
    CONVERSION_CANCELLED = 16
    ADB_ERROR       = 20
    DEPLOY_ERROR    = 21
    EXECUTION_ERROR = 22
    IDB_ERROR = 30
    
    INTERNAL_ERROR = 99


# ---------------------------------------------------------------------
# Error Categories (Explicit, Not Derived)
# ---------------------------------------------------------------------

class ErrorCategory(str, Enum):
    USER = "user_error"
    ENVIRONMENT = "env_error"
    MODEL = "model_error"
    BENCHMARK = "benchmark_error"
    INTERNAL = "internal_error"


# ---------------------------------------------------------------------
# Canonical Error Codes (Single Namespace)
# ---------------------------------------------------------------------

class ErrorCode(str, Enum):
    # 1xxx – User / Input
    INVALID_MODEL = "E1001"
    UNSUPPORTED_MODEL = "E1002"

    # 2xxx – Environment / Platform
    PLATFORM_UNSUPPORTED = "E2001"
    DEPENDENCY_MISSING = "E2002"

    # 3xxx – Model / Conversion
    MODEL_LOAD_FAILED = "E3001"
    CONVERSION_FAILED = "E3002"
    UNSUPPORTED_OPERATOR = "E3003"
    CONVERSION_CANCELLED = "E3004"
    CONVERSION_TIMEOUT   = "E3005"
    CONVERSION_CACHED    = "E3006"
    NO_CONVERSION_PATH   = "E3007"
    STATE_DICT_DETECTED  = "E3008"

    # 4xxx – Benchmark / Device
    BENCHMARK_FAILED = "E4001"
    DEVICE_UNSUPPORTED = "E4002"

    # 5xxx – ADB / Android
    ADB_NOT_FOUND          = "E5001"
    ADB_NO_DEVICE          = "E5002"
    ADB_UNAUTHORIZED       = "E5003"
    ADB_OFFLINE            = "E5004"
    ADB_MULTIPLE_DEVICES   = "E5005"
    ADB_TIMEOUT            = "E5006"
    ADB_UNSUPPORTED_ABI    = "E5007"
    ADB_DEPLOY_FAILED      = "E5008"
    ADB_EXECUTION_FAILED   = "E5009"
    ADB_PARSE_FAILED       = "E5010"

    # 6xxx – IDB / iOS
    IDB_NOT_FOUND              = "E6001"
    IDB_COMPANION_NOT_RUNNING  = "E6002"
    IDB_NO_DEVICE              = "E6003"
    IDB_UNAUTHORIZED           = "E6004"
    IDB_OFFLINE                = "E6005"
    IDB_MULTIPLE_DEVICES       = "E6006"
    IDB_TIMEOUT                = "E6007"
    IDB_UNSIGNED_BINARY        = "E6008"
    IDB_SIMULATOR_BOOT_FAILED  = "E6009"
    IDB_DEPLOY_FAILED          = "E6010"
    IDB_EXECUTION_FAILED       = "E6011"
    IDB_PARSE_FAILED           = "E6012"

    # 9xxx – Internal
    INTERNAL_ERROR = "E9001"


# ---------------------------------------------------------------------
# Deterministic Fingerprint
# ---------------------------------------------------------------------

def compute_fingerprint(
    *,
    error_code: ErrorCode,
    stage: Optional[str],
    signature: Optional[str],
) -> str:
    """
    Deterministic fingerprint derived from:
    - error_code
    - stage
    - optional structural signature (e.g., operator name)

    Message text is NOT included.
    """
    import hashlib

    payload = f"{error_code.value}|{stage or ''}|{signature or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------
# Base Error
# ---------------------------------------------------------------------

@dataclass
class MLBuildError(Exception):
    """
    Base class for all MLBuild domain errors.

    Invariants:
    - error_code is immutable
    - category is explicit
    - exit_code is explicit
    - details dict for structured diagnostic info
    """

    message: str
    error_code: ErrorCode
    category: ErrorCategory
    exit_code: ExitCode
    stage: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None  # NEW: structured diagnostic info
    signature: Optional[str] = None  # used for fingerprint stability

    def __post_init__(self):
        if not isinstance(self.error_code, ErrorCode):
            raise TypeError("error_code must be an ErrorCode enum")

        if not isinstance(self.category, ErrorCategory):
            raise TypeError("category must be an ErrorCategory enum")

        if not isinstance(self.exit_code, ExitCode):
            raise TypeError("exit_code must be an ExitCode enum")

        self.context = self.context or {}
        self.details = self.details or {}

        # Compute deterministic fingerprint
        self.fingerprint = compute_fingerprint(
            error_code=self.error_code,
            stage=self.stage,
            signature=self.signature,
        )

        super().__init__(self.message)

    # -----------------------------------------------------------------
    # Structured Output
    # -----------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """
        JSON-safe representation for CLI or telemetry.
        """
        return {
            "code": self.error_code.value,
            "category": self.category.value,
            "message": self.message,
            "stage": self.stage,
            "context": self.context,
            "details": self.details,
            "fingerprint": self.fingerprint,
        }

    # -----------------------------------------------------------------
    # Plain Text (Machine Safe)
    # -----------------------------------------------------------------

    def format(self) -> str:
        """
        Plain multi-line representation.
        No emoji, no decoration.
        """
        lines = [
            f"{self.__class__.__name__}: {self.message}",
            f"  code: {self.error_code.value}",
            f"  category: {self.category.value}",
            f"  fingerprint: {self.fingerprint}",
        ]

        if self.stage:
            lines.append(f"  stage: {self.stage}")

        if self.context:
            lines.append("  context:")
            for k, v in self.context.items():
                lines.append(f"    {k}: {v}")
        
        if self.details:
            lines.append("  details:")
            for k, v in self.details.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)


# ---------------------------------------------------------------------
# Domain-Specific Errors
# ---------------------------------------------------------------------

class PlatformError(MLBuildError):
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.PLATFORM_UNSUPPORTED,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.PLATFORM_ERROR,
            **kwargs,
        )


class ModelLoadError(MLBuildError):
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOAD_FAILED,
            category=ErrorCategory.MODEL,
            exit_code=ExitCode.MODEL_LOAD_ERROR,
            context={"model_path": model_path} if model_path else None,
            **kwargs,
        )


class UnsupportedModelError(MLBuildError):
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNSUPPORTED_MODEL,
            category=ErrorCategory.USER,
            exit_code=ExitCode.MODEL_LOAD_ERROR,
            context={"model_path": model_path} if model_path else None,
            **kwargs,
        )


class ConversionError(MLBuildError):
    def __init__(self, message: str, operator: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONVERSION_FAILED,
            category=ErrorCategory.MODEL,
            exit_code=ExitCode.CONVERSION_ERROR,
            signature=operator,
            context={"operator": operator} if operator else None,
            **kwargs,
        )

class ConvertError(MLBuildError):
    """
    Pipeline-level conversion error.
    Distinct from ConversionError (operator-level).
    Used by mlbuild convert for routing, timeout, and validation failures.
    """
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CONVERSION_FAILED,
        **kwargs,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.MODEL,
            exit_code=ExitCode.CONVERT_ERROR,
            stage=stage,
            **kwargs,
        )


class ConversionCancelled(MLBuildError):
    """
    Raised on SIGINT (Ctrl+C) during a conversion run.
    Signals clean cancellation — not a failure.
    Temp files are preserved on this exit path.
    """
    def __init__(
        self,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        tmp_dir: Optional[str] = None,
    ):
        super().__init__(
            message="Conversion cancelled by user.",
            error_code=ErrorCode.CONVERSION_CANCELLED,
            category=ErrorCategory.USER,
            exit_code=ExitCode.CONVERSION_CANCELLED,
            stage=stage,
            context={
                "run_id": run_id,
                "tmp_dir": tmp_dir,
            },
        )


class ModelValidationError(MLBuildError):
    def __init__(self, message: str, artifact: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_MODEL,
            category=ErrorCategory.MODEL,
            exit_code=ExitCode.VALIDATION_ERROR,
            context={"artifact": artifact} if artifact else None,
            **kwargs,
        )


class BenchmarkError(MLBuildError):
    def __init__(self, message: str, device: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.BENCHMARK_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.BENCHMARK_ERROR,
            context={"device": device} if device else None,
            **kwargs,
        )


class InternalError(MLBuildError):
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            category=ErrorCategory.INTERNAL,
            exit_code=ExitCode.INTERNAL_ERROR,
            **kwargs,
        )

# ---------------------------------------------------------------------
# Android / ADB Errors
# ---------------------------------------------------------------------

class ADBNotFoundError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message="ADB not found. Install Android SDK Platform Tools and ensure 'adb' is in your PATH.",
            error_code=ErrorCode.ADB_NOT_FOUND,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.ADB_ERROR,
            stage="adb_init",
            **kwargs,
        )


class ADBNoDeviceError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message="No Android device detected. Connect via USB and enable USB debugging.",
            error_code=ErrorCode.ADB_NO_DEVICE,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.ADB_ERROR,
            stage="device_discovery",
            **kwargs,
        )


class ADBUnauthorizedError(MLBuildError):
    def __init__(self, serial: Optional[str] = None, **kwargs):
        super().__init__(
            message="Device not authorized. Accept the USB debugging prompt on your phone.",
            error_code=ErrorCode.ADB_UNAUTHORIZED,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.ADB_ERROR,
            stage="device_discovery",
            context={"serial": serial} if serial else None,
            **kwargs,
        )


class ADBOfflineError(MLBuildError):
    def __init__(self, serial: Optional[str] = None, **kwargs):
        super().__init__(
            message="Device went offline after retries. Try unplugging and reconnecting.",
            error_code=ErrorCode.ADB_OFFLINE,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.ADB_ERROR,
            stage="device_discovery",
            context={"serial": serial} if serial else None,
            **kwargs,
        )


class ADBMultipleDevicesError(MLBuildError):
    def __init__(self, serials: list[str], **kwargs):
        super().__init__(
            message=f"Multiple devices found: {', '.join(serials)}. Use --serial to specify one.",
            error_code=ErrorCode.ADB_MULTIPLE_DEVICES,
            category=ErrorCategory.USER,
            exit_code=ExitCode.ADB_ERROR,
            stage="device_discovery",
            context={"serials": serials},
            **kwargs,
        )


class ADBTimeoutError(MLBuildError):
    def __init__(self, command: Optional[str] = None, **kwargs):
        super().__init__(
            message="ADB command timed out. The device may be unresponsive.",
            error_code=ErrorCode.ADB_TIMEOUT,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.ADB_ERROR,
            stage="adb_transport",
            context={"command": command} if command else None,
            **kwargs,
        )


class UnsupportedABIError(MLBuildError):
    def __init__(self, abi: str, **kwargs):
        super().__init__(
            message=f"No benchmark binary for ABI '{abi}'. Open an issue on GitHub.",
            error_code=ErrorCode.ADB_UNSUPPORTED_ABI,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.DEPLOY_ERROR,
            stage="deploy",
            context={"abi": abi},
            signature=abi,
            **kwargs,
        )


class DeployError(MLBuildError):
    def __init__(self, detail: str, run_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Failed to deploy to device: {detail}",
            error_code=ErrorCode.ADB_DEPLOY_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.DEPLOY_ERROR,
            stage="deploy",
            context={"run_id": run_id} if run_id else None,
            **kwargs,
        )


class ExecutionError(MLBuildError):
    def __init__(self, raw_stdout: str, run_id: Optional[str] = None, **kwargs):
        super().__init__(
            message="Benchmark crashed on device. Raw stdout attached.",
            error_code=ErrorCode.ADB_EXECUTION_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.EXECUTION_ERROR,
            stage="benchmark_execution",
            context={"run_id": run_id} if run_id else None,
            details={"raw_stdout": raw_stdout},
            **kwargs,
        )


class ParseError(MLBuildError):
    def __init__(self, raw_stdout: str, **kwargs):
        super().__init__(
            message="Could not parse benchmark output. Raw stdout attached.",
            error_code=ErrorCode.ADB_PARSE_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.EXECUTION_ERROR,
            stage="result_parsing",
            details={"raw_stdout": raw_stdout},
            **kwargs,
        )

# ---------------------------------------------------------------------
# iOS / IDB Errors
# ---------------------------------------------------------------------

class IDBNotFoundError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message="idb not found. Install via: brew install idb-companion",
            error_code=ErrorCode.IDB_NOT_FOUND,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="idb_init",
            **kwargs,
        )


class IDBCompanionNotRunningError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message="idb_companion is not running. Start with: idb_companion --daemon",
            error_code=ErrorCode.IDB_COMPANION_NOT_RUNNING,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="idb_init",
            **kwargs,
        )


class IDBNoDeviceError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message="No iOS simulator detected. Boot one via: xcrun simctl boot <udid>",
            error_code=ErrorCode.IDB_NO_DEVICE,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="device_discovery",
            **kwargs,
        )


class IDBUnauthorizedError(MLBuildError):
    def __init__(self, udid: Optional[str] = None, **kwargs):
        super().__init__(
            message="Device not trusted. Unlock your iPhone and tap 'Trust' when prompted.",
            error_code=ErrorCode.IDB_UNAUTHORIZED,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="device_discovery",
            context={"udid": udid} if udid else None,
            **kwargs,
        )


class IDBOfflineError(MLBuildError):
    def __init__(self, udid: Optional[str] = None, **kwargs):
        super().__init__(
            message="Device went offline after retries. Try unplugging and reconnecting.",
            error_code=ErrorCode.IDB_OFFLINE,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="device_discovery",
            context={"udid": udid} if udid else None,
            **kwargs,
        )


class IDBMultipleDevicesError(MLBuildError):
    def __init__(self, udids: list[str], **kwargs):
        super().__init__(
            message=f"Multiple targets found: {', '.join(udids)}. Use --udid to specify one.",
            error_code=ErrorCode.IDB_MULTIPLE_DEVICES,
            category=ErrorCategory.USER,
            exit_code=ExitCode.IDB_ERROR,
            stage="device_discovery",
            context={"udids": udids},
            **kwargs,
        )


class IDBTimeoutError(MLBuildError):
    def __init__(self, command: Optional[str] = None, **kwargs):
        super().__init__(
            message="idb command timed out. The target may be unresponsive.",
            error_code=ErrorCode.IDB_TIMEOUT,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="idb_transport",
            context={"command": command} if command else None,
            **kwargs,
        )


class UnsignedBinaryError(MLBuildError):
    def __init__(self, **kwargs):
        super().__init__(
            message=(
                "Real device benchmarking requires a signed MLBuildRunner.app. "
                "See: mlbuild.dev/ios-signing  "
                "Use --signed-app <path> to provide your own signed build, "
                "or run against a simulator instead."
            ),
            error_code=ErrorCode.IDB_UNSIGNED_BINARY,
            category=ErrorCategory.USER,
            exit_code=ExitCode.IDB_ERROR,
            stage="deploy",
            **kwargs,
        )


class SimulatorBootError(MLBuildError):
    def __init__(self, udid: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Simulator failed to boot. Try: xcrun simctl boot {udid or '<udid>'}",
            error_code=ErrorCode.IDB_SIMULATOR_BOOT_FAILED,
            category=ErrorCategory.ENVIRONMENT,
            exit_code=ExitCode.IDB_ERROR,
            stage="device_discovery",
            context={"udid": udid} if udid else None,
            **kwargs,
        )


class IDBDeployError(MLBuildError):
    def __init__(self, detail: str, run_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Failed to deploy to iOS target: {detail}",
            error_code=ErrorCode.IDB_DEPLOY_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.DEPLOY_ERROR,
            stage="deploy",
            context={"run_id": run_id} if run_id else None,
            **kwargs,
        )


class IDBExecutionError(MLBuildError):
    def __init__(self, raw_stdout: str, run_id: Optional[str] = None, **kwargs):
        super().__init__(
            message="Benchmark crashed on iOS target. Raw stdout attached.",
            error_code=ErrorCode.IDB_EXECUTION_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.EXECUTION_ERROR,
            stage="benchmark_execution",
            context={"run_id": run_id} if run_id else None,
            details={"raw_stdout": raw_stdout},
            **kwargs,
        )


class IDBParseError(MLBuildError):
    def __init__(self, raw_stdout: str, **kwargs):
        super().__init__(
            message="Could not parse iOS benchmark output. Raw stdout attached.",
            error_code=ErrorCode.IDB_PARSE_FAILED,
            category=ErrorCategory.BENCHMARK,
            exit_code=ExitCode.EXECUTION_ERROR,
            stage="result_parsing",
            details={"raw_stdout": raw_stdout},
            **kwargs,
        )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "ExitCode",
    "ErrorCategory",
    "ErrorCode",
    "MLBuildError",
    "PlatformError",
    "ModelLoadError",
    "UnsupportedModelError",
    "ConversionError",
    "ConvertError",
    "ConversionCancelled",
    "ModelValidationError",
    "BenchmarkError",
    "InternalError",
    "ADBNotFoundError",
    "ADBNoDeviceError",
    "ADBUnauthorizedError",
    "ADBOfflineError",
    "ADBMultipleDevicesError",
    "ADBTimeoutError",
    "UnsupportedABIError",
    "DeployError",
    "ExecutionError",
    "ParseError",
    "IDBNotFoundError",
    "IDBCompanionNotRunningError",
    "IDBNoDeviceError",
    "IDBUnauthorizedError",
    "IDBOfflineError",
    "IDBMultipleDevicesError",
    "IDBTimeoutError",
    "UnsignedBinaryError",
    "SimulatorBootError",
    "IDBDeployError",
    "IDBExecutionError",
    "IDBParseError",
]