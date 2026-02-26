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

    # 4xxx – Benchmark / Device
    BENCHMARK_FAILED = "E4001"
    DEVICE_UNSUPPORTED = "E4002"

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
    "ModelValidationError",
    "BenchmarkError",
    "InternalError",
]