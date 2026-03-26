"""
Feature → OS compatibility validation for mlbuild convert.

Guarantees:
- No silent incompatibilities
- No cross-platform comparisons (iOS vs macOS)
- Unknown OS versions fail fast (system must be updated)
- Feature requirements strictly enforced (hard vs soft)

Scope:
- v1: feature inference from ConvertParams only (no graph inspection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import logging

from mlbuild.convert.coreml import TARGET_OS_MAP
from mlbuild.core.errors import ConvertError, ErrorCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# OS utilities (future-proof, no static ordering dependence)
# ---------------------------------------------------------------------

Platform = Literal["iOS", "macOS"]


def get_platform(os_str: str) -> Platform:
    if os_str.startswith("iOS"):
        return "iOS"
    if os_str.startswith("macOS"):
        return "macOS"
    raise ConvertError(
        f"Unknown OS platform: '{os_str}'",
        stage="feature_compat",
        error_code=ErrorCode.CONVERSION_FAILED,
    )


def _version_num(os_str: str) -> int:
    try:
        return int(os_str.replace("iOS", "").replace("macOS", ""))
    except Exception:
        raise ConvertError(
            f"Invalid OS version format: '{os_str}'",
            stage="feature_compat",
            error_code=ErrorCode.CONVERSION_FAILED,
        )


def compare_os(a: str, b: str) -> int:
    """Return positive if a > b, negative if a < b, 0 if equal."""
    return _version_num(a) - _version_num(b)


# ---------------------------------------------------------------------
# Feature requirements (strict typing)
# ---------------------------------------------------------------------

class FeatureRequirement(TypedDict):
    min_os: dict[Platform, str]
    type: Literal["hard", "soft"]
    description: str


FEATURE_OS_REQUIREMENTS: dict[str, FeatureRequirement] = {
    "mlprogram_format": {
        "min_os": {"iOS": "iOS15", "macOS": "macOS13"},
        "type": "hard",
        "description": "ML Program format",
    },
    "fp16_weights": {
        "min_os": {"iOS": "iOS16", "macOS": "macOS13"},
        "type": "hard",
        "description": "FP16 weight quantization",
    },
    "int8_quantization": {
        "min_os": {"iOS": "iOS16", "macOS": "macOS13"},
        "type": "hard",
        "description": "INT8 weight quantization",
    },
    "flexible_shapes": {
        "min_os": {"iOS": "iOS17", "macOS": "macOS14"},
        "type": "soft",
        "description": "Flexible input shapes",
    },
    "int4_quantization": {
        "min_os": {"iOS": "iOS18", "macOS": "macOS15"},
        "type": "hard",
        "description": "INT4 weight quantization",
    },
}


# ---------------------------------------------------------------------
# Validation issue
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    feature: str
    description: str
    required_os: str
    target: str
    target_os: str
    requirement_type: Literal["hard", "soft"]

    def format(self) -> str:
        severity = "Error" if self.requirement_type == "hard" else "Warning"
        fixes = self._format_fixes()

        return (
            f"{severity}: {self.description} requires {self.required_os}, "
            f"but {self.target} → {self.target_os}.\n\n"
            f"Fix:\n{fixes}"
        )

    def _format_fixes(self) -> str:
        platform = get_platform(self.target_os)
        required_version = _version_num(self.required_os)

        # Sort targets by OS version (not name)
        valid_targets = sorted(
            TARGET_OS_MAP.items(),
            key=lambda x: _version_num(x[1])
        )

        lines = []

        for t, os_ in valid_targets:
            if t == self.target:
                continue
            if get_platform(os_) != platform:
                continue
            if _version_num(os_) >= required_version:
                lines.append(f"  • Use --target {t} ({os_})")

        lines.append(
            f"  • Or remove {self.feature.replace('_', ' ')} from your model"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------
# Feature inference (strict, no normalization hacks)
# ---------------------------------------------------------------------

def infer_features(params) -> list[str]:
    features = ["mlprogram_format"]

    quantize_val = params.quantize.value if hasattr(params.quantize, "value") else str(params.quantize)

    if quantize_val == "fp16":
        features.append("fp16_weights")
    elif quantize_val == "int8":
        features.append("int8_quantization")
    elif quantize_val == "int4":
        features.append("int4_quantization")

    return features


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_features_against_target(
    features: list[str],
    target: str,
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:

    if not target:
        raise ConvertError(
            "Target must be specified for validation",
            stage="feature_compat",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    if target not in TARGET_OS_MAP:
        raise ConvertError(
            f"Unknown target '{target}'. Known: {sorted(TARGET_OS_MAP)}",
            stage="feature_compat",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    target_os = TARGET_OS_MAP[target]
    platform = get_platform(target_os)

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    for feature in features:
        req = FEATURE_OS_REQUIREMENTS.get(feature)

        if req is None:
            logger.debug(f"Unknown feature skipped: {feature}")
            continue

        required_os = req["min_os"].get(platform)
        if required_os is None:
            continue  # feature not applicable to this platform

        # HARD FAIL on unknown OS format
        cmp = compare_os(target_os, required_os)

        if cmp >= 0:
            continue  # compatible

        issue = ValidationIssue(
            feature=feature,
            description=req["description"],
            required_os=required_os,
            target=target,
            target_os=target_os,
            requirement_type=req["type"],
        )

        if req["type"] == "hard":
            errors.append(issue)
        else:
            warnings.append(issue)

    return errors, warnings