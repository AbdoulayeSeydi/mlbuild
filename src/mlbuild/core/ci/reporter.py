"""
MLBuild CI Report

Immutable CI artifact used by:
- CLI output
- GitHub Actions
- dashboards
- automated performance gates
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# TYPES
# ---------------------------------------------------------------------

Metric = Literal["latency", "size", "accuracy_cosine", "accuracy_top1"]
Rule = Literal["regression", "budget", "accuracy"]
Result = Literal["pass", "fail"]


@dataclass(frozen=True)
class Violation:
    metric: Metric
    rule: Rule
    actual: float
    threshold: float
    unit: str
    message: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "rule": self.rule,
            "actual": self.actual,
            "threshold": self.threshold,
            "unit": self.unit,
            "message": self.message,
        }


# ---------------------------------------------------------------------
# ENVIRONMENT METADATA
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentInfo:
    timestamp: str
    device: Optional[str]
    backend: Optional[str]
    mlbuild_version: Optional[str]

    @staticmethod
    def now(
        device: Optional[str] = None,
        backend: Optional[str] = None,
        mlbuild_version: Optional[str] = None,
    ) -> "EnvironmentInfo":
        return EnvironmentInfo(
            timestamp=datetime.now(timezone.utc).isoformat(),
            device=device,
            backend=backend,
            mlbuild_version=mlbuild_version,
        )


# ---------------------------------------------------------------------
# CI REPORT
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CIReport:
    model: str

    environment: EnvironmentInfo

    # Baseline
    baseline_tag: Optional[str]
    baseline_build_id: str
    baseline_latency_ms: Optional[float]
    baseline_size_mb: float

    # Candidate
    candidate_build_id: str
    candidate_variant: str
    candidate_parent_build_id: Optional[str]
    candidate_latency_ms: Optional[float]
    candidate_size_mb: float

    # Deltas
    latency_delta_pct: Optional[float]
    size_delta_pct: float

    # Thresholds
    latency_regression_pct: float
    size_regression_pct: float
    latency_budget_ms: Optional[float]
    size_budget_mb: Optional[float]

    # Accuracy
    accuracy_cosine: Optional[float] = None
    accuracy_top1: Optional[float] = None
    accuracy_passed: Optional[bool] = None

    # Result
    result: Result = "pass"

    violations: List[Violation] = field(default_factory=list)

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def short_id(build_id: str) -> str:
        return build_id[:16]

    @staticmethod
    def fmt_latency(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.2f} ms"

    @staticmethod
    def fmt_size(value: float) -> str:
        return f"{value:.2f} MB"

    @staticmethod
    def fmt_pct(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:+.2f}%"

    # -----------------------------------------------------------------
    # DATA MODEL
    # -----------------------------------------------------------------

    @property
    def passed(self) -> bool:
        return self.result == "pass"

    def to_dict(self) -> dict:
        accuracy_present = (
            self.accuracy_cosine is not None
            or self.accuracy_top1 is not None
            or self.accuracy_passed is not None
        )

        return {
            "model": self.model,
            "environment": {
                "timestamp": self.environment.timestamp,
                "device": self.environment.device,
                "backend": self.environment.backend,
                "mlbuild_version": self.environment.mlbuild_version,
            },
            "baseline": {
                "tag": self.baseline_tag,
                "build_id": self.baseline_build_id,
                "latency_ms": self.baseline_latency_ms,
                "size_mb": self.baseline_size_mb,
            },
            "candidate": {
                "build_id": self.candidate_build_id,
                "variant": self.candidate_variant,
                "parent_build_id": self.candidate_parent_build_id,
                "latency_ms": self.candidate_latency_ms,
                "size_mb": self.candidate_size_mb,
            },
            "delta": {
                "latency_pct": round(self.latency_delta_pct, 2)
                if self.latency_delta_pct is not None
                else None,
                "size_pct": round(self.size_delta_pct, 2),
            },
            "thresholds": {
                "latency_regression_pct": self.latency_regression_pct,
                "size_regression_pct": self.size_regression_pct,
                "latency_budget_ms": self.latency_budget_ms,
                "size_budget_mb": self.size_budget_mb,
            },
            "accuracy": None
            if not accuracy_present
            else {
                "cosine_similarity": self.accuracy_cosine,
                "top1_agreement": self.accuracy_top1,
                "passed": self.accuracy_passed,
            },
            "result": self.result,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # -----------------------------------------------------------------
    # TEXT FORMAT
    # -----------------------------------------------------------------

    def to_text(self) -> str:
        data = self.to_dict()

        baseline = data["baseline"]
        candidate = data["candidate"]
        delta = data["delta"]

        lines = [
            "",
            "MLBuild CI Report",
            "─" * 50,
            "",
            f"Model:     {self.model}",
        ]

        baseline_line = f"Baseline:  {self.short_id(baseline['build_id'])}"
        if baseline["tag"]:
            baseline_line += f" ({baseline['tag']})"

        lines.append(baseline_line)

        lines.append(
            f"Candidate: {self.short_id(candidate['build_id'])} ({candidate['variant']})"
        )
        lines.append("")

        lines += [
            f"{'':20s} {'Baseline':>12s}  {'Candidate':>12s}  {'Delta':>10s}",
            f"{'Latency (p50)':20s} {self.fmt_latency(baseline['latency_ms']):>12s}  "
            f"{self.fmt_latency(candidate['latency_ms']):>12s}  "
            f"{self.fmt_pct(delta['latency_pct']):>10s}",
            f"{'Size':20s} {self.fmt_size(baseline['size_mb']):>12s}  "
            f"{self.fmt_size(candidate['size_mb']):>12s}  "
            f"{self.fmt_pct(delta['size_pct']):>10s}",
        ]

        acc = data["accuracy"]
        if acc:
            status = "✓" if acc["passed"] else "✗"
            lines.append(
                f"{'Cosine sim':20s} {'—':>12s}  {acc['cosine_similarity']:.4f:>12}  {status:>10}"
            )

        lines.append("")
        lines.append("Thresholds:")
        t = data["thresholds"]

        lines.append(f"  latency regression allowed: {t['latency_regression_pct']}%")
        lines.append(f"  size regression allowed:    {t['size_regression_pct']}%")

        if t["latency_budget_ms"] is not None:
            lines.append(f"  latency budget:             {t['latency_budget_ms']} ms")

        if t["size_budget_mb"] is not None:
            lines.append(f"  size budget:                {t['size_budget_mb']} MB")

        lines.append("")
        if self.result == "pass":
            lines.append("Result: ✓ PASS")
        elif self.result == "skip":
            lines.append("Result: ⚠ SKIPPED (baseline not found)")
        else:
            lines.append("Result: ✗ FAIL")

        for v in data["violations"]:
            lines.append(f"  • {v['message']}")

        lines.append("")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # MARKDOWN FORMAT
    # -----------------------------------------------------------------

    def to_markdown(self) -> str:
        data = self.to_dict()

        status = "✅ PASS" if self.passed else "❌ FAIL"
        baseline = data["baseline"]
        candidate = data["candidate"]
        delta = data["delta"]

        lines = [
            f"## 🤖 MLBuild CI Report — {status}",
            "",
            f"**Model:** `{self.model}`  ",
            f"**Baseline:** `{self.short_id(baseline['build_id'])}`",
            f"**Candidate:** `{self.short_id(candidate['build_id'])}` ({candidate['variant']})",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "|--------|----------|-----------|-------|",
            f"| Latency (p50) | {self.fmt_latency(baseline['latency_ms'])} | "
            f"{self.fmt_latency(candidate['latency_ms'])} | {self.fmt_pct(delta['latency_pct'])} |",
            f"| Size | {self.fmt_size(baseline['size_mb'])} | "
            f"{self.fmt_size(candidate['size_mb'])} | {self.fmt_pct(delta['size_pct'])} |",
        ]

        acc = data["accuracy"]
        if acc:
            status = "✓" if acc["passed"] else "✗"
            lines.append(
                f"| Cosine sim | — | {acc['cosine_similarity']:.4f} | {status} |"
            )

        if data["violations"]:
            lines.append("")
            lines.append("**Violations:**")
            for v in data["violations"]:
                lines.append(f"- {v['message']}")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # FILE WRITE
    # -----------------------------------------------------------------

    def save(self, workspace_root: Path) -> Path:
        """
        Write .mlbuild/ci_report.json using atomic write.
        """

        out_path = workspace_root / ".mlbuild" / "ci_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=out_path.parent)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(self.to_json())

            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        logger.info("CI report written to %s", out_path)
        return out_path