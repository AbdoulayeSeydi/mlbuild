"""
Hylos CloudSync — automatic telemetry push after every command.

Design principles:
- Silent on failure. NEVER crashes or slows the CLI.
- 5 second timeout max. User never waits.
- Skipped silently if not logged in or no project set.
- Never logs sensitive data (passwords, keys, tokens).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPABASE_URL = os.getenv("HYLOS_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("HYLOS_SUPABASE_ANON_KEY")
CONFIG_PATH = Path.home() / ".hylos" / "config.json"

# Args that should never be sent to cloud
_REDACTED_ARGS = {
    "password", "token", "key", "secret", "api_key",
    "email", "credential", "auth"
}


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _get_token() -> str | None:
    try:
        import keyring
        import httpx
        import base64
        import json as _json
        import time

        config = _load_config()
        email = config.get("email")
        if not email:
            return None

        access = keyring.get_password("hylos-cli", f"{email}_access")
        refresh = keyring.get_password("hylos-cli", f"{email}_refresh")

        if not access:
            return None

        # Always check expiry and refresh proactively
        try:
            payload = access.split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            claims = _json.loads(base64.b64decode(payload))
            exp = claims.get("exp", 0)
            now = time.time()

            # Refresh if expired OR expiring within 24 hours
            if exp - now < 86400 and refresh and SUPABASE_URL and SUPABASE_ANON_KEY:
                resp = httpx.post(
                    f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
                    headers={"apikey": SUPABASE_ANON_KEY},
                    json={"refresh_token": refresh},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    new_access = data.get("access_token")
                    new_refresh = data.get("refresh_token")
                    if new_access:
                        keyring.set_password("hylos-cli", f"{email}_access", new_access)
                        access = new_access
                    if new_refresh:
                        keyring.set_password("hylos-cli", f"{email}_refresh", new_refresh)
        except Exception:
            pass

        return access
    except Exception:
        return None


def _get_project_id() -> str | None:
    return _load_config().get("project_id")


def _redact_args(args: dict) -> dict:
    """Strip sensitive keys from args before sending."""
    return {
        k: "[redacted]" if any(s in k.lower() for s in _REDACTED_ARGS) else v
        for k, v in args.items()
    }


def push(table: str, data: dict[str, Any]) -> bool:
    """
    Push one row to Supabase.
    Returns True on success, False on any failure.
    Never raises.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return False

    token = _get_token()
    project_id = _get_project_id()

    if not token or not project_id:
        return False

    data["project_id"] = project_id
    data["created_at"] = datetime.now(timezone.utc).isoformat()

    # Strip None values
    data = {k: v for k, v in data.items() if v is not None}

    try:
        import httpx
        response = httpx.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=data,
            timeout=5.0,
        )
        return response.status_code in (200, 201)
    except Exception:
        return False


def push_activity(
    command: str,
    args: dict,
    duration_ms: int,
    exit_code: int,
    error_message: str | None = None,
    machine_id: str | None = None,
    machine_name: str | None = None,
    subcommand: str | None = None,
    linked_build_id: str | None = None,
) -> bool:
    """Push a command activity row. Called from main.py _record()."""
    return push("command_activity", {
        "command": command,
        "subcommand": subcommand,
        "args": _redact_args(args),
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "error_message": error_message,
        "machine_id": machine_id,
        "machine_name": machine_name,
        "linked_build_id": linked_build_id,
    })


def push_benchmark(
    local_build_id: str,
    build_name: str | None,
    platform: str | None,
    runtime: str | None,
    device_model: str | None,
    os_version: str | None,
    compute_unit: str | None,
    runs: int | None,
    warmup: int | None,
    latency_p50_ms: float | None,
    latency_p95_ms: float | None,
    latency_p99_ms: float | None,
    memory_peak_mb: float | None,
    thermal_state: str | None = None,
    stability_score: float | None = None,
    stability_band: str | None = None,
    passed_budget: bool | None = None,
    latency_mean_ms: float | None = None,
    latency_std_ms: float | None = None,
    p50_ci_low: float | None = None,
    p50_ci_high: float | None = None,
    autocorrelation_lag1: float | None = None,
    latency_drift_pct: float | None = None,
    failures: int | None = None,
) -> bool:
    return push("benchmark_runs", {
        "local_build_id": local_build_id,
        "build_name": build_name,
        "platform": platform,
        "runtime": runtime,
        "device_model": device_model,
        "os_version": os_version,
        "compute_unit": compute_unit,
        "runs": runs,
        "warmup": warmup,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "latency_p99_ms": latency_p99_ms,
        "memory_peak_mb": memory_peak_mb,
        "thermal_state": thermal_state,
        "stability_score": stability_score,
        "stability_band": stability_band,
        "passed_budget": passed_budget,
        "latency_mean_ms": latency_mean_ms,
        "latency_std_ms": latency_std_ms,
        "p50_ci_low": p50_ci_low,
        "p50_ci_high": p50_ci_high,
        "autocorrelation_lag1": autocorrelation_lag1,
        "latency_drift_pct": latency_drift_pct,
        "failures": failures,
    })


def push_comparison(
    comparison_type: str,
    baseline_build_id: str | None,
    baseline_name: str | None,
    candidate_build_id: str | None,
    candidate_name: str | None,
    latency_delta_pct: float | None,
    size_delta_pct: float | None,
    regression_detected: bool | None,
    verdict: str | None,
    metric_used: str | None = None,
    threshold_pct: float | None = None,
    accuracy_delta: float | None = None,
    variants: list | None = None,
) -> bool:
    return push("comparisons", {
        "comparison_type": comparison_type,
        "baseline_build_id": baseline_build_id,
        "baseline_name": baseline_name,
        "candidate_build_id": candidate_build_id,
        "candidate_name": candidate_name,
        "latency_delta_pct": latency_delta_pct,
        "size_delta_pct": size_delta_pct,
        "accuracy_delta": accuracy_delta,
        "regression_detected": regression_detected,
        "verdict": verdict,
        "metric_used": metric_used,
        "threshold_pct": threshold_pct,
        "variants": variants,
    })


def push_accuracy(
    check_type: str,
    baseline_build_id: str | None,
    candidate_build_id: str | None,
    cosine_similarity: float | None,
    top1_agreement: float | None,
    kl_divergence: float | None,
    js_divergence: float | None,
    rmse: float | None,
    mae: float | None,
    max_error: float | None,
    samples: int | None,
    seed: int | None,
    passed: bool | None,
    cosine_threshold: float | None = None,
    top1_threshold: float | None = None,
) -> bool:
    return push("accuracy_checks", {
        "check_type": check_type,
        "baseline_build_id": baseline_build_id,
        "candidate_build_id": candidate_build_id,
        "cosine_similarity": cosine_similarity,
        "top1_agreement": top1_agreement,
        "kl_divergence": kl_divergence,
        "js_divergence": js_divergence,
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "samples": samples,
        "seed": seed,
        "passed": passed,
        "cosine_threshold": cosine_threshold,
        "top1_threshold": top1_threshold,
    })


def push_profile(
    local_build_id: str,
    build_name: str | None,
    platform: str | None,
    device_model: str | None,
    cold_start_ms: float | None,
    warm_p50_ms: float | None,
    warm_p95_ms: float | None,
    memory_peak_mb: float | None,
    bottleneck_layer: str | None,
    top_layers: list | None,
) -> bool:
    return push("profile_results", {
        "local_build_id": local_build_id,
        "build_name": build_name,
        "platform": platform,
        "device_model": device_model,
        "cold_start_ms": cold_start_ms,
        "warm_p50_ms": warm_p50_ms,
        "warm_p95_ms": warm_p95_ms,
        "memory_peak_mb": memory_peak_mb,
        "bottleneck_layer": bottleneck_layer,
        "top_layers": top_layers,
    })


def push_build(
    local_build_id: str,
    name: str | None,
    format: str | None,
    target_device: str | None,
    quantization: str | None,
    size_mb: float | None,
    task_type: str | None,
    subtype: str | None = None,
    source_hash: str | None = None,
    notes: str | None = None,
    pinned: bool = False,
    source_command: str = "import",
) -> bool:
    return push("builds", {
        "local_build_id": local_build_id,
        "name": name,
        "format": format,
        "target_device": target_device,
        "quantization": quantization,
        "size_mb": size_mb,
        "task_type": task_type,
        "subtype": subtype,
        "source_hash": source_hash,
        "notes": notes,
        "pinned": pinned,
        "source_command": source_command,
    })


def push_ci_check(
    baseline_build_id: str | None,
    candidate_build_id: str | None,
    latency_threshold_pct: float | None,
    size_threshold_pct: float | None,
    latency_delta_pct: float | None,
    size_delta_pct: float | None,
    accuracy_delta: float | None,
    passed: bool,
    exit_code: int,
) -> bool:
    return push("ci_checks", {
        "baseline_build_id": baseline_build_id,
        "candidate_build_id": candidate_build_id,
        "latency_threshold_pct": latency_threshold_pct,
        "size_threshold_pct": size_threshold_pct,
        "latency_delta_pct": latency_delta_pct,
        "size_delta_pct": size_delta_pct,
        "accuracy_delta": accuracy_delta,
        "passed": passed,
        "exit_code": exit_code,
    })


def push_tag(
    name: str,
    local_build_id: str | None,
    build_name: str | None,
) -> bool:
    return push("tags", {
        "name": name,
        "local_build_id": local_build_id,
        "build_name": build_name,
    })
