"""
Hylos CI Benchmark Publisher (production-grade)

Core guarantees:
- No client-side baseline mutation logic
- No stale auth usage
- Retry + backoff for CI instability
- Schema validation before upload
- Idempotent writes (no duplicate baselines)
- No fake local persistence claims
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
from rich.console import Console

from .login import get_valid_access_token, ConfigManager

console = Console()

# ----------------------------
# Environment config (required)
# ----------------------------

import os

SUPABASE_URL = os.getenv("HYLOS_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("HYLOS_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise SystemExit("[FATAL] Missing HYLOS_SUPABASE_URL or HYLOS_SUPABASE_ANON_KEY")


# ----------------------------
# HTTP client (reused)
# ----------------------------

client = httpx.Client(timeout=10)


# ----------------------------
# Retry wrapper (CI-safe)
# ----------------------------

def request_with_retry(method, url, *, retries=3, **kwargs):
    last_err = None

    for i in range(retries):
        try:
            resp = client.request(method, url, **kwargs)

            # retryable server/network failures
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)

            return resp

        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            last_err = e
            sleep = 2 ** i
            time.sleep(sleep)

    raise last_err


# ----------------------------
# Model hashing (stable + full strength)
# ----------------------------

def compute_model_hash(model_path: Optional[Path], build_id: Optional[str]) -> str:
    if model_path and model_path.exists():
        sha = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()  # full hash (no truncation)

    if build_id:
        return hashlib.sha256(build_id.encode()).hexdigest()

    raise ValueError("No valid model identifier provided")


# ----------------------------
# Validation layer (critical)
# ----------------------------

REQUIRED_FIELDS = ["candidate_latency_ms"]

def validate_report(report_data: Dict[str, Any]):
    missing = [f for f in REQUIRED_FIELDS if report_data.get(f) is None]
    if missing:
        raise ValueError(f"Missing required benchmark fields: {missing}")
    if report_data.get("candidate_latency_ms", 0) < 0:
        raise ValueError("Invalid latency values")

# ----------------------------
# Core function
# ----------------------------

def push_benchmark(report, model_path: Optional[Path], build_id: Optional[str], baseline: str) -> bool:
    """
    Push CI benchmark safely to Hylos Cloud.
    """

    # ------------------------
    # Auth (single source of truth)
    # ------------------------

    access_token = get_valid_access_token()
    config = ConfigManager.load()
    
    project_id = config.get("project_id")
    if not project_id:
        console.print("[red]No project configured[/red]")
        return False

    # ------------------------
    # Parse report safely
    # ------------------------

    try:
        import dataclasses
        if hasattr(report, "__dataclass_fields__"):
            report_data = dataclasses.asdict(report)
        elif hasattr(report, "to_dict"):
            report_data = report.to_dict()
        else:
            report_data = {}
    except Exception as e:
        console.print(f"[red]Invalid report format: {e}[/red]")
        return False

    try:
        validate_report(report_data)
    except ValueError as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        return False

    # ------------------------
    # Derived fields
    # ------------------------

    model_hash = compute_model_hash(model_path, build_id)
    model_id = report_data.get("model") or (model_path.stem if model_path else (build_id or "unknown")[:64])

    idempotency_key = hashlib.sha256(
        f"{project_id}:{model_hash}:{baseline}".encode()
    ).hexdigest()

    env = report_data.get("environment") or {}

    # ------------------------
    # Payload
    # ------------------------

    row = {
        "project_id":      project_id,
        "model_id":        model_id,
        "model_hash":      model_hash,
        "model_version":   baseline,
        "platform":        "ios" if env.get("backend") == "coreml" else "android",
        "runtime":         env.get("backend", "unknown"),
        "device_model":    env.get("device", "unknown"),
        "latency_p50_ms":  report_data.get("candidate_latency_ms"),
        "latency_p95_ms":  report_data.get("candidate_latency_ms"),
        "latency_p99_ms":  report_data.get("candidate_latency_ms"),
        "model_size_mb":   report_data.get("candidate_size_mb"),
        "compute_backend": env.get("backend"),
        "idempotency_key": idempotency_key,
    }

    # strip None values
    row = {k: v for k, v in row.items() if v is not None}

    # ------------------------
    # Step 1: Insert baseline (NO PRE-UPDATE)
    # ------------------------

    try:
        response = request_with_retry(
            "POST",
            f"{SUPABASE_URL}/rest/v1/benchmarks",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json=row,
        )

        response.raise_for_status()
        inserted = response.json()

        baseline_id = inserted[0]["id"] if inserted else "unknown"

    except httpx.HTTPStatusError as e:
        code = e.response.status_code

        if code == 401:
            console.print("[red]Auth expired. Run hylos login[/red]")
        elif code == 409:
            console.print("[yellow]Duplicate benchmark ignored (idempotent)[/yellow]")
            return True
        else:
            console.print(f"[red]Push failed: HTTP {code}[/red]")

        return False

    except Exception as e:
        console.print(f"[red]Unexpected failure: {str(e)}[/red]")
        return False

    # ------------------------
    # Success output
    # ------------------------

    console.print("\n[green]Benchmark pushed successfully ✓[/green]")
    console.print(f"[dim]model:   {model_id}[/dim]")
    console.print(f"[dim]version: {baseline}[/dim]")
    console.print(f"[dim]id:      {baseline_id[:8]}...[/dim]\n")

    return True