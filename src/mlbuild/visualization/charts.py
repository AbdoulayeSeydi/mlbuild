"""
Enterprise-grade terminal visualization for benchmarking.

Features:
- Deterministic layout (CI-safe)
- Dynamic width scaling
- Statistical context (mean, p50, p95, std)
- Outlier clipping (p99 optional)
- Zero-safe scaling
- Sorted comparison + delta display
- Sparkline-style warmup curve
- Stable-point marking
- ANSI color toggle
"""

from typing import List, Dict, Optional
import numpy as np
from rich.console import Console
from rich.table import Table


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _safe_stats(values: List[float]):
    if not values:
        return 0, 0, 0, 0, 0

    arr = np.array(values)
    mean = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr))
    p99 = float(np.percentile(arr, 99))
    return mean, p50, p95, std, p99


def _get_bar_width(console: Console, fixed_columns: int = 35, max_width: int = 60):
    width = console.size.width
    usable = max(10, width - fixed_columns)
    return min(max_width, usable)


# -------------------------------------------------------------------------
# Histogram
# -------------------------------------------------------------------------

def create_latency_histogram(
    latencies: List[float],
    bins: int = 20,
    clip_p99: bool = False,
    console: Optional[Console] = None,
    use_color: bool = True,
) -> Table:

    console = console or Console(color_system="auto" if use_color else None)

    if not latencies:
        return Table(title="Latency Distribution (no data)")

    mean, p50, p95, std, p99 = _safe_stats(latencies)

    data = np.array(latencies)

    if clip_p99 and len(data) > 1:
        data = data[data <= p99]

    counts, edges = np.histogram(data, bins=bins)

    max_count = max(counts) if counts.any() else 1
    bar_width = _get_bar_width(console)

    title = (
        f"Latency Distribution | mean={mean:.2f}ms "
        f"p50={p50:.2f} p95={p95:.2f} std={std:.2f}"
    )

    table = Table(show_header=True, title=title)
    table.add_column("Range (ms)", style="dim")
    table.add_column("Count", justify="right")
    table.add_column("Distribution")

    for i, count in enumerate(counts):
        range_str = f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        normalized = int((count / max_count) * bar_width)
        bar = "█" * normalized

        table.add_row(range_str, str(int(count)), bar)

    return table


# -------------------------------------------------------------------------
# Comparison Chart
# -------------------------------------------------------------------------

def create_comparison_chart(
    comparisons: Dict[str, float],
    baseline_key: Optional[str] = None,
    console: Optional[Console] = None,
    use_color: bool = True,
) -> Table:

    console = console or Console(color_system="auto" if use_color else None)

    if not comparisons:
        return Table(title="Comparison (no data)")

    # Zero-safe max
    values = list(comparisons.values())
    max_value = max(values) if any(values) else 1.0

    bar_width = _get_bar_width(console)

    # Sort descending by value
    sorted_items = sorted(
        comparisons.items(),
        key=lambda x: x[1],
        reverse=True
    )

    table = Table(show_header=True, title="Comparison")
    table.add_column("Item")
    table.add_column("Value", justify="right")
    table.add_column("Δ vs baseline", justify="right")
    table.add_column("Visual")

    baseline = comparisons.get(baseline_key) if baseline_key else None

    for label, value in sorted_items:
        normalized = int((value / max_value) * bar_width)
        bar = "█" * normalized

        delta_str = ""
        style = None

        if baseline is not None and baseline > 0:
            delta = ((value - baseline) / baseline) * 100
            delta_str = f"{delta:+.1f}%"

            if delta > 5:
                style = "red" if use_color else None
            elif delta < -5:
                style = "green" if use_color else None

        table.add_row(
            label,
            f"{value:.3f}",
            delta_str,
            bar,
            style=style,
        )

    return table


# -------------------------------------------------------------------------
# Warmup Curve (Sparkline Style)
# -------------------------------------------------------------------------

def create_warmup_curve_chart(
    latencies: List[float],
    stable_run_index: Optional[int] = None,
    console: Optional[Console] = None,
    use_color: bool = True,
) -> Table:

    console = console or Console(color_system="auto" if use_color else None)

    if not latencies or len(latencies) < 2:
        return Table(title="Warmup Curve (insufficient data)")

    arr = np.array(latencies)
    mean, p50, p95, std, _ = _safe_stats(latencies)

    # Even sampling across range
    bar_width = _get_bar_width(console)
    sample_indices = np.linspace(
        0, len(arr) - 1, num=min(bar_width, len(arr))
    ).astype(int)

    sampled = arr[sample_indices]

    min_lat = float(np.min(sampled))
    max_lat = float(np.max(sampled))
    lat_range = max(max_lat - min_lat, 1e-6)

    # Unicode sparkline blocks (8 levels)
    blocks = "▁▂▃▄▅▆▇█"

    sparkline = ""

    for i, lat in zip(sample_indices, sampled):
        level = int(((lat - min_lat) / lat_range) * (len(blocks) - 1))
        char = blocks[level]

        if stable_run_index is not None and i >= stable_run_index:
            char = "|" if use_color else "|"

        sparkline += char

    title = (
        f"Warmup Curve | mean={mean:.2f}ms "
        f"p50={p50:.2f} p95={p95:.2f} std={std:.2f}"
    )

    table = Table(show_header=False, title=title)
    table.add_column("Curve")

    table.add_row(sparkline)

    return table
