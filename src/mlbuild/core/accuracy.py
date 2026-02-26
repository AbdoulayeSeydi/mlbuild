"""
Accuracy estimation for quantized models.

Compares model outputs to measure quantization impact.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    import coremltools as ct
except ImportError:
    ct = None


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing two models."""
    mse: float  # Mean Squared Error
    mae: float  # Mean Absolute Error
    max_error: float  # Maximum absolute error
    relative_error: float  # Mean relative error (%)
    sample_count: int


def compute_accuracy_metrics(
    baseline_outputs: List[np.ndarray],
    candidate_outputs: List[np.ndarray],
) -> AccuracyMetrics:
    """
    Compute accuracy metrics between baseline and candidate outputs.
    
    Args:
        baseline_outputs: List of output arrays from baseline model
        candidate_outputs: List of output arrays from candidate model
        
    Returns:
        AccuracyMetrics with comparison stats
    """
    if len(baseline_outputs) != len(candidate_outputs):
        raise ValueError("Output counts must match")
    
    if len(baseline_outputs) == 0:
        raise ValueError("Need at least one sample")
    
    # Flatten all outputs
    all_baseline = np.concatenate([o.flatten() for o in baseline_outputs])
    all_candidate = np.concatenate([o.flatten() for o in candidate_outputs])
    
    # Compute metrics
    diff = all_candidate - all_baseline
    abs_diff = np.abs(diff)
    
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))
    max_error = float(np.max(abs_diff))
    
    # Relative error (avoid division by zero)
    baseline_magnitude = np.abs(all_baseline)
    valid_mask = baseline_magnitude > 1e-7
    
    if np.any(valid_mask):
        relative_errors = abs_diff[valid_mask] / baseline_magnitude[valid_mask]
        relative_error = float(np.mean(relative_errors) * 100)
    else:
        relative_error = 0.0
    
    return AccuracyMetrics(
        mse=mse,
        mae=mae,
        max_error=max_error,
        relative_error=relative_error,
        sample_count=len(baseline_outputs),
    )


def estimate_model_accuracy(
    baseline_model_path: Path,
    candidate_model_path: Path,
    test_samples: List[np.ndarray],
) -> AccuracyMetrics:
    """
    Estimate accuracy loss between two CoreML models.
    
    Args:
        baseline_model_path: Path to baseline model (e.g., FP32)
        candidate_model_path: Path to candidate model (e.g., INT8)
        test_samples: List of input arrays to test
        
    Returns:
        AccuracyMetrics comparing the models
    """
    if ct is None:
        raise RuntimeError("coremltools required for accuracy estimation")
    
    # Load models
    baseline_model = ct.models.MLModel(str(baseline_model_path))
    candidate_model = ct.models.MLModel(str(candidate_model_path))
    
    # Get input name
    baseline_spec = baseline_model.get_spec()
    input_name = baseline_spec.description.input[0].name
    
    # Run inference on all samples
    baseline_outputs = []
    candidate_outputs = []
    
    for sample in test_samples:
        # Prepare input dict
        input_dict = {input_name: sample}
        
        # Run baseline
        baseline_result = baseline_model.predict(input_dict)
        baseline_output = list(baseline_result.values())[0]
        baseline_outputs.append(baseline_output)
        
        # Run candidate
        candidate_result = candidate_model.predict(input_dict)
        candidate_output = list(candidate_result.values())[0]
        candidate_outputs.append(candidate_output)
    
    # Compute metrics
    return compute_accuracy_metrics(baseline_outputs, candidate_outputs)