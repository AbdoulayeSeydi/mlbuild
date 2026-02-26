"""
Shared pytest fixtures for MLBuild tests.
"""

import pytest
from pathlib import Path
import coremltools as ct


@pytest.fixture
def simple_mlpackage(tmp_path: Path) -> Path:
    """
    Create a simple .mlpackage for testing.
    
    Returns path to .mlpackage directory.
    """
    # Create a simple model (identity function)
    input_features = [('input', ct.models.datatypes.Array(1, 10))]
    output_features = [('output', ct.models.datatypes.Array(1, 10))]
    
    builder = ct.models.neural_network.NeuralNetworkBuilder(
        input_features,
        output_features
    )
    
    # Add identity layer (output = input)
    builder.add_activation(
        name='identity',
        non_linearity='LINEAR',
        input_name='input',
        output_name='output'
    )
    
    # Save model
    mlpackage_path = tmp_path / "test_model.mlpackage"
    spec = builder.spec
    model = ct.models.MLModel(spec)
    model.save(str(mlpackage_path))
    
    return mlpackage_path