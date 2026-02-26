"""
Critical tests for CoreML normalization.

These tests MUST pass before proceeding.
Non-determinism is a fatal flaw.
"""

import hashlib
import pytest
from pathlib import Path
from typing import Any

import coremltools as ct
from coremltools.proto import Model_pb2

from mlbuild.core.hash import (
    compute_artifact_hash,
    _strip_metadata,
    _canonicalize_spec,
    _normalize_spec,
)


class TestMetadataStripping:
    """Test that non-deterministic metadata is removed."""
    
    def test_strips_author_info(self, tmp_path: Path) -> None:
        """Verify author info is removed."""
        spec = _create_simple_model_spec()
        
        # Add author info
        spec.description.metadata.author = "test_author"
        spec.description.metadata.shortDescription = "test description"
        spec.description.metadata.versionString = "1.0.0"
        
        # Strip metadata
        _strip_metadata(spec)
        
        # Verify removed (fields are cleared, become empty strings)
        assert spec.description.metadata.author == ""
        assert spec.description.metadata.shortDescription == ""
        assert spec.description.metadata.versionString == ""
    
    def test_preserves_deterministic_metadata(self, tmp_path: Path) -> None:
        """Verify deterministic user metadata is kept."""
        spec = _create_simple_model_spec()
        
        # Add user-defined metadata
        spec.description.metadata.userDefined["model_type"] = "classifier"
        spec.description.metadata.userDefined["version"] = "2"
        
        # Strip non-deterministic metadata
        _strip_metadata(spec)
        
        # Verify user metadata preserved
        assert "model_type" in spec.description.metadata.userDefined
        assert "version" in spec.description.metadata.userDefined
    
    def test_user_defined_metadata_is_deterministic(self, tmp_path: Path) -> None:
        """
        Verify user metadata produces deterministic serialization.
        
        Protobuf maps are unordered, so we can't test iteration order.
        Instead, test that serialization is deterministic regardless of insertion order.
        """
        spec1 = _create_simple_model_spec()
        spec2 = _create_simple_model_spec()
        
        # Insert in different orders
        spec1.description.metadata.userDefined["z_key"] = "z_value"
        spec1.description.metadata.userDefined["a_key"] = "a_value"
        spec1.description.metadata.userDefined["m_key"] = "m_value"
        
        spec2.description.metadata.userDefined["a_key"] = "a_value"
        spec2.description.metadata.userDefined["m_key"] = "m_value"
        spec2.description.metadata.userDefined["z_key"] = "z_value"
        
        # Normalize both
        _strip_metadata(spec1)
        _strip_metadata(spec2)
        
        # Serialized bytes must be identical
        bytes1 = spec1.SerializeToString(deterministic=True)
        bytes2 = spec2.SerializeToString(deterministic=True)
        
        assert bytes1 == bytes2


class TestCanonicalization:
    """Test that model structure is canonicalized."""
    
    def test_sorts_inputs_by_name(self, tmp_path: Path) -> None:
        """Verify inputs are sorted alphabetically."""
        spec = _create_simple_model_spec()
        
        # Add inputs in random order
        _add_input(spec, "z_input")
        _add_input(spec, "a_input")
        _add_input(spec, "m_input")
        
        # Canonicalize
        _canonicalize_spec(spec)
        
        # Verify sorted order
        input_names = [f.name for f in spec.description.input]
        assert input_names == sorted(input_names)
    
    def test_sorts_outputs_by_name(self, tmp_path: Path) -> None:
        """Verify outputs are sorted alphabetically."""
        spec = _create_simple_model_spec()
        
        # Add outputs in random order
        _add_output(spec, "z_output")
        _add_output(spec, "a_output")
        _add_output(spec, "m_output")
        
        # Canonicalize
        _canonicalize_spec(spec)
        
        # Verify sorted order
        output_names = [f.name for f in spec.description.output]
        assert output_names == sorted(output_names)


class TestNormalizationDeterminism:
    """Test that normalization produces identical results."""
    
    def test_normalize_is_deterministic(self, simple_mlpackage: Path) -> None:
        """
        CRITICAL: Normalizing the same model multiple times
        must produce identical bytes.
        """
        hashes = []
        
        for _ in range(10):
            # Load spec directly from protobuf
            spec_path = simple_mlpackage / "Data" / "com.apple.CoreML" / "model.mlmodel"
            with open(spec_path, 'rb') as f:
                spec = Model_pb2.Model()
                spec.ParseFromString(f.read())
            
            normalized_bytes = _normalize_spec(spec)
            hash_val = hashlib.sha256(normalized_bytes).hexdigest()
            hashes.append(hash_val)
        
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, \
            f"Normalization non-deterministic! Got {len(unique_hashes)} hashes"
    
    def test_same_model_same_hash(self, simple_mlpackage: Path) -> None:
        """Same model → same hash."""
        hash1 = compute_artifact_hash(simple_mlpackage)
        hash2 = compute_artifact_hash(simple_mlpackage)
        
        assert hash1 == hash2


# ---------------------------------------------------------------------
# Test Helpers (not fixtures - those are in conftest.py)
# ---------------------------------------------------------------------

def _create_simple_model_spec() -> Any:
    """Create a minimal CoreML model spec for testing."""
    spec = Model_pb2.Model()
    spec.specificationVersion = 5
    
    # Add a simple identity layer
    spec.description.input.add()
    spec.description.input[0].name = "input"
    spec.description.input[0].type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32
    
    spec.description.output.add()
    spec.description.output[0].name = "output"
    spec.description.output[0].type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32
    
    return spec


def _add_input(spec: Any, name: str) -> None:
    """Add an input to spec."""
    inp = spec.description.input.add()
    inp.name = name
    inp.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32


def _add_output(spec: Any, name: str) -> None:
    """Add an output to spec."""
    out = spec.description.output.add()
    out.name = name
    out.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32