"""
CRITICAL determinism tests.

These tests validate the core invariant:
    Same configuration → Same artifact hash (always)

If these tests fail, the entire system is broken.
"""

import pytest
from pathlib import Path
import hashlib

from mlbuild.core.hash import compute_artifact_hash, compute_config_hash


class TestArtifactHashDeterminism:
    """Test that artifact hashing is deterministic."""
    
    def test_artifact_hash_stable_across_loads(self, simple_mlpackage):
        """
        CRITICAL: Loading and hashing same artifact multiple times
        must produce identical hashes.
        """
        hashes = []
        
        for _ in range(100):
            hash_val = compute_artifact_hash(simple_mlpackage)
            hashes.append(hash_val)
        
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, \
            f"DETERMINISM BROKEN: Got {len(unique_hashes)} unique hashes in 100 runs"
    
    def test_different_models_different_hash(self, simple_mlpackage, tmp_path):
        """Different model → different hash."""
        import coremltools as ct
        
        hash1 = compute_artifact_hash(simple_mlpackage)
        
        # Create a slightly different model
        input_features = [('input', ct.models.datatypes.Array(1, 20))]  # Different shape
        output_features = [('output', ct.models.datatypes.Array(1, 20))]
        
        builder = ct.models.neural_network.NeuralNetworkBuilder(
            input_features,
            output_features
        )
        
        builder.add_activation(
            name='identity',
            non_linearity='LINEAR',
            input_name='input',
            output_name='output'
        )
        
        mlpackage_path2 = tmp_path / "test_model2.mlpackage"
        model = ct.models.MLModel(builder.spec)
        model.save(str(mlpackage_path2))
        
        hash2 = compute_artifact_hash(mlpackage_path2)
        
        assert hash1 != hash2, "Different models must have different hashes"


class TestConfigHash:
    """Test configuration hashing."""
    
    def test_config_hash_deterministic(self):
        """Same config → same hash."""
        config = {
            "target": "apple_a17",
            "quantization": {"type": "int8"},
            "optimizer": {"fuse_ops": True}
        }
        
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        assert hash1 == hash2
    
    def test_config_hash_key_order_independent(self):
        """Hash should be independent of key order."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 == hash2
    
    def test_different_configs_different_hash(self):
        """Different configs → different hashes."""
        config1 = {"target": "apple_a17", "quantization": {"type": "int8"}}
        config2 = {"target": "apple_a17", "quantization": {"type": "fp16"}}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 != hash2