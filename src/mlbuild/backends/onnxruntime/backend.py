"""
ONNX Runtime backend implementation.
Cross-platform CPU inference (also supports GPU on CUDA systems).
"""

import platform
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal

from ..base import Backend, BackendCapabilities, EnvironmentValidation
from ...core.types import Build
from ...core.hash import compute_artifact_hash, compute_config_hash, compute_source_hash
from ...core.hash_utils import compute_backend_artifact_hash 
from ...core.environment import collect_environment, hash_environment


class ONNXRuntimeBackend(Backend):
    """ONNX Runtime backend for cross-platform inference."""
    
    @property
    def name(self) -> str:
        return "onnxruntime"
    
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_fp16=True,  # ONNX Runtime supports FP16
            supports_fp32=True,
            supports_int8=True,  # ONNX Runtime supports INT8 quantization
            supports_dynamic_shapes=True,
            compute_units=["CPU", "CUDA", "CoreML", "DML"]  # Execution providers
        )
    
    def validate_environment(self) -> EnvironmentValidation:
        """Check if ONNX Runtime can run."""
        errors = []
        warnings = []
        info = {}
        
        # Platform info
        info["platform"] = f"{platform.system()} {platform.release()}"
        info["architecture"] = platform.machine()
        
        # Check onnxruntime
        try:
            import onnxruntime as ort
            info["onnxruntime"] = ort.__version__
            
            # Check available execution providers
            available_providers = ort.get_available_providers()
            info["execution_providers"] = ", ".join(available_providers)
            
            if "CUDAExecutionProvider" in available_providers:
                info["cuda_support"] = "available"
            
            if "CoreMLExecutionProvider" in available_providers:
                info["coreml_support"] = "available"
                
        except ImportError:
            errors.append("onnxruntime not installed. Run: pip install onnxruntime")
        
        # Check onnx
        try:
            import onnx
            info["onnx"] = onnx.__version__
        except ImportError:
            errors.append("onnx not installed. Run: pip install onnx")
        
        # Check numpy
        try:
            import numpy as np
            info["numpy"] = np.__version__
        except ImportError:
            errors.append("numpy not installed. Run: pip install numpy")
        
        return EnvironmentValidation(
            is_valid=len(errors) == 0,
            backend_name=self.name,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def build(
        self,
        model_path: Path,
        target: str,
        quantize: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Build:
        """..."""
        import onnx
        import hashlib
        import shutil
        
        from ...loaders import load_model
        from ... import __version__ as MLBUILD_VERSION
        from rich.console import Console
        
        console = Console()
        
        # ========== ADD THIS LINE ==========
        model_path = Path(model_path)  # Ensure it's a Path object
        # ===================================
        
        console.print(f"\n[bold]Building ONNX Runtime model[/bold]")
        console.print(f"Source: {model_path.name}")
        console.print(f"Target: {target}")
        console.print(f"Quantization: {quantize}\n")
        
        # Step 1: Collect environment
        env_data = collect_environment()
        env_fingerprint = hash_environment(env_data)
        
        # Step 2: Load ONNX model (for validation)
        ir = load_model(str(model_path))
        
        # Step 3: Hash source
        source_hash = compute_source_hash(model_path)
        
        # Step 4: Build configuration
        config = {
            "target": target,
            "backend": "onnxruntime",
            "quantization": {"type": quantize},
            "optimizer": {},
        }
        
        # Get onnxruntime version
        try:
            import onnxruntime as ort
            onnxruntime_version = ort.__version__
        except ImportError:
            onnxruntime_version = "unknown"
        
        config_hash = compute_config_hash(config, coremltools_version=onnxruntime_version)
        
       # Step 5: For ONNX Runtime, we just copy the original ONNX file
        # No conversion needed - ONNX Runtime runs ONNX directly
        
        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_root = Path(tmp_root)
            
            # Create a directory to hold the model (like CoreML's .mlpackage structure)
            model_dir = tmp_root / "model"
            model_dir.mkdir()
            temp_model_path = model_dir / "model.onnx"
            
            # Copy original ONNX file into directory
            shutil.copy(model_path, temp_model_path)
            
            # Compute artifact hash on the directory
            from ...core.hash_utils import compute_backend_artifact_hash
            artifact_hash = compute_backend_artifact_hash(model_dir, "onnxruntime")
            
            # Compute build ID
            build_id = self._compute_build_id(
                source_hash,
                config_hash,
                artifact_hash,
                env_fingerprint,
                MLBUILD_VERSION
            )
            
            # Move to final location
            artifacts_root = Path(".mlbuild/artifacts").resolve()
            final_dir = artifacts_root / artifact_hash
            
            artifacts_root.mkdir(parents=True, exist_ok=True)
            
            if final_dir.exists():
                console.print(f"[yellow]✓ Reusing existing artifact {artifact_hash[:12]}[/yellow]")
            else:
                shutil.move(str(model_dir), str(final_dir))
                console.print(f"[green]✓ Artifact saved {artifact_hash[:12]}[/green]")
            
            # Compute size
            final_model_path = final_dir / "model.onnx"
            size_bytes = final_model_path.stat().st_size
            size_mb = Decimal(size_bytes) / Decimal(1024 * 1024)
            
            # Create Build object
            backend_versions = {
                "onnxruntime": onnxruntime_version,
                "onnx": onnx.__version__,
            }
            
            if "numpy" in env_data and env_data["numpy"].get("installed"):
                backend_versions["numpy"] = env_data["numpy"]["version"]
            
            build_obj = Build(
                build_id=build_id,
                artifact_hash=artifact_hash,
                source_hash=source_hash,
                config_hash=config_hash,
                env_fingerprint=env_fingerprint,
                name=name,
                notes=notes,
                created_at=datetime.now(timezone.utc),
                source_path=str(model_path.resolve()),
                target_device=target,
                format="onnx",  # ONNX Runtime uses ONNX format
                quantization=config["quantization"],
                optimizer_config=config["optimizer"],
                backend_versions=backend_versions,
                environment_data=env_data,
                mlbuild_version=MLBUILD_VERSION,
                python_version=env_data["python"]["version"],
                platform=env_data["hardware"]["cpu"]["system"],
                os_version=env_data["hardware"]["cpu"]["release"],
                artifact_path=str(final_dir),
                size_mb=size_mb,
            )
            
            # Save to registry
            from ...registry import LocalRegistry
            registry = LocalRegistry()
            registry.save_build(build_obj)
            
            console.print(f"\n[bold green]✓ Build complete[/bold green]")
            console.print(f"Build ID:      {build_id[:16]}...")
            console.print(f"Artifact Hash: {artifact_hash[:16]}...")
            console.print(f"Size:          {size_mb:.2f} MB")
            console.print(f"Backend:       ONNX Runtime")
            console.print(f"Format:        ONNX")
            console.print()
            
            return build_obj
    
    def _compute_build_id(
        self,
        source_hash: str,
        config_hash: str,
        artifact_hash: str,
        env_fingerprint: str,
        mlbuild_version: str,
    ) -> str:
        """Compute deterministic build ID."""
        import hashlib
        
        return hashlib.sha256(
            b"\x00".join([
                bytes.fromhex(source_hash),
                bytes.fromhex(config_hash),
                bytes.fromhex(artifact_hash),
                bytes.fromhex(env_fingerprint),
                mlbuild_version.encode('utf-8'),
            ])
        ).hexdigest()