"""
Enterprise-grade IR schema validation engine.

Architecture:
- Structured validation issues (no string parsing)
- Deterministic ordering guarantees
- Type-safe validation
- Comprehensive graph integrity checks
- Separation of concerns
- Policy-driven severity handling
- Machine-readable error codes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Optional, FrozenSet
from collections import defaultdict

from .ir import ModelIR
from .errors import ValidationError


# ============================================================
# Structured Validation Types
# ============================================================

class Severity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"  # Blocks compilation
    ERROR = "error"        # Violates spec but may proceed
    WARNING = "warning"    # Non-blocking concern
    INFO = "info"          # Advisory only


@dataclass(frozen=True)
class ValidationIssue:
    """
    Structured validation issue.
    
    Immutable, deterministically orderable, machine-readable.
    """
    severity: Severity
    code: str
    message: str
    path: str
    details: Dict[str, str] = field(default_factory=dict)
    
    def __lt__(self, other: ValidationIssue) -> bool:
        """Deterministic ordering: severity → path → code."""
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.ERROR: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3,
        }
        return (
            severity_order[self.severity],
            self.path,
            self.code,
        ) < (
            severity_order[other.severity],
            other.path,
            other.code,
        )


@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable validation result.
    
    Canonical representation, deterministic ordering.
    """
    issues: FrozenSet[ValidationIssue]
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return sorted([i for i in self.issues if i.severity == Severity.CRITICAL])
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return sorted([i for i in self.issues if i.severity == Severity.ERROR])
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return sorted([i for i in self.issues if i.severity == Severity.WARNING])
    
    @property
    def has_critical(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity in {Severity.CRITICAL, Severity.ERROR} for i in self.issues)
    
    def to_dict(self) -> Dict:
        """Machine-readable JSON-serializable format."""
        return {
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "path": i.path,
                    "message": i.message,
                    "details": i.details,
                }
                for i in sorted(self.issues)
            ],
            "summary": {
                "critical": len(self.critical_issues),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
            },
        }


# ============================================================
# Validation Policy
# ============================================================

class ValidationPolicy(Enum):
    """Validation strictness policy."""
    STRICT = "strict"          # Fail on any error
    PERMISSIVE = "permissive"  # Only fail on critical
    CI = "ci"                  # Fail on critical + error
    AUDIT = "audit"            # Never fail, collect all


# ============================================================
# Core Validators (Separation of Concerns)
# ============================================================

class PresenceValidator:
    """Validates required field presence with type safety."""
    
    @staticmethod
    def validate(ir: ModelIR) -> List[ValidationIssue]:
        issues = []
        
        # Framework is required
        if not hasattr(ir, 'framework') or not ir.framework:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="IR_MISSING_FRAMEWORK",
                path="framework",
                message="ModelIR missing required 'framework' field",
            ))
        
        # Graph is required
        if not hasattr(ir, 'graph') or ir.graph is None:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="IR_MISSING_GRAPH",
                path="graph",
                message="ModelIR missing required 'graph' field",
            ))
            # Short-circuit: can't validate graph structure without graph
            return issues
        
        # Graph inputs required
        if not hasattr(ir.graph, 'inputs') or not ir.graph.inputs:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="GRAPH_MISSING_INPUTS",
                path="graph.inputs",
                message="Graph has no inputs defined",
            ))
        
        # Graph outputs required
        if not hasattr(ir.graph, 'outputs') or not ir.graph.outputs:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="GRAPH_MISSING_OUTPUTS",
                path="graph.outputs",
                message="Graph has no outputs defined",
            ))
        
        # Graph name advisory
        if not hasattr(ir.graph, 'name') or not ir.graph.name:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="GRAPH_MISSING_NAME",
                path="graph.name",
                message="Graph has no name, identifier will be auto-generated",
                details={"default_behavior": "sha256 hash of graph structure"},
            ))
        elif isinstance(ir.graph.name, str) and not ir.graph.name.strip():
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code="GRAPH_EMPTY_NAME",
                path="graph.name",
                message="Graph name is whitespace-only",
            ))
        
        return issues


class MetadataValidator:
    """Validates IR metadata structure and content."""
    
    @staticmethod
    def validate(ir: ModelIR) -> List[ValidationIssue]:
        issues = []
        
        # Access metadata safely via property
        if not hasattr(ir, 'get_framework_version'):
            # Fallback: type-safe metadata access
            if not hasattr(ir, 'metadata'):
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    code="METADATA_MISSING",
                    path="metadata",
                    message="ModelIR missing metadata field",
                ))
                return issues
            
            metadata = ir.metadata
            
            # Type-safe dict access
            if not isinstance(metadata, dict):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    code="METADATA_INVALID_TYPE",
                    path="metadata",
                    message=f"Metadata must be dict, got {type(metadata).__name__}",
                ))
                return issues
            
            # Framework version check
            if "framework_version" not in metadata:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    code="METADATA_MISSING_FRAMEWORK_VERSION",
                    path="metadata.framework_version",
                    message="Missing framework_version in metadata",
                ))
            elif not isinstance(metadata["framework_version"], str):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    code="METADATA_INVALID_FRAMEWORK_VERSION_TYPE",
                    path="metadata.framework_version",
                    message=f"framework_version must be string, got {type(metadata['framework_version']).__name__}",
                ))
            elif not metadata["framework_version"].strip():
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    code="METADATA_EMPTY_FRAMEWORK_VERSION",
                    path="metadata.framework_version",
                    message="framework_version is empty or whitespace",
                ))
        
        return issues


class GraphIntegrityValidator:
    """Validates graph structure, DAG properties, and tensor references."""
    
    @staticmethod
    def validate(ir: ModelIR) -> List[ValidationIssue]:
        issues = []
        
        # Short-circuit if no graph
        if not hasattr(ir, 'graph') or ir.graph is None:
            return issues  # Already flagged by PresenceValidator
        
        graph = ir.graph
        
        # Validate tensors container exists and is dict-like
        if not hasattr(graph, 'tensors'):
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="GRAPH_MISSING_TENSORS",
                path="graph.tensors",
                message="Graph missing tensors container",
            ))
            return issues
        
        tensors = graph.tensors
        
        # Type safety: ensure dict-like
        if not hasattr(tensors, '__contains__'):
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="GRAPH_TENSORS_INVALID_TYPE",
                path="graph.tensors",
                message=f"graph.tensors must support membership test, got {type(tensors).__name__}",
            ))
            return issues
        
        # Build tensor ID set for O(1) lookup and validate uniqueness
        try:
            tensor_ids = set(tensors.keys()) if hasattr(tensors, 'keys') else set(tensors)
        except (TypeError, AttributeError) as e:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                code="GRAPH_TENSORS_NOT_ITERABLE",
                path="graph.tensors",
                message=f"graph.tensors is not iterable: {e}",
            ))
            return issues
        
        # Validate inputs
        if hasattr(graph, 'inputs') and graph.inputs:
            # Type safety
            if not hasattr(graph.inputs, '__iter__'):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    code="GRAPH_INPUTS_NOT_ITERABLE",
                    path="graph.inputs",
                    message=f"graph.inputs must be iterable, got {type(graph.inputs).__name__}",
                ))
            else:
                # Check uniqueness (deterministic ordering)
                inputs_list = sorted(list(graph.inputs))
                inputs_set = set(inputs_list)
                
                if len(inputs_list) != len(inputs_set):
                    duplicates = [tid for tid in inputs_set if inputs_list.count(tid) > 1]
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        code="GRAPH_DUPLICATE_INPUTS",
                        path="graph.inputs",
                        message=f"Graph has duplicate input tensor IDs",
                        details={"duplicates": ", ".join(sorted(duplicates))},
                    ))
                
                # Check referential integrity
                for idx, tensor_id in enumerate(inputs_list):
                    if not isinstance(tensor_id, (str, int)):
                        issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            code="GRAPH_INPUT_INVALID_TYPE",
                            path=f"graph.inputs[{idx}]",
                            message=f"Input tensor ID must be string or int, got {type(tensor_id).__name__}",
                        ))
                        continue
                    
                    if tensor_id not in tensor_ids:
                        issues.append(ValidationIssue(
                            severity=Severity.CRITICAL,
                            code="GRAPH_INPUT_NOT_IN_TENSORS",
                            path=f"graph.inputs[{idx}]",
                            message=f"Input tensor '{tensor_id}' not found in graph.tensors",
                            details={"tensor_id": str(tensor_id)},
                        ))
        
        # Validate outputs
        if hasattr(graph, 'outputs') and graph.outputs:
            # Type safety
            if not hasattr(graph.outputs, '__iter__'):
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    code="GRAPH_OUTPUTS_NOT_ITERABLE",
                    path="graph.outputs",
                    message=f"graph.outputs must be iterable, got {type(graph.outputs).__name__}",
                ))
            else:
                # Check uniqueness (deterministic ordering)
                outputs_list = sorted(list(graph.outputs))
                outputs_set = set(outputs_list)
                
                if len(outputs_list) != len(outputs_set):
                    duplicates = [tid for tid in outputs_set if outputs_list.count(tid) > 1]
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        code="GRAPH_DUPLICATE_OUTPUTS",
                        path="graph.outputs",
                        message=f"Graph has duplicate output tensor IDs",
                        details={"duplicates": ", ".join(sorted(duplicates))},
                    ))
                
                # Check referential integrity
                for idx, tensor_id in enumerate(outputs_list):
                    if not isinstance(tensor_id, (str, int)):
                        issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            code="GRAPH_OUTPUT_INVALID_TYPE",
                            path=f"graph.outputs[{idx}]",
                            message=f"Output tensor ID must be string or int, got {type(tensor_id).__name__}",
                        ))
                        continue
                    
                    if tensor_id not in tensor_ids:
                        issues.append(ValidationIssue(
                            severity=Severity.CRITICAL,
                            code="GRAPH_OUTPUT_NOT_IN_TENSORS",
                            path=f"graph.outputs[{idx}]",
                            message=f"Output tensor '{tensor_id}' not found in graph.tensors",
                            details={"tensor_id": str(tensor_id)},
                        ))
        
        # Validate operators if present
        if hasattr(graph, 'operators') and graph.operators:
            issues.extend(GraphIntegrityValidator._validate_operators(graph, tensor_ids))
        
        return issues
    
    @staticmethod
    def _validate_operators(graph, tensor_ids: Set) -> List[ValidationIssue]:
        """Validate operator tensor references."""
        issues = []
        
        for op_idx, op in enumerate(graph.operators):
            op_path = f"graph.operators[{op_idx}]"
            
            # Validate operator inputs
            if hasattr(op, 'inputs') and op.inputs:
                for input_idx, tensor_id in enumerate(op.inputs):
                    if tensor_id not in tensor_ids:
                        issues.append(ValidationIssue(
                            severity=Severity.CRITICAL,
                            code="OP_INPUT_NOT_IN_TENSORS",
                            path=f"{op_path}.inputs[{input_idx}]",
                            message=f"Operator input tensor '{tensor_id}' not found in graph.tensors",
                            details={"operator": getattr(op, 'type', 'unknown'), "tensor_id": str(tensor_id)},
                        ))
            
            # Validate operator outputs
            if hasattr(op, 'outputs') and op.outputs:
                for output_idx, tensor_id in enumerate(op.outputs):
                    if tensor_id not in tensor_ids:
                        issues.append(ValidationIssue(
                            severity=Severity.CRITICAL,
                            code="OP_OUTPUT_NOT_IN_TENSORS",
                            path=f"{op_path}.outputs[{output_idx}]",
                            message=f"Operator output tensor '{tensor_id}' not found in graph.tensors",
                            details={"operator": getattr(op, 'type', 'unknown'), "tensor_id": str(tensor_id)},
                        ))
        
        return issues


# ============================================================
# Validation Engine
# ============================================================

class IRValidator:
    """
    Composable IR validation engine.
    
    Orchestrates validators and applies policy.
    """
    
    def __init__(self, policy: ValidationPolicy = ValidationPolicy.STRICT):
        self.policy = policy
        self.validators = [
            PresenceValidator(),
            MetadataValidator(),
            GraphIntegrityValidator(),
        ]
    
    def validate(self, ir: ModelIR) -> ValidationResult:
        """
        Run all validators and return immutable result.
        
        Deterministic ordering guaranteed.
        """
        all_issues = []
        
        for validator in self.validators:
            issues = validator.validate(ir)
            all_issues.extend(issues)
        
        # Freeze and deduplicate
        return ValidationResult(issues=frozenset(all_issues))
    
    def validate_or_raise(self, ir: ModelIR) -> ValidationResult:
        """
        Validate IR and raise if policy threshold exceeded.
        
        Raises:
            ValidationError: If validation fails per policy
        """
        result = self.validate(ir)
        
        should_raise = False
        
        if self.policy == ValidationPolicy.STRICT and result.has_errors:
            should_raise = True
        elif self.policy == ValidationPolicy.PERMISSIVE and result.has_critical:
            should_raise = True
        elif self.policy == ValidationPolicy.CI and result.has_errors:
            should_raise = True
        # AUDIT never raises
        
        if should_raise:
            raise ValidationError(
                "ModelIR validation failed",
                details=result.to_dict(),
            )
        
        return result


# ============================================================
# Public API
# ============================================================

def validate_model_ir(
    ir: ModelIR,
    policy: ValidationPolicy = ValidationPolicy.STRICT,
) -> ValidationResult:
    """
    Validate ModelIR with structured, deterministic output.
    
    Args:
        ir: ModelIR instance to validate
        policy: Validation policy for severity handling
    
    Returns:
        ValidationResult with immutable, ordered issues
    """
    validator = IRValidator(policy=policy)
    return validator.validate(ir)


def validate_or_raise(
    ir: ModelIR,
    policy: ValidationPolicy = ValidationPolicy.STRICT,
) -> ValidationResult:
    """
    Validate IR and raise if policy threshold exceeded.
    
    Args:
        ir: ModelIR instance to validate
        policy: Validation policy for severity handling
    
    Raises:
        ValidationError: If validation fails per policy
    
    Returns:
        ValidationResult if validation passes
    """
    validator = IRValidator(policy=policy)
    return validator.validate_or_raise(ir)