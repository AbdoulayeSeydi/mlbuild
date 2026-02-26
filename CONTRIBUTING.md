# Contributing to MLBuild

Thank you for your interest in contributing to MLBuild. This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (mlbuild doctor output)
   - Minimal reproducible example

### Suggesting Features

1. Open an issue with:
   - Problem: What problem does this solve?
   - Solution: How should it work?
   - Alternatives: What alternatives did you consider?
   - Use case: Real-world example

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Make your changes
4. Add tests
5. Run the test suite
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.11+
- macOS (for CoreML support)
- Git

### Setup
```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/mlbuild.git
cd mlbuild

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Initialize for development
mlbuild init
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mlbuild --cov-report=html

# Run specific test file
pytest tests/test_benchmark.py

# Run with verbose output
pytest -vv
```

### Code Style

We use:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Lint
flake8 src/ tests/
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 100 characters
- Docstrings: Google style

### Example
```python
def benchmark_model(
    build_id: str,
    runs: int = 100,
    warmup: int = 20,
) -> BenchmarkResult:
    """
    Benchmark a model build.
    
    Args:
        build_id: Unique identifier for the build
        runs: Number of benchmark iterations
        warmup: Number of warmup iterations
    
    Returns:
        BenchmarkResult containing p50, p95, p99 metrics
    
    Raises:
        BuildNotFoundError: If build_id doesn't exist
    """
    pass
```

### Commit Messages

Use Conventional Commits format:
```
feat: add S3 backend support
fix: resolve memory leak in benchmarking
docs: update README with new commands
test: add tests for remote storage
refactor: simplify CLI argument parsing
perf: optimize artifact compression
chore: update dependencies
```

### Branch Naming
```
feature/add-s3-backend
bugfix/fix-memory-leak
docs/update-contributing-guide
test/add-benchmark-tests
```

## Testing Guidelines

### Test Structure
```
tests/
├── test_benchmark.py      # Benchmark tests
├── test_artifact_hash.py  # Hashing tests
├── test_normalization.py  # Normalization tests
└── conftest.py            # Shared fixtures
```

### Writing Tests
```python
def test_benchmark_calculates_percentiles():
    """Test that benchmark correctly calculates p50/p95/p99."""
    # Arrange
    build = create_test_build()
    
    # Act
    result = benchmark_model(build.build_id, runs=10)
    
    # Assert
    assert result.p50_ms > 0
    assert result.p95_ms >= result.p50_ms
    assert result.p99_ms >= result.p95_ms
```

### Test Coverage

- Aim for >80% coverage on new code
- All bug fixes must include regression tests
- Critical paths (CLI, benchmarking) require 100% coverage

## Documentation

### Docstrings

All public functions/classes must have docstrings:
```python
class LocalRegistry:
    """
    SQLite-based local registry for build artifacts.
    
    Stores build metadata, benchmarks, and experiment tracking data
    in a local .mlbuild/registry.db file.
    
    Attributes:
        db_path: Path to SQLite database
    
    Example:
        >>> registry = LocalRegistry()
        >>> registry.save_build(build)
    """
```

### README Updates

If your PR adds features or changes behavior:
- Update README.md
- Add examples
- Update command reference

## Questions

- General questions: GitHub Discussions
- Bug reports: GitHub Issues
- Security issues: Email abdoulayeaseydi@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You

Every contribution helps make MLBuild better. Thank you for taking the time to contribute.