# Testing Suite

Simple, fast tests for the Semantic Distillation project.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_schemas.py

# Run specific test class
pytest tests/test_utils.py::TestFeatureCache

# Run specific test
pytest tests/test_data.py::TestLoadData::test_load_data_from_csv

# Run tests matching a pattern
pytest -k "cache"
```

## Test Coverage

- **test_schemas.py**: Tests for Pydantic model generation and validation
  - Static schemas (FeatureHypothesis, ResidualAnalysis)
  - Dynamic model generation from feature configs
  - Field validation (scale, bool, categorical)
  
- **test_utils.py**: Tests for utility functions
  - FeatureCache (caching, persistence)
  - RateLimiter (token bucket, thread safety)
  - Logging setup
  - Result storage
  
- **test_data.py**: Tests for data loading
  - Sample data structure
  - CSV loading
  - Binary label conversion
  - Custom column names
  
- **test_features.py**: Tests for feature definitions
  - Feature set validation
  - FeatureSet class operations
  - JSON serialization
  - Feature creation helpers

## Test Philosophy

These tests are designed to be:
- **Fast**: No external API calls, minimal I/O
- **Simple**: Focus on core functionality
- **Isolated**: Each test is independent
- **Comprehensive**: Cover main code paths without being exhaustive

## Adding New Tests

When adding new functionality:
1. Add tests to the appropriate test file
2. Follow existing naming conventions (`test_*`)
3. Keep tests fast and focused
4. Use pytest fixtures (`tmp_path`) for file operations
