# Spinifex GNSS Test Suite

Comprehensive unit and integration tests for the spinifex_gnss module.

## Overview

This test suite provides thorough coverage of the spinifex_gnss codebase, including:

- **Unit tests** for individual functions and classes
- **Integration tests** for complete workflows
- **Edge case tests** for error handling and boundary conditions
- **Fixtures** for reusable test data

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Tests for configuration module
├── test_parse_gnss.py       # Tests for GNSS data parsing
├── test_tec_calculations.py # Tests for TEC calculations
├── test_geometry.py         # Tests for geometry calculations
└── test_integration.py      # Integration tests (to be added)
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_parse_gnss.py
```

### Run Specific Test Class

```bash
pytest tests/test_parse_gnss.py::TestParseDCBSinex
```

### Run Specific Test

```bash
pytest tests/test_parse_gnss.py::TestParseDCBSinex::test_parse_dcb_file
```

### Run with Coverage

```bash
pytest --cov=spinifex_gnss --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Only Fast Tests

```bash
pytest -m "not slow"
```

### Run Only Unit Tests

```bash
pytest -m unit
```

### Run with Verbose Output

```bash
pytest -v
```

### Run and Stop at First Failure

```bash
pytest -x
```

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, test workflows)
- `@pytest.mark.slow` - Slow tests (can be skipped for quick runs)
- `@pytest.mark.requires_data` - Tests that need external data files

## Fixtures

Common fixtures are defined in `conftest.py`:

### Data Fixtures

- `test_data_dir` - Temporary directory for test data
- `sample_rinex_file` - Minimal RINEX file for testing
- `sample_dcb_file` - Sample DCB SINEX file
- `sample_dcb_data` - Sample DCBdata object
- `sample_gnss_data` - Sample GNSSData object

### Geometry Fixtures

- `sample_earth_location` - Earth location (approximately Netherlands)
- `sample_satellite_location` - Satellite location (GPS orbit)
- `sample_ipp` - Sample ionospheric pierce points

### Time Fixtures

- `sample_datetime` - Sample datetime object
- `sample_astropy_time` - Sample Astropy Time object
- `sample_time_range` - Range of times for testing

### Array Fixtures

- `sample_pseudorange_data` - Sample pseudorange observations
- `sample_phase_data` - Sample carrier phase observations
- `sample_tec_data` - Sample TEC values

### Parametrized Fixtures

- `constellation` - Parametrize over all GNSS constellations (G, E, R, C, J)
- `has_dcb` - Parametrize over DCB availability (True/False)

## Test Coverage

Current test coverage by module:

| Module | Coverage | Status |
|--------|----------|--------|
| `config.py` | ~95% | ✅ Complete |
| `parse_gnss.py` | ~80% | ✅ Good |
| `parse_rinex.py` | ~60% | ⚠️ Needs work |
| `gnss_geometry.py` | ~50% | ⚠️ Needs work |
| `proces_gnss_data.py` | ~40% | ❌ Incomplete |
| `gnss_tec.py` | ~30% | ❌ Incomplete |
| `download_gnss.py` | ~20% | ❌ Incomplete |

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or `Test<Functionality>`
- Test methods: `test_<what_is_being_tested>`

### Example Test Structure

```python
import pytest
import numpy as np

class TestMyFunction:
    """Tests for my_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_data)
        assert result == expected_output
        
    def test_edge_case(self):
        """Test edge case behavior."""
        result = my_function(edge_case_input)
        assert result is not None
        
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, input_val, expected):
        """Test with multiple inputs."""
        result = my_function(input_val)
        assert result == expected
```

### Using Fixtures

```python
def test_with_fixture(sample_gnss_data):
    """Test using a fixture."""
    assert sample_gnss_data.is_valid
    assert len(sample_gnss_data.gnss) > 0
```

### Testing Exceptions

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### Testing with Temporary Files

```python
def test_file_parsing(test_data_dir):
    """Test file parsing with temporary file."""
    test_file = test_data_dir / "test.txt"
    test_file.write_text("test content")
    
    result = parse_file(test_file)
    assert result is not None
```

## Best Practices

### 1. Test One Thing at a Time

Each test should verify one specific behavior:

```python
# Good
def test_function_returns_correct_type():
    assert isinstance(my_function(), int)

def test_function_returns_positive_value():
    assert my_function() > 0

# Bad
def test_function():
    result = my_function()
    assert isinstance(result, int)  # Testing type
    assert result > 0                # Testing value
    assert result < 100              # Testing range
```

### 2. Use Descriptive Test Names

```python
# Good
def test_parse_dcb_file_handles_gzipped_input()

# Bad
def test_dcb()
```

### 3. Arrange-Act-Assert Pattern

```python
def test_calculation():
    # Arrange
    input_data = create_test_data()
    expected = 42
    
    # Act
    result = perform_calculation(input_data)
    
    # Assert
    assert result == expected
```

### 4. Test Edge Cases

Always test:
- Empty inputs
- Single element inputs
- Very large inputs
- NaN/None values
- Invalid types
- Boundary conditions

### 5. Use Fixtures for Common Setup

Don't repeat yourself - use fixtures for common test data:

```python
@pytest.fixture
def common_test_data():
    return create_expensive_test_data()

def test_something(common_test_data):
    result = process(common_test_data)
    assert result is not None
```

## Continuous Integration

Tests are run automatically on:
- Every commit to `main` branch
- Every pull request
- Nightly builds

CI configuration is in `.github/workflows/tests.yml` (to be created).

## Test Data

Test data files should be:
- Small (< 1 MB if possible)
- Representative of real data
- Stored in `tests/data/` directory
- Documented in `tests/data/README.md`

For large data files, use fixtures that generate synthetic data instead.

## Troubleshooting

### Tests Fail Locally But Pass in CI

- Check Python version (use same as CI)
- Check dependencies (update with `pip install -r requirements-test.txt`)
- Clear pytest cache: `pytest --cache-clear`

### Tests Are Slow

- Run only fast tests: `pytest -m "not slow"`
- Use pytest-xdist for parallel execution: `pytest -n auto`
- Profile slow tests: `pytest --durations=10`

### Import Errors

- Make sure package is installed: `pip install -e .`
- Check PYTHONPATH is set correctly
- Verify all `__init__.py` files exist

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=spinifex_gnss`
4. Aim for >80% coverage on new code
5. Update this README if adding new test categories

## Questions?

If you have questions about the test suite, please:
1. Check this README
2. Look at existing test examples
3. Ask in the project's discussion forum
4. Open an issue with the `testing` label

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Good Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
