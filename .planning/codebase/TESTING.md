# Testing Patterns

**Analysis Date:** 2026-02-20

## Test Framework

**Status:** No automated testing framework detected

**Finding:**
- No test files found (no `*test*.py`, `*spec*.py`, or `conftest.py` in codebase)
- No test configuration files: `pytest.ini`, `pyproject.toml`, `tox.ini` not present
- No testing dependencies in `requirements.txt` (only: flask, pandas, plotly, SPARQLWrapper)
- Manual verification script exists: `verify_properties.py` (one-off SPARQL validation)

**Assertion Library:**
- None configured
- pytest would be natural choice for Python Flask application

**Run Commands:**
- No automated test suite
- Manual testing documented in CLAUDE.md:
  ```bash
  python app.py              # Run application
  flask run                  # Flask development server
  curl http://localhost:5000/health     # Health check
  curl http://localhost:5000/status     # Status endpoint
  ```

## Test File Organization

**Current structure:**
```
AOP-Wiki-RDF-dashboard/
├── app.py                          # Main Flask application
├── config.py                       # Configuration management
├── plots.py                        # Legacy plot functions
├── plots/
│   ├── __init__.py                 # Package exports
│   ├── shared.py                   # Utilities & constants
│   ├── trends_plots.py             # Historical trends
│   └── latest_plots.py             # Current snapshots
├── verify_properties.py            # Manual SPARQL verification (one-off)
└── .planning/codebase/
    └── TESTING.md                  # This file
```

**Testing absence implications:**
- No co-located tests (convention would be `test_*.py` files alongside source)
- No separate `tests/` directory
- Only manual integration testing via Flask endpoints
- SPARQL queries validated through `verify_properties.py` script (not automated)

## Integration Testing via Endpoints

**Manual testing points identified in codebase:**

**Health Monitoring Endpoints:**
- `GET /health` - Health check endpoint with SPARQL validation
- `GET /status` - Real-time status monitoring dashboard
- Returns JSON with endpoint status, plot load counts, timestamps

**API Endpoints:**
- `GET /api/latest-version` - Fetch latest RDF graph version
- `GET /api/versions` - List all available versions
- `GET /api/properties/<entity_type>` - Get properties for entity type (AOP, KE, KER, Stressor)
- `GET /api/plot/<plot_name>` - Lazy-load specific plot
- `GET /health` - Health/readiness check

**Download Endpoints:**
- Pattern: `/download/<plot_name>` with format parameter
- Test via: `curl http://localhost:5000/download/latest_entity_counts?format=csv`
- Formats supported: CSV, PNG, SVG (via Plotly export)

**Expected flows (from code review):**
```python
# Health check validation (from app.py:273)
@app.route("/health")
def health_check():
    """Returns: {"status": "healthy", "sparql_endpoint": "up", "plots_loaded": "22/22"}"""
    try:
        endpoint_healthy = check_sparql_endpoint_health()
        successful_plots = sum(1 for v in plot_results.values() if v is not None)
        # Returns 200 if healthy, 503 if degraded
    except Exception as e:
        # Returns 500 on error
```

## Manual Testing Coverage

**SPARQL Connectivity:**
```python
# verify_properties.py - Manual verification script
def query_predicates(entity_type_uri, entity_label):
    """Query all predicates used by entities of a specific type."""
    # Validates SPARQL endpoint and entity properties
    # Not automated, run manually to verify data integrity
```

**Plot Generation Testing:**
- Implicit test: `compute_plots_parallel()` in `app.py` executes all 22 plot functions at startup
- Individual plot failures caught and logged, don't crash application
- Success measured: "X/Y plots successful" logged at startup

**CSV Export Validation:**
- Each plot generates HTML + caches data in `_plot_data_cache` dictionary
- Download endpoints access cached data, format as CSV/PNG/SVG
- Manual validation through browser or curl required

## Unit Testing Patterns (Observed but not Automated)

**Safe execution wrapper pattern:**
```python
def safe_plot_execution(plot_func, *args, **kwargs) -> Any:
    """Execute plot function with error handling and timing...

    Returns:
        - Plot result on success
        - Fallback plot on error (prevents cascade failures)
    """
    try:
        start_time = time.time()
        result = plot_func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Plot function {plot_func.__name__} executed in {execution_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error in plot function {plot_func.__name__}: {str(e)}")
        return create_fallback_plot(plot_func.__name__, str(e))
```

This pattern enables:
- Graceful degradation: Failed plots return empty visualization
- Error isolation: One plot failure doesn't affect others
- Timing monitoring: Execution time logged automatically
- Fallback mechanism: Fallback plot with error message generated

**Configuration validation pattern:**
```python
class Config:
    @classmethod
    def validate_config(cls) -> bool:
        """Validate all configuration settings for correctness and safety."""
        try:
            # Validates SPARQL endpoint URL format
            parsed_url = urlparse(cls.SPARQL_ENDPOINT)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid SPARQL endpoint URL: {cls.SPARQL_ENDPOINT}")

            # Validates numeric settings
            if cls.SPARQL_TIMEOUT <= 0:
                raise ValueError("SPARQL_TIMEOUT must be positive")
            if cls.PARALLEL_WORKERS <= 0:
                raise ValueError("PARALLEL_WORKERS must be positive")
            # ... more validation
            return True
        except Exception as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            return False
```

Called at startup in `app.py` and `plots/shared.py`:
```python
if not Config.validate_config():
    logger.error("Invalid configuration detected, using defaults")
```

**Data extraction testing pattern:**
```python
def extract_counts(results: List[Dict[str, Any]], var_name: str = "count") -> pd.DataFrame:
    """Extract version and count data from SPARQL query results with robust error handling."""
    if not results:
        logger.warning(f"No results to extract for {var_name}")
        return pd.DataFrame()

    try:
        # Validate each result
        for r in results:
            if "graph" not in r or var_name not in r:
                logger.warning(f"Missing required fields in result: {r}")
                continue
            try:
                # Convert data
                version = r["graph"]["value"].split("/")[-1]
                count = int(r[var_name]["value"])
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(f"Error processing result {r}: {str(e)}")
        # ... more processing
    except Exception as e:
        logger.error(f"Error in extract_counts: {str(e)}")
        return pd.DataFrame()
```

Tests data quality through validation and logging, returns empty DataFrame on error.

## Mocking Pattern

**Current approach:** No mocking framework detected

**How requests are tested:**
1. **SPARQL queries:** Use real SPARQL endpoint, fail gracefully with retries
2. **Plots:** Safe execution wrapper catches exceptions, returns fallback
3. **Flask endpoints:** Manual browser/curl testing against running server

**If mocking were to be introduced:**

```python
# Example pytest + unittest.mock pattern (not currently used)
from unittest.mock import patch, MagicMock
import pytest

@patch('plots.shared.SPARQLWrapper')
def test_sparql_query_with_retry(mock_sparql):
    """Test SPARQL query with mocked endpoint."""
    # Setup mock to return test data
    mock_sparql.return_value.query.return_value.convert.return_value = {
        "results": {"bindings": [{"count": {"value": "42"}}]}
    }

    # Call function
    result = run_sparql_query_with_retry("SELECT ...", max_retries=1)

    # Assert result
    assert result == [{"count": {"value": "42"}}]

@patch('plots.latest_plots.run_sparql_query')
def test_plot_latest_entity_counts(mock_query):
    """Test plot generation with mocked SPARQL data."""
    mock_query.return_value = [
        {"entity": {"value": "AOP"}, "count": {"value": "100"}},
        {"entity": {"value": "KE"}, "count": {"value": "500"}},
    ]

    html = plot_latest_entity_counts()
    assert "AOP" in html
    assert "100" in html
```

**What NOT to mock:**
- Configuration loading (test real env vars)
- Fallback mechanisms (test error paths with real exceptions)

**What to mock:**
- SPARQL endpoints (network-dependent, slow, requires running server)
- File I/O (for configuration file edge cases)
- Plot rendering (slow, not critical to test internals)

## Fixtures and Test Data

**No fixtures framework detected**

**Current test data approach:**
- `property_labels.csv` serves as configuration/test reference data
- Default data used in `safe_read_csv()`: fallback if CSV read fails
- In-memory caches: `_plot_data_cache`, `_plot_figure_cache` for export testing

**Pattern from `shared.py`:**
```python
def safe_read_csv(filename: str, default_data: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Safely read CSV file with comprehensive error handling and fallback data.

    default_data can be used as test fixture:
    """
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        if default_data:
            return pd.DataFrame(default_data)
        return pd.DataFrame()
```

**If pytest fixtures were to be used:**

```python
# conftest.py (hypothetical)
import pytest
import pandas as pd

@pytest.fixture
def sample_sparql_results():
    """Mock SPARQL query results for testing."""
    return [
        {
            "graph": {"value": "http://aopwiki.org/graph/2024-01-01"},
            "count": {"value": "100"}
        },
        {
            "graph": {"value": "http://aopwiki.org/graph/2024-02-01"},
            "count": {"value": "105"}
        },
    ]

@pytest.fixture
def sample_property_data():
    """Test data for property labels."""
    return pd.DataFrame([
        {
            "uri": "http://purl.org/dc/elements/1.1/title",
            "label": "Title",
            "type": "Essential",
            "applies_to": "AOP|KE|KER"
        },
        {
            "uri": "http://example.org/custom-property",
            "label": "Custom",
            "type": "Metadata",
            "applies_to": "AOP"
        }
    ])

@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for testing."""
    monkeypatch.setenv("SPARQL_ENDPOINT", "http://test-endpoint:8890/sparql")
    monkeypatch.setenv("SPARQL_TIMEOUT", "5")
```

## Error Testing Pattern

**Current approach:** Implicit through graceful degradation

```python
# Pattern from safe_plot_execution
try:
    result = plot_func(*args, **kwargs)
    return result
except Exception as e:
    logger.error(f"Error in plot function {plot_func.__name__}: {str(e)}")
    return create_fallback_plot(plot_func.__name__, str(e))
```

**If pytest were used, error testing would be:**

```python
def test_sparql_query_bad_syntax():
    """Test SPARQL query error handling with malformed query."""
    query = "SELECT * WHERE { INVALID SYNTAX }"
    result = run_sparql_query_with_retry(query, max_retries=1)
    assert result == []  # Should return empty list, not raise

def test_sparql_endpoint_down():
    """Test SPARQL query handling when endpoint is unreachable."""
    with patch('plots.shared.SPARQLWrapper') as mock_sparql:
        mock_sparql.return_value.query.side_effect = ConnectionError("Connection refused")

        result = run_sparql_query_with_retry("SELECT ...", max_retries=1)

        assert result == []
        # Verify retry logic was attempted

def test_plot_generation_error_returns_fallback():
    """Test that plot generation errors return fallback plot, not exception."""
    with patch('plots.latest_plots.run_sparql_query') as mock_query:
        mock_query.side_effect = RuntimeError("Unexpected error")

        result = safe_plot_execution(plot_latest_entity_counts)

        # Should return fallback plot HTML, not raise
        assert isinstance(result, str)
        assert "Error" in result or "fallback" in result.lower()
```

## Coverage

**Requirements:** Not enforced (no coverage configuration found)

**Gaps identified:**
1. **No endpoint integration tests** - Health check `/health` endpoint untested
2. **No plot generation tests** - 22 plot functions have no unit/integration tests
3. **No CSV export tests** - Download endpoints and caching untested
4. **No SPARQL query tests** - Query building and version filtering untested
5. **No configuration tests** - Config validation only runs at startup
6. **No data processing tests** - extract_counts() logic untested

**Test coverage if added (estimated at 0%):**
```
src/
  app.py                  (22 route handlers, ~0% covered)
  config.py               (validate_config, ~0% covered)
  plots/
    shared.py             (9 utility functions, ~0% covered)
    trends_plots.py       (15 plot functions, ~0% covered)
    latest_plots.py       (13 plot functions, ~0% covered)
```

**Critical paths to test:**
1. SPARQL query retry logic with exponential backoff
2. Plot generation with graceful degradation
3. CSV export caching mechanism
4. Version filtering (latest vs historical)
5. Property label filtering by entity type
6. Configuration validation and env var loading

## Common Patterns

**Async Testing:** Not applicable (no async code detected)

**Parallel Testing:** Flask is single-threaded, plots generated in ThreadPoolExecutor at startup
- Could benefit from parallel test execution with pytest-xdist
- Individual plot tests could run in parallel

**Timeout Testing:**
```python
# Implicit pattern in config
SPARQL_TIMEOUT = Config.SPARQL_TIMEOUT  # Configurable in tests via env vars
sparql.setTimeout(TIMEOUT)
```

**If timeout testing were implemented:**
```python
def test_sparql_query_timeout():
    """Test SPARQL query timeout handling."""
    with patch('plots.shared.SPARQLWrapper') as mock_sparql:
        mock_sparql.return_value.query.side_effect = TimeoutError("Query took too long")

        result = run_sparql_query_with_retry("SELECT ...", max_retries=2)

        # Verify retries were attempted
        assert mock_sparql.call_count >= 2
        assert result == []
```

## Recommended Testing Structure

**If testing framework were to be added:**

```
AOP-Wiki-RDF-dashboard/
├── app.py
├── config.py
├── plots/
│   ├── shared.py
│   ├── trends_plots.py
│   └── latest_plots.py
├── tests/                          # NEW
│   ├── conftest.py                 # Fixtures and configuration
│   ├── test_config.py              # Configuration validation
│   ├── test_health_endpoints.py    # /health and /status endpoints
│   ├── test_api_endpoints.py       # /api/* endpoints
│   ├── test_download_endpoints.py  # /download/* endpoints
│   ├── test_sparql_operations.py   # SPARQL query execution
│   ├── test_plot_generation.py     # Plot function error handling
│   ├── test_data_export.py         # CSV caching and export
│   ├── test_data_processing.py     # extract_counts, data validation
│   └── fixtures/
│       ├── sample_sparql_data.json
│       └── sample_properties.csv
└── pytest.ini                      # NEW - pytest configuration
```

**pytest.ini configuration (hypothetical):**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    integration: Integration tests requiring SPARQL endpoint
    unit: Unit tests with mocked dependencies
    slow: Slow tests (>5 seconds)
timeout = 30
```

---

*Testing analysis: 2026-02-20*
