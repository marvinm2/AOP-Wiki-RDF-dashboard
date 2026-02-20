# Coding Conventions

**Analysis Date:** 2026-02-20

## Naming Patterns

**Files:**
- Lowercase with underscores: `app.py`, `config.py`, `plots.py`, `verify_properties.py`
- Module packages use underscores: `plots/shared.py`, `plots/trends_plots.py`, `plots/latest_plots.py`
- CSV data files use underscores: `property_labels.csv`

**Functions:**
- snake_case: `plot_latest_entity_counts()`, `run_sparql_query_with_retry()`, `safe_plot_execution()`
- Verb-first naming: `compute_plots_parallel()`, `extract_counts()`, `check_sparql_endpoint_health()`
- Prefixed with intent: `_build_graph_filter()` (private/internal), `safe_read_csv()` (safety-aware)
- Domain-specific prefixes: `plot_*` for visualization functions, `download_*` for download endpoints, `api_*` for API endpoints

**Variables:**
- snake_case for all variables: `plot_tasks`, `future_to_name`, `execution_time`, `sparql_endpoint`
- Constant-like globals use UPPERCASE: `SPARQL_ENDPOINT`, `MAX_RETRIES`, `RETRY_DELAY`, `TIMEOUT`, `BRAND_COLORS`
- Private module-level variables prefixed with underscore: `_plot_data_cache`, `_plot_figure_cache`
- Descriptive names prioritized over abbreviations: `future_to_name` instead of `f2n`, `execution_time` instead of `et`

**Types:**
- Type hints used throughout: `def run_sparql_query_with_retry(query: str, max_retries: int = MAX_RETRIES) -> List[Dict[str, Any]]`
- Union types for flexible returns: `tuple[str, str, pd.DataFrame]` for multi-part returns
- Optional types explicit: `version: str = None` with default None for optional parameters
- See section on Type Hints for detailed patterns

**Configuration/Constants:**
- Class-based config: `Config.SPARQL_ENDPOINT`, `Config.PARALLEL_WORKERS` - all attributes are UPPERCASE
- Environment variable fallbacks: `os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql")`
- Brand color constants organized in dicts: `BRAND_COLORS['primary']`, `BRAND_COLORS['palette']`, `BRAND_COLORS['type_colors']`

## Code Style

**Formatting:**
- 4-space indentation (Python standard)
- Lines typically 80-120 characters, no hard limit enforced
- No automated formatter configured (no black, flake8, or pylint configs found)
- Imports grouped: standard library, third-party, local imports (implicit pattern from `app.py`)

**Linting:**
- No linting configuration found (.flake8, .pylintrc, setup.cfg, pyproject.toml not present)
- No automated format checking in CI/CD
- Code style maintained through convention and code review

**Structure & Whitespace:**
- One blank line between module-level function definitions
- Two blank lines between class and function definitions (standard Python style)
- Docstrings precede all function/class definitions (see Docstring section)
- Clear logical grouping of related functions in modules

## Import Organization

**Order (from `app.py` and `plots/__init__.py`):**
1. Standard library imports: `os`, `time`, `logging`, `functools.reduce`, `concurrent.futures`
2. Third-party imports: `pandas`, `flask`, `plotly`, `SPARQLWrapper`, `requests`
3. Local imports: `from config import Config`, `from plots import ...`, `from .shared import ...`

**Path Aliases:**
- Relative imports within package: `from .shared import (...)`, `from .trends_plots import (...)`
- Absolute imports at package level: `from config import Config`
- Explicit multi-line imports for clarity: Functions imported on separate lines with parentheses for long lists

**Pattern Example from `plots/__init__.py`:**
```python
# Standard library
from typing import List, Dict, Any, Optional, Tuple

# Third-party
import pandas as pd
import plotly.express as px

# Local
from .shared import (
    run_sparql_query,
    run_sparql_query_with_retry,
    BRAND_COLORS,
)

from .trends_plots import (
    plot_main_graph,
    plot_avg_per_aop,
)
```

## Type Hints

**Full typing coverage required:**
- All function parameters have explicit type hints: `query: str`, `max_retries: int`, `version: str = None`
- Return types always specified: `-> List[Dict[str, Any]]`, `-> tuple[str, str, pd.DataFrame]`, `-> bool`
- Complex types imported from `typing`: `List`, `Dict`, `Any`, `Optional`, `Tuple`
- pandas types used directly: `pd.DataFrame`

**Type Pattern Examples:**
```python
def run_sparql_query_with_retry(query: str, max_retries: int = MAX_RETRIES) -> List[Dict[str, Any]]:
    """Execute SPARQL query..."""
    pass

def extract_counts(results: List[Dict[str, Any]], var_name: str = "count") -> pd.DataFrame:
    """Extract version and count data..."""
    pass

def plot_main_graph() -> tuple[str, str, pd.DataFrame]:
    """Generate main AOP entity evolution visualization..."""
    pass

def safe_read_csv(filename: str, default_data: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Safely read CSV file..."""
    pass
```

## Docstrings

**Format:** Google-style docstrings with comprehensive documentation

**Structure (required for all functions):**
1. One-line summary (period required)
2. Blank line
3. Extended description (paragraphs with implementation context)
4. **Args:** section with type and description
5. **Returns:** section with type and structure
6. Optional **Raises:** section (when applicable)
7. Optional **Example:** section with usage code
8. Optional **Note:** section for important considerations
9. Optional **Performance:** section for optimization details

**Example from `shared.py`:**
```python
def check_sparql_endpoint_health() -> bool:
    """Check if SPARQL endpoint is accessible and responsive.

    Performs a lightweight health check on the configured SPARQL endpoint by
    executing a simple test query. This function validates both connectivity
    and basic query functionality, providing essential health monitoring
    capabilities for the application.

    Health Check Process:
        1. Validates SPARQL endpoint URL format and scheme
        2. Executes a minimal test query (COUNT of any triple)
        3. Verifies response within timeout period
        4. Confirms valid JSON response format

    Returns:
        bool: True if endpoint is healthy and responsive, False otherwise.
            Logs appropriate messages for both success and failure cases.

    Test Query:
        Executes: "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o } LIMIT 1"
        - Lightweight query that works on any RDF store
        - Tests connectivity, query processing, and response format
        - Uses short timeout (10 seconds) for quick health checks

    Error Conditions:
        - Invalid URL format or scheme
        - Network connectivity issues
        - SPARQL endpoint not responding
        - Query execution timeout
        - Invalid response format
        - Server errors (500, 503, etc.)

    Example:
        >>> if check_sparql_endpoint_health():
        ...     print("SPARQL endpoint is ready")
        ... else:
        ...     print("SPARQL endpoint is unavailable")

    Performance:
        - Uses 10-second timeout for quick response
        - Minimal query load on the SPARQL endpoint
        - Designed for frequent health monitoring

    Note:
        This function logs all outcomes (success/failure) and is safe
        to call repeatedly for monitoring purposes.
    """
```

**Module-level docstrings:**
Comprehensive triple-quoted strings at file start describing:
- Module purpose and responsibilities
- Key components and exports
- Usage examples
- Integration points
- Author attribution

Example from `app.py`:
```python
"""AOP-Wiki RDF Dashboard Flask Application.

This is the main Flask application that serves the AOP-Wiki RDF Dashboard. It provides
a web-based interface for visualizing AOP (Adverse Outcome Pathway) data evolution
over time, with comprehensive CSV export capabilities for all visualizations.

The application features:
    - Interactive web dashboard with tabbed navigation
    - Real-time visualization of RDF data evolution
    - Parallel plot computation for optimal performance
    - Comprehensive CSV data export system
    - Health monitoring and status endpoints
    - Professional branding and responsive design

Key Components:
    - Parallel plot computation system using ThreadPoolExecutor
    - Global data caching for CSV exports
    - Health monitoring with SPARQL endpoint checking
    - Professional web interface with brand-consistent styling
    - Comprehensive error handling and fallback mechanisms

Web Endpoints:
    /: Main dashboard with interactive visualizations
    /health: Health check endpoint for monitoring
    /status: Real-time status monitoring page
    /download/*: CSV download endpoints for all plots

Author:
    Generated with Claude Code (https://claude.ai/code)
"""
```

## Error Handling

**Pattern:** Centralized exception handling with classification and retry logic

**Strategies:**
1. **Try-except with specific exception types:** Catch specific exceptions first, general `Exception` last
   ```python
   except SPARQLExceptions.QueryBadFormed as e:
       logger.error(f"Bad SPARQL query: {str(e)}")
       break  # Don't retry for syntax errors
   except SPARQLExceptions.EndPointNotFound as e:
       logger.error(f"SPARQL endpoint not found: {str(e)}")
       break  # Don't retry for endpoint errors
   except (SPARQLExceptions.SPARQLWrapperException, requests.exceptions.RequestException) as e:
       # Retry with exponential backoff
   except Exception as e:
       logger.error(f"Unexpected error: {str(e)}")
   ```

2. **Exponential backoff for network errors:**
   ```python
   if attempt < max_retries - 1:
       time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
   ```

3. **Graceful degradation for non-critical failures:**
   - Plot failures return empty strings or None (not raised)
   - CSV read failures return empty DataFrame with fallback data
   - Individual plot timeouts don't crash application

4. **Safe plot execution wrapper** (in `shared.py`):
   ```python
   def safe_plot_execution(plot_func, *args, **kwargs) -> Any:
       """Execute plot function with error handling and timing..."""
       try:
           result = plot_func(*args, **kwargs)
           return result
       except Exception as e:
           logger.error(f"Error in plot function {plot_func.__name__}: {str(e)}")
           return create_fallback_plot(plot_func.__name__, str(e))
   ```

5. **Data validation in extract functions:**
   ```python
   try:
       # Validate presence of required fields
       if "graph" not in r or var_name not in r:
           logger.warning(f"Missing required fields in result: {r}")
           continue
   except (KeyError, ValueError, IndexError) as e:
       logger.warning(f"Error processing result {r}: {str(e)}")
   ```

**Never raised exceptions:**
- Plot generation functions return None/empty string on error, not exceptions
- SPARQL query failures return empty list `[]` instead of raising
- File read failures return DataFrame (empty or with fallback) instead of raising

## Logging

**Framework:** Python's `logging` module

**Configuration (in each module):**
```python
import logging
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

**Levels used:**
- **DEBUG:** Detailed execution flow (not observed in codebase)
- **INFO:** Successful operations: "Query executed successfully, returned X results"
- **WARNING:** Retry attempts, data quality issues: "Slow query execution: 10.5s"
- **ERROR:** Operation failures, exceptions: "SPARQL endpoint health check failed: {error}"

**Patterns:**
- Always log exception details with `str(e)`: `logger.error(f"Error: {str(e)}")`
- Include context in messages: `logger.warning(f"Query attempt {attempt + 1}/{max_retries} failed: {str(e)}")`
- Log success with data: `logger.info(f"Query executed successfully, returned {len(bindings)} results")`
- Use f-strings for all message formatting

**Log locations (mapped to functions):**
- SPARQL operations: `run_sparql_query_with_retry()` in `plots/shared.py`
- Plot generation: `safe_plot_execution()` wrapper
- Application startup: `compute_plots_parallel()` in `app.py`
- Configuration validation: `Config.validate_config()` in `config.py`

## Comments

**When to comment:**
- Complex SPARQL query logic: Explain data extraction strategy and graph filtering
- Non-obvious algorithm choices: Exponential backoff calculation, version parsing
- Important performance notes: "Reduced from 10 queries to 4 (60% reduction)"
- Removed/disabled features: "# REMOVED - hits Virtuoso execution limits"

**When NOT to comment:**
- Self-documenting code: Function names and docstrings sufficient
- Loop/conditional logic: Clear variable names make intent obvious
- Standard library usage: Python idioms don't need explanation

**Style:**
- Single-line comments with `#` for code-adjacent notes
- Multi-line comments for complex sections
- TODO/FIXME comments not found (maintainability tracked in issues)

## Function Design

**Size guidelines:**
- Single-responsibility principle: Each function does one thing well
- Range: 20-100 lines typical (up to 300+ for complex plot functions with extensive SPARQL)
- Large plot functions acceptable due to SPARQL complexity, not duplicated across modules

**Parameter patterns:**
- Limit to 3-4 required parameters; optional parameters with defaults
- Use dictionaries/dataclasses for multiple related parameters (not observed, but convention implied)
- Optional version parameter pattern: `version: str = None` for version-aware functions

**Return values:**
- Single values for simple operations: `bool`, `str`, `pd.DataFrame`
- Tuples for multi-part results (well-documented in docstring):
  - `tuple[str, str]` for (absolute_plot, percentage_plot)
  - `tuple[str, str, pd.DataFrame]` for (absolute_plot, delta_plot, data)
- Empty collections on failure: `[]` for queries, `""` for HTML, `pd.DataFrame()` for data
- Never return None for data (return empty DataFrame instead)

**Example from `trends_plots.py`:**
```python
def plot_main_graph() -> tuple[str, str, pd.DataFrame]:
    """Generate the main AOP entity evolution visualization with absolute and delta views.
    ...
    Returns:
        tuple[str, str, pd.DataFrame]: A 3-tuple containing:
            - str: HTML for absolute counts line chart
            - str: HTML for delta changes line chart
            - pd.DataFrame: Processed data with all entity counts by version
    """
```

## Module Design

**Organization in `plots/` package:**
- `__init__.py`: Central export point, public API definition, module metadata
- `shared.py`: Utilities, constants, SPARQL operations, helpers
- `trends_plots.py`: Historical trend analysis (entities over time)
- `latest_plots.py`: Current snapshot analysis (latest version only)

**Exports (`plots/__init__.py`):**
- `__all__` list explicitly defines public API
- All importable functions documented in module docstring
- Convenience functions: `get_available_functions()`, `get_cached_data_keys()`, `clear_plot_cache()`

**Barrel file pattern:**
```python
from .shared import (run_sparql_query, BRAND_COLORS, _plot_data_cache)
from .trends_plots import (plot_main_graph, plot_avg_per_aop)
from .latest_plots import (plot_latest_entity_counts)

__all__ = [
    'run_sparql_query',
    'BRAND_COLORS',
    'plot_main_graph',
    'plot_latest_entity_counts',
    ...
]
```

**Module metadata:**
- `__version__ = "2.0.0"` at module level
- `__author__` attribution
- Module-level docstring with usage examples

---

*Convention analysis: 2026-02-20*
