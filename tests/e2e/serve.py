"""Serve the dashboard for the browser tests, with no SPARQL endpoint (#181).

The pages and static files come from the real app; every data request
(/api/versions, /api/plot/...) is answered by the tests themselves
(Playwright route stubs), so the tests exercise the frontend runtime, not the
data. Startup fails fast: no readiness wait, one attempt per query.
"""

import os
import sys

os.environ.update({
    "SPARQL_ENDPOINT": "http://127.0.0.1:9/sparql",
    "SPARQL_MAX_RETRIES": "1",
    "STARTUP_READY_TIMEOUT": "0",
    "STARTUP_BUDGET": "5",
    "STARTUP_MIN_PRECOMPUTE": "1",
    "QUERY_STORE_DIR": "",
    "ENABLE_USAGE_ANALYTICS": "False",
    "LOG_LEVEL": "WARNING",
})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import app  # noqa: E402

if __name__ == "__main__":
    app.app.run(host="127.0.0.1", port=int(os.environ.get("E2E_PORT", "5055")), threaded=True)
