# Virtuoso SPARQL Endpoint Tuning Recommendations

These settings are recommended for the Virtuoso instance serving the AOP-Wiki RDF dashboard.
They are NOT applied by the dashboard Docker container (the dashboard queries an external SPARQL endpoint).

## Recommended Settings

### virtuoso.ini

```ini
[SPARQL]
MaxQueryExecutionTime = 120      ; seconds (default varies by deployment)
MaxQueryCostEstimationTime = 30  ; seconds for query plan estimation
ResultSetMaxRows = 1000000       ; allow large result sets for completeness queries

[Parameters]
NumberOfBuffers = 340000         ; ~2.5GB with 8KB page size
MaxDirtyBuffers = 250000         ; ~1.9GB
MaxCheckpointRemap = 2000

[HTTPServer]
MaxClientConnections = 10        ; dashboard uses 5 parallel workers
ServerThreads = 10
```

## Context

- The dashboard executes up to 5 parallel SPARQL queries during startup
- Complex queries (completeness boxplot, OECD trend) may take 30-75 seconds
- The `MaxQueryExecutionTime` setting is the most impactful for this workload
- If queries return empty results after long waits, increase MaxQueryExecutionTime first
