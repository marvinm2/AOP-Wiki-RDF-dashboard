"""Gunicorn configuration for AOP-Wiki RDF Dashboard production deployment.

Key design decisions:
- preload_app: Runs compute_plots_parallel() once in master, shared via COW fork
- gthread: Allows concurrent requests within each worker for I/O-bound SPARQL queries
- 2 workers: Standard for containerized deployment; scale by adding containers
- timeout=120: AOP completeness boxplot takes ~75s, needs headroom
- worker_tmp_dir=/dev/shm: Prevents Docker tmpfs heartbeat failures
"""

# Binding
bind = "0.0.0.0:5000"

# Workers: 2 workers with gthread for Docker containers
# - 2 workers (not CPU*2+1) because this runs in a container
# - gthread allows concurrent requests within each worker
# - threads=4 gives 8 total concurrent request handlers
workers = 2
worker_class = "gthread"
threads = 4

# Preload: Run compute_plots_parallel() once in master, share via COW fork
preload_app = True

# Timeouts: generous because SPARQL queries can be slow
timeout = 120          # Worker timeout (some plots take 75s)
graceful_timeout = 30  # Graceful shutdown
keepalive = 2

# Docker-specific: use memory-backed tmpdir for heartbeat
worker_tmp_dir = "/dev/shm"

# Logging: write to stdout/stderr for Docker log collection
accesslog = "-"
errorlog = "-"
loglevel = "info"
