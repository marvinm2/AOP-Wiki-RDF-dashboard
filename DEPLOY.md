# AOP-Wiki RDF Dashboard -- Deployment Guide

Deployment of the AOP-Wiki RDF Dashboard and Virtuoso SPARQL endpoint to the VHP4Safety cluster (TGX1).

## Prerequisites

Before deploying, ensure:

1. **DNS A records** point to `81.169.246.233`:
   - `aopwiki-dashboard.cloud.vhp4safety.nl`
   - `aopwiki-multirdf.vhp4safety.nl`

2. **GlusterFS directory** exists on TGX1:
   ```bash
   mkdir -p /mnt/gluster/aopwiki-dashboard/virtuoso-data
   ```

3. **RDF data** is loaded into the GlusterFS directory via the AOP-Wiki-RDF Setup pipeline.

4. **SSH access** as `mmartens` (must be in the `docker` group).

5. **Traefik** is running on the Swarm with the `core` overlay network and `letsencrypt` cert resolver.

## Build

Copy the project to TGX1 and build the Docker image:

```bash
# From your local machine
scp -r . mmartens@81.169.246.233:~/aopwiki-dashboard/

# SSH into TGX1
ssh mmartens@81.169.246.233

# Build the dashboard image
cd ~/aopwiki-dashboard
docker build -t aopwiki-dashboard:latest .
```

## Deploy

```bash
# Create .env file with Virtuoso admin password
echo "DBA_PASSWORD=<your-password>" > .env

# Deploy the stack (.env is read automatically from the working directory)
docker stack deploy -c stack.yml aopwiki-dashboard
```

## Verify

```bash
# Check services are running
docker service ls | grep aopwiki

# Expected output (after start_period):
#   aopwiki-dashboard_dashboard   1/1
#   aopwiki-dashboard_virtuoso    1/1

# Check service logs
docker service logs aopwiki-dashboard_dashboard --tail 50
docker service logs aopwiki-dashboard_virtuoso --tail 50

# Test health endpoint
curl https://aopwiki-dashboard.cloud.vhp4safety.nl/health

# Test SPARQL endpoint
curl 'https://aopwiki-multirdf.vhp4safety.nl/sparql?query=ASK+%7B%7D'
```

## Update

To deploy a new version of the dashboard:

```bash
# Copy updated code to TGX1
scp -r . mmartens@81.169.246.233:~/aopwiki-dashboard/

# SSH in and rebuild
ssh mmartens@81.169.246.233
cd ~/aopwiki-dashboard
docker build -t aopwiki-dashboard:latest .

# Force service update to pick up new image
docker service update --force aopwiki-dashboard_dashboard
```

## Troubleshooting

**Services not starting:**
```bash
docker service ps aopwiki-dashboard_dashboard --no-trunc
docker service ps aopwiki-dashboard_virtuoso --no-trunc
```
Check the `ERROR` column for failure reasons.

**DNS not resolving:**
Verify A records are propagated: `dig aopwiki-dashboard.cloud.vhp4safety.nl`. If not resolved, the Traefik router will not match incoming requests.

**Virtuoso not starting:**
- Check that `/mnt/gluster/aopwiki-dashboard/virtuoso-data` exists and has correct permissions.
- Virtuoso needs write access to its data directory. If the directory is owned by root, the container may fail.
- The health check has 20 retries with 60s start_period, giving Virtuoso up to ~6 minutes to initialize.

**Dashboard health check failing:**
- The dashboard precomputes all plots at startup (~75 seconds). The `start_period: 120s` prevents premature restarts.
- If it still fails, check logs for SPARQL connection errors -- the dashboard needs Virtuoso to be healthy first.
- Virtuoso may need more time on first boot. Restart the dashboard service after Virtuoso is healthy:
  ```bash
  docker service update --force aopwiki-dashboard_dashboard
  ```

**TLS certificate not issued:**
Traefik's `letsencrypt` resolver needs port 443 accessible from the internet and correct DNS. Check Traefik logs for ACME errors.

**Memory issues:**
Dashboard is limited to 2GB, Virtuoso to 4GB. If Virtuoso OOMs on large datasets, increase the limit in `stack.yml` under `deploy.resources.limits.memory`.
