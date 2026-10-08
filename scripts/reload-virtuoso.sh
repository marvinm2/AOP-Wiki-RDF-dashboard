#!/usr/bin/env bash
#
# Reload the multi-version AOP-Wiki RDF into the Virtuoso endpoint.
#
# Unlike the legacy positional load.sh, this reloads BOTH the per-version
# data graphs AND the metadata graph (version catalogue + SPARQL service
# description), so a from-scratch reload re-advertises the versions and
# service description instead of silently dropping them (issue
# marvinm2/AOP-Wiki_multi-endpoint#9).
#
# Works for the Swarm deployment (container aopwiki-dashboard_virtuoso.N.*)
# and local docker-compose (container aopwiki-virtuoso). It loads whatever
# TTLs are already present in the container's /database/data:
#
#   AOPWikiRDF-<date>.ttl, AOPWikiRDF-Enriched-<date>.ttl,
#   AOPWikiRDF-Genes-<date>.ttl, AOPWikiRDF-Void-<date>.ttl
#                                  -> graph http://aopwiki.org/graph/<date>
#   ServiceDescription.ttl, AOPWikiRDF-Catalog.ttl
#                                  -> graph http://aopwiki-multirdf.vhp4safety.nl/metadata
#
# Loading uses DB.DBA.TTLP per file (synchronous + deterministic); the
# ld_dir/rdf_loader_run bulk path was observed to silently partial-load on
# this instance. Persistent namespace prefixes and SPARQL grants are NOT
# re-applied here — they survive RDF_GLOBAL_RESET and are set at first deploy.
#
# Usage:  ./scripts/reload-virtuoso.sh [--yes]
#   --yes   Skip the "this deletes all RDF data" confirmation prompt.
#
set -euo pipefail

META_GRAPH="http://aopwiki-multirdf.vhp4safety.nl/metadata"
GRAPH_BASE="http://aopwiki.org/graph"
CONTAINER_DATA_DIR="/database/data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

YES=false
[[ "${1:-}" == "--yes" ]] && YES=true

# --- DBA password -----------------------------------------------------------
# Taken from, in order: DBA_PASSWORD in the environment, the file named by
# DBA_PASSWORD_FILE, or a .env next to the repo root or this script. On the
# cluster the real password is in the swarm secret virtuoso_dba_password; the
# DBA_PASSWORD in the stack env is only used when a new store is created.
if [[ -z "${DBA_PASSWORD:-}" && -n "${DBA_PASSWORD_FILE:-}" ]]; then
    DBA_PASSWORD=$(<"$DBA_PASSWORD_FILE")
fi
if [[ -z "${DBA_PASSWORD:-}" ]]; then
    ENV_FILE=""
    for cand in "${SCRIPT_DIR}/../.env" "${SCRIPT_DIR}/.env" "${HOME}/aopwiki-dashboard/.env"; do
        [[ -f "$cand" ]] && { ENV_FILE="$cand"; break; }
    done
    if [[ -z "$ENV_FILE" ]]; then
        echo "ERROR: no DBA password (set DBA_PASSWORD or DBA_PASSWORD_FILE, or provide a .env)." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
if [[ -z "${DBA_PASSWORD:-}" ]]; then
    echo "ERROR: DBA_PASSWORD is empty." >&2
    exit 1
fi

# --- Locate the Virtuoso container ------------------------------------------
# The service is NOT node-pinned, so on the Swarm it may run on either node.
# `docker ps` is per-node, so if it's not here, point the operator to the node
# that currently hosts it rather than failing obscurely.
CN=$(docker ps --filter name=aopwiki-dashboard_virtuoso --format '{{.Names}}' | head -1)
[[ -z "$CN" ]] && CN=$(docker ps --filter name=aopwiki-virtuoso --format '{{.Names}}' | head -1)
if [[ -z "$CN" ]]; then
    NODE=$(docker service ps aopwiki-dashboard_virtuoso --filter desired-state=running --format '{{.Node}}' 2>/dev/null | head -1 || true)
    if [[ -n "$NODE" && "$NODE" != "$(hostname)" ]]; then
        echo "ERROR: aopwiki-dashboard_virtuoso is running on '${NODE}', not $(hostname)." >&2
        echo "       Re-run this script on that node (ssh to it, then from ~/aopwiki-dashboard)." >&2
    else
        echo "ERROR: no running Virtuoso container found (aopwiki-dashboard_virtuoso* or aopwiki-virtuoso)." >&2
    fi
    exit 1
fi
echo "Virtuoso container: $CN"

isql() { docker exec -i "$CN" /opt/virtuoso-opensource/bin/isql localhost:1111 dba "$DBA_PASSWORD" "$@"; }

# Per-graph AOP counts as "<date> <count>" lines, sorted. Each row is tagged so
# it can be picked out of isql's table output.
aop_counts() {
    # `|| true`: no rows is reported by the caller, not fatal under pipefail.
    isql <<EOF 2>&1 | { grep -oE 'CNT\|[0-9-]+\|[0-9]+' || true; } | awk -F'|' '{print $2, $3}' | sort
SPARQL SELECT (CONCAT("CNT|", REPLACE(STR(?g), "^.*/", ""), "|", STR(COUNT(DISTINCT ?a))) AS ?row) WHERE { GRAPH ?g { ?a a <http://aopkb.org/aop_ontology#AdverseOutcomePathway> } FILTER(STRSTARTS(STR(?g), "${GRAPH_BASE}/")) } GROUP BY ?g;
EOF
}

# --- Preflight (read-only) --------------------------------------------------
# The reload rebuilds the store from ${CONTAINER_DATA_DIR} alone, so anything
# missing there is silently lost. Refuse unless every quarter has all four
# files, both metadata files are present, and every graph now in the store has
# its files on disk (#163).
mapfile -t TTLS < <(docker exec "$CN" sh -c "ls ${CONTAINER_DATA_DIR}/*.ttl" 2>/dev/null | xargs -n1 basename | sort)
if [[ ${#TTLS[@]} -eq 0 ]]; then
    echo "ERROR: no .ttl files in ${CONTAINER_DATA_DIR} inside $CN — nothing to load." >&2
    exit 1
fi

preflight_ok=true
mapfile -t DATES < <(printf '%s\n' "${TTLS[@]}" | grep -oE '^AOPWikiRDF-.*[0-9]{4}-[0-9]{2}-[0-9]{2}\.ttl$' | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u)
for d in "${DATES[@]}"; do
    for kind in "" "Genes-" "Enriched-" "Void-"; do
        f="AOPWikiRDF-${kind}${d}.ttl"
        if ! printf '%s\n' "${TTLS[@]}" | grep -qx "$f"; then
            echo "PREFLIGHT: missing ${f}" >&2
            preflight_ok=false
        fi
    done
done
for f in ServiceDescription.ttl AOPWikiRDF-Catalog.ttl; do
    printf '%s\n' "${TTLS[@]}" | grep -qx "$f" || { echo "PREFLIGHT: missing ${f}" >&2; preflight_ok=false; }
done

BEFORE=$(aop_counts)
if [[ -z "$BEFORE" ]]; then
    echo "PREFLIGHT: could not read per-graph AOP counts from the store (wrong DBA_PASSWORD?)." >&2
    preflight_ok=false
fi
while read -r d _; do
    [[ -z "$d" ]] && continue
    printf '%s\n' "${DATES[@]}" | grep -qx "$d" || { echo "PREFLIGHT: graph ${d} is loaded but has no files in ${CONTAINER_DATA_DIR}; a reload would drop it." >&2; preflight_ok=false; }
done <<< "$BEFORE"

if [[ "$preflight_ok" != "true" ]]; then
    echo "ERROR: preflight failed; nothing was changed. Stage the missing files and re-run." >&2
    exit 1
fi
echo "Preflight OK: ${#DATES[@]} quarters with all 4 files; $(grep -c . <<< "$BEFORE") graphs in the store."

# --- Confirm (destructive) --------------------------------------------------
if [[ "$YES" != "true" ]]; then
    echo "WARNING: this DELETES all RDF data in $CN and reloads from ${CONTAINER_DATA_DIR}."
    read -r -p "Continue? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# --- Build the TTLP load script from the files in the container -------------

SQL=$'DELETE FROM DB.DBA.load_list;\nRDF_GLOBAL_RESET();\nlog_enable(2);\n'
n_ver=0; n_meta=0
for b in "${TTLS[@]}"; do
    case "$b" in
        ServiceDescription.ttl|AOPWikiRDF-Catalog.ttl)
            SQL+="DB.DBA.TTLP(file_to_string_output('${CONTAINER_DATA_DIR}/${b}'), '', '${META_GRAPH}');"$'\n'
            n_meta=$((n_meta+1)) ;;
        AOPWikiRDF-*.ttl)
            v=$(echo "$b" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' || true)
            [[ -z "$v" ]] && { echo "  skip (no date): $b"; continue; }
            SQL+="DB.DBA.TTLP(file_to_string_output('${CONTAINER_DATA_DIR}/${b}'), '', '${GRAPH_BASE}/${v}');"$'\n'
            n_ver=$((n_ver+1)) ;;
        *) echo "  skip (unrecognised): $b" ;;
    esac
done
SQL+=$'checkpoint;\n'

echo "Loading ${n_ver} version-graph files + ${n_meta} metadata files..."

printf '%s' "$SQL" | isql >/tmp/reload-virtuoso.isql.log 2>&1 || { echo "ERROR: isql load failed; see /tmp/reload-virtuoso.isql.log"; tail -20 /tmp/reload-virtuoso.isql.log; exit 1; }

# --- Verify -----------------------------------------------------------------
echo "=== verification ==="
isql <<EOF 2>&1 | grep -E 'graphs|metadata_triples|total|[0-9]{4,}' | grep -vE 'msec|Rows' || true
SPARQL SELECT (COUNT(DISTINCT ?g) AS ?version_graphs) WHERE { GRAPH ?g {?s ?p ?o} FILTER(STRSTARTS(STR(?g),"${GRAPH_BASE}/")) };
SPARQL SELECT (COUNT(*) AS ?metadata_triples) WHERE { GRAPH <${META_GRAPH}> {?s ?p ?o} };
SPARQL SELECT (COUNT(*) AS ?total) WHERE { GRAPH ?g {?s ?p ?o} };
EOF

AFTER=$(aop_counts)
if [[ "$AFTER" != "$BEFORE" ]]; then
    echo "ERROR: per-graph AOP counts changed across the reload (before < > after):" >&2
    diff <(echo "$BEFORE") <(echo "$AFTER") >&2 || true
    exit 1
fi
echo "Per-graph AOP counts identical before and after ($(grep -c . <<< "$AFTER") graphs)."
echo "Done. If metadata_triples is 0 the catalogue/service description failed to load."
