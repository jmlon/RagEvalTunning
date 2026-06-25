#!/usr/bin/env bash
# Manage a local Neo4j container for the `graph-rag` benchmark system.
#
# Data persists in the named Docker volume `ragbench_neo4j_data`, so stopping
# the container does NOT lose the graph. Use `reset` to wipe the database.
#
# Connection (used by ragbench graph-rag, see config.yaml):
#   bolt://localhost:7687   user: neo4j   password: $NEO4J_LOCAL_PASSWORD (default hola1234)
#   Browser UI: http://localhost:7474
set -euo pipefail

CONTAINER="ragbench-neo4j"
VOLUME="ragbench_neo4j_data"
IMAGE="neo4j:5"
PASSWORD="${NEO4J_LOCAL_PASSWORD:-neo4j}"

cmd="${1:-up}"

case "$cmd" in
  up)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
      echo "[neo4j] '${CONTAINER}' already running."
      exit 0
    fi
    docker volume create "${VOLUME}" >/dev/null
    echo "[neo4j] starting ${CONTAINER} (${IMAGE}); data volume=${VOLUME}"
    docker run -d --rm \
      --name "${CONTAINER}" \
      -p 7474:7474 -p 7687:7687 \
      --volume="${VOLUME}:/data" \
      -e NEO4J_AUTH="neo4j/${PASSWORD}" \
      -e NEO4J_PLUGINS='["apoc"]' \
      -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
      -e NEO4J_dbms_security_procedures_allowlist='apoc.*' \
      "${IMAGE}" >/dev/null
    echo "[neo4j] waiting for bolt to accept connections..."
    for _ in $(seq 1 60); do
      if docker exec "${CONTAINER}" cypher-shell -u neo4j -p "${PASSWORD}" "RETURN 1;" >/dev/null 2>&1; then
        echo "[neo4j] ready at bolt://localhost:7687 (UI http://localhost:7474)"
        exit 0
      fi
      sleep 2
    done
    echo "[neo4j] timed out waiting for readiness; check 'docker logs ${CONTAINER}'." >&2
    exit 1
    ;;
  down)
    docker stop "${CONTAINER}" >/dev/null 2>&1 && echo "[neo4j] stopped (data preserved in ${VOLUME})." \
      || echo "[neo4j] not running."
    ;;
  reset)
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    docker volume rm "${VOLUME}" >/dev/null 2>&1 && echo "[neo4j] data volume removed." \
      || echo "[neo4j] volume not present."
    ;;
  status)
    docker ps --filter "name=${CONTAINER}" --format '{{.Names}} {{.Image}} {{.Status}}' \
      || echo "[neo4j] not running."
    ;;
  *)
    echo "usage: $0 {up|down|reset|status}" >&2
    exit 2
    ;;
esac
