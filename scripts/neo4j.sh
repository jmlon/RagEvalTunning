#!/usr/bin/env bash
# Manage a local Neo4j container for the `graph-rag` benchmark system.
#
# Data persists in the named Docker volume `ragbench_neo4j_data`, so stopping
# the container does NOT lose the graph. Use `reset` to wipe the database.
#
# Connection (used by ragbench graph-rag, see config.yaml):
#   Host:      bolt://localhost:7687              (published ports)
#   Container: bolt://ragbench-neo4j:7687         (shared network 'ragbench-net')
#   Browser UI: http://localhost:7474
# The container also joins the user-defined network 'ragbench-net' so other
# containers (e.g. the ragbench image) can reach it by name; pass that name and
# set NEO4J_LOCAL_URI=bolt://ragbench-neo4j:7687 on their `docker run`.
# Auth is off so any username/password the client sends is accepted/ignored.
# Do NOT expose ports 7474/7687 beyond localhost with auth disabled.
set -euo pipefail

CONTAINER="ragbench-neo4j"
VOLUME="ragbench_neo4j_data"
NETWORK="ragbench-net"
IMAGE="neo4j:5"

cmd="${1:-up}"

case "$cmd" in
  up)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
      echo "[neo4j] '${CONTAINER}' already running."
      exit 0
    fi
    docker volume create "${VOLUME}" >/dev/null
    docker network inspect "${NETWORK}" >/dev/null 2>&1 || docker network create "${NETWORK}" >/dev/null
    echo "[neo4j] starting ${CONTAINER} (${IMAGE}); data volume=${VOLUME}; network=${NETWORK}"
    docker run -d --rm \
      --name "${CONTAINER}" \
      --network "${NETWORK}" \
      -p 7474:7474 -p 7687:7687 \
      --volume="${VOLUME}:/data" \
      -e NEO4J_AUTH="none" \
      -e NEO4J_PLUGINS='["apoc"]' \
      -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
      -e NEO4J_dbms_security_procedures_allowlist='apoc.*' \
      "${IMAGE}" >/dev/null
    echo "[neo4j] waiting for bolt to accept connections..."
    for _ in $(seq 1 60); do
      if docker exec "${CONTAINER}" cypher-shell --non-interactive -u neo4j -p ignored "RETURN 1;" >/dev/null 2>&1; then
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
