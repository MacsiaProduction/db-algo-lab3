#!/usr/bin/env bash
# Finish notebook 05 (comparison + scaling) then record HNSW flame graph.
#
#   LAB_N_SWEEP=1281167 LAB_SCALING_FULL=1 ./scripts/finish_and_flame.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/resume_from_05.sh
LAB_N_SWEEP="${LAB_N_SWEEP:-1000000}" ./scripts/record_flame.sh
echo "==> Re-execute 05 to embed flame (reuses scaling.csv, no rebuild):"
SKIP_SCALING_REBUILD=1 ./scripts/resume_from_05.sh
