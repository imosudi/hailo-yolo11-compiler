#!/usr/bin/env bash
# ==============================================================================
# scripts/validate_environment.sh - Doctor validation script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Running Environment Doctor ==="
python3 "${REPO_ROOT}/train_and_compile.py" doctor "$@"
