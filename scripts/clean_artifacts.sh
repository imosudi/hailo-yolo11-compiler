#!/usr/bin/env bash
# ==============================================================================
# scripts/clean_artifacts.sh - Safe artifact cleaner
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts"

DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--force]" >&2
            exit 1
            ;;
    esac
done

echo "=== Cleaning Artifacts in ${ARTIFACTS_DIR} ==="

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] Would remove files in:"
    find "${ARTIFACTS_DIR}" -type f ! -name ".gitkeep"
    exit 0
fi

if [ "${FORCE}" != true ]; then
    read -rp "Are you sure you want to remove all generated artifacts? (y/N) " confirm
    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "Removing run artifacts and keeping directory structure..."
find "${ARTIFACTS_DIR}" -type f ! -name ".gitkeep" -delete
find "${ARTIFACTS_DIR}/runs" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

echo "Artifacts cleaned successfully."
