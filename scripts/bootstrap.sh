#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap.sh - Environment bootstrap script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== hailo-yolo11-compiler: Environment Bootstrapper ==="
echo "Repository Root: ${REPO_ROOT}"

# Detect Python interpreter (3.10+)
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" &> /dev/null; then
        PY_VER=$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$("${candidate}" -c 'import sys; print(sys.version_info.major)')
        PY_MINOR=$("${candidate}" -c 'import sys; print(sys.version_info.minor)')
        if [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -ge 10 ]; then
            PYTHON_BIN="${candidate}"
            echo "Selected Python interpreter: ${PYTHON_BIN} (v${PY_VER})"
            break
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: Python 3.10+ is required but was not found." >&2
    exit 1
fi

VENV_DIR="${REPO_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ] && [ -d "${REPO_ROOT}/venv" ]; then
    VENV_DIR="${REPO_ROOT}/venv"
    echo "Using existing virtual environment at ${VENV_DIR}..."
elif [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing hailo-yolo11-compiler in editable mode with development dependencies..."
pip install -e "${REPO_ROOT}[dev]"

echo ""
echo "=== Bootstrap Complete ==="
echo "To activate your environment, run:"
if [ "${VENV_DIR}" = "${REPO_ROOT}/venv" ]; then
    echo "    source venv/bin/activate"
else
    echo "    source .venv/bin/activate"
fi
echo "To verify system readiness, run:"
echo "    python train_and_compile.py doctor"
