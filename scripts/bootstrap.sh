#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap.sh - Environment bootstrap script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== hailo-yolo11-compiler: Environment Bootstrapper ==="
echo "Repository Root: ${REPO_ROOT}"

# Detect Python interpreter (Prioritize Python 3.10 for Hailo DFC compatibility)
PYTHON_BIN=""
for candidate in python3.10 python3.11 python3.12 python3; do
    if command -v "${candidate}" &> /dev/null; then
        PY_VER=$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$("${candidate}" -c 'import sys; print(sys.version_info.major)')
        PY_MINOR=$("${candidate}" -c 'import sys; print(sys.version_info.minor)')
        if [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -eq 10 ]; then
            PYTHON_BIN="${candidate}"
            echo "Selected canonical Python 3.10 interpreter: ${PYTHON_BIN} (v${PY_VER})"
            break
        elif [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -ge 10 ] && [ -z "${PYTHON_BIN}" ]; then
            PYTHON_BIN="${candidate}"
            echo "Candidate Python interpreter: ${PYTHON_BIN} (v${PY_VER})"
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: Python 3.10+ is required (Python 3.10 strongly recommended for Hailo DFC)." >&2
    echo "Install via: sudo apt-get install -y python3.10 python3.10-venv python3.10-dev" >&2
    exit 1
fi

VENV_DIR="${REPO_ROOT}/venv-dfc3"
if [ ! -d "${VENV_DIR}" ] && [ -d "${REPO_ROOT}/venv" ]; then
    VENV_DIR="${REPO_ROOT}/venv"
    echo "Using existing virtual environment at ${VENV_DIR}..."
elif [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR} using ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Activating virtual environment (${VENV_DIR})..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install local Hailo DFC wheel if present
DFC_WHEEL=$(find "${REPO_ROOT}" -maxdepth 1 -name "hailo_dataflow_compiler-3.34.0-*.whl" -print -quit 2>/dev/null || true)
if [ -n "${DFC_WHEEL}" ] && [ -f "${DFC_WHEEL}" ]; then
    echo "Installing Hailo Dataflow Compiler from ${DFC_WHEEL}..."
    pip install "${DFC_WHEEL}"
else
    echo "NOTICE: No local Hailo DFC wheel found in ${REPO_ROOT}."
    echo "To compile models to Hailo-8L HEF (Phases F, G, H):"
    echo "  1. Download 'hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl' from https://hailo.ai/developer-zone/software-downloads/"
    echo "  2. Place the wheel in this repository root and run: pip install ./hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"
fi

echo "Installing hailo-yolo11-compiler in editable mode with all dependencies..."
pip install -e "${REPO_ROOT}[dev,training,onnx]"

echo ""
echo "=== Bootstrap Complete ==="
echo "To activate your environment, run:"
echo "    source $(basename "${VENV_DIR}")/bin/activate"
echo ""
echo "To verify system readiness, run:"
echo "    python train_and_compile.py doctor"
