#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap.sh - Environment bootstrap script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== hailo-yolo11-compiler: Environment Bootstrapper ==="
echo "Repository Root: ${REPO_ROOT}"

# ------------------------------------------------------------------------------
# System Dependencies: Hardware Telemetry Tools (lm-sensors)
# ------------------------------------------------------------------------------
echo ""
echo "--- Checking System Hardware Telemetry Tools (lm-sensors) ---"
if ! command -v sensors &> /dev/null; then
    echo "NOTICE: 'sensors' (lm-sensors) is not installed."
    if command -v apt-get &> /dev/null; then
        if [ "${EUID:-$(id -u)}" -eq 0 ]; then
            echo "Installing lm-sensors via apt-get..."
            apt-get update -y && apt-get install -y lm-sensors || true
        elif sudo -n true 2>/dev/null; then
            echo "Installing lm-sensors via sudo apt-get..."
            sudo apt-get update -y && sudo apt-get install -y lm-sensors || true
        else
            echo "To enable extensive hardware sensor telemetry (voltage, current, temps, fans):"
            echo "    sudo apt-get update && sudo apt-get install -y lm-sensors"
        fi
    fi
else
    echo "Found lm-sensors: $(sensors --version 2>&1 | head -n1)"
fi
echo ""

# ------------------------------------------------------------------------------
# System Dependencies: Camera & Live Deployment (Picamera2, V4L2, OpenCV support)
# ------------------------------------------------------------------------------
echo "--- Checking Live Deployment Camera & Video Dependencies ---"
IS_RPI=false
if [ -f "/proc/device-tree/model" ] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_RPI=true
    echo "Detected Raspberry Pi platform: $(tr -d '\0' < /proc/device-tree/model)"
fi

if command -v apt-get &> /dev/null; then
    DEPS_TO_INSTALL=()
    
    # 1. Raspberry Pi CSI Camera Module (Picamera2)
    if [ "${IS_RPI}" = true ]; then
        if ! dpkg -s python3-picamera2 &> /dev/null; then
            DEPS_TO_INSTALL+=("python3-picamera2" "libcamera-tools")
        else
            echo "Found Raspberry Pi CSI camera support: python3-picamera2"
        fi
    fi

    # 2. V4L2 webcam utilities
    if ! command -v v4l2-ctl &> /dev/null; then
        DEPS_TO_INSTALL+=("v4l-utils")
    else
        echo "Found V4L2 camera tools: v4l2-ctl"
    fi

    # 3. OpenCV display / graphics prerequisites
    for libpkg in libgl1 libglib2.0-0; do
        if ! dpkg -s "${libpkg}" &> /dev/null; then
            DEPS_TO_INSTALL+=("${libpkg}")
        fi
    done

    if [ ${#DEPS_TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing live deployment packages via apt-get: ${DEPS_TO_INSTALL[*]}..."
        if [ "${EUID:-$(id -u)}" -eq 0 ]; then
            apt-get update -y && apt-get install -y "${DEPS_TO_INSTALL[@]}" || true
        elif sudo -n true 2>/dev/null; then
            sudo apt-get update -y && sudo apt-get install -y "${DEPS_TO_INSTALL[@]}" || true
        else
            echo "NOTICE: To enable camera hardware acceleration and Picamera2 support, run:"
            echo "    sudo apt-get update && sudo apt-get install -y ${DEPS_TO_INSTALL[*]}"
        fi
    else
        echo "All system camera and live deployment prerequisites are installed."
    fi
fi
echo ""

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
pip install -e "${REPO_ROOT}[dev,training,onnx,deploy]"

echo ""
echo "=== Bootstrap Complete ==="
echo "To activate your environment, run:"
echo "    source $(basename "${VENV_DIR}")/bin/activate"
echo ""
echo "To verify system readiness, run:"
echo "    python train_and_compile.py doctor"
