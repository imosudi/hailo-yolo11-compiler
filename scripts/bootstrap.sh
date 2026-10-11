#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap.sh - Environment bootstrap script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== hailo-yolo11-compiler: Environment Bootstrapper ==="
echo "Repository Root: ${REPO_ROOT}"

# Helper to run commands as root or with sudo
run_as_root() {
    if [ "${EUID:-$(id -u)}" -eq 0 ]; then
        "$@"
    elif command -v sudo &> /dev/null; then
        sudo "$@"
    else
        echo "WARNING: Root privileges required for: $*" >&2
        return 1
    fi
}

# ------------------------------------------------------------------------------
# System Dependencies: Hardware Telemetry, Camera & Live Deployment
# ------------------------------------------------------------------------------
echo ""
echo "--- Checking System Hardware Telemetry, Camera & Video Dependencies ---"

IS_RPI=false
if [ -f "/proc/device-tree/model" ] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_RPI=true
    echo "Detected Raspberry Pi platform: $(tr -d '\0' < /proc/device-tree/model)"
fi

if command -v apt-get &> /dev/null; then
    DEPS_TO_INSTALL=()

    # 1. Hardware sensors telemetry (lm-sensors)
    if ! command -v sensors &> /dev/null; then
        DEPS_TO_INSTALL+=("lm-sensors")
    else
        echo "Found lm-sensors: $(sensors --version 2>&1 | head -n1)"
    fi

    # 2. Raspberry Pi CSI Camera Module (python3-picamera2)
    # Check if python3-picamera2 package is available in apt repositories or on Raspberry Pi
    if [ "${IS_RPI}" = true ] || apt-cache show python3-picamera2 &>/dev/null 2>&1; then
        if ! dpkg -s python3-picamera2 &> /dev/null 2>&1; then
            DEPS_TO_INSTALL+=("python3-picamera2")
        else
            echo "Found Raspberry Pi CSI camera support: python3-picamera2"
        fi

        for cam_tool in libcamera-tools libcamera-apps; do
            if apt-cache show "${cam_tool}" &>/dev/null 2>&1 && ! dpkg -s "${cam_tool}" &>/dev/null 2>&1; then
                DEPS_TO_INSTALL+=("${cam_tool}")
            fi
        done
    fi

    # 3. V4L2 USB camera utilities
    if ! command -v v4l2-ctl &> /dev/null; then
        DEPS_TO_INSTALL+=("v4l-utils")
    else
        echo "Found V4L2 camera tools: v4l2-ctl"
    fi

    # 4. OpenCV display / graphics prerequisites
    for libpkg in libgl1 libglib2.0-0; do
        if ! dpkg -s "${libpkg}" &> /dev/null 2>&1; then
            DEPS_TO_INSTALL+=("${libpkg}")
        fi
    done

    # 5. HailoRT Python library for Raspberry Pi AI HAT+ accelerator
    if [ "${IS_RPI}" = true ] || apt-cache show python3-hailort &>/dev/null 2>&1; then
        if ! dpkg -s python3-hailort &> /dev/null 2>&1; then
            DEPS_TO_INSTALL+=("python3-hailort")
        else
            echo "Found HailoRT Python bindings: python3-hailort"
        fi
    fi

    if [ ${#DEPS_TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing required system packages via apt-get: ${DEPS_TO_INSTALL[*]}..."
        run_as_root apt-get update -y && run_as_root apt-get install -y "${DEPS_TO_INSTALL[@]}" || {
            echo "NOTICE: Some apt packages could not be installed automatically. Ensure permissions allow apt-get."
        }
    else
        echo "All system hardware telemetry, camera, and live deployment prerequisites are installed."
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
    echo "Creating virtual environment at ${VENV_DIR} using ${PYTHON_BIN} with system site packages enabled..."
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

# Ensure virtualenv config enables access to system site packages (enables Picamera2 from apt)
if [ -f "${VENV_DIR}/pyvenv.cfg" ]; then
    if grep -q "include-system-site-packages = false" "${VENV_DIR}/pyvenv.cfg"; then
        echo "Enabling system-site-packages in ${VENV_DIR}/pyvenv.cfg for Picamera2 access..."
        sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "${VENV_DIR}/pyvenv.cfg"
    fi
fi

# Ensure system dist-packages is linked in site-packages
for sp_dir in "${VENV_DIR}"/lib/python*/site-packages; do
    if [ -d "${sp_dir}" ]; then
        echo "/usr/lib/python3/dist-packages" > "${sp_dir}/_system_dist_packages.pth"
    fi
done

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
