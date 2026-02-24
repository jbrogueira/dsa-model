#!/usr/bin/env bash
# Sets up a JAX virtual environment for this project.
# Detects platform and installs the appropriate JAX variant:
#   - macOS ARM (Apple Silicon): jax[cpu]
#   - Linux x86_64:              jax[cuda12]  (requires NVIDIA driver)
#
# Usage:
#   bash setup_jax.sh            # creates ~/venvs/jax/
#   VENV=~/my/path bash setup_jax.sh  # custom venv path

set -euo pipefail

VENV="${VENV:-$HOME/venvs/jax}"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Platform: $OS / $ARCH"
echo "Venv:     $VENV"

# Pick JAX extra based on platform
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    JAX_EXTRA="jax[cpu]"
    echo "Backend:  CPU (Apple Silicon)"
elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    JAX_EXTRA="jax[cuda12]"
    echo "Backend:  CUDA 12 (Linux x86_64)"
    echo "NOTE: requires a working NVIDIA driver. If unavailable, set JAX_PLATFORM_NAME=cpu at runtime."
else
    echo "Unrecognised platform $OS/$ARCH — falling back to jax[cpu]"
    JAX_EXTRA="jax[cpu]"
fi

# Create venv if it doesn't exist
if [[ ! -d "$VENV" ]]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Installing dependencies..."
"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install --quiet \
    numpy numba matplotlib quantecon scipy pytest \
    "$JAX_EXTRA"

echo ""
echo "Done. Activate with:"
echo "  source $VENV/bin/activate"
echo ""
echo "Then run:"
echo "  python olg_transition.py --test --backend jax"
