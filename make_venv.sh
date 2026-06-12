#!/usr/bin/env bash
# Create .venv and install dependencies.
# On Linux, installs jax[cuda12]; on macOS, plain jax.
#
#   bash make_venv.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

if [[ "$(uname)" == "Darwin" ]]; then
  "$VENV/bin/pip" install -r "$DIR/requirements.txt"
else
  "$VENV/bin/pip" install "jax[cuda12]" && \
  "$VENV/bin/pip" install -r "$DIR/requirements.txt"
fi

echo "Done: $VENV"
