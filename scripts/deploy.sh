#!/usr/bin/env bash
# Publish package to PyPI
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

echo "Preparing to publish distributions to PyPI..."

# Ensure build artifacts exist
if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
  echo "No distributions found in dist/ — building now..."
  python -m pip install --upgrade build twine
  python -m build
fi

# Ensure credentials available
: "${TWINE_USERNAME:=}" || true
if [ -z "${TWINE_USERNAME:-}" ]; then
  read -rp "Twine username (use __token__ for API tokens): " TWINE_USERNAME
fi
if [ -z "${TWINE_PASSWORD:-}" ]; then
  echo "Enter Twine password or API token (will not be shown):"
  read -rs TWINE_PASSWORD
  echo
fi

echo "Uploading dist/* to PyPI (https://upload.pypi.org/legacy/)"
TWINE_USERNAME="$TWINE_USERNAME" TWINE_PASSWORD="$TWINE_PASSWORD" python -m twine upload dist/*

echo "Upload complete."
