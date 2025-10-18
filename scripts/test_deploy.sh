#!/usr/bin/env bash
# Publish package to TestPyPI for verification
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

echo "Preparing to publish distributions to TestPyPI..."

# Ensure build artifacts exist
if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
  echo "No distributions found in dist/ — building now..."
  python -m pip install --upgrade build twine
  python -m build
fi

if [ -z "${TWINE_USERNAME:-}" ]; then
  TWINE_USERNAME="__token__"
fi

if [ -z "${TWINE_PASSWORD:-}" ]; then
  echo "Enter TestPyPI token (will not be shown):"
  read -rs TWINE_PASSWORD
  echo
fi

echo "Uploading dist/* to TestPyPI (https://test.pypi.org/legacy/)"
TWINE_USERNAME="$TWINE_USERNAME" TWINE_PASSWORD="$TWINE_PASSWORD" python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

echo "TestPyPI upload complete."
