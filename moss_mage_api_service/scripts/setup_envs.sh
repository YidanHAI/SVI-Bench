#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$SERVICE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SERVICE_ROOT/.env"
  set +a
fi

BASE_PYTHON="${BASE_PYTHON:-python3}"
MOSS_PYTHON="${MOSS_PYTHON:-$SERVICE_ROOT/.venv-moss/bin/python}"
MAGE_PYTHON="${MAGE_PYTHON:-$SERVICE_ROOT/.venv-mage/bin/python}"

if [[ "$BASE_PYTHON" != */* ]]; then
  BASE_PYTHON="$(command -v "$BASE_PYTHON" || true)"
fi

if [[ -z "$BASE_PYTHON" || ! -x "$BASE_PYTHON" ]]; then
  echo "BASE_PYTHON not found: $BASE_PYTHON" >&2
  exit 1
fi

setup_profile() {
  local profile="$1"
  local python_bin requirements
  if [[ "$profile" == moss ]]; then
    python_bin="$MOSS_PYTHON"
    requirements="$SERVICE_ROOT/requirements-moss.txt"
  else
    python_bin="$MAGE_PYTHON"
    requirements="$SERVICE_ROOT/requirements-mage.txt"
  fi
  local venv_dir
  venv_dir="$(dirname -- "$(dirname -- "$python_bin")")"
  if [[ ! -x "$python_bin" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$venv_dir"
  fi
  "$python_bin" -m pip install --upgrade \
    fastapi==0.135.2 uvicorn==0.42.0 Pillow==12.1.1
  "$python_bin" -m pip install --no-deps --ignore-installed -r "$requirements"
  cd "$SERVICE_ROOT"
  "$python_bin" -m service.check_environment --profile "$profile"
}

TARGET="${1:-all}"
case "$TARGET" in
  moss|mage) setup_profile "$TARGET" ;;
  all)
    setup_profile moss
    setup_profile mage
    ;;
  *)
    echo "Usage: bash scripts/setup_envs.sh [moss|mage|all]" >&2
    exit 2
    ;;
esac
