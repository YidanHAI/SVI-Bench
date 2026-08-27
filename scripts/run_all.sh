#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

env_file="${VL_INTERACTION_ENV_FILE:-$ROOT/.env}"
if [[ -f "$env_file" ]]; then
  mode="$(stat -c '%a' "$env_file")"
  if (( (8#$mode & 077) != 0 )); then
    printf 'Environment file must not be accessible by group/other (chmod 600): %s\n' "$env_file" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

action="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec python3 "$ROOT/scripts/pipeline_campaign.py" "$action" "$@"
