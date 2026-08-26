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

export JUDGE_OUTPUT_ROOT="${JUDGE_OUTPUT_ROOT:-$ROOT/outputs/judge_campaign}"
export JUDGE_MANIFEST_DIR="${JUDGE_MANIFEST_DIR:-$JUDGE_OUTPUT_ROOT/manifests}"

prepare_manifests() {
  python3 "$ROOT/scripts/prepare_judge_manifests.py" \
    --out-dir "$JUDGE_MANIFEST_DIR"
}

action="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$action" in
  prepare)
    prepare_manifests
    ;;
  validate)
    prepare_manifests
    exec python3 "$ROOT/scripts/judge_campaign.py" validate "$@"
    ;;
  start)
    prepare_manifests
    exec python3 "$ROOT/scripts/judge_campaign.py" start "$@"
    ;;
  status|stop)
    exec python3 "$ROOT/scripts/judge_campaign.py" "$action" "$@"
    ;;
  help|-h|--help)
    printf 'Usage: bash scripts/judge_all.sh [prepare|validate|start|status|stop] [Judge campaign options]\n'
    ;;
  *)
    printf 'Unknown action: %s\n' "$action" >&2
    exit 2
    ;;
esac
