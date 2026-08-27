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

campaign_config="${VL_INTERACTION_CAMPAIGN_CONFIG:-$ROOT/config/recording_campaign.json}"
export VL_INTERACTION_CAMPAIGN_CONFIG="$campaign_config"

action="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$action" in
  start)
    log_root="$ROOT/outputs/recording_campaign/logs"
    mkdir -p "$log_root"
    log_path="$log_root/campaign_$(date +%Y%m%d_%H%M%S).log"
    nohup setsid node "$ROOT/scripts/run_recording_campaign.mjs" run \
      "$@" --config "$campaign_config" --log-path "$log_path" \
      >>"$log_path" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'Recording campaign failed to start. Log: %s\n' "$log_path" >&2
      tail -n 30 "$log_path" >&2 || true
      exit 1
    fi
    printf 'Recording campaign started: pid=%s\nLog: %s\n' "$pid" "$log_path"
    node "$ROOT/scripts/run_recording_campaign.mjs" status \
      "$@" --config "$campaign_config" || true
    ;;
  dry-run|status|stop)
    exec node "$ROOT/scripts/run_recording_campaign.mjs" \
      "$action" "$@" --config "$campaign_config"
    ;;
  help|-h|--help)
    exec node "$ROOT/scripts/run_recording_campaign.mjs" --help
    ;;
  *)
    printf 'Usage: bash scripts/record_all.sh [start|dry-run|status|stop] [options]\n' >&2
    exit 2
    ;;
esac
