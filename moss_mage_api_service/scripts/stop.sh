#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${RUN_DIR:-$SERVICE_ROOT/run}"

stop_service() {
  local name="$1"
  local pid_file="$RUN_DIR/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: no PID file"
    return
  fi
  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    local command_line
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$command_line" != *"-m service.server"* ]]; then
      echo "$name: PID $pid does not belong to this service; refusing to stop it" >&2
      return 1
    fi
    kill "$pid"
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name did not stop within 30 seconds; PID $pid is still running" >&2
      return 1
    fi
    echo "Stopped $name (PID $pid)"
  else
    echo "$name: stale PID $pid"
  fi
  rm -f "$pid_file"
}

TARGET="${1:-all}"
case "$TARGET" in
  moss-realtime|mage) stop_service "$TARGET" ;;
  all)
    status=0
    stop_service moss-realtime || status=1
    stop_service mage || status=1
    exit "$status"
    ;;
  *)
    echo "Usage: bash scripts/stop.sh [moss-realtime|mage|all]" >&2
    exit 2
    ;;
esac
