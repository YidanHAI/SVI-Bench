#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${RUN_DIR:-$SERVICE_ROOT/run}"
LOG_DIR="${LOG_DIR:-$SERVICE_ROOT/logs}"
mkdir -p "$RUN_DIR" "$LOG_DIR"

start_service() {
  local name="$1"
  local pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "$name is already running (PID $(<"$pid_file"))" >&2
    return 1
  fi
  nohup bash "$SCRIPT_DIR/run.sh" "$name" >"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "Started $name (PID $pid, log $LOG_DIR/$name.log)"
}

start_service moss-realtime
start_service mage
echo "Model loading continues in the background. Check logs and /health before sending traffic."
