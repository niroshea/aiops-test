#!/usr/bin/env bash
cd "$(dirname "$0")"
PID_FILE="msg-embedding-service.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Server stopped (PID: $PID)."
    else
        rm -f "$PID_FILE"
        echo "Stale PID file removed. Server was not running."
    fi
else
    echo "PID file not found. Server not running."
fi
