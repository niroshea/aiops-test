#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PID_FILE="msg-embedding-service.pid"

# Stop
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        while kill -0 "$PID" 2>/dev/null; do sleep 0.5; done
        echo "Server stopped."
    fi
    rm -f "$PID_FILE"
fi

# Start
./run.sh
