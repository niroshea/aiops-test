#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install llama-cpp-python flask
else
    source venv/bin/activate
fi

PID_FILE="msg-embedding-service.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server already running (PID: $(cat "$PID_FILE"))"
    exit 1
fi

echo "Starting embedding server on :10911 (background)..."
nohup python3 msg-embedding-service.py </dev/null >/dev/null 2>&1 &
echo $! >"$PID_FILE"
echo "Server started (PID: $!)"
