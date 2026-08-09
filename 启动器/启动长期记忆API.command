#!/bin/zsh
SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h}
exec /opt/homebrew/bin/python3 "$PROJECT_ROOT/应用/后端/memory_api_server.py" --daemon
