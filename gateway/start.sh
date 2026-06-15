#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8080
