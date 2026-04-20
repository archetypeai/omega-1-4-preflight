#!/usr/bin/env bash
# Thin wrapper around preflight.py so callers can invoke the tool from any cwd.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/preflight.py" "$@"
