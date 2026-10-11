#!/usr/bin/env bash
# Point git at the tracked hooks directory. Run once per clone.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
chmod +x .githooks/*
git config core.hooksPath .githooks
echo "hooks installed: $(git config core.hooksPath)"
