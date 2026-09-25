#!/usr/bin/env bash
# Build paper/PAPER.pdf from paper/PAPER.md using a Markdown->HTML conversion
# and headless Chrome print-to-PDF.
#
# Requirements: python3 (>=3.9), Google Chrome, pip (only `markdown` is installed
# into a local .venv on first run).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-pdf"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q markdown
fi

"$VENV/bin/python" tools/md_to_html.py

CHROME=""
for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
         "$(command -v google-chrome 2>/dev/null || true)" \
         "$(command -v chromium 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CHROME="$c"; break; fi
done
if [ -z "$CHROME" ]; then
  echo "Chrome/Chromium not found; PAPER.html is ready for manual printing." >&2
  exit 0
fi

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --no-sandbox \
  --virtual-time-budget=15000 \
  --print-to-pdf="$PWD/paper/PAPER.pdf" "file://$PWD/paper/PAPER.html" 2>/dev/null

ls -lh paper/PAPER.pdf
echo "OK: paper/PAPER.pdf"
