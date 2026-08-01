#!/usr/bin/env bash
# Build the read-only dashboard SPA for this workspace (see the companion
# .github/workflows/publish-dashboard.yml). Scaffolded by
# viva_superpowers.publish_assets.emit(); run it locally to preview the
# bundle.
set -euo pipefail
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$WS_ROOT/reports/published/dashboard}"
BASE_PATH="/viva-fenics/dashboard"
INTERACTIVE_URL="https://github.com/vivarium-collective/viva-fenics"
rm -rf "$OUT"
PYTHONPATH="$WS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  vivarium-workbench-publish \
    --workspace "$WS_ROOT" \
    --out "$OUT" \
    --base-path "$BASE_PATH" \
    --interactive-url "$INTERACTIVE_URL"
find "$OUT" -name '*.map' -delete
touch "$OUT/.nojekyll"
echo "built read-only dashboard bundle at $OUT ($(du -sh "$OUT" | cut -f1))"
