#!/bin/bash
# Point a SmallData (LUTE) config at a new poni (affects runs fired after this).
#   set_poni.sh <path/to/rNNNN_<det>.poni> <config.yaml> [more configs, e.g. the .j2 template ...]
# Edits poni_file in each config, keeping a timestamped backup of each and printing
# the change. Refuses if the poni can't be read, or if a config has an ai_kwargs
# block (which would override poni_file).
set -e
PONI=$(readlink -f "${1:?usage: set_poni.sh <poni> <config.yaml> [more configs ...]}")
shift
(($#)) || { echo "usage: set_poni.sh <poni> <config.yaml> [more configs ...]"; exit 1; }
test -r "$PONI" || { echo "cannot read $PONI"; exit 1; }
grep -q "^Distance:" "$PONI" || { echo "$PONI does not look like a poni file"; exit 1; }
for f in "$@"; do
    [ -f "$f" ] || { echo "no such config: $f"; exit 1; }
    grep -q "ai_kwargs:" "$f" && { echo "$f still has ai_kwargs, which overrides poni_file; fix by hand"; exit 1; }
    grep -q 'poni_file:' "$f" || { echo "$f has no poni_file line"; exit 1; }
done
for f in "$@"; do
    cp -p "$f" "$f.bak_$(date +%Y%m%d_%H%M%S)"
    sed -i "s#^\(\s*poni_file:\s*\).*#\1\"$PONI\"#" "$f"
    echo "== $f"; grep -n "poni_file" "$f"
done
echo "distance in new poni: $(awk '/^Distance:/{printf "%.2f mm", $2*1000}' "$PONI")"
echo "Applies to runs fired from now on; refire older runs to reprocess them."
