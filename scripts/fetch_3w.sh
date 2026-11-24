#!/usr/bin/env bash
# Download only the REAL instances of the 3W Dataset 2.0.0 (about 0.5 GB),
# pinned to the commit used to produce the released outputs.
#   usage: bash scripts/fetch_3w.sh [target_dir]      (default: data/3W)
set -euo pipefail
TARGET="${1:-data/3W}"
COMMIT="826fec47d5ba59e93b1b91be572bb48641043a2e"   # petrobras/3W, 2026-11-10
git clone --filter=blob:none --no-checkout https://github.com/petrobras/3W.git "$TARGET"
cd "$TARGET"
git sparse-checkout init --no-cone
printf '/dataset/*/WELL-*\n/dataset/dataset.ini\n/dataset/README.md\n/dataset/LICENSE-CC-BY\n' \
  > .git/info/sparse-checkout
git checkout -q "$COMMIT"
echo "real instances: $(find dataset -name 'WELL-*.parquet' | wc -l) (expected 1119)"
