#!/usr/bin/env bash
# Remaining Stage 7 Mel `duration` runs, resumed after the grid was trimmed.
#
#   bash scripts/run_mel_duration.sh
#
# Mel utterance and Mel phoneme/vowel were dropped by decision: ~13 h of GPU
# for levels the handcrafted family has already covered end to end (see
# reports/stage7_training.md). Mel `duration` is the second feature family on
# the spine and is the one Stage 8 needs to compare against the MLP.
#
# Folds already finished are listed explicitly rather than re-run: --all-configs
# has no resume, and each Mel run costs ~25 min.

set -u
cd "$(dirname "$0")/.."

run() {
  echo ""
  echo "### $(date +%H:%M:%S)  $*"
  python -u -m src.training.run --family mel --level duration "$@"
}

# mixed baseline: folds 0-3 already done
for i in 4 5 6 7 8 9; do
  run --protocol mixed --split "$i" --regime baseline
done

for i in 0 1 2 3 4 5 6 7 8 9; do
  run --protocol mixed --split "$i" --regime dann
done

for c in ItalianPD MDVR-KCL NeuroVoz; do
  run --protocol lodo --split "$c" --regime baseline
done

for c in ItalianPD MDVR-KCL NeuroVoz; do
  run --protocol lodo --split "$c" --regime dann
done

echo ""
echo "===== MEL DURATION COMPLETE $(date) ====="
