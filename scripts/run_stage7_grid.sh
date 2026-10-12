#!/usr/bin/env bash
# Stage 7 grid, in dependency and value order. One GPU job -- do not start a
# second against the same device.
#
#   bash scripts/run_stage7_grid.sh
#
# Every run selects checkpoints under configs/train.yaml's min_select_epoch,
# so baseline and DANN share one selection rule (see reports/stage7_training.md).
#
# Phase 1 is the whole spec at handcrafted cost (~1.5 h): if anything is wrong
# with a level, it shows up here rather than nine hours into Mel.
#
# Phoneme runs are vowel-only and mixed-only. Phoneme LODO is not a valid
# experiment: MDVR-KCL has no phoneme segments (English is word-only), so
# holding it out is a no-op, and holding out NeuroVoz leaves one training
# corpus -- an adversary with a single domain has nothing to be adversarial
# about. PIPELINE.md Stage 6 says this; it is not an omission.

set -u
cd "$(dirname "$0")/.."

run() {
  echo ""
  echo "### $(date +%H:%M:%S)  python -m src.training.run $*"
  python -u -m src.training.run "$@"
}

echo "===== PHASE 1: handcrafted, every level ====="

for regime in baseline dann; do
  run --family handcrafted --level duration --protocol mixed --all-configs --regime "$regime"
  run --family handcrafted --level duration --protocol lodo  --all-configs --regime "$regime"
done

for regime in baseline dann; do
  run --family handcrafted --level utterance --protocol mixed --all-configs --regime "$regime"
  run --family handcrafted --level utterance --protocol lodo  --all-configs --regime "$regime"
done

for regime in baseline dann; do
  run --family handcrafted --level phoneme --phone-class vowel \
      --protocol mixed --all-configs --regime "$regime"
done

# Single-corpus baselines: the drop when corpora are pooled (PIPELINE 7a).
# Handcrafted only -- the same sweep on Mel is ~10 h of GPU to answer a
# question the 40-feature model already answers.
for corpus in ItalianPD MDVR-KCL NeuroVoz; do
  run --family handcrafted --level duration --protocol mixed --all-configs \
      --regime single --dataset "$corpus"
done

echo ""
echo "===== PHASE 2: mel duration ====="

for regime in baseline dann; do
  run --family mel --level duration --protocol mixed --all-configs --regime "$regime"
  run --family mel --level duration --protocol lodo  --all-configs --regime "$regime"
done

echo ""
echo "===== PHASE 3: mel utterance and phoneme-vowel ====="

for regime in baseline dann; do
  run --family mel --level utterance --protocol mixed --all-configs --regime "$regime"
  run --family mel --level utterance --protocol lodo  --all-configs --regime "$regime"
done

for regime in baseline dann; do
  run --family mel --level phoneme --phone-class vowel \
      --protocol mixed --all-configs --regime "$regime"
done

echo ""
echo "===== GRID COMPLETE $(date) ====="
