#!/usr/bin/env bash
# §4.2 shuffled-domain control, handcrafted duration only.
#
#   bash scripts/run_shuffle_control.sh
#
# Both arms use seed 1. The published `*_dann_s0` directories were trained
# before torch.manual_seed was set, so re-running seed 0 would neither
# reproduce them nor pair with the shuffled arm. Seed 1 leaves them in place
# and gives the real DANN and the control the same init.
#
# One GPU job. About 26 handcrafted runs.

set -u
cd "$(dirname "$0")/.."

run() {
  echo ""
  echo "### $(date +%H:%M:%S)  python -m src.training.run $*"
  python -u -m src.training.run "$@"
}

SEED=1

echo "===== real DANN, seed ${SEED} ====="
run --family handcrafted --level duration --protocol mixed --all-configs \
    --regime dann --seed "$SEED"
run --family handcrafted --level duration --protocol lodo --all-configs \
    --regime dann --seed "$SEED"

echo "===== shuffled corpus labels, seed ${SEED} ====="
run --family handcrafted --level duration --protocol mixed --all-configs \
    --regime dann --shuffle-domains --seed "$SEED"
run --family handcrafted --level duration --protocol lodo --all-configs \
    --regime dann --shuffle-domains --seed "$SEED"

echo ""
echo "===== SHUFFLE CONTROL COMPLETE $(date) ====="
