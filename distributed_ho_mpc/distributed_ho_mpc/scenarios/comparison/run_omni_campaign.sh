#!/bin/bash
# Omnidirectional comparison campaign (colleague's benchmark), run inside the ho_mpc
# container from the workspace root after `source install/setup.bash`.
#
# usage: run_omni_campaign.sh <out_dir> [seeds] [extra run_comparison.py args...]
#   seeds      "start:stop" or comma list (default 0:15)
#   WORKERS    concurrent jobs (default 6)
#   PIN_CPUS   CPUs for the job slots (default 0,2,4,6,8,10: one per P-core on davide-cf)
#   METHODS    method tags (default: all seven)
#   SCENARIOS  scenarios (default: uniform,asymmetric,priority_conflict)
#
# Resumable: runs that already have a run_info.json are skipped.
set -euo pipefail

OUT=${1:?usage: run_omni_campaign.sh <out_dir> [seeds] [args...]}
SEEDS=${2:-0:15}
shift $(($# < 2 ? $# : 2))

HERE=$(dirname "$(readlink -f "$0")")
WORKERS=${WORKERS:-6}
PIN_CPUS=${PIN_CPUS:-0,2,4,6,8,10}
METHODS=${METHODS:-pf,dqp,cbf_omni,orca_omni,nh_orca_omni,dhqp_omni,dhqp_omni@nc4}
SCENARIOS=${SCENARIOS:-uniform,asymmetric,priority_conflict}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg TQDM_DISABLE=1

mkdir -p "$OUT"
{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "cmd: $0 $OUT $SEEDS $*"
    echo "workers: $WORKERS  pin_cpus: $PIN_CPUS"
    echo "methods: $METHODS"
    echo "scenarios: $SCENARIOS"
    echo "threads: OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
    python3 -c 'import casadi, numpy, qpsolvers, scipy, matplotlib; print("python:", __import__("sys").version.split()[0], "casadi:", casadi.__version__, "numpy:", numpy.__version__, "qpsolvers:", qpsolvers.__version__, "scipy:", scipy.__version__, "matplotlib:", matplotlib.__version__)'
} >> "$OUT/provenance.txt"

python3 "$HERE/run_comparison.py" \
    --preset colleague_omni \
    --scenarios "$SCENARIOS" \
    --methods "$METHODS" \
    --params '{"dhqp_omni@nc4": {"n_control": 4}}' \
    --seeds "$SEEDS" \
    --out "$OUT" \
    --workers "$WORKERS" \
    --pin-cpus "$PIN_CPUS" \
    "$@" 2>&1 | tee -a "$OUT/campaign.log"

python3 "$HERE/analyze_comparison.py" "$OUT" 2>&1 | tee -a "$OUT/analysis.log"
