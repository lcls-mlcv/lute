#!/bin/bash
# Slurm job body for submit_powder_distance.sh: runs the distance fit on a
# compute node. Expects EXPERIMENT, RUN_NUM, OUT_BASE in the environment; any
# arguments are passed through to the script. PD_HERE is the directory
# holding the scripts (set by submit_powder_distance.sh). PD_SCRIPT overrides
# the script name (default powder_distance.py).
HERE=${PD_HERE:?PD_HERE not set (submit through submit_powder_distance.sh)}   # Slurm runs a spool copy of this file
SCRIPT=${PD_SCRIPT:-powder_distance.py}

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src:$PYTHONPATH   # for the .poni

mkdir -p "$OUT_BASE" && cd "$OUT_BASE" || exit 1
echo "powder_distance: script=$SCRIPT exp=$EXPERIMENT run=$RUN_NUM args=[$*] ranks=$SLURM_NTASKS host=$(hostname) out=$OUT_BASE"
# Publish to the eLog Summaries tab (and run status) unless --elog-summary is already given.
ELOG=(--elog-summary); [[ " $* " == *" --elog-summary"* ]] && ELOG=()
mpirun -n "$SLURM_NTASKS" python -u "$HERE/$SCRIPT" -e "$EXPERIMENT" -r "$RUN_NUM" "${ELOG[@]}" "$@"
rc=$?
echo "powder_distance finished rc=$rc; results in $OUT_BASE/${EXPERIMENT}_r$(printf %04d "$RUN_NUM")_*"
exit $rc
