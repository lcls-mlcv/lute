#!/bin/bash
# eLog workflow entry point (trigger: MANUAL, fire it on a calibrant run).
# Fits detector distance + beam centre and writes a .poni with powder_distance.py.
# Nothing here is specific to one experiment.
#
# The eLog runs this in a pod with read-only /sdf/data, so it only submits the
# real work as a Slurm job (powder_distance_job.sh). eLog "parameters" are passed
# through to powder_distance.py, e.g.:  -c LaB6 --expected 140 160
# except the wrapper-only --reservation NAME, which sets the Slurm reservation
# (default none; SLURM_RESERVATION in the environment works too).
# EXPERIMENT and RUN_NUM are set by the eLog; set them by hand to run it outside.
#
# Output: $OUT_BASE/<exp>_r<run>_<det>/ (SUMMARY.md, geometry.json, r<run>_<det>.poni,
# figures); job log in $OUT_BASE/logs/, which must already exist (the pod can't create it).
# OUT_BASE defaults to /sdf/data/lcls/ds/<hutch>/<exp>/results/$USER/geometry; the
# Slurm account is lcls:<exp>. Override OUT_BASE or NTASKS through the environment.

HERE=$(dirname "$(readlink -f "$0")")
EXP=${EXPERIMENT:?EXPERIMENT not set}
RUN=${RUN_NUM:?RUN_NUM not set}
USR=${USER:-$(whoami)}
export OUT_BASE=${OUT_BASE:-/sdf/data/lcls/ds/${EXP:0:3}/$EXP/results/$USR/geometry}
RES=${SLURM_RESERVATION-}

ARGS=()
while (($#)); do
    case $1 in
        --reservation) RES=$2; shift 2 ;;
        --reservation=*) RES=${1#*=}; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

echo "submit_powder_distance: $USR@$(hostname) exp=$EXP run=$RUN reservation=${RES:-none} out=$OUT_BASE args=[${ARGS[*]}]"
[ -d "$OUT_BASE/logs" ] || { echo "missing $OUT_BASE/logs: create it once with mkdir -p (the eLog pod can't)"; exit 1; }
sbatch --account="lcls:${EXP}" --partition=milano --constraint=OS_VER:9.6 \
       ${RES:+--reservation=$RES} --ntasks="${NTASKS:-16}" --time=01:00:00 \
       --job-name="powder_dist_r${RUN}" \
       --output="$OUT_BASE/logs/powder_distance_r$(printf %04d "$RUN")_%j.out" \
       --export=ALL,EXPERIMENT="$EXP",RUN_NUM="$RUN",OUT_BASE="$OUT_BASE",PD_HERE="$HERE",PD_SCRIPT="${PD_SCRIPT:-powder_distance.py}" \
       "$HERE/powder_distance_job.sh" "${ARGS[@]}"
