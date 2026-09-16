#!/bin/bash
#SBATCH --job-name=regen
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=regen_%j.out
#
# Rebuild one pipeline stage's CSV from scratch, sharded across cores.
#
#   STAGE=tm_data  sbatch data_scripts/regenerate.sh   # tm_scores.csv  (~15 min)
#   STAGE=fgw_data sbatch data_scripts/regenerate.sh   # fgw_scores.csv (~2.5 h)
#   bash data_scripts/regenerate.sh                    # inside an interactive session
#
# STAGE defaults to fgw_data. tm_data must run first: fgw_data reads its
# output and needs the superposition column the current tm_data.py writes.
# The #SBATCH lines are comments to bash, so the same file works either way.
# Override any setting from the environment, e.g.
#   SHARDS=8 END_ROW=50000 STAGE=tm_data bash data_scripts/regenerate.sh

set -euo pipefail

STAGE="${STAGE:-fgw_data}"
DATA_DIR="${DATA_DIR:-/jet/home/jxu23/OCEANDIR}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHARDS="${SHARDS:-${SLURM_CPUS_PER_TASK:-16}}"
START_ROW="${START_ROW:-0}"
END_ROW="${END_ROW:-181288}"   # one past the last source row processed so far

case "$STAGE" in
    tm_data)
        OUTPUT="$DATA_DIR/tm_scores.csv"
        AUDIT=(python "$REPO_DIR/data_scripts/audit_upstream.py" --tm-csv "$OUTPUT")
        EXPECTED="  duplicated  0   and no rows with mismatched seqxA/seqM/seqyA"
        ;;
    fgw_data)
        OUTPUT="$DATA_DIR/fgw_scores.csv"
        AUDIT=(python "$REPO_DIR/data_scripts/audit_fgw_data.py" --csv "$OUTPUT")
        EXPECTED="  duplicated pairs 0;  rows per pair ~2/18/<=64 (aligned + negatives);
  label percentiles spread over (0, 1] -- if not, adjust the scale in fgw.py"
        ;;
    *)
        echo "unknown STAGE '$STAGE' (tm_data or fgw_data)" >&2
        exit 1
        ;;
esac

SHARD_DIR="$DATA_DIR/${STAGE}_shards"
LOG_DIR="$DATA_DIR/${STAGE}_logs"
PROGRESS="$DATA_DIR/${STAGE}_progress.txt"
BASENAME="$(basename "$OUTPUT" .csv)"

# A batch job does not inherit an interactively-activated venv. Source the
# project setup if it is there, or point VENV at an activate script yourself.
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1090
    source "$VENV"
elif [ -f "$REPO_DIR/psc_interactive_setup.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/psc_interactive_setup.sh"
fi
echo "python: $(command -v python)"

# The FGW patches are 32x32 and TM-align is single-threaded. Threaded BLAS
# is pure overhead at that size, and with one process per core it would
# oversubscribe the node badly.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Both stages append to their output and resume past whatever source rows are
# already in it. Started on top of an old file, a stage would skip almost
# every pair and append onto stale rows -- so refuse rather than guess.
if [ -e "$OUTPUT" ]; then
    echo "REFUSING TO RUN: $OUTPUT already exists." >&2
    echo "" >&2
    echo "  $STAGE.py appends, and resumes from the source rows already" >&2
    echo "  present. Archive it first:" >&2
    echo "" >&2
    echo "    mv $OUTPUT ${OUTPUT%.csv}.legacy.csv" >&2
    echo "    mv $PROGRESS ${OUTPUT%.csv}.legacy.progress.txt" >&2
    exit 1
fi
if [ -e "$SHARD_DIR" ] && [ -n "$(ls -A "$SHARD_DIR" 2>/dev/null)" ]; then
    echo "REFUSING TO RUN: $SHARD_DIR is not empty (shards would resume, not restart)." >&2
    echo "  rm -rf $SHARD_DIR" >&2
    exit 1
fi

mkdir -p "$SHARD_DIR" "$LOG_DIR"

STEP=$(( (END_ROW - START_ROW + SHARDS - 1) / SHARDS ))   # ceiling division

echo "=================================================================="
echo "  regenerating $OUTPUT  (stage $STAGE)"
echo "=================================================================="
echo "  repo        $REPO_DIR"
echo "  source rows [$START_ROW, $END_ROW)"
echo "  shards      $SHARDS x $STEP rows"
echo "  shard dir   $SHARD_DIR"
echo "  logs        $LOG_DIR"
echo "  started     $(date)"
echo ""

pids=()
labels=()
for i in $(seq 0 $((SHARDS - 1))); do
    s=$(( START_ROW + i * STEP ))
    e=$(( s + STEP ))
    [ "$e" -gt "$END_ROW" ] && e="$END_ROW"
    [ "$s" -ge "$END_ROW" ] && break

    tag=$(printf "%02d" "$i")
    python "$REPO_DIR/data_scripts/$STAGE.py" \
        --start-row "$s" --end-row "$e" \
        --output "$SHARD_DIR/$BASENAME.$tag.csv" \
        > "$LOG_DIR/shard.$tag.log" 2>&1 &

    pid=$!
    pids+=("$pid")
    labels+=("$tag [$s, $e)")
    echo "  launched shard $tag  rows [$s, $e)  pid $pid"
done

echo ""
echo "waiting for ${#pids[@]} shards..."

failed=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "  shard ${labels[$idx]}  OK"
    else
        echo "  shard ${labels[$idx]}  FAILED -- see $LOG_DIR/shard.$(printf "%02d" "$idx").log" >&2
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "" >&2
    echo "$failed shard(s) failed. Not merging -- a partial merge would look" >&2
    echo "like a complete dataset. Fix the cause, delete the bad shard files," >&2
    echo "and rerun those windows individually." >&2
    exit 1
fi

echo ""
echo "merging shards..."

# Shard windows are disjoint and each stage writes one pair's rows in one
# block, so after this concatenation every key is a single contiguous run --
# which is what pair_data.iter_pair_groups requires of fgw_scores.csv.
first=1
for f in "$SHARD_DIR"/"$BASENAME".??.csv; do
    if [ "$first" -eq 1 ]; then
        head -1 "$f" > "$OUTPUT"
        first=0
    fi
    tail -n +2 "$f" >> "$OUTPUT"
done
cat "$SHARD_DIR"/"$BASENAME".??.csv.runs.jsonl > "$OUTPUT.runs.jsonl" 2>/dev/null || true

echo "  wrote $OUTPUT  ($(du -h "$OUTPUT" | cut -f1))"
echo "  rows  $(( $(wc -l < "$OUTPUT") - 1 ))"
echo ""

echo "=================================================================="
echo "  audit"
echo "=================================================================="
"${AUDIT[@]}"

echo ""
echo "finished $(date)"
echo ""
echo "Check the audit above before the next stage. Expected:"
echo "$EXPECTED"
