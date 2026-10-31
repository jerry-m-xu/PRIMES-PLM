#!/bin/bash
#SBATCH --job-name=scope40
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=scope40_%j.out
#
# The SCOPe40 benchmark as batch jobs. Submit from the repository root, with
# the steps to run as the argument:
#
#   sbatch scope_benchmark/scope.sh prepare,baselines
#   sbatch -p GPU-shared --gpus=v100-32:1 --cpus-per-task=5 scope_benchmark/scope.sh student
#   sbatch scope_benchmark/scope.sh evaluate
#
# There is no default step, so a job submitted without one fails at once
# instead of running something else. The steps, one or a comma-separated
# list, run in order:
#   prepare     prepare_scope.py, with the training-overlap check
#   baselines   search_baselines.py, Foldseek and MMseqs2
#   student     search_student.py; needs a GPU, so override the partition as above
#               (options on the sbatch command line win over the #SBATCH lines)
#   evaluate    evaluate_search.py on every hit file, all queries and below MAX_ID
#
# Settings, through the environment:
#   FOLDSEEK, MMSEQS   binaries (default $HOME/tools/<name>/bin/<name>)
#   CHECKPOINT         student checkpoint (default the dense_650M run's student_best.pt)
#   NAME               prefix of the student's hit files (default student)
#   MAX_ID             training-identity cut for the held-out queries (default 0.3)

set -euo pipefail

# the argument, or STEP exported in the environment
STEP="${1:-${STEP:-}}"
if [ -z "$STEP" ]; then
    echo "no step given: sbatch scope_benchmark/scope.sh <prepare|baselines|student|evaluate>" >&2
    exit 1
fi
OCEAN=/jet/home/jxu23/OCEANDIR
SCOPE_DIR="$OCEAN/scope40"
FOLDSEEK="${FOLDSEEK:-$HOME/tools/foldseek/bin/foldseek}"
MMSEQS="${MMSEQS:-$HOME/tools/mmseqs/bin/mmseqs}"
CHECKPOINT="${CHECKPOINT:-$OCEAN/student_checkpoints_dense_650M/student_best.pt}"
NAME="${NAME:-student}"
MAX_ID="${MAX_ID:-0.3}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

# under sbatch this file runs from SLURM's spool directory, so the repository
# is where the job was submitted from; set REPO_DIR to override
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$REPO_DIR/scope_benchmark/scope40.py" ]; then
    echo "REPO_DIR=$REPO_DIR is not the repository; submit from its root or set REPO_DIR." >&2
    exit 1
fi

# a batch job does not inherit an activated venv
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1090
    source "$VENV"
elif [ -f "$REPO_DIR/psc_interactive_setup.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/psc_interactive_setup.sh"
elif [ -f "$HOME/psc_interactive_setup.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/psc_interactive_setup.sh"
fi

run() {
    echo "+ $*"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        # SLURM 22.05+ does not pass --cpus-per-task on to srun; without it the
        # searches would share one CPU
        srun --ntasks=1 --cpus-per-task="$THREADS" "$@"
    else
        "$@"
    fi
}

echo "steps : $STEP"
echo "repo  : $REPO_DIR"
echo "cpus  : $THREADS; gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "start : $(date)"

for step in ${STEP//,/ }; do
    echo ""
    echo "== $step ($(date +%H:%M:%S))"
    case "$step" in
        prepare)
            run python -u "$REPO_DIR/scope_benchmark/prepare_scope.py" --dir "$SCOPE_DIR" \
                --mmseqs "$MMSEQS" --threads "$THREADS" ;;
        baselines)
            run python -u "$REPO_DIR/scope_benchmark/search_baselines.py" --dir "$SCOPE_DIR" \
                --foldseek "$FOLDSEEK" --mmseqs "$MMSEQS" --threads "$THREADS" --max-train-identity "$MAX_ID" ;;
        student)
            if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                echo "no GPU; submit with -p GPU-shared --gpus=v100-32:1 --cpus-per-task=5" >&2
                exit 1
            fi
            run python -u "$REPO_DIR/scope_benchmark/search_student.py" --dir "$SCOPE_DIR" \
                --checkpoint "$CHECKPOINT" --name "$NAME" --max-train-identity "$MAX_ID" ;;
        evaluate)
            shopt -s nullglob
            hits=("$SCOPE_DIR"/hits/*.m8 "$SCOPE_DIR"/hits/*.tsv.gz)
            shopt -u nullglob
            if [ ${#hits[@]} -eq 0 ]; then
                echo "no hit files in $SCOPE_DIR/hits" >&2
                exit 1
            fi
            run python -u "$REPO_DIR/scope_benchmark/evaluate_search.py" --dir "$SCOPE_DIR" "${hits[@]}" \
                --per-query "$SCOPE_DIR/per_query_all.tsv"
            run python -u "$REPO_DIR/scope_benchmark/evaluate_search.py" --dir "$SCOPE_DIR" "${hits[@]}" \
                --max-train-identity "$MAX_ID" --per-query "$SCOPE_DIR/per_query_heldout.tsv" ;;
        *)
            echo "unknown step '$step' (prepare, baselines, student or evaluate)" >&2
            exit 1 ;;
    esac
done

echo ""
echo "end   : $(date)"
