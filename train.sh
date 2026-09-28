#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=train_%j.out
#
# Train the teacher or the student on one GPU.
#
#   MODEL=teacher sbatch train.sh
#   MODEL=student sbatch train.sh          # needs teacher_checkpoints/teacher_best.pt
#   MODEL=teacher bash train.sh            # inside an interactive GPU session
#
# --signal sends SIGTERM five minutes before the wall clock. srun forwards it
# to the trainer, which checkpoints at the next buffer boundary and exits;
# submit the same command again to continue from that checkpoint. Both
# trainers resume from their rolling checkpoint by default (RESUME = True).
# The #SBATCH lines are comments to bash, so the same file works either way.

set -euo pipefail

MODEL="${MODEL:-teacher}"
# Under sbatch, BASH_SOURCE is a copy in SLURM's spool directory, so the repo
# root is taken from the submitted script's path (scontrol) or, failing that,
# from the directory sbatch was run in. Set REPO_DIR to override.
if [ -z "${REPO_DIR:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
    if [ -n "$_submitted" ] && [ -f "${SLURM_SUBMIT_DIR:-.}/$_submitted" ]; then
        _submitted="${SLURM_SUBMIT_DIR:-.}/$_submitted"
    fi
    if [ -f "$_submitted" ]; then
        REPO_DIR="$(cd "$(dirname "$_submitted")" && pwd)"
    else
        REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
    fi
fi
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$REPO_DIR/data_scripts/common.py" ]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repository (no data_scripts/common.py)." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/repo." >&2
    exit 1
fi

case "$MODEL" in
    teacher) SCRIPT="$REPO_DIR/teacher_model/train_teacher.py" ;;
    student) SCRIPT="$REPO_DIR/student_model/train_student.py" ;;
    *) echo "unknown MODEL '$MODEL' (teacher or student)" >&2; exit 1 ;;
esac

# A batch job does not inherit an interactively-activated venv. Source the
# project setup from the repo or the home directory, or point VENV at an
# activate script yourself.
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
if ! command -v python >/dev/null 2>&1; then
    echo "python not found after environment setup." >&2
    echo "Set VENV=/path/to/venv/bin/activate, or put psc_interactive_setup.sh in the repo or your home directory." >&2
    exit 1
fi
echo "python: $(command -v python)"
echo "model : $MODEL ($SCRIPT)"
echo "gpu   : ${CUDA_VISIBLE_DEVICES:-none visible}"
echo "cpus  : ${SLURM_CPUS_PER_TASK:-unset} (loader workers use all but one)"
echo "start : $(date)"
echo ""

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # SLURM 22.05+ no longer passes --cpus-per-task from sbatch to srun; without
    # this the step sees one CPU and the trainer starts no loader workers
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" python -u "$SCRIPT"
else
    python -u "$SCRIPT"
fi

echo ""
echo "end   : $(date)"
