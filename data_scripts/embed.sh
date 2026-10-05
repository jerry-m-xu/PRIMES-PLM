#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=embed_%j.out
#
# ESM-2 per-residue embeddings for every protein in a parquet row window,
# or for a list of proteins.
#
#   sbatch data_scripts/embed.sh                                  # rows [0, 4000000): the whole protein pool
#   sbatch --export=ALL,START_ROW=181288,END_ROW=1000000 data_scripts/embed.sh
#   sbatch --export=ALL,ESM_MODEL=650M,PART=0/4 data_scripts/embed.sh   # one of 4 parallel jobs
#   bash data_scripts/embed.sh                                    # inside an interactive GPU session
#
# ESM_MODEL (35M, 150M, 650M, 3B; default 35M) picks the checkpoint. Models
# other than 35M write to embeddings_<model>/ and never touch embeddings/,
# and embed every protein in fgw_scores.csv (PROTEINS_IN to choose another
# CSV, PROTEIN_LIST for a text file of ids) rather than a parquet window.
# PART=K/N takes every N-th protein from the K-th, so N jobs split the work;
# submit K=0..N-1. Their sequences come from tm_scores.csv, not the PDBs.
#
# The stage skips any protein that already has an embedding, so re-running a
# window is safe, and a job cut off by the wall clock is resumed by
# submitting it again. Structures must be downloaded first
# (download_pdbs_from_parquet.py): the sequence is read from the PDB file.
# The #SBATCH lines are comments to bash, so the same file works either way.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/jet/home/jxu23/OCEANDIR}"
# Under sbatch, BASH_SOURCE is a copy in SLURM's spool directory, so the repo
# root is taken from the submitted script's path (scontrol) or, failing that,
# from the directory sbatch was run in. Set REPO_DIR to override.
if [ -z "${REPO_DIR:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
    if [ -n "$_submitted" ] && [ -f "${SLURM_SUBMIT_DIR:-.}/$_submitted" ]; then
        _submitted="${SLURM_SUBMIT_DIR:-.}/$_submitted"
    fi
    if [ -f "$_submitted" ]; then
        REPO_DIR="$(cd "$(dirname "$_submitted")/.." && pwd)"
    else
        REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
    fi
fi
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO_DIR/data_scripts/common.py" ]; then
    echo "REPO_DIR=$REPO_DIR does not look like the repository (no data_scripts/common.py)." >&2
    echo "Submit from the repository root, or set REPO_DIR=/path/to/repo." >&2
    exit 1
fi
START_ROW="${START_ROW:-0}"
END_ROW="${END_ROW:-4000000}"   # distinct proteins saturate well before this
ESM_MODEL="${ESM_MODEL:-35M}"
PROTEIN_LIST="${PROTEIN_LIST:-}"
PROTEINS_IN="${PROTEINS_IN:-}"
PART="${PART:-}"
if [ "$ESM_MODEL" != "35M" ] && [ -z "$PROTEIN_LIST" ] && [ -z "$PROTEINS_IN" ]; then
    PROTEINS_IN="$DATA_DIR/fgw_scores.csv"
fi
if [ "$ESM_MODEL" = "35M" ]; then
    EMB_DIR="$DATA_DIR/embeddings"
else
    EMB_DIR="$DATA_DIR/embeddings_$ESM_MODEL"
fi

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

echo "=================================================================="
if [ -n "$PROTEIN_LIST" ] || [ -n "$PROTEINS_IN" ]; then
    if [ -n "$PROTEIN_LIST" ]; then
        ARGS=(--protein-list "$PROTEIN_LIST")
    else
        ARGS=(--proteins-in "$PROTEINS_IN")
    fi
    echo "  embedding the proteins of ${PROTEIN_LIST:-$PROTEINS_IN}${PART:+ (part $PART)} with ESM-2 $ESM_MODEL"
    ARGS+=(--model "$ESM_MODEL" --sequences-from "$DATA_DIR/tm_scores.csv")
    if [ -n "$PART" ]; then
        ARGS+=(--part "$PART")
    fi
else
    echo "  embedding rows [$START_ROW, $END_ROW) with ESM-2 $ESM_MODEL"
    ARGS=(--model "$ESM_MODEL" --start-row "$START_ROW" --end-row "$END_ROW")
fi
echo "=================================================================="
echo "  repo        $REPO_DIR"
echo "  embeddings  $EMB_DIR  ($(ls "$EMB_DIR" 2>/dev/null | wc -l) files before)"
echo "  gpu         ${CUDA_VISIBLE_DEVICES:-none visible}"
echo "  started     $(date)"
echo ""

python "$REPO_DIR/data_scripts/precompute_esm.py" "${ARGS[@]}"

echo ""
echo "  embeddings  $(ls "$EMB_DIR" | wc -l) files after"
echo "  finished    $(date)"
echo ""
echo "Next: python data_scripts/audit_upstream.py"
