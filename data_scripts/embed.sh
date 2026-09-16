#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=embed_%j.out
#
# ESM-2 per-residue embeddings for every protein in a parquet row window.
#
#   sbatch data_scripts/embed.sh                                  # rows [0, 4000000): the whole protein pool
#   START_ROW=181288 END_ROW=1000000 sbatch data_scripts/embed.sh
#   bash data_scripts/embed.sh                                    # inside an interactive GPU session
#
# The stage skips any protein that already has an embedding, so re-running a
# window is safe, and a job cut off by the wall clock is resumed by
# submitting it again. Structures must be downloaded first
# (download_pdbs_from_parquet.py): the sequence is read from the PDB file.
# The #SBATCH lines are comments to bash, so the same file works either way.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/jet/home/jxu23/OCEANDIR}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
START_ROW="${START_ROW:-0}"
END_ROW="${END_ROW:-4000000}"   # distinct proteins saturate well before this

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

echo "=================================================================="
echo "  embedding rows [$START_ROW, $END_ROW)"
echo "=================================================================="
echo "  repo        $REPO_DIR"
echo "  embeddings  $DATA_DIR/embeddings  ($(ls "$DATA_DIR/embeddings" 2>/dev/null | wc -l) files before)"
echo "  gpu         ${CUDA_VISIBLE_DEVICES:-none visible}"
echo "  started     $(date)"
echo ""

python "$REPO_DIR/data_scripts/precompute_esm.py" \
    --start-row "$START_ROW" --end-row "$END_ROW"

echo ""
echo "  embeddings  $(ls "$DATA_DIR/embeddings" | wc -l) files after"
echo "  finished    $(date)"
echo ""
echo "Next: python data_scripts/audit_upstream.py"
