#!/bin/bash
#SBATCH --job-name=hposrc
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=job.%j.out

#   sbatch --gres=gpu:v100:1 --job-name=src-dermamnist scripts/hpo_sources.sbatch dermamnist
#   sbatch --gres=gpu:v100:1 scripts/hpo_sources.sbatch dermamnist radimagenet medmnist
#
# Dataset-transfer benchmark for one target: every source in src/sources.py except the
# target itself and 'imagenet', or only the ones named after it, one after another.
# 'imagenet' is the model benchmark's resnet50 run, copied rather than retrained:
#
#   python src/hpo_sources_reuse.py
#
# Same design as scripts/hpo.sbatch -- 100-trial search on fold 1, top 10 configurations rerun with 16
# seeds on fold 1, winner on mean validation AUC -- with the architecture fixed at
# resnet50 and the pretrained weights varying. See src/hpo_sources.py for why.
#
# Results land in results/hpo_sources/<source>/<target>/, one study per source, so the
# thirteen trained sources of a target can be split across several jobs. Never run the same
# target and source in two jobs at once: each works on a node-local copy and the last
# to sync back wins. An interrupted job resumes -- submit the same command again.
#
# The pretrained checkpoints are read from models/ (src/sources.py); the job refuses to
# start if one it needs is missing.

TARGET="$1"
[ -n "$TARGET" ] || { echo "usage: sbatch scripts/hpo_sources.sbatch <target> [source ...]" >&2; exit 2; }
shift

echo "Running on $(hostname): $TARGET"
export TMPDIR=$HOME/tmp
ulimit -n "$(ulimit -Hn)"

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/similarity" || exit 1
export PYTHONNOUSERSITE=1

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

SOURCES=("$@")
if [ ${#SOURCES[@]} -eq 0 ]; then
    read -r -a SOURCES < <(python -c "import sys; sys.path.insert(0, 'src'); import sources; \
print(*sources.to_train('$TARGET'))") || exit 1
fi
echo "sources: ${SOURCES[*]}"

SHARED="$REPO/results/hpo_sources"
LOCAL="/scratch/$USER/hposrc-$SLURM_JOB_ID"
mkdir -p "$LOCAL" 2>/dev/null || LOCAL="/tmp/hposrc-$SLURM_JOB_ID"
echo "working in $LOCAL"

# Resume: bring any previous study of these sources over before starting.
for s in "${SOURCES[@]}"; do
    mkdir -p "$LOCAL/$s/$TARGET" "$SHARED/$s/$TARGET" || exit 1
    rsync -a "$SHARED/$s/$TARGET/" "$LOCAL/$s/$TARGET/" || exit 1
done

# optuna.db is copied with SQLite's backup API rather than rsync, for the reason given in
# scripts/hpo.sbatch: in WAL mode a file copy can catch the database and its log at
# different moments.
sync_out() {
    for s in "${SOURCES[@]}"; do
        python - "$LOCAL/$s/$TARGET/optuna.db" "$SHARED/$s/$TARGET/optuna.db" <<'PY'
import os, sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(src):
    tmp = dst + '.tmp'
    con = sqlite3.connect(src, timeout=60)
    out = sqlite3.connect(tmp)
    try:
        con.backup(out)
        out.execute('PRAGMA journal_mode=DELETE')
    finally:
        out.close(); con.close()
    for leftover in (tmp + '-wal', tmp + '-shm'):
        if os.path.exists(leftover):
            os.remove(leftover)
    os.replace(tmp, dst)
PY
        rsync -a --exclude='optuna.db*' "$LOCAL/$s/$TARGET/" "$SHARED/$s/$TARGET/" || return 1
    done
}

( while true; do sleep 300; sync_out; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_out' EXIT

python src/hpo_sources.py --target "$TARGET" --sources "${SOURCES[@]}" \
    --out "$LOCAL" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --trials 100 --final-folds 1 --final-seeds 16 --final-topk 10
RC=$?

# the validation loss the selection breaks exact AUC ties on, from the saved predictions
for s in "${SOURCES[@]}"; do
    [ -f "$LOCAL/$s/$TARGET/final_runs.csv" ] && python src/hpo_plain_loss.py --out "$LOCAL/$s" --target "$TARGET"
done

kill $SYNC_PID 2>/dev/null
trap - EXIT
if sync_out; then
    rm -rf "$LOCAL"
else
    echo "final copy back failed; leaving $LOCAL on $(hostname)" >&2
fi
exit $RC
