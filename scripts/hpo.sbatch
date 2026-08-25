#!/bin/bash
#SBATCH --job-name=hpo
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=job.%j.out

#   sbatch --gres=gpu:v100:1 --job-name=hpo-dermamnist scripts/hpo.sbatch dermamnist
#
# One target, one GPU, run to completion. Results land in results/hpo/<target>/.
# An interrupted run resumes from the optuna.db there -- submit the same command again.
#
# The study is written on node-local disk and copied back, for one reason: optuna.db is
# SQLite in WAL mode, and WAL needs a shared-memory segment that NFS and Lustre do not
# provide. A run works on /scratch and the results are copied out every few minutes.
#
# Not done here, and why: the conda environment is loaded from shared storage rather than
# staged local, because that costs a few minutes once per job against days of training;
# the pretrained weights use the default cache in $HOME, because a shared cache is
# downloaded once for all eleven targets where a per-job one is downloaded eleven times;
# and there is no restart-on-kill wrapper, because the measured peak is ~10 GiB against
# the 32G asked for above.

TARGET="$1"
[ -n "$TARGET" ] || { echo "usage: sbatch scripts/hpo.sbatch <target>" >&2; exit 2; }

echo "Running on $(hostname): $TARGET"
export TMPDIR=$HOME/tmp

# The DataLoader keeps several loaders alive with persistent workers, and each shared
# tensor handed back to the parent costs a descriptor. At the default soft limit of 1024
# the run dies part-way through a study on 'Too many open files'.
ulimit -n "$(ulimit -Hn)"

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
# the conda shell function is not defined in a batch job; conda activate needs it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/similarity" || exit 1
export PYTHONNOUSERSITE=1

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

SHARED="$REPO/results/hpo"
# /tmp, not $TMPDIR, as the fallback: TMPDIR is set to $HOME/tmp above, and falling back
# there would put the study back on the shared filesystem, which is what this avoids.
LOCAL="/scratch/$USER/hpo-$SLURM_JOB_ID"
mkdir -p "$LOCAL/$TARGET" 2>/dev/null || LOCAL="/tmp/hpo-$SLURM_JOB_ID"
mkdir -p "$LOCAL/$TARGET" "$SHARED/$TARGET" || exit 1
echo "working in $LOCAL"

# Resume: bring any previous study over before starting.
rsync -a "$SHARED/$TARGET/" "$LOCAL/$TARGET/" || exit 1

# The database is copied by SQLite rather than by rsync. In WAL mode its committed state
# is split between optuna.db and its log, so an rsync of the two can catch them at
# different moments and produce a file that does not open -- which would matter exactly
# when the shared copy is all that is left, after a node failure. The backup API holds a
# read transaction for the length of the copy, and journal_mode=DELETE on the result means
# the copy can be read from a login node while the run continues.
sync_out() {
    python - "$LOCAL/$TARGET/optuna.db" "$SHARED/$TARGET/optuna.db" <<'PY'
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
    rsync -a --exclude='optuna.db*' "$LOCAL/$TARGET/" "$SHARED/$TARGET/"
}

# Every five minutes, so a node failure costs minutes rather than days, and so the run can
# be watched from a login node while it is going.
( while true; do sleep 300; sync_out; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_out' EXIT

python src/hpo_finetune.py --target "$TARGET" \
    --out "$LOCAL" --workers "${SLURM_CPUS_PER_TASK:-8}"
RC=$?

kill $SYNC_PID 2>/dev/null
trap - EXIT
if sync_out; then
    rm -rf "$LOCAL"
else
    echo "final copy back failed; leaving $LOCAL on $(hostname)" >&2
fi
exit $RC
