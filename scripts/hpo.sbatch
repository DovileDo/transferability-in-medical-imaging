#!/bin/bash
# Run one MedMNIST target's architecture benchmark, start to finish, on one GPU.
#
#   sbatch --gres=gpu:v100:1 --job-name=hpo-dermamnist scripts/hpo.sbatch dermamnist
#
# One job per target, and it runs until the target is done. This assumes the wall-clock
# limit is extended once the job is running; --time below is only what is asked for at
# submission. Set HPO_MAX_HOURS if the limit is real and enforced, and the run will stop
# ten minutes short of it with its results copied back.
#
# Nothing resubmits itself. If a job does end early -- a node failure, a cancellation, a
# ceiling that was enforced after all -- the study in results/hpo/<target>/optuna.db is
# resumable, and resuming is the same sbatch command again: it picks up where the last
# one stopped and loses at most the trial that was in flight.
#
# The figures below are the only measured ones there are, and they are stale in three
# ways at once. They were taken on a GB10, under the pilot's budget of 100 trials + 20
# final runs per architecture, over a search space whose batch sizes were {32..256}.
# The current defaults spend 100 trials + 160 final runs (--final-topk 10 on four folds
# with four seeds), which is 2.2x the runs, and batch 16 costs more per image than the
# 256 it replaced. Treat them as a lower bound and as an ordering across targets -- the
# ordering is driven by how much data each one has, which has not changed.
#
#     organamnist     324 h      pathmnist      265 h      octmnist         130 h
#     organsmnist     319 h      tissuemnist    260 h      pneumoniamnist    77 h
#     organcmnist     319 h      bloodmnist     237 h      breastmnist       77 h
#     dermamnist      210 h      retinamnist    156 h
#
# For real numbers, read results/hpo/<target>/arch_cost.json once the first job has
# started: probe_arch_cost times a training step per image on the node the job actually
# landed on, per architecture, and the projection follows from that.
#
# ---------------------------------------------------------------------------------
#SBATCH --job-name=hpo
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#
# Check the partition before submitting eleven of these -- a request the queue cannot
# satisfy sits pending rather than failing, and there is nothing in the log to read:
#
#     sinfo -p acltr -o '%P %l %c %m %G'    limits, cores, memory, and how GPUs are named
#     scontrol show partition acltr         MaxTime, and whether preemption is on
#
# --mem is 32G because asking for more costs cards, not because a run needs it. The
# datasets held in RAM top out at 2.2 GB (tissuemnist's 47,280 test images) and a run
# peaks near 10 GiB with four workers, so this is 3x headroom. What it buys is nodes
# whose RAM is the binding constraint rather than their GPUs: at 64G only one of cn13's
# four A100s is usable and only three of cn7's five cards, which across the cluster is
# eight GPUs that cannot be had for no gain. scripts/cluster.md has the arithmetic.
#
# On a queue that holds more than one kind of card, name the kind rather than taking what
# is free. At 224px in fp32 the largest batch in the space needs about 18 GB for the
# heaviest backbones -- measured, vgg16 17.9 GiB and densenet121 17.3 at batch 128,
# against 3.8 for shufflenet -- so call it 20 GiB for the full space. A card under that
# does not stop the run: those trials are caught and marked failed, and the search goes
# on with large batches available to the light architectures and not the heavy ones. The
# search space then differs by architecture, which is a per-architecture perturbation of
# exactly what this benchmark measures. scripts/gpu_check.py reports it before the run
# starts and names the --batch-sizes that do fit, which is the honest alternative:
# restrict the space deliberately and equally rather than by whichever card was free.
#
#     sbatch --gres=gpu:a30:1 --job-name=hpo-bloodmnist scripts/hpo.sbatch bloodmnist
#
# --gres on the command line is enough: with nothing resubmitting itself there is no
# second job to carry it into, and a Slurm requeue re-runs this one with the resources it
# was originally given.
#
# %x-%j in the output file is the job name and id, so the eleven targets land in separate
# files rather than one interleaved one.

set -u -o pipefail

TARGET="${1:-${HPO_TARGET:-}}"
if [ -z "$TARGET" ]; then
    echo "usage: sbatch scripts/hpo.sbatch <target>" >&2
    exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

# `set -u` is on and neither of these is guaranteed to be exported -- a site that does
# not set USER would abort the job on the scratch lookup rather than fall through it.
JOBID="${SLURM_JOB_ID:-$$}"
USERNAME="${USER:-$(id -un)}"

# ---- node-local scratch ----------------------------------------------------------
# Everything this job writes, and the environment it runs in, live here. That ordering is
# deliberate: the environment has to be staged before it can be activated, so scratch has
# to be resolved first.
#
# SLURM_TMPDIR is not a Slurm variable. Some sites set it, many do not, and the ones that
# do not use their own name, so the candidates are tried in turn and the one chosen is
# printed. Getting this wrong is quiet rather than loud: everything still runs, on the
# shared filesystem, where SQLite's locking is unreliable and the study can end up corrupt
# instead of merely slow. Set HPO_LOCAL_ROOT if your site's per-job scratch is elsewhere.
pick_scratch() {
    local c
    for c in "${HPO_LOCAL_ROOT:-}" "${SLURM_TMPDIR:-}" \
             "/scratch/$USERNAME" "/scratch/$JOBID" /scratch \
             "/local/$JOBID" "${TMPDIR:-}" /tmp; do
        [ -n "$c" ] || continue
        mkdir -p "$c" 2>/dev/null || continue
        [ -w "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}
ROOT=$(pick_scratch) || { echo "no writable scratch directory" >&2; exit 1; }
SCRATCH="$ROOT/hpo-$JOBID"
mkdir -p "$SCRATCH" || exit 1
FREE_GB=$(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[$(date +%T)] node-local scratch: $SCRATCH (${FREE_GB:-?} GB free)"
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 20 ]; then
    echo "[warn] under 20 GB free here. Measured: the unpacked environment is 5.9 GB and" >&2
    echo "       the pretrained weights 0.9 GB, before the study and the saved test" >&2
    echo "       predictions, which approach 1 GB on the larger targets" >&2
fi

# ---- environment -----------------------------------------------------------------
# Run from scratch, not from shared storage. A conda environment is a few hundred thousand
# small files and importing torch stats most of them; on a shared filesystem that is slow
# every time, for every job, and it competes with every other job doing the same. The
# environment is therefore unpacked onto the local disk here and thrown away with the rest
# of the scratch directory at the end.
#
# scripts/env.sbatch builds it once and writes the archive. Unpacking is a minute or two
# against ten to twenty for a fresh conda install, and it is the same bytes every job --
# a per-job `conda create` would silently re-resolve and could hand different targets
# different versions of torch.
ENV_PACK="${HPO_ENV_PACK:-$HOME/hpo-env/${HPO_ENV:-similarity}.tar.gz}"
if [ -f "$ENV_PACK" ]; then
    ENV_DIR="$SCRATCH/env"
    mkdir -p "$ENV_DIR" || exit 1
    echo "[$(date +%T)] unpacking $ENV_PACK -> $ENV_DIR"
    tar -xzf "$ENV_PACK" -C "$ENV_DIR" || { echo "could not unpack $ENV_PACK" >&2; exit 1; }
    source "$ENV_DIR/bin/activate" || exit 1
    # conda-pack bakes the build-time prefix into scripts and .pth files; this rewrites
    # them to where the archive actually landed. Only conda-pack archives have it.
    [ -x "$ENV_DIR/bin/conda-unpack" ] && "$ENV_DIR/bin/conda-unpack"
    echo "[$(date +%T)] python: $(command -v python)"
else
    # No archive: fall back to a named conda environment on shared storage. It works, and
    # it is what a first run before env.sbatch will hit, but it is the slow path.
    echo "[warn] $ENV_PACK not found; falling back to the shared conda environment." >&2
    echo "       Build the archive with scripts/env.sbatch to run off local disk." >&2
    if [ -n "${HPO_CONDA_MODULE:-}" ]; then
        module load "$HPO_CONDA_MODULE" || { echo "module load $HPO_CONDA_MODULE failed" >&2; exit 1; }
    elif [ -f "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    fi
    if conda shell.bash hook >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${HPO_ENV:-similarity}" || exit 1
    else
        source activate "${HPO_ENV:-similarity}" || exit 1
    fi
fi

# A torch that cannot run on this card fails at the first trial, hours in, after the
# pretrained weights have been fetched and the data staged. It costs a second to find out
# now. Volta (V100, sm_70) is the case that bites: recent wheels start at sm_80.
python "$REPO/scripts/gpu_check.py" || exit 1

# ---- data ------------------------------------------------------------------------
# Two directories of data have to be on the machine before this runs, and only one of
# them comes out of git. The bundles under data/splits/bundles/ hold the training folds
# and the validation split, 2.6 GB for all eleven, and are gitignored -- copy them in or
# rebuild them with src/make_splits.py. The test split is not in them: it is read from
# <target>_224.npz under MEDMNIST_ROOT, which is 25 GB for all eleven and 12 GB for
# pathmnist alone, so on a site with a small home quota point this somewhere else.
#
# Neither is staged to scratch. Both are read once, at start-up, and numpy pulls only the
# members it is asked for out of the npz -- so copying them local would mean reading the
# same bytes and writing them again. It is the environment that is read over and over,
# and that is the one that moves.
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-$HOME/.medmnist}"
export BUNDLE_DIR="${BUNDLE_DIR:-$REPO/data/splits/bundles}"
for d in "$MEDMNIST_ROOT/${TARGET}_224.npz" "$BUNDLE_DIR/${TARGET}_splits_224.npz"; do
    [ -f "$d" ] || { echo "missing: $d" >&2; exit 1; }
done

# Pretrained weights are fetched from torchvision. With a shared home every job would
# download the same nine files at once, so the cache goes on the scratch disk and is
# warmed once here instead.
export TORCH_HOME="$SCRATCH/torch"
mkdir -p "$TORCH_HOME"
python - <<'PY' || exit 1
import sys, torchvision
sys.path.insert(0, 'src')
import hpo_finetune as H
for name in H.ARCHS.values():
    # progress=False: the download bar is a carriage-returned redraw, which in a log file
    # is several screens of noise per checkpoint
    getattr(torchvision.models, name)(weights='IMAGENET1K_V1', progress=False)
print(f'warmed {len(H.ARCHS)} pretrained checkpoints')
PY

# ---- staging ---------------------------------------------------------------------
# The study is a SQLite database. SQLite's WAL mode needs a shared-memory segment that
# network filesystems do not provide, and its locking on NFS/Lustre is unreliable, so
# the run works on node-local disk and the results are copied back. The copy-back also
# runs periodically, because it is what makes the run visible to hpo_watch.py from a
# login node while it is still going.
SHARED="$REPO/results/hpo"
LOCAL="$SCRATCH/run"
mkdir -p "$LOCAL/$TARGET" "$SHARED/$TARGET"

echo "[$(date +%T)] staging in $SHARED/$TARGET -> $LOCAL/$TARGET"
rsync -a "$SHARED/$TARGET/" "$LOCAL/$TARGET/" || exit 1

# The study database is copied by SQLite rather than by rsync. It is being written to
# while this runs, and in WAL mode its committed state is split between optuna.db and its
# log, so an rsync of the two can land torn -- which would matter exactly when the shared
# copy is all that is left, after a node failure. scripts/db_snapshot.py takes a
# consistent snapshot and renames it into place; everything else rsyncs as usual.
sync_out() {
    local ok=0
    python "$REPO/scripts/db_snapshot.py" \
        "$LOCAL/$TARGET/optuna.db" "$SHARED/$TARGET/optuna.db" || ok=1
    rsync -a --exclude='*.tmp' --exclude='optuna.db*' \
        "$LOCAL/$TARGET/" "$SHARED/$TARGET/" || ok=1
    return $ok
}

# periodic copy-back so progress is visible from outside the node
( while true; do sleep 300; sync_out; done ) &
SYNC_PID=$!
cleanup() { kill "$SYNC_PID" 2>/dev/null; sync_out; }
trap cleanup EXIT

# ---- run -------------------------------------------------------------------------
# The run gets as long as it needs. --signal above still delivers TERM 15 minutes before
# whatever limit is in force, and the guard treats a signal as "stop, do not restart" and
# exits 2, so a job that does get stopped stops cleanly and copies back rather than being
# cut off mid-trial.
#
# Set HPO_MAX_HOURS to put a ceiling back on -- on a partition that enforces a hard limit
# and does not honour --signal, that is the only way to leave time for the final copy
# back. Unset means no ceiling.
#
# --mem below is not enforced here. The guard caps memory with a systemd scope, which
# needs a user session bus that compute nodes do not have, so it detects that, skips the
# scope and says so in the log. #SBATCH --mem above is the limit that actually binds;
# the guard's number only sets the threshold at which it reports "this run hit its own
# limit", so keep it a little under the Slurm figure. What the guard still does here is
# restart the run after a kill, and stop after --max-unproductive attempts that get
# nowhere rather than holding the node through a crash loop.
LIMIT=()
if [ -n "${HPO_MAX_HOURS:-}" ]; then
    # ten minutes short of the ceiling, for the final copy back
    RUN_SECONDS=$(( HPO_MAX_HOURS * 3600 - 600 ))
    [ "$RUN_SECONDS" -gt 0 ] || { echo "HPO_MAX_HOURS must exceed 10 minutes" >&2; exit 2; }
    LIMIT=(timeout --signal=TERM "${RUN_SECONDS}s")
    echo "[$(date +%T)] stopping after ${HPO_MAX_HOURS}h less ten minutes"
fi

"${LIMIT[@]+"${LIMIT[@]}"}" \
    python scripts/run_hpo_guarded.py \
        --target "$TARGET" \
        --out "$LOCAL" \
        --log "$REPO/hpo_${TARGET}.log" \
        --mem "${HPO_MEM:-28G}" \
        --max-attempts "${HPO_ATTEMPTS:-8}" \
        -- --workers "${SLURM_CPUS_PER_TASK:-8}" --pruner median
RC=$?

kill "$SYNC_PID" 2>/dev/null
trap - EXIT
if sync_out; then
    # the whole scratch directory -- results, the unpacked environment and the weight
    # cache -- and only once the shared copy is known good. A site that does not clear
    # /scratch itself, or a fallback to /tmp, otherwise accumulates several GB per job.
    rm -rf "$SCRATCH"
else
    echo "[warn] the final copy back failed; leaving $SCRATCH on $(hostname)" >&2
fi

# ---- finish ------------------------------------------------------------------------
# progress.json reaches stage 'done' only after every architecture has finished its
# search and its final runs, so it is the one marker that means the target is complete.
STAGE=$(python - "$SHARED/$TARGET/progress.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('stage', 'unknown'))
except Exception:
    print('unknown')
PY
)

echo "[$(date +%T)] guard exited $RC, progress stage '$STAGE'"

if [ "$STAGE" = "done" ]; then
    echo "$TARGET finished. Collect with: python src/hpo_collect.py --out results/hpo"
    exit 0
fi

# Not done, and nothing resubmits it. The study is in results/hpo/<target>/optuna.db and
# resuming is the same command again -- it picks up where this left off and loses at most
# the trial that was in flight.
echo "$TARGET is not finished (stage '$STAGE'). Resume with the same command:" >&2
echo "  sbatch --gres=<same card> --job-name=hpo-$TARGET scripts/hpo.sbatch $TARGET" >&2
exit "$RC"
