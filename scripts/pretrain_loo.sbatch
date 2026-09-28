#!/bin/bash
#SBATCH --job-name=loo50
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=loo50.%j.out

#   sbatch --gres=gpu:a100_80gb:1 scripts/pretrain_loo.sbatch
#   sbatch --gres=gpu:a100_80gb:1 scripts/pretrain_loo.sbatch dermamnist octmnist
#
# Leave-target-out ResNet-50 pre-training, one held-out target after another -- all
# eleven benchmark targets unless some are named. Checkpoints go to models/loo_resnet50/.
# A target whose .pt already exists is skipped, and an interrupted one resumes from its
# last epoch, so resubmitting the same command picks up where the last job stopped.
#
# Why these resources. The pool is every MedMNIST training split at 224 except the
# target -- up to ~40 GB of uint8 held in memory, plus the validation splits and the
# batches in flight between worker processes, hence 160G. Augmentation runs on the CPU,
# and a ResNet-50 in bfloat16 on an A100 consumes a few thousand images a second, so the
# loaders need many cores; check img_per_s in the first epoch's line of the log, and if
# it is far below that, the job is starved of CPUs rather than of GPU.
#
# The pool includes chestmnist, which no benchmark target fetches, so the job downloads it
# on first use (MedMNIST verifies the file's MD5 against its own record).

TARGETS=("$@")
[ ${#TARGETS[@]} -gt 0 ] || TARGETS=(bloodmnist breastmnist dermamnist octmnist organamnist \
    organcmnist organsmnist pathmnist pneumoniamnist retinamnist tissuemnist)

echo "Running on $(hostname): ${TARGETS[*]}"
export TMPDIR=$HOME/tmp
ulimit -n "$(ulimit -Hn)"

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/similarity" || exit 1
export PYTHONNOUSERSITE=1

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
OUT=models/loo_resnet50
mkdir -p "$OUT"

MEDMNIST_ROOT="${MEDMNIST_ROOT:-$HOME/.medmnist}"
if [ ! -f "$MEDMNIST_ROOT/chestmnist_224.npz" ]; then
    echo "=== fetching chestmnist_224.npz"
    python -c "from medmnist import ChestMNIST; ChestMNIST(split='train', download=True, size=224, root='$MEDMNIST_ROOT')" \
        || { echo "chestmnist download failed"; exit 1; }
fi

RC=0
for t in "${TARGETS[@]}"; do
    if [ -f "$OUT/$t.pt" ]; then
        echo "=== $t: already finished, skipping"
        continue
    fi
    echo "=== $t: $(date '+%F %T')"
    python src/pretrain_loo.py --target "$t" --out "$OUT" --data "$MEDMNIST_ROOT" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" || { echo "=== $t FAILED"; RC=1; }
done
exit $RC
