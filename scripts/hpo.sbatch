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
# Writes results/hpo/<target>/. Interrupted runs resume from the optuna.db in there:
# submit the same command again.

TARGET="$1"
[ -n "$TARGET" ] || { echo "usage: sbatch scripts/hpo.sbatch <target>" >&2; exit 2; }

echo "Running on $(hostname): $TARGET"
module load Anaconda3
source activate similarity

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
python src/hpo_finetune.py --target "$TARGET" --workers "${SLURM_CPUS_PER_TASK:-8}"
