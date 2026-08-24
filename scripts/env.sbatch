#!/bin/bash
#SBATCH --job-name=hpo-env
#SBATCH --output=hpo-env.%j.out
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -e
set -x

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

ENV=$HOME/envs/similarity
export CONDA_PKGS_DIRS=$HOME/conda-pkgs
export TMPDIR=$HOME/tmp

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -p "$ENV" python=3.10

# the conda shell function is not defined in a batch job; conda activate needs it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# ~/.local holds packages for the module python, including a torch built for a CUDA 13
# driver. Nothing from there may leak into this environment.
export PYTHONNOUSERSITE=1

# cu118: the driver on these nodes is 12.6 and the V100s are sm_70. This build runs on
# both; the default wheels are cu13 and run on neither.
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

# only-if-needed so medmnist does not pull a different torch over the one above.
# numpy<2 because the code uses np.Inf, which 2.0 removed.
pip install --upgrade-strategy only-if-needed \
    'numpy<2' 'pandas<3' scipy scikit-learn 'optuna>=3.0' medmnist psutil

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
