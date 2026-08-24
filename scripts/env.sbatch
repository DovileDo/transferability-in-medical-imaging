#!/bin/bash
#SBATCH --job-name=hpo-env
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out

echo "Running on $(hostname):"
module load Anaconda3 || exit 1

# conda-forge only. The Anaconda default channels require a Terms of Service acceptance
# that a batch job cannot give, and --override-channels keeps them out of the solve.
conda create -y -n similarity python=3.10 -c conda-forge --override-channels || exit 1
source activate similarity || exit 1

# ~/.local holds packages for the module's python 3.13, including a torch built for a
# CUDA 13 driver. Nothing from there may leak into this environment.
export PYTHONNOUSERSITE=1

# cu118: the driver on these nodes is 12.6, and the V100s are sm_70. This build runs on
# both; the cu13 wheels pip installs by default run on neither.
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118 || exit 1

# only-if-needed so medmnist does not pull a different torch on top of the one above.
# numpy<2 because the code uses np.Inf, which 2.0 removed.
pip install --upgrade-strategy only-if-needed \
    'numpy<2' 'pandas<3' scipy scikit-learn 'optuna>=3.0' medmnist psutil || exit 1

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
