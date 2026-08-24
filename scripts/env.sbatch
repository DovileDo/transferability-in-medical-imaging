#!/bin/bash
#SBATCH --job-name=hpo-env
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out

echo "Running on $(hostname):"
module load Anaconda3
conda create -y --name similarity python=3.10
source activate similarity

# cuda 11.8 because the V100s are sm_70, which newer torch builds no longer carry
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# numpy<2 because the code uses np.Inf, which 2.0 removed
pip install 'numpy<2' 'pandas<3' scipy scikit-learn 'optuna>=3.0' medmnist psutil

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
