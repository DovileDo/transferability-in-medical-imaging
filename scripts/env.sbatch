#!/bin/bash
# Build the conda environment the benchmark runs in, as a job rather than on the login
# node.
#
#     sbatch --gres=gpu:v100:1 scripts/env.sbatch
#
# Run this once, before the first hpo.sbatch. It is a job for two reasons: sites ask for
# it (installs are heavy and login nodes are shared), and the verification at the end has
# to see a GPU -- which is the whole point of doing it this way. An environment that
# imports torch successfully on a login node still tells you nothing about whether it can
# run on the card you were given.
#
# It ends by packing the environment into a single archive, which is what the training
# jobs actually run from: each one unpacks it onto its own node-local disk instead of
# importing a few hundred thousand small files across a shared filesystem. Measured here,
# packing takes about two minutes and produces 3.2 GB from a 5.9 GB environment, and
# unpacking it takes 25 seconds -- against ten to twenty minutes for a fresh conda
# install in every job, which would also re-resolve and could hand different targets
# different versions of torch.
#
# Ask for the same kind of card you will train on. `--gres=gpu` takes whatever is free,
# and on a queue holding several kinds that verifies the wrong one:
#
#     sbatch --gres=gpu:v100:1 scripts/env.sbatch      # and then HPO_GRES=gpu:v100:1
#
# ---------------------------------------------------------------------------------
#SBATCH --job-name=hpo-env
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#
# What can be changed from outside:
#
#     HPO_ENV             environment name                       (default similarity)
#     HPO_CONDA_MODULE    module providing conda                 (default Anaconda3)
#     HPO_TORCH_INSTALL   the whole torch install command, if the default is wrong
#     HPO_ENV_PACK        where the archive goes, and where the training jobs look for
#                         it  (default ~/hpo-env/<env>.tar.gz). Shared storage: every
#                         compute node has to be able to read it.
#
# The torch build is the one thing that is genuinely site-specific, because it has to
# match the driver on the compute nodes and the compute capability of the card. The
# default follows the site documentation. If it turns out to be wrong the verification
# below says so, in terms of what to install instead.

set -u -o pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
ENV_NAME="${HPO_ENV:-similarity}"

echo "Running on $(hostname):"

# ---- conda ------------------------------------------------------------------------
module load "${HPO_CONDA_MODULE:-Anaconda3}" || {
    echo "could not load the conda module; set HPO_CONDA_MODULE to whatever your site" >&2
    echo "calls it (module avail 2>&1 | grep -i -e conda -e python)" >&2
    exit 1
}

# `conda activate` needs a shell hook that `module load` does not always install. Set it
# up when it is available and fall back to the older `source activate`, which needs only
# conda's bin directory on PATH -- that is the form the site documentation uses.
if conda shell.bash hook >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    activate() { conda activate "$1"; }
else
    activate() { source activate "$1"; }
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[$(date +%T)] environment '$ENV_NAME' already exists; verifying it as-is."
    echo "                to rebuild from scratch: conda env remove -n $ENV_NAME"
else
    echo "[$(date +%T)] creating '$ENV_NAME'"
    # python is pinned because numpy is: the analysis code uses np.Inf, which numpy 2.0
    # removed, and the last numpy 1.x has no wheels past 3.12
    conda create -y -n "$ENV_NAME" python=3.10 || exit 1
fi

activate "$ENV_NAME" || { echo "could not activate $ENV_NAME" >&2; exit 1; }
echo "[$(date +%T)] python: $(python -V 2>&1), $(command -v python)"

# ---- torch ------------------------------------------------------------------------
# Volta (V100, sm_70) is the reason this is not just `pip install torch`. Recent torch
# wheels are built for sm_80 and up and will not run on a V100 at all -- the failure is
# "no kernel image is available for execution on the device", at the first trial, hours
# into a job. The CUDA 11.8 build still carries sm_70.
if python -c 'import torch' 2>/dev/null; then
    echo "[$(date +%T)] torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "[$(date +%T)] installing torch"
    eval "${HPO_TORCH_INSTALL:-conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia}" || {
        echo "torch install failed. Set HPO_TORCH_INSTALL to the command your site wants," >&2
        echo "e.g. the pip index for a given CUDA:" >&2
        echo "  export HPO_TORCH_INSTALL='pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118'" >&2
        exit 1
    }
fi

# ---- everything else --------------------------------------------------------------
# Only what the benchmark itself imports. conda.yaml is the full environment for the
# analysis and the notebooks as well, and it carries packages (mlflow, radt, migedit,
# opencv, jupyterlab) that a compute node does not need and that can fail to resolve and
# take the whole install down with them. Build the full one separately if you want to run
# the notebooks here too.
#
# --upgrade-strategy only-if-needed keeps pip from replacing the torch just installed
# while resolving something that merely depends on it -- medmnist is the one that tries.
echo "[$(date +%T)] installing the rest"
pip install --upgrade-strategy only-if-needed \
    'numpy<2' 'pandas<3' scipy scikit-learn 'optuna>=3.0' medmnist psutil || exit 1

# ---- verify -----------------------------------------------------------------------
echo "[$(date +%T)] verifying"
python - <<'PY' || exit 1
import sys
import importlib.metadata as md

sys.path.insert(0, 'src')

missing = []
for p in ('torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn',
          'optuna', 'medmnist', 'psutil'):
    try:
        print(f'  {p:16} {md.version(p)}')
    except Exception:
        print(f'  {p:16} MISSING')
        missing.append(p)
if missing:
    sys.exit(1)

import numpy as np
if not hasattr(np, 'Inf'):
    sys.exit('numpy has no np.Inf: the analysis scripts need numpy < 2')

# the imports the benchmark makes at start-up, in the order it makes them
import splits, hpo_select, hpo_finetune          # noqa: F401
print('  the benchmark imports cleanly')
PY

# and the check that matters: not whether torch imports, but whether it can run a kernel
# on the card this job was given
python "$REPO/scripts/gpu_check.py" --run || exit 1

# ---- pack ---------------------------------------------------------------------------
# The training jobs run the environment off node-local disk rather than out of shared
# storage: a conda environment is a few hundred thousand small files, importing torch
# stats most of them, and on a shared filesystem that cost is paid by every job every
# time -- while every other job is paying it too.
#
# So the environment is packed once, here, and each job unpacks the archive onto its own
# scratch. That is a minute or two against ten to twenty for a fresh conda install per
# job, and -- the part that matters more -- it is the same bytes every time. A per-job
# `conda create` re-resolves, so two targets started a month apart could get different
# versions of torch, which is precisely the kind of difference this benchmark measures.
#
# The archive goes to shared storage, since every compute node has to read it.
PACK="${HPO_ENV_PACK:-$HOME/hpo-env/${ENV_NAME}.tar.gz}"
mkdir -p "$(dirname "$PACK")" || exit 1

command -v conda-pack >/dev/null 2>&1 || {
    echo "[$(date +%T)] installing conda-pack"
    # conda-pack ships a console script and no __main__, so it is invoked by name and
    # has to be on PATH -- installing it into this environment puts it there
    pip install conda-pack || exit 1
}

echo "[$(date +%T)] packing -> $PACK (a couple of minutes)"
# --ignore-missing-files: pip installs over a conda environment leave the conda metadata
# describing files pip has since replaced, which conda-pack reports as missing.
# --quiet: the progress bar is a carriage-returned redraw per 100 ms, which is unreadable
# in a log file and tens of kilobytes of it.
rm -f "$PACK"
conda-pack --name "$ENV_NAME" --output "$PACK" --force --ignore-missing-files --quiet || {
    echo "packing failed. The jobs will still run, from shared storage and more slowly." >&2
    echo "Delete $PACK to take that path deliberately." >&2
    exit 1
}

echo
echo "[$(date +%T)] '$ENV_NAME' is ready: $(du -h "$PACK" | cut -f1) at $PACK"
echo
echo "Next, once the bundles and MEDMNIST_ROOT are in place:"
echo "  export HPO_GRES=${SLURM_JOB_GRES:-gpu:1}"
echo "  sbatch --gres=\$HPO_GRES --job-name=hpo-breastmnist $REPO/scripts/hpo.sbatch breastmnist"
echo
echo "The jobs find the archive at \$HPO_ENV_PACK, defaulting to the path above. Rebuild"
echo "it by re-running this script after removing the environment:"
echo "  conda env remove -n $ENV_NAME && sbatch --gres=... scripts/env.sbatch"
