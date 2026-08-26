# Dataset transferability in medical image classification

This repository contains the code and results from the paper [*On dataset transferability in medical image classification*](https://arxiv.org/pdf/2412.20172). The study introduces a novel transferability metric that integrates feature quality and gradients to assess the suitability and adaptability of source model features for various target tasks. 

This PyTorch implementation enables the calculation of transferability scores and provides pre-training and fine-tuning scripts for medical image classification tasks, using datasets from [MedMNIST](https://medmnist.com/).

## Features

- Implements a novel transferability metric combining feature quality and gradients.
- Supports comparison with existing metrics such as LEEP, NLEEP, and others.
- Provides tools for leave-target-out pretraining on MedMNIST, ResNet18 pretraining on [RadImageNet](https://github.com/BMEII-AI/RadImageNet), and fine-tuning models on target tasks from MedMNIST. 
- Designed specifically for medical imaging datasets.

For access to the models and data used in the paper, please refer to:
* [MedMNIST datasets and pre-trained models](https://medmnist.com/) (we used ResNet18 224x224),
* [RadImageNet](https://github.com/BMEII-AI/RadImageNet),
* [Leave-target-out MedMNIST pretraining](https://osf.io/4zgrd/), and
* [Subsets of MedMNIST datasets used in our paper](https://osf.io/4zgrd/).

## Setup

### Requirements
Dependencies are provided in the `conda.yaml` file. To set up the environment:

```bash
conda env create -f conda.yaml
conda activate your-environment-name
```

## Usage

### Fine-Tuning

To fine-tune a source model for a target task, use the following command:

```bash
python -m fine_tuning.py --source_flag <MedMNIST source flag> --target_flag <MedMNIST target flag> --lr <learning rate> --epochs <number of epochs> --batch_size <batch size>
```
Replace `<MedMNIST source flag>` and `<MedMNIST target flag>` with the appropriate dataset flags from MedMNIST. Adjust learning rate, epochs, and batch size as needed.

### Fine-Tuning Benchmark (Optuna + AdamW)

`src/hpo_finetune.py` rebuilds the architecture-transferability ground truth with a
hyperparameter search instead of a fixed grid: per architecture it runs an Optuna TPE
study on one training fold, retrains the best few configurations on the folds it did not
search on, and reports the one that holds up. The test set is never used for selection.

The search covers the optimizer itself, on per-optimizer learning-rate ranges:

| Hyperparameter | Range / values | `--space` |
|---|---|---|
| `optimizer` | {SGD, AdamW} | core |
| `lr` (SGD) | 1e-3 – 1.0, log | core |
| `lr` (AdamW) | 1e-5 – 1e-2, log | core |
| `momentum` (SGD) | {0, 0.9, 0.99}, Nesterov when > 0 | core |
| `weight_decay` | 1e-8 – 1e-1, log | core |
| `batch_size` | {16, 32, 64, 128} | core |
| `head_lr_mult` | {1, 10} | core |
| `aug` | {none, light, heavy} | core |
| `train_epochs` | {25, 50, 100, 200, 400} | core |
| `dropout` before the head | {0, 0.5} | wide |
| `label_smoothing` | {0, 0.1} | wide |
| `warmup_epochs` | {0, 1, 3} | wide |
| `scheduler` | cosine; `--schedulers cosine constant` to search it | wide |

An arm narrowed to a single value is treated as a constant rather than a dimension: it is
not suggested to Optuna, does not appear in the study as a searched axis, and is recovered
from the configuration when a trial is replayed. With a fixed trial budget, every
dimension removed makes the search over the remaining ones twice as dense.

Three protocol choices are worth knowing about:

- **Training length is searched, not capped.** A fixed cap combined with cosine annealing
  and early stopping is a confound: a trial that stops at epoch 40 of a 400-epoch cosine
  never reaches the low-LR phase. Each trial instead draws its own `train_epochs` and its
  schedule completes over exactly that budget. `budget_bound` is recorded per trial, so
  you can check whether the longest arm was binding.
- **Everything is counted in optimizer steps, not epochs.** An epoch is 43 steps at batch
  size 16 and 5 at batch size 128, so epoch-based early stopping and pruning would mean
  very different things across the search space. Each trial is validated a fixed number of
  times over its own budget (`--validations-per-trial`).
- **Training and validation come from the official split each belongs to.** Training is
  `n_classes*100` images drawn uniformly from the official training split, with nothing
  removed from it afterwards, so it carries the target's class imbalance exactly as drawn.
  Validation is `n_classes*25` images from the official validation split, drawn
  *conditional on the training draw*: its places are handed out in proportion to that
  draw's class mix, so a class that is rare in training is rare in validation too. That
  is what keeps validation measuring what the test split measures. A class the proportion
  leaves with fewer than 3 images takes half of its training count instead — self-limiting,
  so no class is ever handed more validation images than training ones. Both parts are
  redrawn per fold, so validation sampling error averages down over the folds instead of
  sitting on all of them as the same offset.

The ranges are wide on purpose. Selection has to resolve differences of well under a
percentage point of AUC between architectures, so the space has to contain the optimum
for each of them rather than stop near it: in the earlier fixed grid, 33 of the 88
architecture×target cells settled on the largest learning rate available to them and 57
on the larger momentum. `python src/grid_evidence.py` recomputes that, and also how much
the choice among near-equivalent configurations moves the result — ranking architectures
by their best validation run rather than by the mean test AUC of their five best gives a
Kendall τ of 0.714 on median (0.286 at worst), and those five span 1.24pp of test AUC
against a median gap of 0.16pp between consecutively ranked architectures.

#### The frozen splits

The benchmark trains and selects on fixed subsets that are built once, ahead of time, by
`src/make_splits.py` and then read from `data/splits/bundles/` — one archive per target
holding the images themselves. `src/hpo_finetune.py` never draws anything, never touches
MedMNIST for its training or validation data, and refuses to start if the bundle it needs
is not there.

Each of the 5 folds draws **both parts from the official split each belongs to**:
`n_classes*100` training images taken uniformly at random from the official training
split, and `n_classes*25` validation images from the official validation split.

| | drawn |
|---|---|
| `train` | `n_classes*100`, uniform from the official train split, unstratified — nothing is removed from it for validation |
| `val` | `n_classes*25` from the official val split, allocated in proportion to the train draw's class mix, with any class left under 3 images taking half its training count; a different fold is a different validation set |

The training part is deliberately *not* stratified: the natural class imbalance is part of
the task being studied, and because validation comes from the other pool the training set
keeps that imbalance untouched. The validation allocation follows the training draw so that it
estimates the same quantity the test split does — a macro one-vs-rest AUC is invariant to
each positive class's prevalence but not to the mixture of negatives that class is ranked
against. It is drawn conditional on the training draw for exactly that reason: what a
class is owed in validation is set by how much of it was drawn to train on, not by how
much of it the official validation split happens to hold.

There is no flat floor. A class the proportion leaves with fewer than 3 validation images
instead takes **half of its training count**, and the largest classes pay for it. Below 3
positives a one-vs-rest AUC is not an estimate of anything, and that class still enters
the macro average with the same weight as every other; a standard error near 0.21 at 1
positive and 0.15 at 2 falls to 0.12 at 3. Taking half of the training count rather than a
fixed number is what makes the rule self-limiting — a class holding 7 training images gets
3, one holding 5 gets 2, and no class is ever handed more validation images than it has
training ones, which is exactly what a flat floor of 10 did to a class holding 7.

The rule fires on `dermamnist` and nowhere else: every other target's smallest validation
class already sits at 5 or more. On `dermamnist` it costs about 1.5pp of agreement between
the training and validation class mixes and buys 7–17% off the fold's macro-AUC standard
error. Where the official validation split is smaller than the budget it is used whole —
`retinamnist` has 120 validation images against a budget of 125. A class the training draw
misses entirely gets no validation images, and a class whose AUC is undefined is dropped
from the macro average by `auc_per_class` rather than propagated.

Folds are drawn from independently spawned PCG64 streams keyed on `(seed 24, fold)`, so a
fold's indices depend only on its own number: rebuilding one, or adding a sixth, cannot
perturb the others.

**The transferability scores in `results/` were computed on `train_imgs_fold1` of the
published OSF subsets.** Those subsets came out of `src/data_split.py`, which drew
`n_classes*100` training images per fold from a legacy global `RandomState` seeded with
24. Only the `dermamnist` folds can be reproduced from it byte for byte; the other targets'
files were written from a different state of that RNG, and no ordering of the draws in the
script recovers them. So for ten of the eleven targets, those scores and a ground truth
rebuilt here do not share their training images.

##### Building them (once, before fine-tuning)

```bash
python src/make_splits.py --all         # 11 bundles, ~2.3 GB, skips what already exists
python src/make_splits.py --all --verify    # which bundles are present
```

This is the only step that *draws* anything, but not the only one that needs the official
MedMNIST files: the bundles hold the training folds and the validation split, and the test
split is read from `<target>_224.npz` at run time. So a machine that fine-tunes needs both
the bundles (2.6 GB for eleven targets) and `$MEDMNIST_ROOT` (25 GB for eleven, 12 GB of
it pathmnist). `src/hpo_finetune.py` checks for both before it spends anything.

Per target this step writes:

| file | | |
|---|---|---|
| `data/splits/<target>.json` | ~50 kB | indices, per-class counts, sha256 of the images and labels, and the MD5 of the MedMNIST release they were cut from — **tracked in git** |
| `data/splits/bundles/<target>_splits_224.npz` | 0.04–0.6 GB | the pixels: `val` plus all five training folds, with the manifest embedded — gitignored, archive it |

plus a `SHA256SUMS` over the bundles. Images are read out of `<target>_224.npz` as a
stream, so cutting 900 images out of pathmnist never materialises its 13.5 GB training
array.

An index only reproduces a sample for as long as the file it indexes into stays available
and unchanged — which is why the pixels, not just the manifest, are the artifact to
archive. Upload the bundles alongside the paper and a MedMNIST re-release cannot change
what the published numbers were produced from.

##### Using them

`src/hpo_finetune.py` reads the bundle for its target (`--bundle-dir`, or `$BUNDLE_DIR`;
default `data/splits/bundles/`). It checks up front that the archive holds every fold the
run will need, hashes each part as it loads it against the manifest inside the bundle
*and* against the git-tracked `data/splits/<target>.json`, prints both checksums, and
records them in `meta.json`:

```
  train fold 1 (sha256 9eea4236ead6): 700 images, per-class [26, 34, 85, 7, 76, 463, 9]
  val fold 1 (n_classes*25 budget, sha256 62125738f67f): 175 images, per-class [7, 8, 21, 3, 19, 113, 4]
```

It refuses to resume a study whose trials were trained or selected on different images.
So a damaged archive, a bundle from another draw, or a checkout whose manifests disagree
with the bundles all fail loudly instead of quietly changing the experiment.

A bundle is a plain compressed npz and needs none of this code to read:

```python
d = np.load('dermamnist_splits_224.npz')
d['val_images'], d['val_labels']                    # (151, 224, 224, 3), (151, 1)
d['train_fold1_images'], d['train_fold1_labels']    # (700, 224, 224, 3), (700, 1)
d['train_fold1_index']                              # rows of the official train split
json.loads(str(d['manifest']))                      # checksums, counts, provenance
```

Rebuilding a bundle gives byte-identical *arrays*, but not a byte-identical *file* — the
manifest carries a build timestamp and zip entries carry mtimes. `SHA256SUMS` is for
checking a download; the per-part `images_sha256` inside the manifest is what identifies
the sample.

#### Running it

```bash
export BUNDLE_DIR=                 # optional; defaults to data/splits/bundles/
export MEDMNIST_ROOT=~/.medmnist   # <target>_224.npz, used only for the test split

python src/make_splits.py --targets dermamnist                   # once, if not built yet
python src/hpo_finetune.py --target dermamnist --estimate-only   # projected runtime
python src/hpo_finetune.py --target dermamnist                   # --space wide, 100 trials
python src/hpo_finetune.py --target dermamnist --space core      # 8 parameters, 50 trials
```

The defaults *are* the published protocol — 100 trials, 25 random before TPE takes over,
no pruning, the top 10 configurations retrained on folds 2–5 with four seeds each — so the
plain command above reproduces the paper rather than a cheaper variant of it. That is 260
runs per architecture-target pair, 25,740 in total.

The split between searching and repeating is the protocol's main design decision, and it
was measured rather than assumed. On the dermamnist pilot, going from 100 trials to 200
bought +0.0083 validation AUC but only +0.0034 on test, and within the top 20 trials by
validation AUC — the region selection actually operates in — the rank correlation with
test AUC is τ = 0.069. The search separates good configurations from bad ones and then
stops resolving anything. Replication is what the ground truth is short of instead: at
4 runs per architecture the reliability of the reported means is 0.857, which caps any
transferability metric correlated against them at Kendall τ ≈ 0.75; at 16 runs it is
0.960, for a ceiling of 0.87. `--trials 200 --final-topk 5 --final-seeds 1` restores the
older allocation.

The one place that costs real time for little return is `--test-eval all` (the default),
which scores the test split on every trial so the validation-to-test relationship can be
studied afterwards; nothing in the protocol reads those numbers, and on the large test
splits (`tissuemnist`, `organamnist`) `--test-eval best` saves hours per architecture.

`--estimate-only` prints the projected runtime before anything is committed; always run it
first on a new machine, since the cost varies by an order of magnitude across
architectures.

One command covers all nine architectures for one target. Progress goes to stdout and to
`results/hpo/<target>/`; from another shell:

```bash
python src/hpo_watch.py --target dermamnist -w   # progress and ETA, refreshing
```

The search runs on one training fold (`--fold`, default 1), and what comes out of it is a
shortlist, not a winner. The top `--final-topk` configurations (default 10) are each
retrained over `--final-folds` x `--final-seeds` — by default folds 2–5 with four seeds
each, so 160 runs — and the winner is the one with the highest validation AUC *averaged
over those folds*. The ground truth is that configuration's mean test AUC over the same
16 runs, with the spread alongside.

Selecting on one run is the problem this solves. The search ranks ~4000 validation AUCs
(100 trials x up to 40 checkpoints), each measured on at most 25 images per class, where
the standard error of an AUC is on the order of a percentage point — while the
architectures being compared sit a fraction of that apart. The argmax of that ranking is
substantially noise: on the pilot, the best-on-test configuration sat at median rank 10
of the top 20 by validation AUC, so a shortlist of 5 caught it in 44% of architectures
and one of 10 in 56%. Re-estimating ten candidates on folds none of them was tuned on
puts the choice on an average rather than an extreme, and it is the retrain rather than
the search that does the selecting. `--final-topk 1` restores the old top-1 rule.

The default folds deliberately exclude the search fold, so no fold both tunes a
configuration and contributes to the number reported for it; putting it back
(`--final-folds 1 2 3 4 5`) prints a warning. Trials that diverged are excluded from the
shortlist: their value depends on stopping before the blow-up, which does not survive a
change of fold or seed. A run counts as diverged if its training or validation loss goes
non-finite, or if either exceeds `hpo_finetune.DIVERGE_LOSS_MULT` (500) times the loss of
a uniform predictor. The magnitude bound is what does the work: float32 reaches ~3e38, so
a run at an aggressive learning rate saturates into a constant predictor
long before it overflows, and a non-finiteness test on its own lets it keep the checkpoint
it held on the way up. On the dermamnist search that bound caught 21 blown-up runs the old
test missed. Validation loss is checked as well as training loss because batch-norm
statistics accumulated under an exploding rate can blow up in eval mode while the training
loss, computed from each batch's own statistics, still looks finite. Each architecture's
share of trials that never beat chance is
printed after its search and recorded as `near_chance` in `trials_all.csv` — equal trial
counts are not equal search quality, and that share is the part that differs. A trial
counts if it diverged or if its validation AUC came in at or below
`hpo_select.CHANCE_AUC` (0.55): at high learning rates a run usually decays towards
chance rather than exploding, so counting only the divergences reports a search as
healthier than it was. The two numbers are printed separately.

Because nothing of a finished run is kept but its record — the weights are never written
to disk — the record has to carry enough to answer a question nobody asked while the run
was alive. Two things are stored for that reason. Every trial and every final run carries
the **per-class one-vs-rest AUCs** and the support of each class, for validation and test,
so any reweighting of them can be recomputed afterwards; the macro average the protocol
selects on is the unweighted mean of exactly those numbers. And the **raw predictions** of
the checkpoint a run is reported on are written to
`results/hpo/<target>/predictions/<arch>/*.npz` (`val_score`, `val_true`, `test_score`,
`test_true`), which covers the metrics that are not a function of the per-class AUCs at
all — average precision, a different operating point, calibration, or a paired test
between two architectures on the same images. `--save-scores` controls the second: `all`
(default) keeps them for every trial as well as the 160 final runs per architecture,
`final` for the final runs only, `none` for neither. `all` costs more on the large test splits,
since it adds the test predictions of 100 search trials — across all eleven targets and
nine architectures that is roughly 8.6 GB against 5.3 GB. It is the default because it is
cheap for what it buys: any AUC variant, average precision, calibration or paired
per-image test can then be recomputed after the run instead of requiring it again, and
the trials are what the validation-to-test relationship has to be studied on.

For a long run, launch it through the guard rather than directly:

```bash
python scripts/run_hpo_guarded.py --target dermamnist -- --workers 8 --pruner median
```

This puts the search in a systemd scope with a memory cap and no swap, so a run that
grows without bound is killed by its own cgroup instead of by the kernel taking down the
machine — and it restarts the run when that happens, since the search resumes from
`optuna.db` and loses at most the trial that was in flight. It also reaps DataLoader
forkservers orphaned by earlier runs and clears a `run.lock` left by a killed one. The
peak memory each attempt reached is reported, which is what `--mem` should be set from
(the default is 48G; a single-architecture DermaMNIST run with 4 workers peaks near
9.4 GiB, and the largest test splits materialise several GB more). Use `--jobs N` to be
warned when N concurrent caps would oversubscribe the machine's RAM.

Restarting is only worth doing when the failure was transient, so each attempt is judged
by what it finished, counted from `optuna.db` and `final_runs.csv`. Attempts that finish
nothing repeat — a target whose test split does not fit in the memory it was given fails
the same way every time — and `--max-unproductive` of them in a row (3) ends the run,
while an attempt that got somewhere resets both that count and the backoff. `--max-attempts`
(8) is then the budget for failures that had been making progress. On a batch scheduler
this is the difference between releasing the node in minutes and holding a GPU for hours
re-running the same first minute.

Under a batch scheduler, build the environment first — as a job, because the check that
matters needs to see a GPU:

```bash
sbatch --gres=gpu:v100:1 scripts/env.sbatch     # the card you will train on
```

That also packs the environment into a single archive (`$HPO_ENV_PACK`, by default
`~/hpo-env/similarity.tar.gz`), which is what the training jobs actually run from: each
one unpacks it onto node-local scratch rather than importing a few hundred thousand small
files across shared storage, where the cost is paid by every job every time and every
other job is paying it concurrently. Measured: packing 2 min for 3.2 GB from a 5.9 GB
environment, unpacking 27 s. A fresh `conda create` per job would be ten to twenty minutes
and — worse — would re-resolve, so two targets started a month apart could get different
versions of torch. Without an archive the jobs fall back to a shared conda environment and
say so.

Data is *not* staged to scratch. The bundle and the test `npz` are read once at start-up
and numpy pulls only the members it is asked for, so copying them local would mean reading
the same bytes and writing them again. It is the environment that is read over and over.

An environment that imports torch and reports a GPU can still be unable to run a kernel on
it. A wheel carries compiled code for a set of compute capabilities, and a card outside
that set fails with *no kernel image is available for execution on the device* — at the
first trial, hours in, after the weights are fetched and the data staged. Volta is the case
that bites: V100 is `sm_70` and recent wheels start at `sm_80`, so `pip install torch`
gives an environment that works everywhere except on the card it has to run on. The CUDA
11.8 build still carries `sm_70`. `scripts/gpu_check.py` is the test, matching by major
compute capability rather than exact string — `sm_80` code does run on an `sm_86` card —
and `hpo.sbatch` re-runs it before every job. Override the torch install with
`HPO_TORCH_INSTALL` and the conda module with `HPO_CONDA_MODULE` if your site differs.

Then `sbatch --gres=<card> scripts/hpo.sbatch <target>` runs one target on one GPU, start
to finish — one job per target, on the assumption that the wall-clock limit is extended
once the job is running. `--time` is only what is asked for at submission; set
`HPO_MAX_HOURS` if the limit is real and enforced, and the run stops ten minutes short of
it with its results copied back. Nothing resubmits itself, and if a job does end early the
study is resumable by re-running the same command.

It works on node-local disk and copies back every five minutes — the study database
through `scripts/db_snapshot.py` rather than rsync, so the shared copy is always one that
opens — which is both what makes the run watchable from a login node and what survives a
node failure.

On a queue holding more than one kind of card, name the kind in `--gres`.
`scripts/submit_all.sh` carries a
target-to-card mapping and submits all eleven; it prints and validates the plan by default
and submits only with `--go`, because a gres name the partition does not offer does not
fail — the job sits pending indefinitely with nothing in any log. `scripts/cluster.md`
records what is on each node and why each pairing.

##### Does the hardware have to match?

Not between targets. Every Kendall τ is computed within a target, across architectures, so
what has to hold is that the nine architectures of one target met the same conditions —
not that all eleven targets did. Two things could break that, and neither is left to the
machine: TF32 is pinned off, so the arithmetic is the same on any card, and the seed and
the split are fixed in the manifest.

What is left is memory, and it is not a fair constraint. A card too small for the largest
batch does not stop the run — the trial raises, Optuna catches it, and the search
continues over what is left. At batch 128 and 224px in fp32, measured:

| | 64 | 128 | | | 64 | 128 |
|---|---|---|---|---|---|---|
| `vgg` | 9.1 | **17.9** | | `resnet` | 7.8 | 13.1 |
| `densenet` | 8.9 | **17.3** | | `efficientnet` | 6.4 | 11.8 |
| `convnext` | 8.4 | **15.1** | | `shufflenet` | 2.4 | 3.8 |

So a 16 GB card removes batch 128 from `vgg`, `densenet` and `convnext` and leaves it for
`shufflenet` — a per-architecture change to the search space, which is a per-architecture
perturbation of the quantity being measured. **≈20 GiB is the threshold for the full
space**; `scripts/gpu_check.py` warns below it and names the `--batch-sizes` that do fit,
which is the honest alternative: restrict the space explicitly and equally for all nine.

Within a target the hardware is constant as long as the target finishes in one job, which
is what the one-job-per-target arrangement is for. A run that has to be resumed after a
node failure can land elsewhere, so every host and card a target has trained on is
accumulated in `ran_on` in its `meta.json` — what actually happened is on the record
rather than inferred afterwards. `cudnn.benchmark` still picks algorithms per card, so
runs on different nodes differ in the last bits; that is the same order as changing the
seed, and 16 runs stand behind each reported number.

The search is stored in `results/hpo/<target>/optuna.db`, so an interrupted run resumes by
re-issuing the same command. Architectures can be split across GPUs with `--archs`, e.g.
`CUDA_VISIBLE_DEVICES=0 ... --archs densenet efficientnet convnext vgg` alongside
`CUDA_VISIBLE_DEVICES=1 ... --archs googlenet mnasnet mobilenet shufflenet resnet`.

Finally, turn the studies into tables:

```bash
python src/hpo_collect.py --out results/hpo
```

which writes `results/AUCs_model_hpo.csv` (the selected configuration's mean test AUC over
the held-out folds, in percent, same layout as `AUCs_model.csv`), `AUCs_model_hpo_sd.csv`
(its spread over those runs), `AUCs_model_hpo_paired.csv` (per target, every pair of
architectures differenced fold by fold — the tighter and more relevant statistic for
comparing two of them), and `AUCs_model_hpo_besttrial.csv` (the single-best-validation-run
rule, for comparison). `results/hpo/selection.csv` records what each choice was between,
including what top-1 would have reported instead.

Where the runs carry per-class AUCs, it also writes `AUCs_model_hpo_weighted.csv`: the
same selected runs scored with a prevalence-weighted average instead of the macro one,
and a printed summary of how many architecture pairs swap order between the two and
whether the top of any target's table changes. Nothing is re-selected for it — the
configuration is still the one the protocol chose — so what it isolates is the effect of
the metric on the reported number, not on which run gets reported. The difference is not
cosmetic on an imbalanced target: on DermaMNIST one class holds two thirds of the test
split, so a weighted average is close to a report on that class alone while the macro
average gives it a seventh of the weight.

The selection rule itself lives in `src/hpo_select.py` and is recomputed from
`final_runs.csv` rather than trusted, so a study extended after its final stage cannot
leave a stale winner behind, and a different rule can be applied to the released runs
without rerunning anything.

### Calculating Transferability Scores

To compute transferability scores for dataset or model evaluation, run:
```bash
python -m transferability_scores.py --source <dataset or model> --method <FU | LP | LEEP | NLEEP | others>
```
`--source`: Specify `dataset` for dataset transferability or `model` for architecture transferability.
* Source datasets: 12 datasets from MedMNIST, leave-target-out MedMNIST, RadImageNet, ImageNet
* Implemented architectures: densenet, efficientnet, googlenet, mnasnet, mobilenet, vgg, convnext, shufflenet, resnet
  
`--method`: Choose a transferability metric. Options include:
* FU or LP proposed in the paper,
* LEEP, NLEEP, LogME, PARC, SFDA, NCTI.

### Results

The results from the experiments, including transferability scores and fine-tuning performance, are available in the `results` folder and are analyzed in the `dataset_transferability`, `architecture_transferability`, and `finetuned_AUCs` notebooks.

## Contact

Feel free to contact us for help with reproducing our experiments or if you have any questions about this repository.

## Citation
If you find our method useful in your research, please cite:

```yaml
@article{juodelyte2024dataset,
  title={On dataset transferability in medical image classification},
  author={Juodelyte, Dovile and Ferrante, Enzo and Lu, Yucheng and Singh, Prabhant and Vanschoren, Joaquin and Cheplygina, Veronika},
  journal={arXiv preprint arXiv:2412.20172},
  year={2024}
}
```

