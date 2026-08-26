#!/usr/bin/env python
# coding: utf-8
"""Fine-tuning benchmark with Optuna hyperparameter search over CNN backbones.

Replacement ground truth for results/AUCs_model.csv. For one target dataset it runs,
per architecture:

  1. an Optuna TPE study, maximising validation AUC;
  2. a final stage that retrains a shortlist of the best configurations on the held-out
     folds, picks the winner by mean validation AUC over them, and reports its mean test
     AUC.

The default split of the budget -- 100 search trials, then the top 10 configurations on
4 folds x 4 seeds -- comes from measuring both halves on the dermamnist pilot. The search
stops resolving anything once it reaches the plateau (among the top 20 trials by
validation AUC, the rank correlation with test AUC is tau 0.069), while replication is
what the ground truth is short of: at 4 runs per architecture the reliability of the
reported means is 0.857, which caps any transferability metric correlated against them at
Kendall tau 0.75, and at 16 runs it is 0.960 for a ceiling of 0.87.

The search covers the optimizer itself -- SGD (with momentum) and AdamW -- on
per-optimizer learning-rate ranges, plus weight decay, batch size, the head
learning-rate multiplier and augmentation strength; --space wide (the default) also
searches dropout, label smoothing, warmup and the schedule.

The ranges are wide deliberately. Selection has to resolve sub-percentage-point
differences between architectures, so the space has to contain each one's optimum rather
than stop near it: in the earlier fixed grid (finetuned_AUCs/, lr in {1e-4..1e-1},
momentum in {0, 0.9}, wd in {0..1e-2}) 33 of 88 architecture-target cells settled on the
largest learning rate available to them and 57 on the larger momentum.
src/grid_evidence.py recomputes that and the sensitivity of the selection rule.

Selection never touches the test set: trials are ranked on validation AUC, and the
test set is only evaluated for the running-best trial (--test-eval all to score every
trial instead) and for the final multi-seed runs.

Everything is stored in results/hpo/<target>/optuna.db, so an interrupted run resumes
by re-issuing the same command. Progress is mirrored to results/hpo/<target>/progress.json
and can be followed from another shell with src/hpo_watch.py, which also projects how
long the remaining work will take.

Both the training folds and the validation split are frozen, and come from the bundles in
data/splits/bundles/ -- one archive per target holding the images themselves, checksum-
verified on load. Build them once with

  python src/make_splits.py --all

(that is the only step that touches MedMNIST and the only one that draws anything), or
copy them in from wherever they are archived. Nothing is drawn at run time, so two
machines with the same bundles fine-tune on the same images.

Paths come from the environment so the job can move between machines:
  BUNDLE_DIR     the frozen subsets (default data/splits/bundles in the repo)
  MEDMNIST_ROOT  directory holding <target>_224.npz, for the test split
                 (default ~/.medmnist)

Examples
--------
# what will this cost before committing to it?
python src/hpo_finetune.py --target dermamnist --estimate-only

# full run for one target, all nine architectures
python src/hpo_finetune.py --target dermamnist --trials 200   # a longer search

# cheaper: the seven parameters that matter most, fewer trials
python src/hpo_finetune.py --target dermamnist --space core

# shard across two GPUs
CUDA_VISIBLE_DEVICES=0 python src/hpo_finetune.py --target dermamnist \
    --archs densenet efficientnet googlenet convnext vgg
CUDA_VISIBLE_DEVICES=1 python src/hpo_finetune.py --target dermamnist \
    --archs mnasnet mobilenet shufflenet resnet
"""

import argparse
import copy
import json
from collections import OrderedDict
import math
import os
import socket
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from medmnist import INFO
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo_select                                                     # noqa: E402
import splits                                                         # noqa: E402

MEDMNIST_ROOT = os.environ.get('MEDMNIST_ROOT', os.path.expanduser('~/.medmnist'))

# the nine backbones of the architecture setting. 'resnet' was resnet18 up to and
# including the dermamnist pilot, which made it the same network the dataset setting
# calls the 'imagenet' source -- the two settings shared that row exactly. It is
# resnet50 from here on, so that link no longer holds and the dataset setting's
# 'imagenet' row is no longer this row: src/transferability_scores.py still builds
# resnet18 there, deliberately, since changing it would redefine every source.
#
# Each slot holds the variant people normally cite, rather than a reduced one: the
# pilot ran mobilenet_v3_small, shufflenet_v2_x0_5, vgg11 and efficientnet_v2_s, which
# are the cut-down members of their families and made the set harder to compare with
# published results. googlenet stays: its natural successor, inception_v3, cannot take
# 224px input, and feeding one architecture different images would break the property
# that every architecture sees the same ones.
ARCHS = {
    'densenet': 'densenet121',
    'efficientnet': 'efficientnet_b0',
    'googlenet': 'googlenet',
    'mnasnet': 'mnasnet1_0',
    'mobilenet': 'mobilenet_v3_large',
    'vgg': 'vgg16',
    'convnext': 'convnext_tiny',
    'shufflenet': 'shufflenet_v2_x1_0',
    'resnet': 'resnet50',
}


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

class MyDataset(Dataset):
    """MedMNIST split held in shared memory.

    Pixels live in a shared-memory uint8 tensor so that DataLoader workers started
    with 'forkserver'/'spawn' map them instead of receiving a pickled copy -- without
    this, one worker per loader would copy the whole split (gigabytes for the larger
    test sets).
    """

    def __init__(self, data, targets, transform=None, as_rgb=False):
        self.data = torch.from_numpy(np.ascontiguousarray(data)).share_memory_()
        self.targets = np.ascontiguousarray(targets)
        self.transform = transform
        self.as_rgb = as_rgb

    def __getitem__(self, index):
        x = Image.fromarray(self.data[index].numpy())
        y = self.targets[index].astype(int)
        if self.as_rgb:
            x = x.convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def train_transform(strength='heavy'):
    """Augmentation pipeline.

    'heavy' is the pipeline from src/fine-tuning_imagenet.py. Note its
    RandomResizedCrop keeps torchvision's default scale of (0.08, 1.0), i.e. crops
    down to 8% of the image -- aggressive for medical images, which is why the
    milder settings are worth searching over.

    Augmentations run on uint8 (the v2-recommended order) and the cast to float
    happens last.
    """
    if strength == 'none':
        aug = []
    elif strength == 'light':
        aug = [v2.RandomHorizontalFlip(p=0.5),
               v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), antialias=True)]
    elif strength == 'heavy':
        aug = [v2.RandomHorizontalFlip(p=0.5),
               v2.RandomResizedCrop(size=(224, 224), antialias=True),
               v2.RandomRotation(degrees=(0, 5)),
               v2.RandomAdjustSharpness(sharpness_factor=2),
               v2.RandomAutocontrast(),
               v2.RandomEqualize()]
    else:
        raise ValueError(f'unknown augmentation strength {strength!r}')
    return v2.Compose([v2.ToImage()] + aug +
                      [v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=_MEAN, std=_STD)])


def eval_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_MEAN, std=_STD),
    ])


def load_train(target_flag, fold=1, bundle_dir=None):
    """Training part of one fold, from the bundle.

    `n_classes*100` images from the official training split, stratified to that split's
    own class distribution, so the target's natural imbalance -- which is part of the task
    being studied -- is carried exactly rather than in expectation. `load_val` holds the
    other part, drawn from the official *validation* split.
    """
    imgs, labels, entry = splits.load_train(target_flag, fold, bundle_dir)
    as_rgb = INFO[target_flag]['n_channels'] == 1
    return MyDataset(imgs, labels, transform=train_transform('heavy'),
                     as_rgb=as_rgb), entry


def load_val(target_flag, fold=1, bundle_dir=None):
    """Validation part of one fold, checksum-verified on load.

    `n_classes*25` images from the official validation split, stratified to that split's
    own class distribution, so selection is scored on the prevalence the test split has.
    Because it varies per fold, its sampling error averages down over the final stage
    rather than sitting on every fold as the same offset.
    """
    imgs, labels, entry = splits.load_val(target_flag, fold, bundle_dir)
    as_rgb = INFO[target_flag]['n_channels'] == 1
    return MyDataset(imgs, labels, transform=eval_transform(), as_rgb=as_rgb), entry


def describe_labels(dataset, target_flag):
    """Per-class counts, printed at startup so the selection signal is auditable."""
    info = INFO[target_flag]
    labels = dataset.targets
    if info['task'] == 'multi-label, binary-class':
        return f'{len(labels)} images, {labels.sum(axis=0).tolist()} positives per label'
    counts = np.bincount(labels.reshape(-1), minlength=len(info['label']))
    return f'{len(labels)} images, per-class {counts.tolist()}'


def load_test(target_flag):
    as_rgb = INFO[target_flag]['n_channels'] == 1
    path = os.path.join(MEDMNIST_ROOT, f'{target_flag}_224.npz')
    d = np.load(path)
    return MyDataset(d['test_images'], d['test_labels'],
                     transform=eval_transform(), as_rgb=as_rgb)


def resolve_worker_context(name, workers, device):
    """multiprocessing context for the DataLoader workers.

    Forking a worker after the parent has initialised CUDA deadlocks on some
    driver/platform combinations (the worker never returns from its first futex
    wait), so on CUDA the default is 'forkserver': workers are forked from a clean
    server process that never touched the GPU.
    """
    import multiprocessing

    if workers == 0:
        return None
    if name == 'auto':
        name = 'forkserver' if (device.type == 'cuda' and sys.platform == 'linux') else 'fork'
    ctx = multiprocessing.get_context(name)
    if name == 'forkserver':
        try:
            ctx.set_forkserver_preload(['torch', 'torchvision', 'PIL.Image'])
        except Exception:
            pass
    return ctx


class Loaders:
    """Built once and reused across trials.

    Starting workers costs seconds under 'forkserver'/'spawn', which would dominate
    if every trial (let alone every epoch) rebuilt its loaders. Only the training
    loader varies, and only over the handful of batch sizes in the search space.
    """

    #: training loaders kept alive at once; each holds `--workers` worker processes,
    #: and the search moves between (augmentation, batch size) combinations
    MAX_TRAIN_LOADERS = 3

    def __init__(self, cfg, ctx, mp_context):
        self.cfg = cfg
        self.ctx = ctx
        self.kw = dict(num_workers=cfg.workers, pin_memory=ctx['device'].type == 'cuda',
                       multiprocessing_context=mp_context,
                       persistent_workers=cfg.workers > 0)
        self._cache = OrderedDict()

    def _get(self, key, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        self._cache.move_to_end(key)
        return self._cache[key]

    def train(self, fold, aug, batch_size):
        def build():
            # copy.copy shares the shared-memory pixels, so a view per augmentation
            # setting costs nothing
            view = copy.copy(self.ctx['split'](fold))
            view.transform = train_transform(aug)
            return DataLoader(view, batch_size=batch_size, shuffle=True,
                              drop_last=len(view) > batch_size, **self.kw)

        loader = self._get(('train', fold, aug, batch_size), build)
        live = [k for k in self._cache if k[0] == 'train']
        for key in live[:max(0, len(live) - self.MAX_TRAIN_LOADERS)]:
            self.drop(key)
        return loader

    def val(self, fold):
        # one loader per fold, kept alive: the validation parts are small and every
        # trial on a fold reuses the same one
        return self._get(('val', fold), lambda: DataLoader(
            self.ctx['valsplit'](fold), batch_size=self.cfg.test_batch_size,
            shuffle=False, **self.kw))

    def test(self):
        return self._get(('test',), lambda: DataLoader(
            self.ctx['testset'](), batch_size=self.cfg.test_batch_size, shuffle=False,
            **self.kw))

    def probe(self):
        """Inference-cost probe: the training split without augmentation.

        The validation split is too small to time (a couple of batches, so iterator
        restarts dominate). copy.copy shares the same shared-memory pixels, so this
        view costs nothing.
        """
        def build():
            view = copy.copy(self.ctx['split'](self.cfg.fold))
            view.transform = eval_transform()
            return DataLoader(view, batch_size=self.cfg.test_batch_size, shuffle=False,
                              **self.kw)
        return self._get(('probe',), build)

    def drop(self, key):
        loader = self._cache.pop(key, None)
        del loader


# --------------------------------------------------------------------------------------
# metrics -- kept identical to src/fine-tuning_imagenet.py so numbers stay comparable
# --------------------------------------------------------------------------------------

def _safe_auc(y_true_binary, y_score_binary):
    """One-vs-rest AUC, or None when the class has no positives (or no negatives).

    The frozen validation split has every class in it by construction, so this should
    not trigger there -- but a class with a single image would still make its AUC
    undefined for some batches of predictions, and averaging a NaN over the remaining
    classes would poison the whole score. Undefined classes are skipped and the macro
    average is taken over the ones that are defined. Test splits are large enough that
    nothing is ever skipped there.
    """
    if y_true_binary.min() == y_true_binary.max():
        return None
    return roc_auc_score(y_true_binary, y_score_binary)


def auc_per_class(y_true, y_score, task):
    """The per-class one-vs-rest AUCs behind `getAUC`, and the support of each class.

    `getAUC` averages these unweighted, and that macro average is what the protocol
    selects on. Keeping the parts is what makes the choice auditable: a prevalence-
    weighted average, or one over a subset of the classes, can then be recomputed from
    the record instead of requiring the run to be repeated. That matters here because
    nothing of a finished run survives except its metrics -- the weights are never
    written to disk -- so a per-class number not recorded while the run is alive is
    only recoverable by training it again.

    Returns `(aucs, support)`, both indexed by class, with `None` for a class whose AUC
    is undefined so that the positions still line up with the class indices. Support is
    the number of positives, which is the weight a prevalence-weighted average needs.
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    # a diverged run (the search reaches learning rates that do diverge) produces
    # non-finite scores; that is a result, not an error
    finite = bool(np.isfinite(y_score).all())

    if task == 'multi-label, binary-class':
        cols = [(y_true[:, i], y_score[:, i]) for i in range(y_score.shape[1])]
    elif task == 'binary-class':
        score = y_score[:, -1] if y_score.ndim == 2 else y_score
        cols = [(y_true, score)]
    else:
        cols = [((y_true == i).astype(float), y_score[:, i])
                for i in range(y_score.shape[1])]

    # plain floats, not numpy scalars: these are written straight into Optuna's user
    # attributes and into CSV, and both go through json.dumps
    aucs = [None if not finite or (a := _safe_auc(t, s)) is None else float(a)
            for t, s in cols]
    return aucs, [int(np.sum(t)) for t, _ in cols]


def macro_auc(aucs):
    """Mean over the classes whose AUC is defined -- the protocol's objective."""
    defined = [a for a in aucs if a is not None]
    return float(np.mean(defined)) if defined else float('nan')


def weighted_auc(aucs, support):
    """Prevalence-weighted mean of the per-class AUCs, over the defined ones.

    Recorded alongside the macro average rather than replacing it. The two differ by
    exactly how much the rare classes matter, which on an imbalanced target is the
    whole question: on the DermaMNIST training fold one class holds 468 of the 700
    images and another holds 3, so a weighted average is close to a report on the
    majority class alone while the macro average gives that class a seventh of the
    weight. Which is the better ground truth is a judgement, and recording both lets
    it be made -- and checked against the ranking it produces -- after the fact.
    """
    return hpo_select.weighted_mean(aucs, support)


def getAUC(y_true, y_score, task):
    """Macro one-vs-rest AUC, kept bit-identical to the pre-existing definition."""
    if not np.isfinite(np.squeeze(y_score)).all():
        return float('nan')
    return macro_auc(auc_per_class(y_true, y_score, task)[0])


def getACC(y_true, y_score, task, threshold=0.5):
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if not np.isfinite(y_score).all():
        return float('nan')

    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            acc += accuracy_score(y_true[:, label], y_pre[:, label])
        ret = acc / y_true.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))
    return ret


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------

def _last_linear(module):
    """Index of the last nn.Linear inside a Sequential-like classifier head."""
    idx = [i for i, m in enumerate(module) if isinstance(m, nn.Linear)]
    if not idx:
        raise ValueError(f'no nn.Linear found in {type(module).__name__}')
    return idx[-1]


def build_model(arch, n_classes, dropout=0.0):
    """Return (net, head_module). head_module is the freshly initialised classifier."""
    name = ARCHS.get(arch, arch)
    ctor = getattr(torchvision.models, name, None)
    if ctor is None:
        raise ValueError(f'unknown architecture {arch!r} (torchvision has no {name})')
    net = ctor(weights='IMAGENET1K_V1')

    def make_head(in_features):
        linear = nn.Linear(in_features, n_classes)
        return nn.Sequential(nn.Dropout(p=dropout), linear) if dropout > 0 else linear

    if hasattr(net, 'fc') and isinstance(net.fc, nn.Linear):          # resnet, googlenet, shufflenet
        net.fc = make_head(net.fc.in_features)
        head = net.fc
    elif hasattr(net, 'classifier'):
        clf = net.classifier
        if isinstance(clf, nn.Linear):                                 # densenet
            net.classifier = make_head(clf.in_features)
            head = net.classifier
        else:                                                          # vgg, mnasnet, mobilenet, ...
            i = _last_linear(clf)
            clf[i] = make_head(clf[i].in_features)
            head = clf[i]
    elif hasattr(net, 'heads'):                                        # vision transformers
        i = _last_linear(net.heads)
        net.heads[i] = make_head(net.heads[i].in_features)
        head = net.heads[i]
    else:
        raise ValueError(f'do not know how to replace the head of {name}')
    return net, head


def _logits(out):
    """googlenet/inception return named tuples while training."""
    return out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)


# --------------------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------------------

def evaluate(model, loader, task, criterion, device):
    model.eval()
    total_loss, y_score, y_true = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = _logits(model(inputs)).float()

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                outputs = torch.sigmoid(outputs)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                outputs = torch.softmax(outputs, dim=1)
                targets = targets.float().reshape(-1, 1)

            total_loss.append(loss.item())
            y_score.append(outputs.cpu())
            y_true.append(targets.cpu())

    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()
    aucs, support = auc_per_class(y_true, y_score, task)
    # the scores come back with the summary: they are already in memory here, and the
    # caller that keeps them is the only thing standing between a finished run and a
    # metric nobody thought to compute while the weights still existed
    return {'loss': sum(total_loss) / len(total_loss),
            'auc': macro_auc(aucs),
            'acc': getACC(y_true, y_score, task),
            'auc_per_class': aucs,
            'support': support,
            'y_score': y_score,
            'y_true': y_true}


def make_optimizer(net, head, params):
    """The freshly initialised head gets lr * head_lr_mult, the backbone gets lr."""
    lr, wd = params['lr'], params['wd']
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in net.parameters() if id(p) not in head_ids and p.requires_grad]
    head_params = [p for p in net.parameters() if id(p) in head_ids and p.requires_grad]
    groups = [{'params': backbone, 'lr': lr},
              {'params': head_params, 'lr': lr * params['head_lr_mult']}]

    if params['optimizer'] == 'sgd':
        momentum = params['momentum']
        return torch.optim.SGD(groups, lr=lr, momentum=momentum,
                               nesterov=momentum > 0, weight_decay=wd)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def make_scheduler(optimizer, total_steps, warmup_steps, kind):
    """Stepped once per optimizer step, so the schedule completes over the trial's
    own budget rather than over a cap shared with trials of a different length."""
    if kind == 'cosine':
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps))
    else:
        main = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if warmup_steps <= 0:
        return main
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, main], milestones=[warmup_steps])


def make_criterion(task, label_smoothing=0.0):
    if task == 'multi-label, binary-class':
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


#: how many times the loss of a uniform predictor counts as a blown-up run. Measured on
#: the completed dermamnist search, all 1800 trials over the 9 architectures: the worst
#: validation loss belonging to a run that had learned anything (AUC above 0.6) was 54x
#: chance, and 3.4x for the ones above 0.8, while the runs that had destroyed themselves
#: began at 1362x and ran up to 1e31x. Between those lies a band of at-chance runs that
#: had not exploded, the highest at 235x. Every multiplier from 250x to 1000x sorts the
#: 1800 trials identically, so the constant is picked at the middle of that plateau
#: rather than at either edge; 2000x already starts to miss a genuine blow-up.
DIVERGE_LOSS_MULT = 500.0


def diverge_threshold(task, n_classes, mult=DIVERGE_LOSS_MULT):
    """The loss above which a run is treated as diverged.

    Non-finite losses are the obvious case, but they are the rare one: float32 reaches
    ~3e38, so a run at an aggressive learning rate saturates into a constant predictor
    long before anything overflows. Such a run keeps a finite loss and a validation AUC
    of exactly 0.5 -- every per-class AUC of a constant score is exactly 0.5 -- and a
    non-finiteness test on its own lets it through with its pre-blow-up checkpoint
    intact. Nothing here is an artefact of reduced precision: bfloat16 carries the same
    8-bit exponent as float32 and so the same dynamic range, which is why dropping mixed
    precision did not remove the need for this bound. It was measured under bfloat16 and
    the thresholds are unchanged, since the range they are scaled against is unchanged.

    This bound catches the runs that blew up, not every run that failed. A model can also
    collapse to a constant at a modest loss, and there the two populations interleave: in
    the dermamnist search one trial sat at chance with a loss of 60x while another, 54x,
    had reached AUC 0.60 and was still learning. No threshold separates those without
    being fitted to the handful of points that happen to lie between them. They are
    counted instead by `hpo_select.near_chance` when the search is reported, which asks
    what the run achieved rather than how it failed; this bound exists for the narrower
    job the flag actually does -- keeping a lucky checkpoint taken just before a blow-up
    out of the objective and out of the shortlist.

    Scaled to the loss of a uniform predictor so the same rule reads the same on a
    binary target and a nine-class one: log(C) for cross-entropy, log(2) per label for
    the multi-label BCE case.
    """
    chance = math.log(2.0) if task == 'multi-label, binary-class' \
        else math.log(max(2, int(n_classes)))
    return mult * chance


def _cpu_state_dict(net, into=None):
    """Snapshot of the weights on CPU, keeping the _metadata that load_state_dict needs.

    MNASNet's _load_from_state_dict refuses a state dict whose metadata carries no
    version, so a plain dict comprehension over state_dict().items() is not enough.
    Passing a previous snapshot as `into` copies into it rather than allocating a
    fresh one -- for vgg16 that is half a gigabyte per improving epoch.
    """
    from collections import OrderedDict

    sd = net.state_dict()
    if into is not None:
        for k, v in sd.items():
            into[k].copy_(v)
        return into
    out = OrderedDict((k, v.detach().to('cpu', copy=True)) for k, v in sd.items())
    metadata = getattr(sd, '_metadata', None)
    if metadata is not None:
        out._metadata = copy.deepcopy(metadata)
    return out


def run_one(cfg, arch, params, seed, ctx, fold=1, trial=None, on_val=None):
    """Train one configuration to its own budget.

    The budget is `train_epochs` passes over the training split, but progress,
    validation, early stopping and Optuna's intermediate values are all counted in
    optimizer steps: an epoch is 21 steps at batch size 32 and 2 at batch size 256,
    so anything measured in epochs would not be comparable across the search space.

    Returns a dict of metrics; raises TrialPruned if Optuna prunes it.
    """
    import optuna

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = ctx['device']
    task = ctx['task']
    criterion = make_criterion(task, params['label_smoothing'])
    diverge_loss = diverge_threshold(task, ctx['n_classes'])

    net, head = build_model(arch, ctx['n_classes'], dropout=params['dropout'])
    net.to(device)

    train_loader = ctx['loaders'].train(fold, params['aug'], params['batch_size'])
    val_loader = ctx['loaders'].val(fold)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, params['train_epochs'] * steps_per_epoch)
    warmup_steps = min(params['warmup_epochs'] * steps_per_epoch, total_steps // 2)
    # a fixed number of validations per trial rather than a fixed step interval: an
    # epoch is 21 steps at batch size 32 and 2 at batch size 256, so a fixed interval
    # would give a 400-epoch large-batch trial a handful of checkpoints and a
    # small-batch one several hundred, and patience would mean different things
    val_every = (max(1, min(cfg.val_every_steps, total_steps)) if cfg.val_every_steps
                 else max(1, total_steps // cfg.validations_per_trial))

    optimizer = make_optimizer(net, head, params)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, params['scheduler'])

    best_auc, best_state, best_step, since_best = -np.inf, None, 0, 0
    best_val = {'val_loss': np.nan, 'val_auc': -np.inf, 'val_acc': np.nan,
                'val_auc_per_class': None, 'val_support': None}
    best_scores = None
    step, n_val, stopped_early, diverged = 0, 0, False, False
    t0 = time.time()
    net.train()

    while step < total_steps and not stopped_early and not diverged:
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = _logits(net(inputs))
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            # the search does reach learning rates that blow up; stop burning the
            # budget and let the objective score it at chance. Reading the loss once
            # into Python costs the same synchronisation the isfinite test already
            # forced, and lets both bounds be checked for it.
            train_loss = loss.item()
            if not math.isfinite(train_loss) or train_loss > diverge_loss:
                diverged = True
                break

            if step % val_every == 0 or step >= total_steps:
                ev = evaluate(net, val_loader, task, criterion, device)
                val_loss, val_auc, val_acc = ev['loss'], ev['auc'], ev['acc']
                net.train()
                n_val += 1

                # eval mode blows up on its own: batch-norm statistics accumulated
                # under an exploding learning rate can make the validation loss
                # diverge while the training loss, computed in train mode from the
                # batch's own statistics, still looks finite
                if not np.isfinite(val_loss) or val_loss > diverge_loss:
                    diverged = True
                    break

                # only a finite AUC can take the checkpoint, and a non-finite one must
                # not be allowed to *stay* the best either: eval-mode logits can go
                # non-finite (exploded batch-norm statistics) while the training loss
                # is still finite, and `val_auc > nan` is False forever after, which
                # would freeze the checkpoint for the rest of the trial
                if np.isfinite(val_auc) and (best_state is None
                                             or not np.isfinite(best_auc)
                                             or val_auc > best_auc):
                    best_auc, best_step, since_best = val_auc, step, 0
                    best_state = _cpu_state_dict(net, into=best_state)
                    best_val = {'val_loss': val_loss, 'val_auc': val_auc,
                                'val_acc': val_acc,
                                'val_auc_per_class': ev['auc_per_class'],
                                'val_support': ev['support']}
                    # the predictions of the checkpoint the run is reported on, kept so
                    # the objective can be recomputed under another average later
                    best_scores = (ev['y_score'], ev['y_true'])
                else:
                    since_best += 1

                if on_val is not None:
                    on_val(step, total_steps, n_val, train_loss, val_loss, val_auc,
                           time.time() - t0)

                if trial is not None:
                    trial.report(val_auc, n_val)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                if since_best >= cfg.patience:
                    stopped_early = True

            if step >= total_steps or stopped_early:
                break

    if best_state is None:      # diverged, or never scored a finite AUC
        best_state = _cpu_state_dict(net)
    net.load_state_dict(best_state)
    out = dict(best_val)
    if not np.isfinite(out['val_auc']):     # never validated, or never finite
        out['val_auc'] = float('nan')
    out.update(steps_run=step, total_steps=total_steps, best_step=best_step,
               validations=n_val, steps_per_epoch=steps_per_epoch,
               epochs_run=step / steps_per_epoch, stopped_early=stopped_early,
               diverged=diverged,
               # did the trial use its whole budget? if this is almost always true the
               # budget is the binding constraint and train_epochs needs a longer arm
               budget_bound=not stopped_early,
               train_seconds=time.time() - t0)
    return out, net, criterion, best_scores


def test_metrics(net, ctx, cfg, criterion):
    ev = evaluate(net, ctx['loaders'].test(), ctx['task'], criterion, ctx['device'])
    return ({'test_loss': ev['loss'], 'test_auc': ev['auc'], 'test_acc': ev['acc'],
             'test_auc_per_class': ev['auc_per_class'], 'test_support': ev['support']},
            (ev['y_score'], ev['y_true']))


def write_scores(out_dir, arch, name, val=None, test=None):
    """Store one run's predictions, so a metric can be chosen after the run is over.

    Per-class AUCs already travel with every run, and any reweighting of the one-vs-rest
    AUCs -- prevalence-weighted against macro, say -- can be recomputed from those alone.
    What they cannot give is a metric that is not a function of them: average precision,
    a different operating point, calibration error, or a paired test between two
    architectures on the same images. Those need the predictions, and since the weights
    are discarded when a run ends, they need them written down while it is alive.

    One compressed file per run, named for the run, written through a temporary path so
    an interrupted write leaves no half-file for a resumed run to trip over.
    """
    arrays = {}
    for kind, pair in (('val', val), ('test', test)):
        if pair is None:
            continue
        score, true = pair
        arrays[f'{kind}_score'] = np.asarray(score, dtype=np.float32)
        arrays[f'{kind}_true'] = np.asarray(true, dtype=np.float32)
    if not arrays:
        return None

    d = os.path.join(out_dir, 'predictions', arch)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{name}.npz')
    tmp = f'{path}.tmp'
    # written through a handle rather than a path: given a path, savez appends '.npz' to
    # it, and the temporary file would not be where os.replace goes looking for it
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------
# cost probe -- gives the ETA something to work with before any trial has finished
# --------------------------------------------------------------------------------------

def _time_batches(loader, step, device, iters=6, warmup=3):
    """Seconds per image over `iters` timed batches, after `warmup` untimed ones.

    Goes through the real DataLoader: for the cheap backbones the augmentation
    pipeline, not the GPU, is what sets the epoch time.
    """
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    seen, imgs, t0, elapsed = 0, 0, None, 0.0
    for _ in range(100):  # re-iterate if the split is shorter than warmup + iters
        for inputs, targets in loader:
            step(inputs, targets)
            seen += 1
            if seen == warmup:
                sync()
                t0 = time.time()
            elif seen > warmup:
                imgs += len(inputs)
                if seen >= warmup + iters:
                    sync()
                    elapsed = time.time() - t0
                    return elapsed / max(imgs, 1)
        if seen == 0:
            return 0.0
    sync()
    return (time.time() - t0) / max(imgs, 1) if t0 else 0.0


def probe_arch_cost(arch, ctx, cfg, batch_size=32):
    """Per-image training and inference cost, measured through the real input pipeline."""
    device = ctx['device']
    # every trial pays this again: weights off disk, onto the GPU, plus the first
    # CPU snapshot of the best state
    t0 = time.time()
    net, head = build_model(arch, ctx['n_classes'])
    net.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    _cpu_state_dict(net)
    build_s = time.time() - t0

    optimizer = make_optimizer(net, head, PARAM_DEFAULTS)
    criterion = make_criterion(ctx['task'])

    def train_step(inputs, targets):
        net.train()
        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = _logits(net(inputs))
        if ctx['task'] == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    def eval_step(inputs, targets):
        net.eval()
        with torch.no_grad():
            net(inputs.to(device, non_blocking=True))

    # timed with the most expensive augmentation, so the projection is an upper bound
    train_s = _time_batches(ctx['loaders'].train(cfg.fold, 'heavy', batch_size),
                            train_step, device)
    # inference is timed at the batch size the test set will actually use
    eval_s = _time_batches(ctx['loaders'].probe(), eval_step, device, iters=4, warmup=1)

    del net, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'train_s_per_img': train_s, 'eval_s_per_img': eval_s, 'build_s': build_s}


# --------------------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------------------

def _read_json_or(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def atomic_write_json(path, obj):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


class Progress:
    """Heartbeat file describing the in-flight trial, read by src/hpo_watch.py."""

    def __init__(self, path, target, archs):
        self.path = path
        self.state = {'target': target, 'archs': archs, 'pid': os.getpid(),
                      'host': socket.gethostname(), 'started': time.time(),
                      'updated': time.time(), 'stage': 'starting'}
        self.flush()

    def update(self, **kw):
        self.state.update(kw)
        self.state['updated'] = time.time()
        self.flush()

    def flush(self):
        atomic_write_json(self.path, self.state)


def acquire_lock(path, force=False):
    if os.path.exists(path):
        try:
            with open(path) as f:
                info = json.load(f)
            alive = info.get('host') == socket.gethostname() and _pid_alive(info.get('pid'))
        except Exception:
            info, alive = {}, False
        if alive and not force:
            raise SystemExit(
                f'another run holds {path} (pid {info.get("pid")} on {info.get("host")}).\n'
                f'wait for it, or pass --force if you are sure it is dead.')
    atomic_write_json(path, {'pid': os.getpid(), 'host': socket.gethostname(),
                             'started': time.time()})


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def enable_wal(db_path):
    """WAL lets hpo_watch.py read the study while this process writes to it.

    The mode that was actually reached is reported rather than assumed. WAL needs a
    shared-memory segment beside the database, which network filesystems do not
    provide, so on a shared scratch or home directory SQLite quietly stays on the
    rollback journal -- the PRAGMA returns the old mode instead of raising. Everything
    still works there, because one process writes a given study and readers open it
    with a timeout, but a reader can block behind a write for as long as that timeout,
    which is worth knowing before it looks like a hang.
    """
    try:
        con = sqlite3.connect(db_path, timeout=30)
        mode = con.execute('PRAGMA journal_mode=WAL').fetchone()[0]
        con.close()
        if str(mode).lower() != 'wal':
            print(f'[warn] {db_path} is in {mode!r} mode, not WAL -- likely a network '
                  f'filesystem. Reads from hpo_watch.py may block while a trial is '
                  f'being written; put --out on node-local storage to avoid it')
    except Exception as exc:  # pragma: no cover - non-fatal
        print(f'[warn] could not enable WAL on {db_path}: {exc}')


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

#: everything a trial needs; conditional parameters fall back to these
PARAM_DEFAULTS = {'optimizer': 'adamw', 'lr': 1e-4, 'momentum': 0.0, 'wd': 1e-4,
                  'batch_size': 32, 'head_lr_mult': 1.0, 'aug': 'heavy',
                  'train_epochs': 50, 'dropout': 0.0, 'label_smoothing': 0.0,
                  'warmup_epochs': 1, 'scheduler': 'cosine'}


#: `cfg` attribute holding the arm for each categorical parameter
ARMS = {'optimizer': 'optimizers', 'momentum': 'momentums', 'batch_size': 'batch_sizes',
        'head_lr_mult': 'head_lr_mults', 'aug': 'augs', 'train_epochs': 'train_epochs',
        'dropout': 'dropouts', 'warmup_epochs': 'warmups', 'scheduler': 'schedulers',
        'label_smoothing': 'label_smoothings'}


def fixed_arms(cfg):
    """`{parameter: value}` for every arm that has been narrowed to one choice.

    A one-element arm is a constant, not a dimension, and is what `--head-lr-mults 1`
    means. Keeping them here is what lets a stored trial be replayed faithfully: the
    parameter is never written to the study, so the final stage has to recover it from
    the configuration rather than from Optuna.
    """
    out = {}
    for param, attr in ARMS.items():
        choices = getattr(cfg, attr, None)
        if choices is not None and len(choices) == 1:
            out[param] = choices[0]
    return out


def _pick(trial, name, choices):
    """Sample a categorical arm, or return its only value without searching it.

    Suggesting a one-element categorical would add a parameter for TPE to model and
    would show up in the study as a searched axis, which is exactly what fixing an arm
    is meant to avoid: with the same trial budget, every dimension removed doubles the
    density of the search over the ones that remain.
    """
    return choices[0] if len(choices) == 1 else trial.suggest_categorical(name, choices)


def suggest_params(trial, cfg):
    """Sample one configuration.

    The learning rate is conditional on the optimizer: SGD and AdamW live on ranges
    two orders of magnitude apart, so a shared range would spend most of the budget
    in a region that is hopeless for one of them. Optuna treats 'lr_sgd' and
    'lr_adamw' as separate parameters, and the sampler is grouped so it models each
    branch on its own trials.
    """
    p = dict(PARAM_DEFAULTS)
    opt = _pick(trial, 'optimizer', cfg.optimizers)
    p['optimizer'] = opt
    if opt == 'sgd':
        p['lr'] = trial.suggest_float('lr_sgd', cfg.sgd_lr_min, cfg.sgd_lr_max, log=True)
        p['momentum'] = _pick(trial, 'momentum', cfg.momentums)
    else:
        p['lr'] = trial.suggest_float('lr_adamw', cfg.adamw_lr_min, cfg.adamw_lr_max,
                                      log=True)

    p['wd'] = trial.suggest_float('wd', cfg.wd_min, cfg.wd_max, log=True)
    p['batch_size'] = _pick(trial, 'batch_size', cfg.batch_sizes)
    p['head_lr_mult'] = _pick(trial, 'head_lr_mult', cfg.head_lr_mults)
    p['aug'] = _pick(trial, 'aug', cfg.augs)
    # how long to train is searched rather than capped: the schedule then completes
    # over the trial's own budget, and the pilot measures how long is actually needed
    p['train_epochs'] = _pick(trial, 'train_epochs', cfg.train_epochs)

    if cfg.space == 'wide':
        p['dropout'] = _pick(trial, 'dropout', cfg.dropouts)
        p['warmup_epochs'] = _pick(trial, 'warmup_epochs', cfg.warmups)
        p['scheduler'] = _pick(trial, 'scheduler', cfg.schedulers)
        if cfg.task != 'multi-label, binary-class':   # BCE has no label smoothing
            p['label_smoothing'] = _pick(trial, 'label_smoothing', cfg.label_smoothings)
    return p


def normalize_params(raw, cfg=None):
    """Rebuild a trial's configuration from what Optuna stored for it.

    Arms fixed to a single value were never stored, so they come from `cfg` when it is
    given and from PARAM_DEFAULTS otherwise -- the two agree for the defaults, and
    passing `cfg` is what keeps a non-default fixed arm from silently reverting.
    """
    p = dict(PARAM_DEFAULTS)
    if cfg is not None:
        p.update(fixed_arms(cfg))
    for k, v in raw.items():
        if k in p:
            p[k] = v
    p['lr'] = raw.get(f"lr_{p['optimizer']}", raw.get('lr', p['lr']))
    return p


def run_arch(arch, cfg, ctx, out_dir, progress):
    import optuna
    from optuna.trial import TrialState

    storage_url = f'sqlite:///{os.path.join(out_dir, "optuna.db")}'
    storage = optuna.storages.RDBStorage(
        storage_url, engine_kwargs={'connect_args': {'timeout': 60}})
    # multivariate + group: the space is conditional (momentum and the learning-rate
    # range depend on the optimizer), and grouping models each branch on its own trials
    sampler = optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=cfg.startup_trials,
                                         multivariate=True, group=True)
    if cfg.pruner == 'none':
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=cfg.startup_trials,
                                             n_warmup_steps=cfg.prune_warmup)
    study = optuna.create_study(study_name=arch, storage=storage, direction='maximize',
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    study.set_user_attr('target', cfg.target)
    study.set_user_attr('train_epochs', cfg.train_epochs)
    study.set_user_attr('n_trials', cfg.trials)
    study.set_user_attr('final_topk', cfg.final_topk)
    study.set_user_attr('final_seeds', cfg.final_seeds)
    study.set_user_attr('final_folds', cfg.final_folds)

    # a crash leaves trials stuck in RUNNING; one process owns a study at a time
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        try:
            storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
            print(f'[{arch}] marked stale running trial {t.number} as failed')
        except Exception:
            pass

    def objective(trial):
        params = suggest_params(trial, cfg)
        progress.update(stage='hpo', arch=arch, trial=trial.number,
                        trials_total=cfg.trials, frac=0.0, params=params,
                        trial_started=time.time())

        on_val = _make_val_logger(cfg, progress, f'[{arch}] trial {trial.number}')

        net = None
        try:
            res, net, criterion, val_scores = run_one(
                cfg, arch, params, cfg.seed, ctx,
                fold=cfg.fold, trial=trial, on_val=on_val)

            # a diverged or otherwise undefined run scores at chance rather than
            # failing, so the sampler learns to avoid that region. Divergence counts
            # against a configuration even when it had a good checkpoint before it
            # blew up: what that checkpoint was worth depends on stopping at exactly
            # the right step, and the final stage retrains on other folds and seeds
            # where the blow-up lands somewhere else. Its pre-divergence best is kept
            # in the trial's user attributes, out of the objective.
            usable = np.isfinite(res['val_auc']) and not res['diverged']
            score = res['val_auc'] if usable else 0.5

            best_so_far = _best_value(study)
            is_best = usable and (best_so_far is None or score > best_so_far)
            test_scores = None
            if usable and (cfg.test_eval == 'all'
                           or (cfg.test_eval == 'best' and is_best)):
                progress.update(stage='hpo-test')
                test_res, test_scores = test_metrics(net, ctx, cfg, criterion)
                res.update(test_res)

            if cfg.save_scores == 'all':
                write_scores(out_dir, arch, f'trial{trial.number}',
                             val=val_scores, test=test_scores)

            for k, v in res.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr('arch', arch)
            trial.set_user_attr('scored_at_chance', not usable)
            return score
        finally:
            del net
            if ctx['device'].type == 'cuda':
                torch.cuda.empty_cache()

    # failed trials do not count against the budget: a crash that had to be cleaned
    # up above, or an out-of-memory trial, should not silently shrink the search
    trials = study.get_trials(deepcopy=False)
    finished = [t for t in trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    failed = sum(1 for t in trials if t.state == TrialState.FAIL)
    if failed:
        print(f'[{arch}] {failed} earlier trial(s) failed and will be retried', flush=True)
    remaining = cfg.trials - len(finished)
    if remaining > 0:
        print(f'[{arch}] {len(finished)}/{cfg.trials} trials already done, '
              f'running {remaining} more', flush=True)
        # an out-of-memory trial (a large batch size on a small GPU) is recorded as
        # failed and the search moves on; any other exception stops the run so a
        # systematic problem cannot quietly consume the whole budget
        study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                       catch=(torch.cuda.OutOfMemoryError,))
    else:
        print(f'[{arch}] search already complete ({len(finished)} trials)', flush=True)

    # equal trial counts are not equal search quality: the share of the space an
    # architecture can train in at all differs between them, and the parity claim is
    # only auditable if that share is reported
    done_trials = [t for t in study.get_trials(deepcopy=False)
                   if t.state == TrialState.COMPLETE]
    diverged = sum(1 for t in done_trials if t.user_attrs.get('scored_at_chance'))
    at_chance = sum(1 for t in done_trials
                    if hpo_select.near_chance(t.value,
                                              t.user_attrs.get('scored_at_chance')))
    if done_trials:
        print(f'[{arch}] {at_chance}/{len(done_trials)} completed trials failed to beat '
              f'chance (val AUC <= {hpo_select.CHANCE_AUC}), of which {diverged} '
              f'diverged or scored undefined', flush=True)

    run_final(arch, study, cfg, ctx, out_dir, progress)


def _make_val_logger(cfg, progress, prefix):
    """Heartbeat + stdout at each validation, thinned so the log stays readable."""
    state = {'best': -np.inf}

    def on_val(step, total_steps, n_val, train_loss, val_loss, val_auc, elapsed):
        frac = step / total_steps
        progress.update(frac=frac, step=step, total_steps=total_steps,
                        validations=n_val, last_val_auc=val_auc)
        improved = val_auc > state['best']
        state['best'] = max(state['best'], val_auc)
        if improved or n_val % cfg.log_every == 0 or step >= total_steps:
            print(f'{prefix} step {step}/{total_steps} ({100 * frac:.0f}%) '
                  f'train_loss {train_loss:.4f} val_loss {val_loss:.4f} '
                  f'val_auc {val_auc:.4f}{" *" if improved else ""} '
                  f'({elapsed:.0f}s)', flush=True)

    return on_val


def _best_value(study):
    from optuna.trial import TrialState
    vals = [t.value for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if t.value is not None]
    return max(vals) if vals else None


def candidate_trials(study, k, cfg=None):
    """The k best distinct configurations of a finished study.

    Trials that diverged or never scored a finite AUC are excluded outright rather than
    relied on to rank low: on a target where most of the space is unstable they could
    otherwise fill the shortlist, and a configuration that has to be stopped before it
    blows up is not one the final stage can reproduce.

    Distinct: TPE concentrates, so the tail of a search re-proposes configurations it
    has already tried, and retraining the same one twice would spend the final budget
    without widening the choice.
    """
    from optuna.trial import TrialState

    trials = [t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
              if t.value is not None and not t.user_attrs.get('scored_at_chance')]
    picked, seen = [], set()
    for t in sorted(trials, key=lambda t: -t.value):
        params = normalize_params(t.params, cfg)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        picked.append((t, params))
        if len(picked) >= k:
            break
    return picked


def run_final(arch, study, cfg, ctx, out_dir, progress):
    """Retrain the shortlist on the held-out folds -- this is the ground truth.

    The search ranks configurations on one run each, scored on 25 images per class; the
    top of that ranking is decided by a margin far smaller than the noise in it. So the
    shortlist, not the argmax, is what leaves the search: the top `--final-topk`
    configurations are each retrained on every fold in `--final-folds`, none of which
    they were tuned on, and the one with the highest validation AUC *averaged over those
    folds* wins. Its mean test AUC over the same runs is the number we report.

    Every candidate sees the same folds with the same seeds, so the comparison between
    them is paired, and so is the comparison between architectures.
    """
    cands = candidate_trials(study, cfg.final_topk, cfg)
    if not cands:
        # nothing this architecture can be reported on: either the search produced no
        # completed trial, or every one of them diverged. Leaving the cell empty is the
        # honest outcome -- the per-architecture at-chance share above says which
        print(f'[{arch}] no usable trial to retrain, skipping the final stage',
              flush=True)
        return
    print(f'[{arch}] shortlist of {len(cands)} by search val_auc: '
          + ', '.join(f'trial {t.number} ({t.value:.4f})' for t, _ in cands), flush=True)

    path = os.path.join(out_dir, 'final_runs.csv')
    done = hpo_select.done_keys(hpo_select.read_final_runs(path, arch))
    # fold-major: every candidate is run on a fold before the next fold starts, so an
    # interrupted final stage still holds a complete, paired comparison over the folds
    # it did reach, rather than one candidate measured everywhere and the rest nowhere
    plan = [(t.number, params, fold, cfg.seed + 1000 + i)
            for fold in cfg.final_folds for i in range(cfg.final_seeds)
            for t, params in cands]
    for n, (trial_no, params, fold, seed) in enumerate(plan):
        if (trial_no, fold, seed) in done:
            continue
        progress.update(stage='final', arch=arch, trial=n, trials_total=len(plan),
                        frac=0.0, params=params, trial_started=time.time())
        on_val = _make_val_logger(cfg, progress,
                                  f'[{arch}] final t{trial_no}f{fold}s{seed}')

        res, net, criterion, val_scores = run_one(cfg, arch, params, seed, ctx,
                                                  fold=fold, on_val=on_val)
        progress.update(stage='final-test')
        test_res, test_scores = test_metrics(net, ctx, cfg, criterion)
        res.update(test_res)
        # these are the runs the reported number is computed from, so their predictions
        # are kept unless asked otherwise: 20 per architecture, against 200 trials
        if cfg.save_scores != 'none':
            write_scores(out_dir, arch, f'final_t{trial_no}f{fold}s{seed}',
                         val=val_scores, test=test_scores)
        del net
        if ctx['device'].type == 'cuda':
            torch.cuda.empty_cache()

        row = {'target': cfg.target, 'arch': arch, 'trial': trial_no, 'fold': fold,
               'seed': seed, 'train_sha256': ctx['train_sha'].get(f'train_fold{fold}', ''),
               **params, **res,
               'finished': datetime.now().isoformat(timespec='seconds')}
        _append_csv(path, row)
        print(f'[{arch}] final trial {trial_no} fold {fold} seed {seed}: '
              f'val_auc {res["val_auc"]:.4f} test_auc {res["test_auc"]:.4f}', flush=True)

    # selected from everything on disk for this architecture, not just the rows this
    # invocation produced, so a resumed run reaches the same decision as an unbroken one
    choice = hpo_select.select(hpo_select.read_final_runs(path, arch))
    if choice is None:
        return
    print(hpo_select.format_choice(arch, choice), flush=True)
    sd = f'+- {choice["test_sd"]:.4f}' if choice['test_sd'] is not None else ''
    print(f'[{arch}] ground truth test AUC {choice["test_auc"]:.4f} {sd} '
          f'(trial {choice["trial"]}, {choice["n"]} runs on folds {choice["folds"]})',
          flush=True)
    _record_choice(out_dir, cfg.target, arch, choice)


def _record_choice(out_dir, target, arch, choice):
    """Keep each architecture's decision next to the runs it was made from."""
    path = os.path.join(out_dir, 'selection.json')
    all_choices = _read_json_or(path, {})
    all_choices[arch] = {'target': target, **{k: v for k, v in choice.items()
                                              if k != 'candidates'},
                         'candidates': choice['candidates'],
                         'decided': datetime.now().isoformat(timespec='seconds')}
    atomic_write_json(path, all_choices)


def _append_csv(path, row):
    import csv
    # per-class vectors go in as JSON, not as a Python repr: `None` has to come back out
    # of the file as null for anything that is not Python to read the column
    row = {k: json.dumps(v) if isinstance(v, (list, tuple)) else v
           for k, v in row.items()}
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', required=True, help='MedMNIST target, e.g. dermamnist')
    p.add_argument('--archs', nargs='*', default=list(ARCHS),
                   help=f'architectures to benchmark (default: all {len(ARCHS)})')
    p.add_argument('--out', default='results/hpo', help='output root')
    p.add_argument('--space', choices=['core', 'wide'], default='wide',
                   help="'core' searches optimizer, learning rate, momentum, weight "
                        "decay, batch size, head lr multiplier and augmentation; "
                        "'wide' (default) adds dropout, label smoothing, warmup and "
                        'the schedule')
    p.add_argument('--trials', type=int, default=None,
                   help='Optuna trials per architecture (default: 50 core, 100 wide). '
                        'Not 200: on the pilot, doubling 100 to 200 bought +0.0083 '
                        'validation AUC but only +0.0034 on test, and among the top 20 '
                        'trials by validation AUC the correlation with test AUC is '
                        'tau 0.069 -- the search cannot rank its own plateau, so the '
                        'budget does more good in the final stage')
    p.add_argument('--final-topk', type=int, default=10,
                   help='how many of the best configurations are retrained in the final '
                        'stage. The search ranks them on one run each, by a margin much '
                        'smaller than the noise in it, so the shortlist rather than the '
                        'argmax leaves the search: all k are rerun on the held-out '
                        'folds and the winner is the one with the best validation AUC '
                        'averaged over them (1 reproduces the old top-1 rule). On the '
                        'pilot the best-on-test configuration sat at median rank 10 of '
                        'the top 20 by validation AUC -- a shortlist of 5 caught it in '
                        '44%% of architectures, 10 in 56%%')
    p.add_argument('--final-seeds', type=int, default=4,
                   help='seeds per fold per candidate in the final stage. With four '
                        'folds this puts 16 runs behind the reported number, which is '
                        'what the ground truth needs to be worth correlating against: '
                        'at 4 runs the reliability of the per-architecture means is '
                        '0.857, capping any transferability metric at Kendall tau 0.75, '
                        'and at 16 it is 0.960 for a ceiling of 0.87. Seeds rather than '
                        'more folds because only five folds exist and one is spent on '
                        'the search')
    p.add_argument('--final-folds', type=int, nargs='*', default=[2, 3, 4, 5],
                   help='folds of the fine-tuning split the shortlist is rerun on; the '
                        'ground truth averages over folds x seeds. The default excludes '
                        'the search fold (--fold), so no fold both selects the '
                        'configuration and contributes to the reported number')
    p.add_argument('--validations-per-trial', type=int, default=40,
                   help='how many times a trial is validated over its own budget; the '
                        'step interval follows from it, so early stopping and the '
                        'learning curves mean the same thing at every batch size')
    p.add_argument('--val-every-steps', type=int, default=0,
                   help='override: validate every N optimizer steps instead')
    p.add_argument('--patience', type=int, default=10,
                   help='early stopping patience, in validations (default 10 of 40, '
                        'i.e. a quarter of the budget with no improvement)')
    p.add_argument('--pruner', choices=['median', 'none'], default='none',
                   help="'none' (default) runs every trial to completion, which keeps "
                        'full learning curves so a pruning rule can be simulated offline '
                        "instead of being baked in; 'median' is cheaper but the curves "
                        'are then truncated by a rule that cannot be revisited')
    p.add_argument('--log-every', type=int, default=20,
                   help='print every Nth validation (improvements always print)')
    p.add_argument('--fold', type=int, default=1,
                   help='fold of the fine-tuning split used for the search (1-5)')
    p.add_argument('--bundle-dir', default=None,
                   help='directory holding <target>_splits_224.npz, the frozen train '
                        f'and validation subsets (default {splits.bundle_dir()})')
    p.add_argument('--no-build', dest='build_missing', action='store_false',
                   help='do not fetch or build anything that is missing; check for it '
                        'and fail instead. The default fetches <target>_224.npz from '
                        'MedMNIST when it is absent (URL and MD5 come from medmnist.INFO '
                        'and the MD5 is verified) and cuts the bundle from it. The '
                        'manifest is tracked in git and is reused, never redrawn, so a '
                        'bundle built here holds the same images as one copied in -- '
                        'which is checked, since every part is re-hashed against the '
                        'manifest as it is read')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-eval', choices=['best', 'all', 'none'], default='all',
                   help="score the test set for every trial (default), the running-best "
                        'trial, or no trial; final runs are always scored. Nothing in '
                        'the protocol reads these -- they are recorded so the '
                        'validation-to-test relationship can be studied afterwards, '
                        "which needs every trial: 'best' records the improving "
                        'subsequence of the search, a biased sample for that question. '
                        "On the large test splits (tissuemnist, organamnist) 'best' is "
                        'much cheaper')
    p.add_argument('--save-scores', choices=['final', 'all', 'none'], default='all',
                   help='keep the raw predictions of the best checkpoint, so a metric '
                        'can be chosen after the run rather than before it: for every '
                        'trial as well as the final runs (default), for the final runs '
                        'the reported number comes from, or not at all. Per-class AUCs '
                        'are recorded either way and already settle any reweighting of '
                        'the one-vs-rest AUCs; these files are what a metric that is '
                        'not a function of those needs -- average precision, '
                        'calibration, or a paired test between two architectures on the '
                        "same images. 'all' is the default because the trials are what "
                        'the validation-to-test relationship has to be studied on, and '
                        'nothing of a finished run survives except what was written '
                        'while it was alive: across all 11 targets it costs about '
                        '8.6 GB against 5.3 GB')
    p.add_argument('--test-batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader workers; 0 keeps augmentation in the main process '
                        '(slower, but fully seed-reproducible)')
    p.add_argument('--worker-context', choices=['auto', 'fork', 'forkserver', 'spawn'],
                   default='auto',
                   help="worker start method; 'auto' uses forkserver on CUDA because "
                        'forking after CUDA init deadlocks on some platforms')
    p.add_argument('--deterministic', action='store_true',
                   help='disable cudnn autotuning for bit-reproducible runs (slower)')
    p.add_argument('--estimate-only', action='store_true',
                   help='probe per-architecture cost, print a runtime projection, exit')
    p.add_argument('--force', action='store_true', help='ignore a stale lock file')
    # search space. The published grid (finetuned_AUCs/) was lr in {1e-4..1e-1},
    # momentum in {0, 0.9}, wd in {0..1e-2}, batch size in {32..256}; its selected
    # optima piled up against the top of the lr and batch-size ranges and the bottom
    # of the weight-decay range, so the ranges below extend past it on both sides.
    p.add_argument('--optimizers', nargs='*', default=['sgd', 'adamw'],
                   choices=['sgd', 'adamw'])
    p.add_argument('--sgd-lr-min', type=float, default=1e-3)
    p.add_argument('--sgd-lr-max', type=float, default=1e-1,
                   help='the pilot put the median validation AUC of every SGD trial '
                        'above lr 0.18 at 0.51 -- that is the divergence floor, and it '
                        'is arithmetic rather than a property of one target, so the top '
                        'decade is cut (was 1.0)')
    p.add_argument('--adamw-lr-min', type=float, default=1e-5)
    p.add_argument('--adamw-lr-max', type=float, default=1e-2)
    p.add_argument('--momentums', type=float, nargs='*', default=[0.0, 0.9, 0.99])
    p.add_argument('--wd-min', type=float, default=1e-6,
                   help='the low end stands in for "no weight decay". The pilot could '
                        'not tell 1e-8, 1e-7 and 1e-6 apart -- all three bins had the '
                        'same distribution of scores -- so the floor comes up (was 1e-8)')
    p.add_argument('--wd-max', type=float, default=1e-1)
    p.add_argument('--batch-sizes', type=int, nargs='*', default=[16, 32, 64, 128],
                   help='32 was the modal winner at 1.40x lift and it was also the '
                        'floor, so the arm is extended down; 128 and 256 came in at '
                        '0.26x and 0.65x')
    p.add_argument('--head-lr-mults', type=float, nargs='*', default=[1.0],
                   help='fixed rather than searched: among the 90 best pilot '
                        'configurations 1 and 10 came in at 1.01x and 0.99x lift, which '
                        'is a dimension doing no work. Pass both to search it again')
    p.add_argument('--augs', nargs='*', default=['none', 'light', 'heavy'],
                   choices=['none', 'light', 'heavy'])
    p.add_argument('--train-epochs', type=int, nargs='*',
                   default=[25, 50, 100, 200, 400],
                   help='training-length arms, in passes over the training split. The '
                        'schedule completes over whichever arm a trial draws. Left as '
                        'it is: 50 and 400 carry identical lift among the pilot winners, '
                        'which is two convergence regimes rather than a ceiling pressing '
                        'on the search')
    # --space wide only
    p.add_argument('--dropouts', type=float, nargs='*', default=[0.0, 0.5])
    p.add_argument('--label-smoothings', type=float, nargs='*', default=[0.0, 0.1])
    p.add_argument('--warmups', type=int, nargs='*', default=[0, 1, 3, 5],
                   help='3 was both the top arm and the modal winner at 1.34x lift, so '
                        'the arm is extended (was 0 1 3)')
    p.add_argument('--schedulers', nargs='*', default=['cosine'],
                   choices=['cosine', 'constant'],
                   help='fixed rather than searched: cosine and constant came in at '
                        '1.01x and 0.99x lift on the pilot. Pass both to search it again')
    p.add_argument('--startup-trials', type=int, default=None,
                   help='random trials before TPE and before pruning kicks in '
                        '(default: a quarter of --trials, capped at 25, so the default '
                        'budget of 100 gives exactly 25). Identical across '
                        'architectures, so every architecture is handed the same random '
                        'configurations to start from')
    p.add_argument('--prune-warmup', type=int, default=20,
                   help='validations a trial is safe from pruning (default 20 = 500 '
                        'optimizer steps)')

    cfg = p.parse_args(argv)
    if cfg.trials is None:
        cfg.trials = 50 if cfg.space == 'core' else 100
    if cfg.startup_trials is None:
        # NB: pin --startup-trials explicitly if you want the first N trials of a long
        # run to be exactly what an N-trial run would have produced
        cfg.startup_trials = min(25, max(8, cfg.trials // 4))
    if cfg.space == 'core':
        cfg.augs = [a for a in cfg.augs if a != 'none']
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    unknown = [a for a in cfg.archs if a not in ARCHS and not hasattr(torchvision.models, a)]
    if unknown:
        raise SystemExit(f'unknown architectures: {unknown}')

    out_dir = os.path.join(cfg.out, cfg.target)
    os.makedirs(out_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = not cfg.deterministic
    torch.backends.cudnn.deterministic = cfg.deterministic
    # Single precision has to mean single precision. PyTorch leaves cudnn's TF32 path on
    # by default, which computes convolutions with a 10-bit mantissa; measured here that
    # shifts resnet's logits by 2.2e-2 relative and convnext's by 4.3e-4, against a
    # rerun floor of exactly zero -- a per-architecture perturbation of the quantity this
    # benchmark exists to measure. MLPerf's training rules require a reference
    # convergence point to be FP32 or BF16 for the same reason. The matmul flag already
    # defaults to False; it is pinned so the protocol does not rest on that default.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    info = INFO[cfg.target]
    cfg.task = info['task']
    train_cache, val_cache, testset_cache, part_sha = {}, {}, {}, {}

    # Everything the run reads, present and verified before it spends anything: the
    # bundle for the training folds and the validation split, and <target>_224.npz for
    # the test split, which the bundle does not carry. Whatever is missing is downloaded
    # or built here rather than reported -- the manifest is tracked, so building reuses
    # it and cannot draw different images, and load_part re-hashes every part against it
    # as it is read. --no-build turns this back into a check that fails.
    #
    # A missing fold, or a test split that is not there, has to surface now: with
    # --test-eval best or none nothing would touch the test file until the final stage,
    # and _test_size reports a missing file as zero test images rather than raising, so
    # even the estimate would look fine.
    if cfg.build_missing:
        import make_splits
        try:
            make_splits.ensure(cfg.target, dest=cfg.bundle_dir)
        except Exception as e:
            raise SystemExit(f'{cfg.target}: could not prepare the data -- '
                             f'{type(e).__name__}: {e}')

    folds_used = sorted({cfg.fold, *cfg.final_folds})
    manifest = splits.read_manifest(cfg.target)
    needed = splits.parts_of(manifest, folds_used)
    bundle = splits.check_bundle(cfg.target, needed, cfg.bundle_dir)
    test_npz = os.path.join(MEDMNIST_ROOT, f'{cfg.target}_224.npz')
    if not os.path.exists(test_npz):
        raise SystemExit(
            f'{test_npz} not found, and the run needs it for the test split -- the '
            f'bundles hold only the training folds and the validation split.\n'
            f'Point MEDMNIST_ROOT at the directory holding <target>_224.npz, or drop '
            f'--no-build and it will be fetched.')
    # what every part this run will touch is supposed to be, from the tracked manifest.
    # The folds of the final stage decide the reported number, so recording only the
    # search fold would leave most of the ground truth traceable to nothing; each is
    # re-hashed against this when it is actually loaded.
    expected_sha = splits.part_sha256(cfg.target, needed)

    def split(fold):
        if fold not in train_cache:
            train_cache[fold], entry = load_train(cfg.target, fold, cfg.bundle_dir)
            part_sha[f'train_fold{fold}'] = entry['images_sha256']
        return train_cache[fold]

    def valsplit(fold):
        """The validation part of one fold."""
        if fold not in val_cache:
            val_cache[fold], entry = load_val(cfg.target, fold, cfg.bundle_dir)
            part_sha[f'val_fold{fold}'] = entry['images_sha256']
            val_cache[(fold, 'entry')] = entry
        return val_cache[fold]

    def testset():
        if 'ds' not in testset_cache:
            testset_cache['ds'] = load_test(cfg.target)
        return testset_cache['ds']

    trainset = split(cfg.fold)
    valset = valsplit(cfg.fold)
    val_entry = val_cache[(cfg.fold, 'entry')]
    ctx = {'device': device, 'task': info['task'],
           'n_classes': len(info['label']), 'split': split, 'valsplit': valsplit,
           'testset': testset, 'train_sha': part_sha}
    mp_context = resolve_worker_context(cfg.worker_context, cfg.workers, device)
    ctx['loaders'] = Loaders(cfg, ctx, mp_context)

    n_test = _test_size(cfg.target)
    meta_path = os.path.join(out_dir, 'meta.json')
    # a sharded run (one process per GPU, each with --archs) must not drop the other
    # shard's architectures from the picture the watcher builds
    previous = _read_json_or(meta_path, {})
    archs = list(previous.get('archs', [])) if previous.get('target') == cfg.target else []
    archs += [a for a in cfg.archs if a not in archs]
    archs.sort(key=lambda a: list(ARCHS).index(a) if a in ARCHS else len(ARCHS))

    # resuming into a study whose trials were trained or selected on different images
    # would make the comparison between them meaningless, and it would not be visible
    # anywhere in the output
    if previous.get('target') == cfg.target:
        was_parts = previous.get('split_sha256_by_part',
                                 previous.get('train_sha256_by_part', {}))
        checks = [(f'training fold {cfg.fold}', previous.get('train_sha256'),
                   part_sha[f'train_fold{cfg.fold}'])]
        # every part this run would touch, not just the one it searches on: a bundle
        # whose fold 3 differs produces a different reported number and nothing else
        # would notice. Validation is in here too now that it is drawn per fold.
        checks += [(p.replace('_fold', ' fold '), was_parts[p], expected_sha[p])
                   for p in needed if p in was_parts]
        for what, was, now in checks:
            if was != now:
                raise SystemExit(
                    f'{out_dir} holds trials trained against {what} '
                    f'{was[:12] if was else "(unrecorded -- predating data/splits)"}, '
                    f'but this run would use {now[:12]}. Comparing them would be '
                    f'meaningless. Point --bundle-dir at the subsets those trials '
                    f'used, or move {out_dir} aside and start fresh.')

    final_runs = cfg.final_topk * len(cfg.final_folds) * cfg.final_seeds
    meta = {'target': cfg.target, 'archs': archs, 'trials': cfg.trials,
            'space': cfg.space, 'optimizers': cfg.optimizers,
            'final_runs': final_runs, 'final_topk': cfg.final_topk,
            'final_folds': cfg.final_folds, 'seeds_per_fold': cfg.final_seeds,
            'train_epochs': cfg.train_epochs, 'val_every_steps': cfg.val_every_steps,
            'validations_per_trial': cfg.validations_per_trial, 'pruner': cfg.pruner,
            'batch_sizes': cfg.batch_sizes, 'patience': cfg.patience,
            'startup_trials': cfg.startup_trials, 'fold': cfg.fold,
            'bundle': bundle,
            'val_sha256': val_entry['images_sha256'],
            'val_class_counts': val_entry['class_counts'],
            'train_sha256': part_sha[f'train_fold{cfg.fold}'],
            'split_sha256_by_part': expected_sha,
            'train_sha256_by_part': {p: h for p, h in expected_sha.items()
                                     if p.startswith('train_')},
            'train_class_counts': splits.class_counts(trainset.targets, cfg.target),
            'n_train': len(trainset), 'n_val': len(valset), 'n_test': n_test,
            'test_eval': cfg.test_eval, 'save_scores': cfg.save_scores,
            'device': str(device), 'precision': 'fp32',
            'workers': cfg.workers,
            'worker_context': mp_context.get_start_method() if mp_context else 'none',
            'started': previous.get('started', time.time()), 'host': socket.gethostname(),
            'ran_on': _ran_on(previous, device, cfg.target),
            'cmd': ' '.join(sys.argv)}
    atomic_write_json(meta_path, meta)

    print(f'target {cfg.target}: {len(trainset)} train / {len(valset)} val / '
          f'{n_test} test images, {ctx["n_classes"]} classes, task {info["task"]!r}')
    print(f'  train fold {cfg.fold} (sha256 '
          f'{part_sha[f"train_fold{cfg.fold}"][:12]}): '
          f'{describe_labels(trainset, cfg.target)}')
    print(f'  val fold {cfg.fold} (n_classes*{splits.VAL_PER_CLASS} budget, sha256 '
          f'{val_entry["images_sha256"][:12]}): '
          f'{describe_labels(valset, cfg.target)}')
    print(f'  train and val are each stratified to the class mix of the official split '
          f'they are drawn from; folds {folds_used} each have their own of both')
    print(f'device {device}, fp32, {len(cfg.archs)} architectures, '
          f'{cfg.trials} trials + {final_runs} final runs each (search on fold '
          f'{cfg.fold}; top {cfg.final_topk} configurations rerun on folds '
          f'{cfg.final_folds} x {cfg.final_seeds} seeds, the winner picked on mean '
          f'validation AUC over them)')
    if cfg.fold in cfg.final_folds:
        print(f'[warn] fold {cfg.fold} both selects the configuration and contributes '
              f'to the reported AUC, so that fold is tuned on its own training draw')
    print(f'budget: train_epochs arms {cfg.train_epochs}, '
          f'{cfg.validations_per_trial} validations per trial, early stop after '
          f'{cfg.patience} without improvement, pruner {cfg.pruner}')
    print(f"search space '{cfg.space}': optimizer {cfg.optimizers}, "
          f'lr {cfg.sgd_lr_min:g}-{cfg.sgd_lr_max:g} (sgd) / '
          f'{cfg.adamw_lr_min:g}-{cfg.adamw_lr_max:g} (adamw), '
          f'wd {cfg.wd_min:g}-{cfg.wd_max:g}, batch {cfg.batch_sizes}, '
          f'head lr x{cfg.head_lr_mults}, aug {cfg.augs}'
          + (f', dropout {cfg.dropouts}, smoothing {cfg.label_smoothings}, '
             f'warmup {cfg.warmups}, schedule {cfg.schedulers}'
             if cfg.space == 'wide' else ''), flush=True)

    cost_path = os.path.join(out_dir, 'arch_cost.json')
    costs = {}
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            costs = json.load(f)
    missing = [a for a in cfg.archs if a not in costs]
    if missing:
        print(f'probing per-architecture cost for {len(missing)} architectures ...', flush=True)
        for a in missing:
            costs[a] = probe_arch_cost(a, ctx, cfg)
            print(f'  {a:<13} {costs[a]["train_s_per_img"] * 1e3:6.2f} ms/img train, '
                  f'{costs[a]["eval_s_per_img"] * 1e3:6.2f} ms/img eval, '
                  f'{costs[a]["build_s"]:5.1f} s/model', flush=True)
        atomic_write_json(cost_path, costs)
        ctx['loaders'].drop(('probe',))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hpo_watch

    if cfg.estimate_only:
        print()
        print(hpo_watch.render(hpo_watch.collect(cfg.out, cfg.target)))
        return

    lock = os.path.join(out_dir, 'run.lock')
    acquire_lock(lock, force=cfg.force)
    enable_wal(os.path.join(out_dir, 'optuna.db'))
    progress = Progress(os.path.join(out_dir, 'progress.json'), cfg.target, cfg.archs)
    print()
    print(hpo_watch.render(hpo_watch.collect(cfg.out, cfg.target)), flush=True)

    try:
        for arch in cfg.archs:
            print(f'\n=== {arch} ({cfg.target}) ===', flush=True)
            run_arch(arch, cfg, ctx, out_dir, progress)
            progress.update(stage='between')
            print(hpo_watch.render(hpo_watch.collect(cfg.out, cfg.target)), flush=True)
        progress.update(stage='done')
    finally:
        if os.path.exists(lock):
            os.remove(lock)

    print(f'\nfinished {cfg.target}. collect results with:\n'
          f'  python src/hpo_collect.py --out {cfg.out}', flush=True)


def _ran_on(previous, device, target_flag):
    """Every distinct host and GPU this target has been trained on, oldest first.

    A target does not finish in one allocation: the job resubmits itself, and the next
    link can land on another node. 'host' is only ever the latest one, so on its own it
    misdescribes any run that took more than one job.

    What the comparison between architectures rests on is that they all met the same
    hardware, and that has to be recorded while it happens or it cannot be checked at all
    afterwards. The card matters more than the node: TF32 is off, so arithmetic does not
    vary between them, but memory does, and a card too small for the largest batch turns
    those trials into failures -- narrowing the search space for the heavy architectures
    and not for the light ones.
    """
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        gpu = f'{props.name} ({props.total_memory / 2**30:.0f} GiB)'
    else:
        gpu = 'cpu'
    entry = {'host': socket.gethostname(), 'gpu': gpu}
    seen = (list(previous.get('ran_on', []))
            if previous.get('target') == target_flag else [])
    if entry not in seen:
        seen.append(entry)
    return seen


def _test_size(target_flag):
    """Number of test images, read from the npz header without materialising it."""
    try:
        d = np.load(os.path.join(MEDMNIST_ROOT, f'{target_flag}_224.npz'))
        with d.zip.open('test_labels.npy') as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
        return int(shape[0])
    except Exception:
        return 0


if __name__ == '__main__':
    main()
