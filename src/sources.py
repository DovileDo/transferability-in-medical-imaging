#!/usr/bin/env python
# coding: utf-8
"""The pretrained source checkpoints the dataset-transfer benchmark ranks.

Every source is the same resnet18. Only the weights differ, which is what makes this
benchmark well posed in a way the architecture one is not: a statistic computed on the
network -- a gradient norm at a named layer, a feature covariance -- is comparable across
sources because the layers are literally the same layers.

Three kinds of source:

    imagenet        torchvision's ImageNet-1k weights
    radimagenet     RadImageNet, 165 classes
    medmnist        pretrained on every MedMNIST dataset except this target (the count
                    of source classes therefore depends on the target)
    <dataset>       pretrained on that one MedMNIST dataset

`source == target` is excluded by `for_target`: a model pretrained on the target's own
training data is not a transfer setting, and results/AUCs_dataset.csv leaves that cell
empty for the same reason.

Checkpoints are loaded with strict=False, because the classifier head never matches and
is replaced anyway. That is also how a wrong path fails silently, so `load_backbone`
refuses a checkpoint that does not cover the backbone and says what it did cover.
"""

import os

import torch
import torchvision
from medmnist import INFO

#: root of the shared store; SHARE_ROOT overrides it
DEFAULT_ROOT = '/home/doju/medmnist_share'

#: the one architecture every source shares. The checkpoints in the store are resnet18
#: -- 'resnet18_224_1.pth' and the leave-one-out '.pt' files -- so this is what they
#: load into. Moving the benchmark to resnet50 means re-pretraining every source at
#: resnet50 and changing this line; load_backbone will refuse a mismatched checkpoint
#: rather than quietly keep a random initialisation.
BACKBONE = 'resnet18'

#: the 12 MedMNIST datasets that were pretrained on individually
PRETRAINED = ['bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist',
              'organamnist', 'organcmnist', 'organsmnist', 'pathmnist',
              'pneumoniamnist', 'retinamnist', 'tissuemnist']

SOURCES = ['imagenet', 'radimagenet', 'medmnist'] + PRETRAINED

#: source classes of the leave-one-out models. The pretraining pooled the label sets of
#: every other dataset, so the count depends on which target was held out -- and the
#: organ datasets and pneumoniamnist share their labels with datasets that remain, which
#: is why they do not lose any.
_LOO_FULL = 69
_LOO_SHARED = ('pneumoniamnist', 'organamnist', 'organcmnist', 'organsmnist')


def root():
    return os.environ.get('SHARE_ROOT') or DEFAULT_ROOT


def for_target(target, sources=None):
    """The sources that may be ranked on `target`, excluding self-transfer."""
    return [s for s in (sources or SOURCES) if s != target]


def n_source_classes(source, target):
    if source == 'radimagenet':
        return 165
    if source == 'medmnist':
        if target == 'chestmnist':
            return 56
        if target in _LOO_SHARED:
            return _LOO_FULL
        return _LOO_FULL - len(INFO[target]['label'])
    return len(INFO[source]['label'])


#: input normalisation each source was pretrained with. src/leave-one-out-pretrain.py
#: trains on mean 0.5 / std 0.5, and the RadImageNet and MedMNIST baselines follow the
#: same convention; only torchvision's ImageNet weights expect ImageNet statistics. The
#: published src/fine-tuning.py branched on exactly this, and it matters here: feeding a
#: source inputs normalised differently from its pretraining handicaps it, and the
#: sources are what this benchmark compares.
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
MEDMNIST_NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def normalization(source):
    """(mean, std) the given source expects, as three-channel lists."""
    return IMAGENET_NORM if source == 'imagenet' else MEDMNIST_NORM


def checkpoint_path(source, target):
    if source == 'imagenet':
        return None
    if source in ('radimagenet', 'medmnist'):
        name = 'radimagenet' if source == 'radimagenet' else target
        return os.path.join(root(), 'models', 'doju_pre-trained_for', f'{name}.pt')
    return os.path.join(root(), 'models', 'doju_sim_pretrained', source,
                        'resnet18_224_1.pth')


def load_backbone(source, target, min_coverage=0.99):
    """A resnet18 carrying `source`'s weights, with its original classifier still on.

    Returns (net, coverage). `coverage` is the fraction of the backbone's parameter
    tensors the checkpoint actually supplied, ignoring the classifier -- strict=False
    hides a path or key-prefix mistake as a network that quietly kept its random
    initialisation, and that is indistinguishable from a real result afterwards.
    """
    ctor = getattr(torchvision.models, BACKBONE)
    if source == 'imagenet':
        return ctor(weights='IMAGENET1K_V1'), 1.0

    path = checkpoint_path(source, target)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{source} for {target}: no checkpoint at {path}')
    blob = torch.load(path, map_location='cpu')
    state = blob['net'] if isinstance(blob, dict) and 'net' in blob else blob
    if any(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}

    net = ctor(num_classes=n_source_classes(source, target))
    own = net.state_dict()
    backbone = [k for k in own if not k.startswith('fc.')]
    hit = sum(1 for k in backbone if k in state and state[k].shape == own[k].shape)
    coverage = hit / len(backbone)
    if coverage < min_coverage:
        raise RuntimeError(
            f'{source} for {target}: checkpoint covers {coverage:.0%} of the backbone '
            f'({hit}/{len(backbone)} tensors) -- wrong file, or keys named differently')
    net.load_state_dict(state, strict=False)
    return net, coverage
