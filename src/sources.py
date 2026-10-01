#!/usr/bin/env python
# coding: utf-8
"""The pretrained source checkpoints the dataset-transfer benchmark ranks.

Every source is the same resnet50. Only the weights differ, which is what makes this
benchmark well posed in a way the architecture one is not: a statistic computed on the
network -- a gradient norm at a named layer, a feature covariance -- is comparable across
sources because the layers are literally the same layers. It is also the resnet50 of the
architecture benchmark, with the same ImageNet weights, so the two benchmarks share a
point: the 'imagenet' source here is the 'resnet' architecture there.

Three kinds of source:

    imagenet        torchvision's ImageNet-1k weights (IMAGENET1K_V1, as in
                    src/hpo_finetune.py)
    radimagenet     RadImageNet's official PyTorch resnet50 (backbone only, no head)
    medmnist        pretrained on every MedMNIST dataset except this target, by
                    src/pretrain_loo.py --target
    <dataset>       pretrained on that one MedMNIST dataset, by src/pretrain_loo.py --only

`source == target` is excluded by `for_target`: a model pretrained on the target's own
training data is not a transfer setting.

The checkpoints live under models/ in the repository, where src/pretrain_loo.py writes
them; MODELS_ROOT overrides that:

    models/loo_resnet50/<target>.pt
    models/single_resnet50/<dataset>.pt
    models/radimagenet/resnet50_torch.pt     as downloaded from the RadImageNet release

Checkpoints are loaded with strict=False, because the classifier head never matches and
is replaced anyway. That is also how a wrong path fails silently, so `load_backbone`
refuses a checkpoint that does not cover the backbone and says what it did cover.
"""

import os
import re

import torch
import torchvision
from medmnist import INFO

#: the one architecture every source shares
BACKBONE = 'resnet50'

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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def root():
    return os.environ.get('MODELS_ROOT') or os.path.join(_REPO, 'models')


#: sources whose results are taken from the architecture benchmark instead of trained:
#: 'imagenet' is that benchmark's 'resnet' run exactly. src/hpo_sources_reuse.py copies it.
REUSED = {'imagenet': 'resnet'}


def for_target(target, sources=None):
    """The sources that may be ranked on `target`, excluding self-transfer."""
    return [s for s in (sources or SOURCES) if s != target]


def to_train(target):
    """The sources of `target` the dataset-transfer benchmark has to train itself."""
    return [s for s in for_target(target) if s not in REUSED]


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


#: input normalisation each source was pretrained with. src/pretrain_loo.py trains on
#: mean 0.5 / std 0.5, and RadImageNet's PyTorch example scales to [-1, 1] by
#: (x - 127.5) * 2 / 255, which is the same thing; only torchvision's ImageNet weights
#: expect ImageNet statistics. Feeding a source inputs normalised differently from its
#: pretraining handicaps it, and the sources are what this benchmark compares.
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
MEDMNIST_NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def normalization(source):
    """(mean, std) the given source expects, as three-channel lists."""
    return IMAGENET_NORM if source == 'imagenet' else MEDMNIST_NORM


def channel_order(source):
    """Input channel order the source expects, as indices into RGB; None for RGB.

    RadImageNet's own fine-tuning code (pytorch_example.ipynb) reads images with
    cv2.imread, which returns BGR, and never converts them, so its inputs are BGR.
    Greyscale targets, replicated to three identical channels, are unaffected.
    """
    return [2, 1, 0] if source == 'radimagenet' else None


class ReorderChannels:
    """Permute the channels of a CHW image tensor. A class in an importable module
    rather than a lambda, so a transform holding it pickles into DataLoader workers."""

    def __init__(self, order):
        self.order = list(order)

    def __call__(self, x):
        return x[self.order]


def checkpoint_path(source, target):
    if source == 'imagenet':
        return None
    if source == 'radimagenet':
        return os.path.join(root(), 'radimagenet', 'resnet50_torch.pt')
    if source == 'medmnist':
        return os.path.join(root(), 'loo_resnet50', f'{target}.pt')
    return os.path.join(root(), 'single_resnet50', f'{source}.pt')


#: RadImageNet saves nn.Sequential(*resnet50.children()[:9]), so its keys are positions
#: in that list rather than torchvision's names
_SEQUENTIAL = {'0': 'conv1', '1': 'bn1', '4': 'layer1', '5': 'layer2',
               '6': 'layer3', '7': 'layer4'}


def _torchvision_keys(state):
    out = {}
    for k, v in state.items():
        k = re.sub(r'^(module\.|backbone\.)+', '', k)
        head, _, rest = k.partition('.')
        out[f'{_SEQUENTIAL[head]}.{rest}' if head in _SEQUENTIAL else k] = v
    return out


def load_backbone(source, target, min_coverage=0.99):
    """A resnet50 carrying `source`'s weights, with its original classifier still on.

    Returns (net, coverage). `coverage` is the fraction of the backbone's parameter
    tensors the checkpoint actually supplied, ignoring the classifier -- strict=False
    hides a path or key-prefix mistake as a network that quietly kept its random
    initialisation, and that is indistinguishable from a real result afterwards.

    `net.source_head` says whether the classifier is the source's own. RadImageNet ships
    without one, and anything that reads source predictions -- LEEP -- has to skip it
    rather than read a random head.
    """
    ctor = getattr(torchvision.models, BACKBONE)
    if source == 'imagenet':
        net = ctor(weights='IMAGENET1K_V1')
        net.source_head = True
        return net, 1.0

    path = checkpoint_path(source, target)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{source} for {target}: no checkpoint at {path}')
    blob = torch.load(path, map_location='cpu')
    state = blob['net'] if isinstance(blob, dict) and 'net' in blob else blob
    state = _torchvision_keys(state)

    has_head = 'fc.weight' in state
    if has_head and state['fc.weight'].shape[0] != n_source_classes(source, target):
        raise RuntimeError(f'{source} for {target}: head has {state["fc.weight"].shape[0]} '
                           f'classes, expected {n_source_classes(source, target)}')
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
    net.source_head = has_head
    return net, coverage

