#!/usr/bin/env python
# coding: utf-8
"""Dataset transfer under tuning: which pretraining source transfers best to a target.

Same protocol as src/hpo_finetune.py, with the axis swapped. There the architecture
varied and the weights were always ImageNet; here the architecture is always resnet50
and the *weights* vary across the sources in src/sources.py. Every part of the runner is
reused -- the search, the top-k final phase, resume, divergence handling, the CSVs -- so
the two benchmarks are directly comparable and there is one implementation to trust.

What is deliberately different, and why. All of it comes out of what the architecture
benchmark measured about itself:

  One training subset, not five.  Redrawing the training subset moves test AUC about a
  third as much as changing the random seed does, and pairing a metric to the fold its
  ground truth came from was worth nothing (median 0.000 over 320 metric-fold pairs). So
  the estimand here is the ranking on one fixed training set, search and final share
  fold 1, and the compute the folds were costing goes into seeds instead.

  Sixteen seeds, not four.  Seed noise is the dominant term: 0.645 AUC points against
  0.197 for the subset draw. At four seeds the architecture ranking reproduced at split
  half 0.65; sixteen seeds on one fold reaches 0.90 or better on every target, for the
  same number of runs the four-fold design spent.

  A search per source.  Selecting a configuration per source changed it in 94 of 99
  architecture-target pairs and moved test AUC by -2.65 to +5.53 points, so a shared
  recipe would be ranking the recipe's suitability as much as the source's.

  Ten candidates, not five.  The search ranking is close to uninformative inside its own
  top ten -- the configuration that eventually wins sits at rank 1 in 5 of 99 pairs and
  at ranks 6-10 in 55 of them. Usually that costs nothing, because those candidates are
  near-equivalent and a tie is broken differently; the median loss from a shortlist of
  five is 0.00 points. The tail is the reason: five is worse in 26 of 99 pairs and by up
  to 1.6 points, which is the size of the entire between-architecture spread on targets
  like organsmnist.

  Self-transfer excluded.  src/sources.py drops source == target.

  One runner call per source.  Each source expects the input normalisation it was
  pretrained with -- ImageNet statistics for torchvision's weights, mean 0.5 / std 0.5
  for everything else, as src/sources.py records and the published src/fine-tuning.py
  also branched on -- and RadImageNet expects BGR, since its own fine-tuning code feeds
  cv2.imread's output unconverted. src/hpo_finetune.py bakes the transform into the
  dataset when it loads, once, before it iterates architectures, because for
  architecture transfer every source is ImageNet and one normalisation is correct. So
  the sources are run one per call, which is the sharded mode the runner already
  supports through --archs, with the transform constants set for that source. It costs
  re-reading the bundle and the test split per source; feeding a source the wrong
  normalisation would cost it accuracy, in a comparison that is about the sources.

  One study per source, in its own directory: results/hpo_sources/<source>/<target>/.
  The architecture benchmark keeps every architecture of a target in one optuna.db, which
  is fine for one job per target. Here a target is thirteen resnet50 runs end to end --
  days to weeks -- so the sources of one target have to be splittable across jobs, and
  two jobs syncing back one SQLite file would overwrite each other's trials. Every
  downstream tool takes --out, so each source directory is read like a benchmark of its
  own, with the source in the 'arch' column.

Reliability is measured the way the estimand demands: split-half over seeds, which
src/hpo_reliability.py already computes with --split-on seed.

  ImageNet is not retrained.  The 'imagenet' source is the architecture benchmark's
  resnet50 run exactly -- same weights, normalisation, fold, search and seeds -- so
  src/hpo_sources_reuse.py copies that into results/hpo_sources/imagenet/ instead.

    python src/hpo_sources.py --target dermamnist
    python src/hpo_sources.py --target dermamnist --sources radimagenet medmnist

Everything else -- flags, outputs, resume -- is src/hpo_finetune.py's; see its --help.
"""

import argparse
import os
import sys

import torch.nn as nn
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo_finetune                                                  # noqa: E402
import sources                                                       # noqa: E402

#: the design above, as defaults. Each is overridable on the command line.
DEFAULTS = {'final_folds': [1], 'final_seeds': 16, 'final_topk': 10}


def build_model(source, n_classes, dropout=0.0):
    """sources.BACKBONE with `source`'s weights and a fresh head, in hpo_finetune's contract.

    hpo_finetune calls this with whatever it is iterating over, so the first argument is
    a source name here rather than an architecture. The target is read from the module
    global the wrapper sets, because the leave-one-out checkpoints and their class counts
    depend on which target is held out.
    """
    net, _ = sources.load_backbone(source, _TARGET)
    head = nn.Linear(net.fc.in_features, n_classes)
    net.fc = nn.Sequential(nn.Dropout(p=dropout), head) if dropout > 0 else head
    return net, (head if dropout > 0 else net.fc)


_TARGET = None
#: channel order of the source being run, set per source in main
_CHANNELS = None


def _in_source_order(make):
    """Wrap one of hpo_finetune's transform builders to put channels in the order the
    current source expects, right after the image becomes a tensor. Patched in the same
    way as build_model, so the runner's data pipeline is otherwise untouched."""
    def build(*args, **kwargs):
        t = make(*args, **kwargs)
        if not _CHANNELS:
            return t
        first, *rest = t.transforms
        assert isinstance(first, v2.ToImage), first
        return v2.Compose([first, sources.ReorderChannels(_CHANNELS), *rest])
    return build


def main(argv=None):
    global _TARGET, _CHANNELS
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--target', required=True)
    ap.add_argument('--sources', nargs='*', default=None)
    ap.add_argument('--out', default='results/hpo_sources',
                    help='each source writes to <out>/<source>/<target>/')
    known, rest = ap.parse_known_args(argv)
    _TARGET = known.target

    # by default only what has to be trained; 'imagenet' comes from the architecture
    # benchmark (src/hpo_sources_reuse.py), but naming it still runs it
    chosen = (sources.for_target(known.target, known.sources) if known.sources
              else sources.to_train(known.target))
    if not chosen:
        raise SystemExit(f'no sources left for {known.target}')
    missing = [s for s in chosen
               if s != 'imagenet' and not os.path.exists(sources.checkpoint_path(s, known.target))]
    if missing:
        raise SystemExit(f'no checkpoint for: {", ".join(missing)} '
                         f'(looked under {sources.root()}; set MODELS_ROOT to move it)')

    # the runner iterates '--archs'; here those names are sources, and build_model reads
    # them as such. Patching rather than forking keeps one protocol implementation.
    hpo_finetune.build_model = build_model
    hpo_finetune.train_transform = _in_source_order(hpo_finetune.train_transform)
    hpo_finetune.eval_transform = _in_source_order(hpo_finetune.eval_transform)

    base = ['--target', known.target]
    for flag, value in (('--final-folds', DEFAULTS['final_folds']),
                        ('--final-seeds', [DEFAULTS['final_seeds']]),
                        ('--final-topk', [DEFAULTS['final_topk']])):
        if flag not in rest:
            base += [flag, *[str(v) for v in value]]

    print(f'dataset transfer: {known.target} <- {len(chosen)} sources '
          f'({", ".join(chosen)})', flush=True)

    rc = None
    for source in chosen:
        mean, std = sources.normalization(source)
        hpo_finetune._MEAN, hpo_finetune._STD = mean, std
        _CHANNELS = sources.channel_order(source)
        hpo_finetune.ARCHS = {source: sources.BACKBONE}
        print(f'\n########## {known.target} <- {source}  (normalise {mean}, '
              f'{"BGR" if _CHANNELS else "RGB"})', flush=True)
        rc = hpo_finetune.main(base + ['--archs', source,
                                       '--out', os.path.join(known.out, source)] + rest)
    return rc


if __name__ == '__main__':
    main()
