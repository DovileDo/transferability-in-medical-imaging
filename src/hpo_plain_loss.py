#!/usr/bin/env python
# coding: utf-8
"""Unsmoothed validation loss for every final run, written beside final_runs.csv.

src/hpo_select.py breaks ties in validation AUC on this -- see its select(). The runner
records its training loss, which includes the searched label smoothing and so is not
comparable across configurations; this recomputes plain cross-entropy from the
validation predictions each final run already saved.

Writes <out>/<target>/val_loss_plain.csv, keyed by arch, trial, fold and seed. Run it
when no job is writing to the target -- it reads final_runs.csv and the predictions,
and rewrites the sidecar in full.

    python src/hpo_plain_loss.py --out results/hpo16
    python src/hpo_plain_loss.py --out results/hpo16 --target pneumoniamnist
"""

import argparse
import csv
import os
import sys

import numpy as np
from medmnist import INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo_select                                                    # noqa: E402


def one_target(root, target):
    path = os.path.join(root, target, 'final_runs.csv')
    rows = hpo_select.read_final_runs(path)
    if not rows:
        return 0, 0
    task = INFO[target]['task']
    out, missing = [], 0
    for r in rows:
        name = f"final_t{r['trial']}f{r['fold']}s{r['seed']}.npz"
        pred = os.path.join(root, target, 'predictions', r['arch'], name)
        if not os.path.exists(pred):
            missing += 1
            continue
        d = np.load(pred)
        if 'val_score' not in d.files:
            missing += 1
            continue
        out.append({**{k: r[k] for k in hpo_select.PLAIN_LOSS_KEYS},
                    'val_loss_plain': hpo_select.plain_loss(d['val_true'], d['val_score'], task)})
    side = os.path.join(root, target, hpo_select.PLAIN_LOSS_FILE)
    tmp = side + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[*hpo_select.PLAIN_LOSS_KEYS, 'val_loss_plain'])
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, side)
    return len(out), missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/hpo16')
    ap.add_argument('--target', nargs='*')
    args = ap.parse_args(argv)
    targets = args.target or sorted(d for d in os.listdir(args.out)
                                    if os.path.isfile(os.path.join(args.out, d, 'final_runs.csv')))
    for t in targets:
        n, miss = one_target(args.out, t)
        print(f'{t:16s} {n:6d} runs' + (f'   {miss} without saved validation predictions' if miss else ''))


if __name__ == '__main__':
    main()
