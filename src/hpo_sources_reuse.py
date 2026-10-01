#!/usr/bin/env python
# coding: utf-8
"""The 'imagenet' source of the dataset-transfer benchmark, taken from the model benchmark.

That source is torchvision's resnet50 with IMAGENET1K_V1 weights and ImageNet
normalisation, searched for 100 trials on fold 1 and rerun top-10 x 16 seeds on fold 1 --
exactly the 'resnet' architecture of results/hpo16. Training it again would repeat that
run, so its study, final runs, predictions and selection are copied instead, renamed to
'imagenet', into the layout src/hpo_sources.py writes:

    results/hpo16/<target>/            arch 'resnet'
    results/hpo_sources/imagenet/<target>/    arch 'imagenet'

A target is copied only once its resnet final stage is complete. Rerunning replaces the
copy, so it can be run again after a target finishes.

    python src/hpo_sources_reuse.py                       # every completed target
    python src/hpo_sources_reuse.py --target dermamnist
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sources                                                       # noqa: E402


def _rows(path, arch, as_name):
    if not os.path.exists(path):
        return None, []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = [dict(row, arch=as_name) for row in r if row['arch'] == arch]
        return r.fieldnames, rows


def _write_rows(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _copy_study(src_db, dst_db, arch, as_name):
    """One study out of src_db, renamed. Copied with the backup API, as the sbatch does."""
    import optuna
    tmp = dst_db + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    con, out = sqlite3.connect(src_db, timeout=60), sqlite3.connect(tmp)
    try:
        con.backup(out)
        out.execute('PRAGMA journal_mode=DELETE')
    finally:
        out.close()
        con.close()
    storage = optuna.storages.RDBStorage(f'sqlite:///{tmp}')
    for s in optuna.get_all_study_summaries(storage, include_best_trial=False):
        if s.study_name != arch:
            optuna.delete_study(study_name=s.study_name, storage=storage)
    storage.engine.dispose()
    con = sqlite3.connect(tmp)
    con.execute('UPDATE studies SET study_name = ? WHERE study_name = ?', (as_name, arch))
    con.commit()
    con.execute('VACUUM')
    con.close()
    os.replace(tmp, dst_db)


def one_target(target, bench, out, source, arch):
    src = os.path.join(bench, target)
    dst = os.path.join(out, source, target)
    sel_path = os.path.join(src, 'selection.json')
    if not os.path.exists(sel_path):
        return 'no selection.json yet'
    with open(sel_path) as f:
        sel = json.load(f)
    if not sel.get(arch, {}).get('complete'):
        return f'{arch} final stage not complete'
    with open(os.path.join(src, 'meta.json')) as f:
        meta = json.load(f)
    fields, finals = _rows(os.path.join(src, 'final_runs.csv'), arch, source)
    expected = meta['final_topk'] * len(meta['final_folds']) * meta['seeds_per_fold']
    if len(finals) != expected:
        return f'{len(finals)} of {expected} final runs'

    tmp = dst + '.partial'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    _copy_study(os.path.join(src, 'optuna.db'), os.path.join(tmp, 'optuna.db'), arch, source)
    _write_rows(os.path.join(tmp, 'final_runs.csv'), fields, finals)
    pfields, plain = _rows(os.path.join(src, 'val_loss_plain.csv'), arch, source)
    if pfields:
        _write_rows(os.path.join(tmp, 'val_loss_plain.csv'), pfields, plain)
    preds = os.path.join(src, 'predictions', arch)
    if os.path.isdir(preds):
        shutil.copytree(preds, os.path.join(tmp, 'predictions', source))
    with open(os.path.join(tmp, 'selection.json'), 'w') as f:
        json.dump({source: sel[arch]}, f, indent=2)
    cost_path = os.path.join(src, 'arch_cost.json')
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            cost = json.load(f)
        if arch in cost:
            with open(os.path.join(tmp, 'arch_cost.json'), 'w') as f:
                json.dump({source: cost[arch]}, f, indent=2)
    meta.update(archs=[source], reused_from=f'{src} (arch {arch})')
    with open(os.path.join(tmp, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return f'copied ({len(finals)} final runs, {len(plain)} plain losses)'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default='results/hpo16', help='the model benchmark')
    ap.add_argument('--out', default='results/hpo_sources')
    ap.add_argument('--target', nargs='*')
    args = ap.parse_args(argv)
    targets = args.target or sorted(d for d in os.listdir(args.bench)
                                    if os.path.isdir(os.path.join(args.bench, d)))
    for t in targets:
        for source, arch in sources.REUSED.items():
            print(f'{t:16s} {source} <- {arch}: '
                  f'{one_target(t, args.bench, args.out, source, arch)}', flush=True)


if __name__ == '__main__':
    main()
