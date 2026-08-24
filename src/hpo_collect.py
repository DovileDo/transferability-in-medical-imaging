#!/usr/bin/env python
# coding: utf-8
"""Turn finished src/hpo_finetune.py runs into ground-truth AUC tables.

Writes, under --out:
    trials_all.csv   every Optuna trial (params, val/test metrics, state), with
                     `near_chance` marking the ones that never beat chance
    final_all.csv    every final run: the top-k configurations of each study, each on
                     every held-out fold, with `selected` marking the winning one
    selection.csv    one row per architecture-target pair: which configuration won,
                     what each candidate scored, and how far apart they were

and, under --results:
    AUCs_model_hpo.csv      architectures x targets, mean test AUC of the selected
                            configuration over the held-out folds, in percent --
                            drop-in replacement for AUCs_model.csv
    AUCs_model_hpo_sd.csv   its standard deviation over those runs, same layout
    AUCs_model_hpo_paired.csv
                            per target, the difference between every pair of
                            architectures taken fold by fold -- what a comparison
                            between two of them actually rests on
    AUCs_model_hpo_besttrial.csv
                            test AUC of the single best-validation trial -- the top-1
                            rule, kept for comparison with the top-k selection
    AUCs_model_hpo_weighted.csv
                            the same selected runs scored with a prevalence-weighted
                            average of the per-class AUCs instead of the macro average,
                            so the ranking can be checked against the choice of metric.
                            Only written when the runs carry per-class AUCs

The selection rule lives in src/hpo_select.py and is the same one the runner applied;
it is recomputed here rather than trusted, so a study that was extended after its final
stage ran cannot leave a stale winner behind.

    python src/hpo_collect.py --out results/hpo
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo_select                                                     # noqa: E402

# column order of results/AUCs_model.csv
TARGET_ORDER = ['bloodmnist', 'breastmnist', 'dermamnist', 'octmnist', 'organamnist',
                'organcmnist', 'organsmnist', 'pathmnist', 'pneumoniamnist',
                'retinamnist', 'tissuemnist']
SOURCE_ORDER = ['densenet', 'efficientnet', 'googlenet', 'mnasnet', 'mobilenet',
                'vgg', 'convnext', 'shufflenet', 'resnet']


def read_trials(out_root, target):
    """All trials of one target's studies as a list of flat dicts."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db = os.path.join(out_root, target, 'optuna.db')
    if not os.path.exists(db):
        return []
    storage = optuna.storages.RDBStorage(f'sqlite:///{db}',
                                         engine_kwargs={'connect_args': {'timeout': 30}})
    rows = []
    for summary in optuna.get_all_study_summaries(storage=storage):
        arch = summary.study_name
        study = optuna.load_study(study_name=arch, storage=storage)
        for t in study.get_trials(deepcopy=False):
            row = {'target': target, 'arch': arch, 'trial': t.number,
                   'state': t.state.name, 'val_auc_objective': t.value}
            row.update({k: v for k, v in t.params.items()})
            row.update({k: v for k, v in t.user_attrs.items() if k != 'arch'})
            if t.datetime_start and t.datetime_complete:
                row['seconds'] = (t.datetime_complete - t.datetime_start).total_seconds()
            rows.append(row)
    return rows


def read_finals(out_root, target):
    return hpo_select.read_final_runs(os.path.join(out_root, target, 'final_runs.csv'))


def select_all(rows, top1=None):
    """Apply the selection rule per architecture-target pair.

    Returns every final run marked with `selected`, and one summary row per pair
    recording what the choice was between -- the spread among the candidates is the
    quantity a single-run argmax would have gambled on, so it is worth keeping.

    `top1` maps a pair to the trial the search ranked first, so the summary can carry
    what the old top-1 rule would have reported alongside what the top-k rule chose.
    """
    top1 = top1 or {}
    pairs = {}
    for r in rows:
        pairs.setdefault((r.get('target'), r.get('arch')), []).append(r)

    winners, summary = [], []
    for (target, arch), rs in sorted(pairs.items()):
        choice = hpo_select.select(rs)
        if choice is None:
            continue
        keep = {(hpo_select.trial_of(r), r.get('fold')) for r in rs
                if hpo_select.trial_of(r) == choice['trial']
                and int(r['fold']) in choice['compared_on']}
        for r in rs:
            sel = (hpo_select.trial_of(r), r.get('fold')) in keep
            winners.append({**r, 'selected': sel})
        tests = [c['test_auc'] for c in choice['candidates'] if c['test_auc'] is not None]
        first = top1.get((target, arch))
        summary.append({
            'target': target, 'arch': arch, 'trial': choice['trial'],
            'n_candidates': len(choice['candidates']), 'n_runs': choice['n'],
            'folds': ' '.join(str(f) for f in choice['folds']),
            'complete': choice['complete'],
            'val_auc': choice['val_auc'], 'test_auc': choice['test_auc'],
            'test_sd': choice['test_sd'],
            # what the top-1 rule would have reported instead, and the range the
            # shortlist spanned on test -- the cost of getting the choice wrong
            'top1_trial': first,
            'test_auc_top1': next((c['test_auc'] for c in choice['candidates']
                                   if c['trial'] == first), None),
            'candidate_test_spread': (max(tests) - min(tests)) if len(tests) > 1 else None,
            **choice['params'],
        })
    return winners, summary


def _per_class(df, kind):
    """Pull the per-class AUC and support columns out of their JSON encoding.

    Returns None when the columns are absent, which is what a study run before they
    were recorded looks like. Nothing else in the collector depends on them, so the
    rest of the tables are produced either way.
    """
    cols = (f'{kind}_auc_per_class', f'{kind}_support')
    if not all(c in df for c in cols):
        return None
    def load(v):
        if isinstance(v, str):
            return json.loads(v)
        return v if isinstance(v, list) else None
    aucs = df[cols[0]].map(load)
    support = df[cols[1]].map(load)
    if aucs.isna().all():
        return None
    return aucs, support


def weighted_column(df, kind):
    """Prevalence-weighted mean of each row's per-class AUCs, or None if unavailable."""
    parts = _per_class(df, kind)
    if parts is None:
        return None
    aucs, support = parts
    return pd.Series(
        [np.nan if a is None or s is None else hpo_select.weighted_mean(a, s)
         for a, s in zip(aucs, support)], index=df.index)


def ranking_shift(macro, weighted):
    """How far the architecture order moves between two scorings of the same runs.

    Both arguments are target -> arch -> value. Reports, per target, how many of the
    architecture pairs swap order and whether the top of the table changes, because
    those are the two things a ground-truth ranking is read for. A metric that moves
    the numbers but not the order is a different claim from one that moves the order.
    """
    out = []
    for target in sorted(set(macro) & set(weighted)):
        a, b = macro[target], weighted[target]
        archs = sorted(set(a) & set(b))
        if len(archs) < 2:
            continue
        swaps = sum(1 for i, x in enumerate(archs) for y in archs[i + 1:]
                    if (a[x] - a[y]) * (b[x] - b[y]) < 0)
        pairs = len(archs) * (len(archs) - 1) // 2
        top_a = max(archs, key=lambda k: a[k])
        top_b = max(archs, key=lambda k: b[k])
        out.append({'target': target, 'n_arch': len(archs), 'pairs': pairs,
                    'swapped_pairs': swaps, 'top_macro': top_a,
                    'top_weighted': top_b, 'top_changed': top_a != top_b})
    return pd.DataFrame(out)


def paired_differences(won, scale=100.0):
    """Per-target architecture-vs-architecture differences, measured within a fold.

    The final stage runs every architecture on the same folds with the same seeds, so
    the difference between two of them can be taken fold by fold. That paired difference
    is what a comparison actually rests on, and its spread is much tighter than the
    marginal standard deviations, which also carry the fold-to-fold variation common to
    both architectures.
    """
    per_fold = (won.groupby(['target', 'arch', 'fold'])['test_auc'].mean()
                .unstack('arch'))
    rows = []
    for target, block in per_fold.groupby('target'):
        archs = [a for a in block.columns if block[a].notna().any()]
        for i, a in enumerate(archs):
            for b in archs[i + 1:]:
                d = (block[a] - block[b]).dropna() * scale
                if not len(d):
                    continue
                rows.append({'target': target, 'arch_a': a, 'arch_b': b,
                             'n_folds': len(d), 'mean_diff': d.mean(),
                             'sd_diff': d.std(ddof=1) if len(d) > 1 else np.nan,
                             'sd_a': block[a].dropna().std(ddof=1) * scale,
                             'sd_b': block[b].dropna().std(ddof=1) * scale})
    return pd.DataFrame(rows)


def pivot(df, value, index='arch', columns='target', scale=100.0):
    """arch x target matrix in the layout of results/AUCs_model.csv."""
    m = df.pivot_table(index=index, columns=columns, values=value, aggfunc='mean') * scale
    m = m.reindex(index=[a for a in SOURCE_ORDER if a in m.index]
                  + [a for a in m.index if a not in SOURCE_ORDER])
    m = m.reindex(columns=[t for t in TARGET_ORDER if t in m.columns]
                  + [c for c in m.columns if c not in TARGET_ORDER])
    return m.rename_axis('source').reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/hpo')
    ap.add_argument('--results', default='results')
    ap.add_argument('--target', nargs='*', help='default: every target under --out')
    args = ap.parse_args(argv)

    os.makedirs(args.results, exist_ok=True)
    targets = args.target or sorted(
        d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)))

    trials, finals = [], []
    for t in targets:
        trials += read_trials(args.out, t)
        finals += read_finals(args.out, t)

    if not trials:
        raise SystemExit(f'no studies found under {args.out}')

    tdf = pd.DataFrame(trials)

    # equal trial counts are not equal search quality: how much of the space an
    # architecture can train in at all differs between them, and the parity claim is
    # only auditable if that share is reported. `scored_at_chance` alone undercounts it
    # -- a run at the top of the learning-rate range usually decays towards chance
    # rather than exploding -- so the recorded share is the wider one and the
    # divergences are reported as the subset of it that they are.
    done = tdf['state'] == 'COMPLETE'
    div = (tdf['scored_at_chance'].astype('boolean').fillna(False)
           if 'scored_at_chance' in tdf
           else pd.Series(False, index=tdf.index, dtype='boolean'))
    tdf['near_chance'] = pd.array(
        [hpo_select.near_chance(v, d) if c else None
         for v, d, c in zip(tdf['val_auc_objective'], div, done)], dtype='boolean')

    # the per-class vectors arrive from Optuna as lists; written as JSON rather than as
    # a Python repr so that `None` leaves the file as null and the column is readable by
    # something other than Python. final_runs.csv is already JSON on disk
    for col in tdf.columns:
        if tdf[col].map(lambda v: isinstance(v, list)).any():
            tdf[col] = tdf[col].map(lambda v: json.dumps(v) if isinstance(v, list) else v)

    tdf.to_csv(os.path.join(args.out, 'trials_all.csv'), index=False)
    print(f'{len(tdf)} trials -> {os.path.join(args.out, "trials_all.csv")}')

    if done.any():
        share = tdf[done].groupby(['target', 'arch'])['near_chance'].mean()
        worst = share.sort_values(ascending=False).head(3)
        print(f'share of completed trials that failed to beat chance (val AUC <= '
              f'{hpo_select.CHANCE_AUC}), worst pairs: '
              + ', '.join(f'{a}/{t} {100 * v:.0f}%' for (t, a), v in worst.items()))
        n_near, n_div = int(tdf.loc[done, 'near_chance'].sum()), int(div[done].sum())
        print(f'  {n_near}/{int(done.sum())} overall, of which {n_div} diverged or '
              f'scored undefined and {n_near - n_div} trained to completion at chance')

    if finals:
        # the trial the search ranked first, i.e. what a top-1 rule would have taken
        complete = tdf[tdf['state'] == 'COMPLETE'].dropna(subset=['val_auc_objective'])
        top1 = {k: int(complete.loc[i, 'trial'])
                for k, i in complete.groupby(['target', 'arch'])['val_auc_objective']
                .idxmax().items()}

        marked, summary = select_all(finals, top1)
        fdf = pd.DataFrame(marked)
        for c in ('test_auc', 'test_acc', 'val_auc', 'seed', 'train_seconds'):
            if c in fdf:
                fdf[c] = pd.to_numeric(fdf[c], errors='coerce')
        fdf.to_csv(os.path.join(args.out, 'final_all.csv'), index=False)
        print(f'{len(fdf)} final runs -> {os.path.join(args.out, "final_all.csv")}')

        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(args.out, 'selection.csv'), index=False)
        print(f'{len(sdf)} selections -> {os.path.join(args.out, "selection.csv")}')
        if 'complete' in sdf and (~sdf['complete']).any():
            print(f'[warn] {(~sdf["complete"]).sum()} pairs chose between candidates '
                  f'with unequal fold coverage -- the final stage is unfinished')

        # only the winning configuration's runs make the reported tables; averaging a
        # pair's rows across configurations would report a number no configuration has
        won = fdf[fdf['selected']]
        n = won.groupby(['target', 'arch'])['test_auc'].size()
        if (n < 2).any():
            print(f'[warn] {(n < 2).sum()} target/arch pairs rest on a single run')

        mean = pivot(won, 'test_auc')
        sd = won.pivot_table(index='arch', columns='target', values='test_auc',
                             aggfunc=lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan)
        sd = (sd * 100).rename_axis('source').reset_index()
        mean.to_csv(os.path.join(args.results, 'AUCs_model_hpo.csv'), index=False)
        sd.to_csv(os.path.join(args.results, 'AUCs_model_hpo_sd.csv'), index=False)
        print(f'-> {os.path.join(args.results, "AUCs_model_hpo.csv")}')
        print(mean.to_string(index=False))

        # the same runs, scored by a prevalence-weighted average of the per-class AUCs
        # instead of the macro average. Nothing here re-selects: the configuration is
        # the one the protocol chose, so what this isolates is the effect of the metric
        # on the reported number, separately from its effect on which run is reported
        wcol = weighted_column(won, 'test')
        if wcol is not None and wcol.notna().any():
            wdf = won.assign(test_auc_weighted=wcol)
            wmean = pivot(wdf, 'test_auc_weighted')
            wmean.to_csv(os.path.join(args.results, 'AUCs_model_hpo_weighted.csv'),
                         index=False)
            print(f'-> {os.path.join(args.results, "AUCs_model_hpo_weighted.csv")}')

            def as_map(frame):
                m = frame.set_index('source')
                return {t: {a: v for a, v in m[t].dropna().items()} for t in m.columns}

            shift = ranking_shift(as_map(mean), as_map(wmean))
            if len(shift):
                tot, sw = shift['pairs'].sum(), shift['swapped_pairs'].sum()
                changed = shift[shift['top_changed']]
                print(f'macro vs prevalence-weighted AUC: {sw}/{tot} architecture pairs '
                      f'swap order, the top architecture changes on '
                      f'{len(changed)}/{len(shift)} targets'
                      + (': ' + ', '.join(f'{r.target} {r.top_macro}->{r.top_weighted}'
                                          for r in changed.itertuples())
                         if len(changed) else ''))
                gap = 100 * (wdf.groupby(['target', 'arch'])['test_auc_weighted'].mean()
                             - wdf.groupby(['target', 'arch'])['test_auc'].mean())
                print(f'  weighted minus macro, per pair: median {gap.median():+.2f} pp, '
                      f'min {gap.min():+.2f}, max {gap.max():+.2f}')
        elif 'test_auc_per_class' not in won:
            print('[note] no per-class AUCs in these runs, skipping the weighted-AUC '
                  'comparison (they are recorded by runs from this version onwards)')

        pdf = paired_differences(won)
        if len(pdf):
            pdf.to_csv(os.path.join(args.results, 'AUCs_model_hpo_paired.csv'),
                       index=False)
            tight = (pdf['sd_diff'] < pdf[['sd_a', 'sd_b']].max(axis=1)).mean()
            print(f'-> {os.path.join(args.results, "AUCs_model_hpo_paired.csv")} '
                  f'({len(pdf)} pairs; the paired SD is the tighter statistic in '
                  f'{100 * tight:.0f}% of them)')

        moved = sdf.dropna(subset=['test_auc', 'test_auc_top1'])
        if len(moved):
            d = 100 * (moved['test_auc'] - moved['test_auc_top1'])
            print(f'top-k selection vs the top-1 trial: changed the configuration in '
                  f'{(moved["trial"] != moved["top1_trial"]).sum()}/{len(moved)} pairs, '
                  f'test AUC median {d.median():+.2f} pp, min {d.min():+.2f}, '
                  f'max {d.max():+.2f}')
    else:
        print('[warn] no final runs yet, skipping AUCs_model_hpo.csv')

    # the top-1 rule, for comparison: test AUC of the single best-validation trial.
    # Ranked on the objective rather than the recorded val_auc, so a trial that scored
    # at chance cannot come first on the checkpoint it had before it diverged.
    scored = tdf[(tdf['state'] == 'COMPLETE') & tdf.get('test_auc').notna()] \
        if 'test_auc' in tdf else tdf.iloc[0:0]
    if len(scored):
        idx = scored.groupby(['target', 'arch'])['val_auc_objective'].idxmax()
        best = scored.loc[idx]
        pivot(best, 'test_auc').to_csv(
            os.path.join(args.results, 'AUCs_model_hpo_besttrial.csv'), index=False)
        print(f'-> {os.path.join(args.results, "AUCs_model_hpo_besttrial.csv")}')


if __name__ == '__main__':
    main()
