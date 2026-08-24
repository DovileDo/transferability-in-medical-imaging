"""The rule that turns final runs into one ground-truth number.

The search proposes; this decides. `src/hpo_finetune.py` retrains the top few
configurations of a study on the held-out training folds, and the winner among them is
the one with the highest validation AUC *averaged over those folds* -- an average of
several runs rather than the single run that happened to top the search.

Why not simply report the best trial: with a validation split of 25 images per class,
the AUC of one run carries a standard error of a percentage point or more, and the
search takes an argmax over a few thousand such numbers (every trial, every checkpoint).
The winner of that argmax is partly whichever configuration got the luckiest draw, while
the architectures being compared are separated by tenths of a point. Re-estimating the
top few on folds none of them was tuned on costs a fraction of the search and puts the
choice on an average instead of an extreme.

Kept free of torch, numpy and optuna so the runner, the watcher and the collector can
all apply the identical rule -- a selection rule that differs between the process that
writes the numbers and the process that reads them is worse than no rule at all. The
definition of a trial that failed to beat chance lives here for the same reason: it is
reported by the runner as each search ends and recomputed by the collector afterwards.
"""

import csv
import os

#: column holding the Optuna trial a final run came from ('best_trial' in runs written
#: before the top-k stage existed, when there was only ever one configuration)
TRIAL_COLS = ('trial', 'best_trial')

#: written into final_runs.csv by the runner, echoed back when reporting a winner
PARAM_COLS = ('optimizer', 'lr', 'momentum', 'wd', 'batch_size', 'head_lr_mult', 'aug',
              'train_epochs', 'dropout', 'label_smoothing', 'warmup_epochs', 'scheduler')

#: a trial no better than this has not shown it can train the architecture at all.
#: Reporting only; the objective always keeps a trial's own validation AUC.
CHANCE_AUC = 0.55


def near_chance(value, diverged=False):
    """Did this trial fail to beat chance by a usable margin?

    True for a run that diverged or scored undefined, and for one that trained to
    completion but landed at or below `CHANCE_AUC`. The second case is the common one:
    at the top of the learning-rate range a run more often decays towards chance than
    it explodes into NaNs, so the divergence flag on its own counts far too few of them
    and would report a search as healthier than it was.

    This is a description of the search, not a rule applied to it. A near-chance trial
    still enters the objective at its own value, so the sampler sees how bad the region
    is rather than a flat floor, and it is still eligible for the shortlist -- where it
    loses on validation AUC anyway, unless nothing else did better either.
    """
    if diverged:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return v != v or v <= CHANCE_AUC          # NaN is not a score above chance


def weighted_mean(values, weights):
    """Support-weighted mean over the entries that are defined, or NaN if none are.

    Used to re-average per-class AUCs by class prevalence. It lives beside the selection
    rule, not beside the training code, because the point of recording the per-class
    parts is that the average over them can be chosen after the runs are finished --
    and whichever average is chosen has to be the same one wherever it is applied.

    A class whose AUC was undefined is skipped rather than counted as zero, matching
    what the macro average does with it, so the two differ only in the weights.
    """
    pairs = [(float(v), float(w)) for v, w in zip(values, weights)
             if v is not None and w is not None and float(w) > 0]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return float('nan')
    return sum(v * w for v, w in pairs) / total


def read_final_runs(path, arch=None):
    """Rows of a final_runs.csv, optionally for one architecture."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if arch is None or r.get('arch') == arch]


def trial_of(row):
    for c in TRIAL_COLS:
        if row.get(c) not in (None, ''):
            return int(row[c])
    return -1


def _num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def done_keys(rows):
    """`(trial, fold, seed)` of every run already on disk, for resuming."""
    out = set()
    for r in rows:
        try:
            out.add((trial_of(r), int(r.get('fold', 1)), int(r['seed'])))
        except (TypeError, ValueError):
            continue
    return out


def group_runs(rows):
    """`{trial: [row, ...]}`, one group per configuration that was retrained."""
    groups = {}
    for r in rows:
        groups.setdefault(trial_of(r), []).append(r)
    return groups


def select(rows):
    """Apply the selection rule to one architecture's final runs.

    Returns None if there is nothing to choose from, otherwise a dict with the winning
    trial number, the folds it was compared on, its mean validation AUC over them, and
    the mean and standard deviation of its test AUC -- the ground truth for that
    architecture-target pair.

    Candidates are compared only on folds every one of them has a run for, so a search
    interrupted part-way through the final stage still compares like with like; the
    reported test AUC is the winner's mean over exactly those folds. `complete` says
    whether every candidate had a run on every fold.

    Every configuration with runs on disk is a candidate, which for a study that was
    extended after its final stage includes ones only the earlier, shorter search had
    shortlisted. That widens the choice rather than corrupting it -- the comparison is
    still on validation AUC over held-out folds -- but it does mean an extended study
    chooses from more than `--final-topk` configurations.
    """
    groups = group_runs(rows)
    if not groups:
        return None

    folds = {t: {int(r['fold']) for r in rs if r.get('fold') not in (None, '')}
             for t, rs in groups.items()}
    common = set.intersection(*folds.values()) if folds else set()
    if not common:                    # nothing overlaps yet: fall back to per-config means
        common = set().union(*folds.values()) if folds else set()

    cands = []
    for t, rs in groups.items():
        on = [r for r in rs if r.get('fold') not in (None, '') and int(r['fold']) in common]
        val = [v for v in (_num(r, 'val_auc') for r in on) if v is not None]
        test = [v for v in (_num(r, 'test_auc') for r in on) if v is not None]
        if not val:
            continue
        cands.append({
            'trial': t,
            'n': len(val),
            'folds': sorted({int(r['fold']) for r in on}),
            'val_auc': sum(val) / len(val),
            'test_auc': sum(test) / len(test) if test else None,
            'test_sd': _sd(test),
            'params': {k: rs[0][k] for k in PARAM_COLS if k in rs[0]},
        })
    if not cands:
        return None

    cands.sort(key=lambda c: -c['val_auc'])
    seen_folds = set().union(*folds.values())
    best = dict(cands[0])
    best['candidates'] = cands
    best['compared_on'] = sorted(common)
    # complete means every candidate ran on every fold any of them ran on, so the
    # choice was made on the whole planned comparison rather than a truncated one
    best['complete'] = all(fs == seen_folds for fs in folds.values())
    return best


def format_choice(arch, choice, scale=100.0):
    """One line per candidate, winner first -- what the runner prints and stores."""
    lines = []
    for i, c in enumerate(sorted(choice['candidates'], key=lambda c: -c['val_auc'])):
        test = f"{c['test_auc'] * scale:6.2f}" if c['test_auc'] is not None else '     -'
        sd = f"+-{c['test_sd'] * scale:4.2f}" if c['test_sd'] is not None else '      '
        lines.append(f"  {'*' if i == 0 else ' '} trial {c['trial']:>4}  "
                     f"val {c['val_auc'] * scale:6.2f}  test {test} {sd}  "
                     f"({c['n']} folds)")
    head = (f'[{arch}] selection over {len(choice["candidates"])} candidates on folds '
            f'{choice["compared_on"]}'
            + ('' if choice['complete'] else ' (incomplete -- some runs missing)'))
    return '\n'.join([head] + lines)
