#!/usr/bin/env python
# coding: utf-8
"""Follow an src/hpo_finetune.py run and project how long the rest will take.

Reads the Optuna study, the heartbeat file and the per-architecture cost probe that
the runner writes, then reports progress per architecture plus an ETA for the whole
job. Safe to run against a live run -- it only reads -- and needs no GPU.

    python src/hpo_watch.py                        # every target under results/hpo
    python src/hpo_watch.py --target dermamnist    # one target
    python src/hpo_watch.py -w                     # refresh until finished

Before any trial has finished the ETA comes from the cost probe and is marked '~';
it tightens as real trial durations accumulate.
"""

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo_select                                                     # noqa: E402

RUNNING, PENDING, DONE = 'running', 'pending', 'done'


# --------------------------------------------------------------------------------------
# reading state
# --------------------------------------------------------------------------------------

def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _read_final_runs(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _load_trials(db_path, archs):
    """Per-architecture trial records from the Optuna DB, or {} if unreadable."""
    if not os.path.exists(db_path):
        return {}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from optuna.trial import TrialState
    except ImportError:
        return {}

    url = f'sqlite:///{db_path}'
    try:
        storage = optuna.storages.RDBStorage(
            url, engine_kwargs={'connect_args': {'timeout': 30}}, skip_compatibility_check=True)
        names = {s.study_name for s in optuna.get_all_study_summaries(storage=storage)}
    except Exception:
        return {}

    out = {}
    for arch in archs:
        if arch not in names:
            continue
        try:
            study = optuna.load_study(study_name=arch, storage=storage)
            trials = study.get_trials(deepcopy=False)
        except Exception:
            continue
        recs = []
        for t in trials:
            dur = None
            if t.datetime_start and t.datetime_complete:
                dur = (t.datetime_complete - t.datetime_start).total_seconds()
            recs.append({
                'number': t.number,
                'state': t.state.name,
                'finished': t.state in (TrialState.COMPLETE, TrialState.PRUNED,
                                        TrialState.FAIL),
                'value': t.value,
                'duration': dur,
                'epochs_run': t.user_attrs.get('epochs_run'),
                'test_auc': t.user_attrs.get('test_auc'),
            })
        out[arch] = recs
    return out


# --------------------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------------------

def _expected_new_maxima(done, total):
    """Expected number of remaining record-breaking trials among total i.i.d. draws.

    Trial k is a new maximum with probability 1/k, so the count left after `done`
    trials is sum_{k=done+1}^{total} 1/k. Used to price the test evaluations that
    --test-eval best will still trigger.
    """
    return sum(1.0 / k for k in range(max(done, 1) + 1, total + 1))


def _probe_costs(cost, meta, arch):
    """(per average trial, per test evaluation, per trial setup) from the cost probe.

    A trial trains for one of the `train_epochs` arms and validates every
    `val_every_steps` optimizer steps, so the average cost is taken over the arms and
    over the batch sizes that set how many steps an epoch is.
    """
    c = cost.get(arch)
    if not c:
        return None, None, 0.0

    n_train = meta.get('n_train', 0)
    n_val = meta.get('n_val', 0)
    arms = meta.get('train_epochs') or [meta.get('max_epochs', 60)]

    mean_epochs = sum(arms) / len(arms)
    validations = meta.get('validations_per_trial', 40)

    trial = (mean_epochs * n_train * c['train_s_per_img']
             + validations * n_val * c['eval_s_per_img'])
    test = meta.get('n_test', 0) * c['eval_s_per_img']
    return trial, test, c.get('build_s', 0.0)


def estimate_arch(arch, recs, finals, meta, cost, inflight):
    """Progress and remaining seconds for one architecture."""
    trials_total = meta.get('trials', 0)
    # 'final_seeds' is what runs before the top-k stage recorded the same quantity as
    final_total = meta.get('final_runs', meta.get('final_seeds', 0))

    # the runner retries failed trials, so they do not count towards the budget
    finished = [r for r in recs if r['state'] in ('COMPLETE', 'PRUNED')]
    pruned = sum(1 for r in finished if r['state'] == 'PRUNED')
    failed = [r for r in recs if r['state'] == 'FAIL']
    complete = [r for r in finished if r['state'] == 'COMPLETE']

    probe_trial_s, probe_test_s, probe_build_s = _probe_costs(cost, meta, arch)
    durations = [r['duration'] for r in finished if r['duration']]
    measured = len(durations) > 0

    if measured:
        mean_trial_s = sum(durations) / len(durations)
        full_durations = [r['duration'] for r in complete if r['duration']]
        full_trial_s = (sum(full_durations) / len(full_durations)
                        if full_durations else mean_trial_s)
    elif probe_trial_s is not None:
        # cold start: an average training-length arm, with no early stopping and no
        # pruning, so this is an upper bound
        mean_trial_s = probe_build_s + probe_trial_s
        full_trial_s = mean_trial_s + (probe_test_s or 0)
        if meta.get('test_eval') == 'all':      # every trial pays for the test split
            mean_trial_s = full_trial_s
    else:
        mean_trial_s, full_trial_s = None, None

    elapsed = sum(durations) + sum(r['duration'] or 0 for r in failed)
    elapsed += sum(float(f.get('train_seconds') or 0) for f in finals)

    trials_left = max(0, trials_total - len(finished))
    finals_left = max(0, final_total - len(finals))

    remaining = None
    if mean_trial_s is not None:
        remaining = trials_left * mean_trial_s + finals_left * full_trial_s
        # test evaluations still to come from new record trials
        if meta.get('test_eval') == 'best' and probe_test_s and trials_left:
            remaining += _expected_new_maxima(len(finished), trials_total) * probe_test_s

    # the trial currently on the GPU is already counted in trials_left; replace its
    # share with what the heartbeat says is actually left of it
    note = ''
    if inflight and inflight.get('arch') == arch:
        stage = inflight.get('stage', '')
        frac = inflight.get('frac') or 0.0
        started = inflight.get('trial_started')
        # what is left of the trial on the GPU, from how far into its own budget it is
        spent = (time.time() - started) if started else 0.0
        if frac > 0.02:
            left = max(0.0, spent * (1.0 / frac - 1.0))
        else:
            left = (mean_trial_s or 0.0)
        if remaining is not None:
            baseline = full_trial_s if stage.startswith('final') else mean_trial_s
            remaining = max(0.0, remaining - (baseline or 0.0) + left)
        step, total = inflight.get('step', 0), inflight.get('total_steps', 0)
        which = (f'final {inflight.get("trial", 0) + 1}/{final_total}'
                 if stage.startswith('final') else f'trial {inflight.get("trial")}')
        note = f'{which} step {step}/{total} ({100 * frac:.0f}%)'
        status = RUNNING
    elif trials_left == 0 and finals_left == 0:
        status, remaining = DONE, 0.0
    else:
        status = PENDING

    best_val = max((r['value'] for r in complete if r['value'] is not None), default=None)
    # the final stage runs several candidates, so the ground truth is the winner's mean
    # and not the mean over every row -- averaging across configurations would show a
    # number that is not any configuration's performance
    choice = hpo_select.select(finals)
    final_auc = choice['test_auc'] if choice else None
    final_sd = choice['test_sd'] if choice else None

    return {
        'arch': arch, 'status': status, 'measured': measured,
        'trials_done': len(finished), 'trials_total': trials_total,
        'pruned': pruned, 'failed': len(failed),
        'finals_done': len(finals), 'finals_total': final_total,
        'elapsed': elapsed, 'remaining': remaining,
        'best_val_auc': best_val, 'final_test_auc': final_auc, 'final_test_sd': final_sd,
        'note': note,
    }


def collect(out_root, target):
    """Everything the renderer needs for one target."""
    d = os.path.join(out_root, target)
    meta = _read_json(os.path.join(d, 'meta.json'))
    cost = _read_json(os.path.join(d, 'arch_cost.json'))
    progress = _read_json(os.path.join(d, 'progress.json'))
    finals = _read_final_runs(os.path.join(d, 'final_runs.csv'))
    archs = meta.get('archs') or progress.get('archs') or []
    trials = _load_trials(os.path.join(d, 'optuna.db'), archs)

    stage = progress.get('stage')
    inflight = progress if stage not in (None, 'done', 'between', 'starting') else None
    stale = inflight is not None and (time.time() - progress.get('updated', 0)) > 900
    if stale:
        inflight = None

    rows = [estimate_arch(a, trials.get(a, []),
                          [f for f in finals if f.get('arch') == a],
                          meta, cost, inflight)
            for a in archs]

    return {'target': target, 'dir': d, 'meta': meta, 'progress': progress,
            'rows': rows, 'inflight': inflight, 'stale': stale}


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def fmt_dur(seconds):
    if seconds is None:
        return '?'
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        return f'{h // 24}d {h % 24}h'
    if h:
        return f'{h}h {m:02d}m'
    return f'{m}m {s:02d}s'


#: a run whose heartbeat is older than this is treated as gone when its PID cannot be
#: checked. The runner touches progress.json at every validation, so on the slowest
#: architecture measured this is still many heartbeats.
HEARTBEAT_DEAD = 900


def _alive(progress):
    """Is the run still going?

    A PID is only meaningful on the machine that owns it. When the run is on another
    host -- a scheduler put it on a compute node and this is a login node -- os.kill
    would either report a local process that has nothing to do with it, or report
    nothing and call a healthy run dead. There the heartbeat is the only evidence
    available, so it is what gets used.
    """
    pid, host = progress.get('pid'), progress.get('host')
    age = time.time() - progress.get('updated', 0)
    if host and host != socket.gethostname():
        return age <= HEARTBEAT_DEAD
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def render(state, width=96):
    meta, rows = state['meta'], state['rows']
    lines = []
    head = f"{state['target']}  ({state['dir']})"
    if meta.get('started'):
        head += f"  running {fmt_dur(time.time() - meta['started'])}"
    lines.append(head)

    p = state['progress']
    if p:
        age = time.time() - p.get('updated', 0)
        if p.get('stage') == 'done':
            live = 'finished'
        else:
            live = 'alive' if _alive(p) else 'not running'
        note = f"pid {p.get('pid')} on {p.get('host')} [{live}], heartbeat {fmt_dur(age)} ago"
        if state['stale']:
            note += '  -- stale, ignoring in-flight trial'
        lines.append(note)
    if not rows:
        lines.append('no run found yet (start one with src/hpo_finetune.py)')
        return '\n'.join(lines)

    lines.append('')
    lines.append(f"{'arch':<13} {'trials':>10} {'pruned':>7} {'final':>6} "
                 f"{'best val':>9} {'test AUC':>15} {'elapsed':>9} {'left':>10}  note")
    lines.append('-' * width)

    total_elapsed = total_left = 0.0
    any_estimated = False
    for r in rows:
        left = r['remaining']
        total_elapsed += r['elapsed'] or 0.0
        total_left += left or 0.0
        if not r['measured'] and r['status'] != DONE:
            any_estimated = True
        mark = '' if r['measured'] else '~'
        val = f"{r['best_val_auc']:.4f}" if r['best_val_auc'] is not None else '-'
        if r['final_test_auc'] is None:
            test = '-'
        elif r['final_test_sd'] is not None and r['finals_done'] > 1:
            test = f"{r['final_test_auc']:.4f}+-{r['final_test_sd']:.4f}"
        else:
            test = f"{r['final_test_auc']:.4f}"
        note = r['note'] or ('done' if r['status'] == DONE else '')
        if r['failed']:
            note = f"{note} ({r['failed']} failed)".strip()
        lines.append(
            f"{r['arch']:<13} {r['trials_done']:>4}/{r['trials_total']:<5} "
            f"{r['pruned']:>7} {r['finals_done']:>2}/{r['finals_total']:<3} "
            f"{val:>9} {test:>15} {fmt_dur(r['elapsed']):>9} "
            f"{mark + fmt_dur(left):>10}  {note}")

    lines.append('-' * width)
    done = sum(1 for r in rows if r['status'] == DONE)
    summary = (f"{done}/{len(rows)} architectures done   "
               f"elapsed {fmt_dur(total_elapsed)}   remaining {fmt_dur(total_left)}")
    lines.append(summary)
    if total_left > 0:
        eta = datetime.now() + timedelta(seconds=total_left)
        lines.append(f"projected finish {eta:%Y-%m-%d %H:%M}"
                     + ('   (~ = from the cost probe, no finished trial yet)'
                        if any_estimated else ''))
    return '\n'.join(lines)


def discover_targets(out_root):
    if not os.path.isdir(out_root):
        return []
    return sorted(d for d in os.listdir(out_root)
                  if os.path.isdir(os.path.join(out_root, d)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/hpo')
    ap.add_argument('--target', nargs='*', help='default: every target under --out')
    ap.add_argument('-w', '--watch', action='store_true', help='refresh until all done')
    ap.add_argument('-n', '--interval', type=float, default=30.0, help='refresh seconds')
    args = ap.parse_args(argv)

    while True:
        targets = args.target or discover_targets(args.out)
        if not targets:
            print(f'nothing under {args.out} yet')
            if not args.watch:
                return
        blocks, left = [], 0.0
        for t in targets:
            st = collect(args.out, t)
            blocks.append(render(st))
            left += sum(r['remaining'] or 0.0 for r in st['rows'])
        text = '\n\n'.join(blocks)
        if args.watch:
            print('\033[2J\033[H', end='')
            print(f'{datetime.now():%Y-%m-%d %H:%M:%S}\n')
        print(text, flush=True)
        if len(targets) > 1:
            print(f'\nall targets: remaining {fmt_dur(left)}, '
                  f'projected finish {datetime.now() + timedelta(seconds=left):%Y-%m-%d %H:%M}')
        if not args.watch or left <= 0:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
