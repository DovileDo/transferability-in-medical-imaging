#!/usr/bin/env python
# coding: utf-8
"""Run one target's HPO search under a memory cap, and restart it if it dies.

The pilot's only failure was not a bug in the search: the machine ran out of memory,
the kernel's OOM killer took the run down along with half the desktop session, and
nothing noticed for eleven hours. Two things were wrong with that, and this script
fixes both.

*Containment.* The run is launched inside a systemd scope with `MemoryMax` and
`MemorySwapMax=0`. If it exceeds its budget the cgroup OOM killer kills that run and
nothing else -- no other target, no system service. Forbidding swap is deliberate:
before it died the pilot spent over an hour thrashing, at one point taking 288s for a
stretch of steps it had been doing in 16s. Compute spent swapping is compute wasted,
and a run that would swap is better killed and restarted.

*Recovery.* The search is resumable by design -- the study is in optuna.db, finished
final runs are in final_runs.csv, and a resumed run marks stale RUNNING trials failed
and carries on -- so a restart costs at most the trial that was in flight. This script
restarts on any abnormal exit, with backoff, until the run reports completion or the
attempt limit is reached.

Restarting is only worth doing when the failure was transient, and the way to tell is
to ask what the attempt actually finished. An attempt that completed no trials at all
will almost certainly complete none on the next try either -- a target whose test split
does not fit in the memory it was given fails the same way every time -- so a few of
those in a row end the run instead of working through the retry schedule. Under a batch
scheduler that matters more than it does locally: the alternative is a GPU held for
hours re-running the same first minute.

It also reaps orphaned DataLoader forkservers before starting. Nine of them were found
holding memory on this machine after earlier runs died; their parents were long gone
and nothing else was going to clean them up.

    python scripts/run_hpo_guarded.py --target dermamnist -- --workers 8 --pruner median

Everything after `--` is passed to src/hpo_finetune.py unchanged.
"""
import argparse
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

try:
    import psutil
except ImportError:                                            # pragma: no cover
    psutil = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: exit statuses that mean "killed", not "failed". A shell reports a signalled child as
#: 128+signum, but Popen.returncode reports it as -signum, and both forms turn up here
#: depending on whether systemd-run or the child itself was the one signalled. 9 is
#: SIGKILL, which is what the cgroup OOM killer sends; 15 is SIGTERM, from a shutdown.
KILLED = {137, 143, -9, -15}


def reap_orphan_workers(dry_run=False):
    """Kill leftover multiprocessing helpers whose owning run is gone.

    A DataLoader forkserver is reparented to init when its job dies, and it keeps the
    memory it had. Only processes with PPID 1 are touched: a forkserver belonging to a
    live run is a child of that run, never of init, so this cannot disturb a job that
    is still going -- including a sibling target on another GPU.
    """
    if psutil is None:
        return []
    killed = []
    for p in psutil.process_iter(['pid', 'ppid', 'cmdline', 'username']):
        try:
            if p.info['ppid'] != 1 or p.info['username'] != os.environ.get('USER'):
                continue
            cmd = ' '.join(p.info['cmdline'] or [])
            if ('multiprocessing.forkserver' in cmd
                    or 'multiprocessing.resource_tracker' in cmd):
                killed.append(p.info['pid'])
                if not dry_run:
                    p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def clear_dead_lock(out_dir):
    """Remove a run.lock left behind by a killed run.

    hpo_finetune refuses to start when the lock names a live PID, which is right. After
    a SIGKILL the lock still names the dead one, and a PID can be reused, so identity is
    checked rather than liveness: the lock is only cleared when nothing is running under
    that PID that looks like this search.
    """
    import json
    lock = os.path.join(out_dir, 'run.lock')
    if not os.path.exists(lock):
        return False
    try:
        with open(lock) as f:
            info = json.load(f)
        pid = int(info.get('pid', 0))
    except Exception:
        os.remove(lock)
        return True
    alive = False
    if psutil is not None and pid:
        try:
            alive = 'hpo_finetune' in ' '.join(psutil.Process(pid).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            alive = False
    if alive:
        return False
    os.remove(lock)
    return True


def scope_available():
    """Can we actually put the run in a systemd scope?

    Checked by trying it, not by looking for the binary: on a batch-scheduler compute
    node systemd-run is installed but there is no user session bus behind it, and the
    call fails with 'Failed to connect to bus'. That is the normal state under Slurm,
    where the scheduler has already placed the job in its own cgroup with whatever
    --mem the batch script asked for -- so containment is handled and only the restart
    half of this script is still needed.
    """
    if shutil.which('systemd-run') is None:
        return False
    probe = subprocess.run(
        ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=64M', '--',
         'true'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return probe.returncode == 0


def tree_rss_gib(pid):
    """Resident memory of a process and its children, in GiB, for the no-scope case.

    An approximation, not the cgroup's exact high-water mark: pages shared between the
    DataLoader workers and their parent are counted once per process, so this reads
    higher than the scope figure for the same run. Useful for spotting a trend or a
    runaway, not for setting a cap to the megabyte.
    """
    if psutil is None:
        return None
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / 2**30


def scope_cgroup(unit):
    uid = os.getuid()
    return (f'/sys/fs/cgroup/user.slice/user-{uid}.slice/'
            f'user@{uid}.service/app.slice/{unit}.scope')


def release_unit(unit):
    """Clear a scope left loaded by a previous attempt.

    A scope whose process was killed stays registered, and systemd-run then refuses the
    name: 'Unit ... was already loaded or has a fragment file'. Without this the guard
    contains the first OOM and then fails to restart, which is the half of the job that
    matters. The name is kept stable rather than made unique per attempt so that the
    cgroup path stays predictable and peak memory can be read from it.
    """
    for verb in ('stop', 'reset-failed'):
        subprocess.run(['systemctl', '--user', verb, f'{unit}.scope'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def read_peak(unit):
    """Peak memory the scope reached, in GiB, or None if the kernel does not expose it."""
    for name in ('memory.peak', 'memory.max_usage_in_bytes'):
        try:
            with open(os.path.join(scope_cgroup(unit), name)) as f:
                return int(f.read().strip()) / 2**30
        except (OSError, ValueError):
            continue
    return None


def work_done(out_dir):
    """Finished work the study holds, as (trials, final runs), or None if unreadable.

    Compared across an attempt this says whether the attempt accomplished anything,
    which is what separates a crash loop from a flake. Elapsed time does not: a run that
    dies materialising the first test split can burn five minutes and be exactly as
    hopeless as one that dies in two seconds.

    COMPLETE and PRUNED trials both count, because both advance the search. FAILED ones
    do not: a trial killed mid-flight is left RUNNING and marked failed by the next
    attempt when it clears stale rows, so counting those would make a crash loop look
    like progress.
    """
    db = os.path.join(out_dir, 'optuna.db')
    if not os.path.exists(db):
        trials = 0
    else:
        try:
            # read-write, not mode=ro: a database whose writer was SIGKILLed may need its
            # write-ahead log replayed before it can be read, and a read-only connection
            # cannot do that
            con = sqlite3.connect(db, timeout=10)
            try:
                trials = con.execute("SELECT COUNT(*) FROM trials "
                                     "WHERE state IN ('COMPLETE', 'PRUNED')").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error:
            return None
    finals = 0
    path = os.path.join(out_dir, 'final_runs.csv')
    if os.path.exists(path):
        try:
            with open(path) as f:
                finals = max(sum(1 for _ in f) - 1, 0)     # minus the header
        except OSError:
            return None
    return trials, finals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', required=True)
    p.add_argument('--mem', default='48G',
                   help='memory cap for this run (systemd MemoryMax syntax). The pilot '
                        'peaked near 10G on dermamnist, but the cap has to cover the '
                        'largest test split a target materialises -- tissuemnist is '
                        '47k images at 224px -- so the default is generous. The peak '
                        'actually reached is printed after every attempt; tighten it '
                        'once you have seen a real number for the target')
    p.add_argument('--jobs', type=int, default=1,
                   help='how many of these you intend to run on THIS machine, used only '
                        'to warn when the caps oversubscribe RAM')
    p.add_argument('--max-attempts', type=int, default=8,
                   help='restarts allowed for failures that had made progress first. '
                        'Each one costs at most the trial that was in flight')
    p.add_argument('--max-unproductive', type=int, default=3,
                   help='stop after this many attempts in a row that finish no trial '
                        'and no final run. Those repeat, so retrying them only holds '
                        'the machine; 0 disables the check')
    p.add_argument('--unproductive-seconds', type=int, default=300,
                   help='fallback used only when the trial counts cannot be read: an '
                        'attempt shorter than this counts as unproductive')
    p.add_argument('--backoff', type=int, default=60,
                   help='seconds before the first restart, doubling to a 30 min ceiling. '
                        'Reset once an attempt gets somewhere')
    p.add_argument('--out', default='results/hpo')
    p.add_argument('--log', default=None, help='default: hpo_<target>.log in the repo')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='arguments after -- are passed to src/hpo_finetune.py')
    args = p.parse_args()

    extra = args.rest[1:] if args.rest[:1] == ['--'] else args.rest
    out_dir = os.path.join(REPO, args.out, args.target)
    log_path = args.log or os.path.join(REPO, f'hpo_{args.target}.log')
    unit = f'hpo-{args.target}'

    use_scope = scope_available()
    if psutil is not None and args.jobs > 1 and use_scope:
        total = psutil.virtual_memory().total / 2**30
        want = args.jobs * _to_gib(args.mem)
        if want > 0.8 * total:
            print(f'[warn] {args.jobs} jobs x {args.mem} = {want:.0f}G exceeds 80% of '
                  f'{total:.0f}G of RAM; lower --mem or run fewer at once', flush=True)

    cmd_inner = [args.python, os.path.join(REPO, 'src', 'hpo_finetune.py'),
                 '--target', args.target, '--out', args.out] + extra
    if use_scope:
        cmd = ['systemd-run', '--user', '--scope', '-q', '--unit', unit,
               '-p', f'MemoryMax={args.mem}', '-p', 'MemorySwapMax=0',
               '--'] + cmd_inner
        how = f'cap {args.mem}, no swap'
    else:
        cmd = cmd_inner
        how = ('no systemd scope available -- relying on the scheduler or the OS for '
               'containment, restarting on failure either way')

    print(f'guarded run: target {args.target}, {how}, '
          f'up to {args.max_attempts} attempts', flush=True)
    print(f'  log  {log_path}', flush=True)
    print(f'  cmd  {" ".join(cmd_inner)}', flush=True)

    delay = args.backoff
    stopping = {'asked': False}
    unproductive = 0
    for attempt in range(1, args.max_attempts + 1):
        before = work_done(out_dir)
        orphans = reap_orphan_workers()
        if orphans:
            print(f'[attempt {attempt}] reaped {len(orphans)} orphaned worker(s): '
                  f'{orphans}', flush=True)
        if clear_dead_lock(out_dir):
            print(f'[attempt {attempt}] cleared a stale run.lock', flush=True)
        if use_scope:
            release_unit(unit)

        started = time.time()
        with open(log_path, 'a') as log:
            log.write(f'\n=== guarded attempt {attempt} at '
                      f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            log.flush()
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)

            # a signal from outside means stop, not fail: a batch scheduler sending
            # SIGTERM at the wall-clock limit wants the job to end, and restarting the
            # run inside a job that is already being torn down would waste whatever
            # time is left. The child is asked to stop and the loop exits with 2, which
            # the submitting script reads as "not finished, resubmit me"
            def forward(signum, _frame):
                stopping['asked'] = True
                proc.send_signal(signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, forward)

            # sampled often, because a run that dies during startup dies in seconds and
            # its peak is the number that says whether the cap was the reason. With a
            # scope the kernel keeps the high-water mark for us and the number is exact.
            # Without one it is sampled from the process tree, which is only
            # approximate in both directions: summing RSS counts pages shared between
            # the workers and their parent once per process, while sampling once a
            # second misses anything shorter than that
            peak = None
            while proc.poll() is None:
                now = read_peak(unit) if use_scope else tree_rss_gib(proc.pid)
                peak = max(now or 0.0, peak or 0.0) or None
                time.sleep(1)
            rc = proc.returncode

        elapsed = time.time() - started
        hours = elapsed / 3600
        peak_s = f', peak {peak:.1f} GiB' if peak else ''
        after = work_done(out_dir)
        gain = ''
        if before is not None and after is not None:
            parts = [f'+{n} {what}'
                     for n, what in zip((after[0] - before[0], after[1] - before[1]),
                                        ('trials', 'final runs')) if n]
            gain = (', ' + ', '.join(parts)) if parts else ''
        if stopping['asked']:
            print(f'[attempt {attempt}] stopped on request after {hours:.1f} h'
                  f'{peak_s}{gain}; the search is resumable, resubmit to continue',
                  flush=True)
            return 2
        if rc == 0:
            print(f'[attempt {attempt}] finished cleanly after {hours:.1f} h'
                  f'{peak_s}{gain}', flush=True)
            return 0
        why = ('killed (out of memory, or terminated)' if rc in KILLED
               else f'exited {rc}')
        print(f'[attempt {attempt}] {why} after {hours:.1f} h{peak_s}{gain}', flush=True)
        # only blame the cap when the run was actually near it: a kill at 6 GiB under a
        # 48 GiB cap came from somewhere else, and saying otherwise sends you tuning the
        # wrong number
        if rc in KILLED and peak and peak > 0.8 * _to_gib(args.mem):
            print(f'           peak {peak:.1f} GiB against a {args.mem} cap: this run '
                  f'hit its own limit, raise --mem', flush=True)
        elif rc in KILLED and peak:
            print(f'           peak {peak:.1f} GiB is well under the {args.mem} cap, so '
                  f'the kill came from outside this run', flush=True)

        # a failure that finished nothing will finish nothing next time either. Give
        # those their own small budget so a run that cannot start at all releases the
        # machine in minutes, and keep the full budget for failures that had been making
        # progress -- an attempt that got somewhere resets both the count and the backoff,
        # since a run that died after six hours says nothing about one that died at start
        if before is None or after is None:
            productive = elapsed >= args.unproductive_seconds
        else:
            productive = after > before
        if productive:
            unproductive = 0
            delay = args.backoff
        elif args.max_unproductive > 0:
            unproductive += 1
            print(f'           this attempt finished no trial and no final run '
                  f'({unproductive} of {args.max_unproductive} in a row)', flush=True)
            if unproductive >= args.max_unproductive:
                print(f'giving up on {args.target}: {unproductive} attempts in a row got '
                      f'nowhere, so the failure is not transient and retrying it would '
                      f'only hold the machine. See {log_path}', flush=True)
                return 1

        if attempt == args.max_attempts:
            break
        print(f'           restarting in {delay}s (the search resumes from optuna.db)',
              flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 1800)

    print(f'giving up on {args.target} after {args.max_attempts} attempts', flush=True)
    return 1


def _to_gib(spec):
    spec = str(spec).strip().upper()
    mult = {'K': 2**-20, 'M': 2**-10, 'G': 1, 'T': 1024}
    if spec and spec[-1] in mult:
        return float(spec[:-1]) * mult[spec[-1]]
    return float(spec) / 2**30


if __name__ == '__main__':
    sys.exit(main())
