#!/usr/bin/env python
# coding: utf-8
"""Copy a SQLite database that is being written to, without tearing it.

The study database lives on node-local disk while a job runs and is copied back to
shared storage every few minutes. That copy is what survives a node failure, so it has
to be a database that opens.

rsync cannot make it one. In WAL mode the committed state of the study is split between
optuna.db and its write-ahead log, and a copy that catches the two at slightly different
moments is torn between them -- the resulting file may open and be missing trials, or
may not open at all. SQLite's own backup API holds a read transaction for the length of
the copy, so what it writes is consistent by construction. The result goes through a
temporary file and an atomic rename, so a copy interrupted half way leaves the previous
good one in place rather than replacing it with a fragment.

    python scripts/db_snapshot.py results/hpo/x/optuna.db /shared/hpo/x/optuna.db

Prints nothing and exits 0 when the source does not exist yet: a run that has not
created its study is not an error, it is the first few minutes of the job.
"""
import os
import sqlite3
import sys


def snapshot(src, dst, timeout=60.0):
    """Write a consistent copy of the SQLite database at `src` to `dst`.

    Returns False if there is nothing to copy yet. Raises sqlite3.Error if the source
    cannot be read, which the caller should treat as "keep the copy you already have".
    """
    if not os.path.exists(src):
        return False
    parent = os.path.dirname(os.path.abspath(dst))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f'{dst}.tmp'
    # read-write, not mode=ro: read-only access to a WAL database needs its shared-memory
    # file, and a database whose writer was killed needs its log replayed first. Opening
    # normally alongside the live writer is what SQLite's concurrency is for.
    con = sqlite3.connect(src, timeout=timeout)
    try:
        out = sqlite3.connect(tmp)
        try:
            con.backup(out)
            # the copy inherits WAL from the source, and WAL needs a shared-memory file
            # that network filesystems do not provide -- so merely opening the shared
            # copy to look at it, from a login node, could fail. A rollback journal has
            # no such requirement and reads fine from anywhere. The next job switches it
            # back to WAL when it stages the database onto local disk and opens it.
            out.execute('PRAGMA journal_mode=DELETE')
        finally:
            out.close()
    finally:
        con.close()
    for leftover in (f'{tmp}-wal', f'{tmp}-shm'):
        try:
            os.remove(leftover)
        except OSError:
            pass
    os.replace(tmp, dst)
    return True


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        if not snapshot(argv[1], argv[2]):
            return 0
    except (sqlite3.Error, OSError) as exc:
        # the copy already on disk stays as it is; the next sync will try again
        print(f'db_snapshot: {argv[1]}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
