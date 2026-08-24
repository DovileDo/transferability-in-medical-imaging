"""Build the frozen train/validation subsets the fine-tuning benchmark runs on.

Run this once, before fine-tuning, if `data/splits/bundles/` is not already populated.
It needs the official MedMNIST files; `src/hpo_finetune.py` afterwards does not.

  1. draws the samples and writes `data/splits/<target>.json` -- the indices, per-class
     counts, checksums and the MD5 of the MedMNIST release they were cut from. A few tens
     of kB, tracked in git, and the definition of what the benchmark trains on.
  2. cuts those images out of `<target>_224.npz` and writes
     `data/splits/bundles/<target>_splits_224.npz`, plus a `SHA256SUMS` for the lot.

What gets drawn (scheme 'capped', the default):

  Per fold, one draw of `n_classes*125` images taken uniformly at random from the
  official training split -- not stratified, because the natural class imbalance is part
  of the task being studied -- and then cut into a training and a validation part. That
  is the whole point of the scheme: the benchmark simulates having one small dataset, so
  training and validation have to come out of one collection.

  The cut holds out up to 25 images per class, and never more than half of what that
  class has in the draw. The cap is what keeps the sample realisable: drawing validation
  from the official validation split instead, as this script used to, gave DermaMNIST
  fold 2 seven training images of its rarest class and twelve validation images of it --
  a state no single collection of 875 images could produce. The per-class budget is what
  keeps a macro AUC estimable: a plain proportional holdout leaves the rarest classes
  with one or two positives, and their one-vs-rest AUC is then pure noise.

  Validation is drawn per fold, so unlike the single shared split this replaces, its
  sampling error averages down over the folds of the final stage instead of sitting on
  every fold as the same offset.

  Scheme 'legacy' reproduces the older draw -- `n_classes*100` training images per fold
  from the official training split and one shared `25`-per-class validation split from
  the official validation split -- for rebuilding the bundles earlier results were
  produced against.

The bundles are what the benchmark reads and what should be archived (OSF): an index
reproduces a sample only for as long as the file it indexes into stays available and
unchanged, and a re-release of MedMNIST would silently change the experiment. Keeping the
pixels is the only thing that survives that -- and it is checked, not assumed, on every
load.

Usage
-----
python src/make_splits.py --all                      # everything, skipping what exists
python src/make_splits.py --targets dermamnist       # one target
python src/make_splits.py --all --verify             # report, write nothing
python src/make_splits.py --targets dermamnist --rebuild   # redraw from scratch
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime

import numpy as np
from medmnist import INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import (BUNDLE_SUFFIX, FOLDS, SCHEME, TARGETS,  # noqa: E402
                    TOTAL_PER_CLASS, VAL_CAP_FRACTION, VAL_PER_CLASS, bundle_dir,
                    bundle_path, class_counts, entry_of, manifest_path, parts_of,
                    read_manifest, scheme_of, sha256)

MEDMNIST_ROOT = os.environ.get('MEDMNIST_ROOT', os.path.expanduser('~/.medmnist'))

TRAIN_PER_CLASS = 100          # scheme 'legacy' only: n_classes*100 training images
SEED = 24                      # the seed src/data_split.py used, kept for both schemes


def source_path(target_flag):
    return os.path.join(MEDMNIST_ROOT, f'{target_flag}_224.npz')


def fetch_source(target_flag, verbose=True):
    """The 224px MedMNIST file, downloaded if it is not already here.

    medmnist.INFO carries the URL and the MD5 for every resolution, so this needs nothing
    the package does not already know. The MD5 is checked before the file is put in
    place: a truncated download that kept its name would otherwise be indistinguishable
    from a good one, and would surface much later as a hash mismatch inside a bundle.
    """
    path = source_path(target_flag)
    if os.path.exists(path):
        return path

    from medmnist import INFO
    info = INFO[target_flag]
    url, want = info['url_224'], info['MD5_224']
    os.makedirs(MEDMNIST_ROOT, exist_ok=True)
    tmp = f'{path}.part'
    if verbose:
        print(f'{target_flag}: fetching {url}', flush=True)

    import urllib.request
    # a line per gigabyte rather than a redrawn bar: pathmnist is 12 GB and this ends up
    # in a Slurm log, where carriage returns are unreadable and a progress bar is tens of
    # kilobytes of it
    STEP = 1 << 30
    h = hashlib.md5()
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = next_mark = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
            done += len(chunk)
            if verbose and done >= next_mark + STEP:
                next_mark = done - done % STEP
                of = f' of {total / 2**30:.0f}' if total else ''
                print(f'  {done / 2**30:.0f}{of} GiB', flush=True)

    got = h.hexdigest()
    if got != want:
        os.remove(tmp)
        raise ValueError(f'{target_flag}: downloaded file has MD5 {got}, MedMNIST says '
                         f'{want}. Not keeping it.')
    os.replace(tmp, path)
    if verbose:
        mb = os.path.getsize(path) / 2**20
        size = f'{mb / 1024:.1f} GiB' if mb >= 1024 else f'{mb:.0f} MiB'
        print(f'{target_flag}: {path} ({size}, MD5 ok)', flush=True)
    return path


def ensure(target_flag, split_dir=None, dest=None, verbose=True):
    """Everything one target needs to train, built or downloaded if it is not there.

    The manifest is tracked in git and is not redrawn: `build` reuses it when it exists,
    so what comes out here holds exactly the images the manifest names, and load_part
    re-hashes them against it when they are read. Nothing about which images are used
    depends on whether this had to build anything.
    """
    from splits import check_bundle

    try:
        check_bundle(target_flag, dest=dest, split_dir=split_dir)
        have_bundle = True
    except (FileNotFoundError, ValueError):
        have_bundle = False

    if have_bundle and os.path.exists(source_path(target_flag)):
        return False                                   # nothing to do

    fetch_source(target_flag, verbose)                 # needed either way: build reads it,
    if have_bundle:                                    # and the test split is read from it
        return True
    if verbose:
        print(f'{target_flag}: building the bundle', flush=True)
    build(target_flag, split_dir=split_dir, dest=dest, verbose=verbose)
    return True


# --------------------------------------------------------------------------------------
# reading rows out of a MedMNIST npz without decompressing it into memory
# --------------------------------------------------------------------------------------

def _read_header(f):
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f'unsupported .npy version {version}')


def _skip(f, n, chunk=1 << 23):
    while n > 0:
        got = f.read(min(n, chunk))
        if not got:
            raise EOFError('unexpected end of array data')
        n -= len(got)


def read_rows(npz_path, member, index):
    """Rows `index` of one array inside an npz, read as a stream.

    The MedMNIST files are deflate-compressed, so they cannot be memory-mapped, and
    `np.load(...)[member]` would materialise the whole array -- 13.5 GB of pathmnist
    training images to pull out 900 of them. Reading the member in ascending index order
    costs one pass of decompression and holds only the selected rows.
    """
    index = np.asarray(index, dtype=np.int64)
    order = np.argsort(index, kind='stable')
    with zipfile.ZipFile(npz_path) as z, z.open(f'{member}.npy') as f:
        shape, fortran, dtype = _read_header(f)
        if fortran:
            raise ValueError(f'{member} is Fortran-ordered, not supported')
        if index.size and (index.min() < 0 or index.max() >= shape[0]):
            raise IndexError(f'index out of range for {member} with {shape[0]} rows -- '
                             f'this is not the file the split was drawn from')
        row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        out = np.empty((len(index),) + tuple(shape[1:]), dtype=dtype)
        pos = 0
        for j in order:
            i = int(index[j])
            _skip(f, i * row - pos)
            buf = f.read(row)
            if len(buf) != row:
                raise EOFError(f'short read for row {i} of {member}')
            out[j] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
            pos = (i + 1) * row
    return out


def _labels(target_flag, split):
    """Labels for one official split, from the 28px file when it is the only one here.

    The resolutions are the same dataset at different sizes: same order, same labels
    (checked for every target with both files present), and the 28px files are a few tens
    of MB. That is what lets the manifest be drawn for a target whose 224px file has not
    been downloaded yet -- only the bundle needs it.
    """
    for name in (f'{target_flag}_224.npz', f'{target_flag}.npz'):
        path = os.path.join(MEDMNIST_ROOT, name)
        if os.path.exists(path):
            with np.load(path) as d:
                return d[f'{split}_labels']
    raise FileNotFoundError(
        f'neither {target_flag}_224.npz nor {target_flag}.npz found in {MEDMNIST_ROOT}; '
        f'MedMNIST is needed to draw the split (the 28px file is enough for that, the '
        f'224px file for the bundle)')


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------

def draw_train(target_flag):
    """Per-fold training indices: exactly what src/data_split.py drew.

    That script seeded the legacy global RandomState with 24 and, per fold, drew the
    training subset and then the validation subsample, so the validation draws are
    replayed here purely to keep the stream aligned. Legacy RandomState is stream-stable
    across NumPy versions by policy, and the draws depend only on the split sizes, so
    this reproduces the published dermamnist folds byte for byte. It does not reproduce
    the other targets' published subsets -- those were written from a different state of
    the RNG, and no ordering of the draws in that script recovers them.
    """
    n_classes = len(INFO[target_flag]['label'])
    n = INFO[target_flag]['n_samples']
    k_train, k_val = n_classes * TRAIN_PER_CLASS, n_classes * VAL_PER_CLASS

    np.random.seed(SEED)
    folds = {}
    for fold in range(1, FOLDS + 1):
        folds[fold] = np.sort(np.random.choice(n['train'], k_train, replace=False))
        if n['val'] > k_val:
            np.random.choice(n['val'], k_val, replace=False)     # stream alignment only
    return folds


def draw_val(target_flag):
    """Validation indices: VAL_PER_CLASS per class, or all of a class that has fewer."""
    info = INFO[target_flag]
    n_classes = len(info['label'])
    labels = _labels(target_flag, 'val')
    rng = np.random.default_rng(SEED)

    if info['task'] == 'multi-label, binary-class':
        # no single class per image to stratify on; keep the budget, draw uniformly
        idx = rng.choice(len(labels), min(len(labels), VAL_PER_CLASS * n_classes),
                         replace=False)
    else:
        y = labels.reshape(-1)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(y == c), min(int((y == c).sum()), VAL_PER_CLASS),
                       replace=False)
            for c in range(n_classes) if (y == c).any()])
    return np.sort(idx).astype(np.int64)


def _cut_one_draw(y, n_classes, index, task, rng):
    """Split one natural-prevalence draw into a training and a validation part.

    Validation takes up to VAL_PER_CLASS images of each class and never more than
    VAL_CAP_FRACTION of what that class holds in the draw. The cap is the part that
    matters: it is what makes the result a split of one collection rather than two
    samples of different things, and it guarantees no class ends up with more
    validation images than training images.

    A class with a single image in the draw keeps it for training. Its one-vs-rest AUC
    is then undefined and `auc_per_class` drops it from the macro average -- which is
    the honest outcome, because at that point the collection genuinely cannot measure
    that class.
    """
    drawn = y[index]
    train, val = [], []

    if task == 'multi-label, binary-class':
        # no single class per image to hold out on; keep the budget, draw uniformly
        n_val = min(int(len(index) * VAL_CAP_FRACTION), VAL_PER_CLASS * n_classes)
        perm = rng.permutation(len(index))
        val = index[perm[:n_val]]
        train = index[perm[n_val:]]
        return np.sort(train).astype(np.int64), np.sort(val).astype(np.int64)

    for c in range(n_classes):
        members = np.flatnonzero(drawn == c)
        if len(members) == 0:
            continue
        n_val = min(VAL_PER_CLASS, int(len(members) * VAL_CAP_FRACTION))
        perm = rng.permutation(members)
        val.extend(index[perm[:n_val]])
        train.extend(index[perm[n_val:]])
    return (np.sort(train).astype(np.int64), np.sort(val).astype(np.int64))


def draw_capped(target_flag):
    """`({fold: train_index}, {fold: val_index})` -- the default scheme.

    Each fold is one draw of `n_classes*TOTAL_PER_CLASS` images taken uniformly from the
    official training split and then cut in two. Folds are drawn from independently
    spawned streams rather than one shared sequence, so a fold's indices depend only on
    the seed, the target and its own number: adding a sixth fold, or rebuilding one,
    cannot perturb the others.
    """
    info = INFO[target_flag]
    n_classes = len(info['label'])
    y = np.asarray(_labels(target_flag, 'train')).reshape(-1)
    n_total = min(n_classes * TOTAL_PER_CLASS, len(y))

    train, val = {}, {}
    for fold in range(1, FOLDS + 1):
        rng = np.random.default_rng([SEED, fold])
        index = rng.choice(len(y), n_total, replace=False)
        train[fold], val[fold] = _cut_one_draw(y, n_classes, index, info['task'], rng)
    return train, val


def _part_manifest(target_flag, split, index):
    """One part of the manifest. `split` is the official split its indices point into."""
    labels = _labels(target_flag, split)[np.asarray(index, dtype=np.int64)]
    return {
        'split': split,                 # which official split `index` indexes into
        'n_images': int(len(index)),
        'class_counts': class_counts(labels, target_flag),
        'labels_sha256': sha256(labels),
        'images_sha256': None,          # filled when the pixels are cut
        'index': np.asarray(index, dtype=np.int64).tolist(),
    }


def build_manifest(target_flag, split_dir=None, scheme=SCHEME):
    """Draw the splits for one target and write `<target>.json`."""
    info = INFO[target_flag]
    common = {
        'target': target_flag,
        'n_classes': len(info['label']),
        'task': info['task'],
        'source': {'file': f'{target_flag}_224.npz', 'md5': info['MD5_224'],
                   'n_samples': info['n_samples']},
    }

    if scheme == 'capped':
        train, val = draw_capped(target_flag)
        man = dict(common, draw={
            'scheme': 'capped',
            'total_per_class': TOTAL_PER_CLASS,
            'val_per_class': VAL_PER_CLASS,
            'val_cap_fraction': VAL_CAP_FRACTION,
            'folds': FOLDS, 'seed': SEED, 'stratified': False,
            'source_split': 'train',
            'procedure': 'src/make_splits.py draw_capped (PCG64, spawned per fold)',
        })
        man['train'] = {'folds_index': {str(f): _part_manifest(target_flag, 'train', idx)
                                        for f, idx in train.items()}}
        man['val'] = {'folds_index': {str(f): _part_manifest(target_flag, 'train', idx)
                                      for f, idx in val.items()}}
    elif scheme == 'legacy':
        train = draw_train(target_flag)
        man = dict(common, draw={
            'scheme': 'legacy',
            'train_per_class': TRAIN_PER_CLASS, 'val_per_class': VAL_PER_CLASS,
            'folds': FOLDS, 'seed': SEED, 'stratified': False,
            'procedure': 'src/data_split.py replay (legacy RandomState)',
        })
        man['train'] = {'per_class': TRAIN_PER_CLASS, 'folds': FOLDS, 'seed': SEED,
                        'stratified': False,
                        'procedure': 'src/data_split.py replay (legacy RandomState)',
                        'folds_index': {str(f): _part_manifest(target_flag, 'train', idx)
                                        for f, idx in train.items()}}
        man['val'] = dict(_part_manifest(target_flag, 'val', draw_val(target_flag)),
                          per_class=VAL_PER_CLASS, seed=SEED, stratified=True)
    else:
        raise ValueError(f'unknown scheme {scheme!r} (capped or legacy)')

    man['created'] = datetime.now().isoformat(timespec='seconds')
    write_manifest(man, manifest_path(target_flag, split_dir))
    return man


def write_manifest(man, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(man, f, indent=1)
    os.replace(path + '.tmp', path)


# --------------------------------------------------------------------------------------
# cutting the pixels and writing the bundle
# --------------------------------------------------------------------------------------

def _part(man, part):
    return entry_of(man, part)


def _source_split(man, part):
    """Which official split a part's indices point into.

    Recorded per part since the capped draw takes validation out of the training pool
    too; manifests written before the field existed had validation, and only
    validation, coming from the official validation split.
    """
    entry = entry_of(man, part)
    if 'split' in entry:
        return entry['split']
    return 'val' if part == 'val' else 'train'


def cut(target_flag, man, verbose=True):
    """Every part of one target, cut out of the official npz and checked.

    One decompression pass per split: the pass costs the same whether one fold is wanted
    or five, and it is 12.6 GB for pathmnist.
    """
    src = source_path(target_flag)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{src} not found. The official MedMNIST file is what the subsets are cut '
            f'from -- download it (medmnist.INFO has the URL and MD5), or copy an '
            f'already-built {target_flag}{BUNDLE_SUFFIX} into {bundle_dir()}.')

    out = {}
    parts = parts_of(man)
    by_split = {}
    for part in parts:
        by_split.setdefault(_source_split(man, part), []).append(part)

    for split, want in sorted(by_split.items()):
        per_part = {p: np.asarray(_part(man, p)['index'], dtype=np.int64) for p in want}
        union = np.unique(np.concatenate(list(per_part.values())))
        if verbose:
            print(f'  cutting {len(union)} {split} images out of '
                  f'{os.path.basename(src)} ...', flush=True)
        pool = read_rows(src, f'{split}_images', union)
        with np.load(src) as d:
            all_labels = d[f'{split}_labels']

        for p, index in per_part.items():
            entry = _part(man, p)
            images, labels = pool[np.searchsorted(union, index)], all_labels[index]
            if sha256(labels) != entry['labels_sha256']:
                raise ValueError(
                    f'{target_flag} {p}: labels from {src} do not match the manifest. '
                    f'That file is not the MedMNIST release the split was drawn from '
                    f'(the manifest expects md5 {man["source"]["md5"][:12]}).')
            got = sha256(images)
            if entry['images_sha256'] is None:
                entry['images_sha256'] = got
            elif got != entry['images_sha256']:
                raise ValueError(
                    f'{target_flag} {p}: images hash to {got[:12]}, the manifest says '
                    f'{entry["images_sha256"][:12]}. Same indices, different pixels -- '
                    f'check that {src} is the release recorded in the manifest.')
            out[p] = (images, labels)
        del pool
    return out


def write_bundle(target_flag, man, parts, dest=None, verbose=True):
    """One self-contained npz per target: the pixels, plus the manifest that defines them."""
    arrays = {'manifest': np.array(json.dumps(man))}
    for part, (images, labels) in parts.items():
        arrays[f'{part}_images'] = images
        arrays[f'{part}_labels'] = labels
        arrays[f'{part}_index'] = np.asarray(_part(man, part)['index'], dtype=np.int64)

    path = bundle_path(target_flag, dest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if verbose:
        print(f'  compressing -> {path}', flush=True)
    np.savez_compressed(path + '.tmp.npz', **arrays)
    os.replace(path + '.tmp.npz', path)

    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return path, digest.hexdigest(), os.path.getsize(path)


def build(target_flag, split_dir=None, dest=None, rebuild=False, verbose=True,
          scheme=SCHEME):
    """Manifest and bundle for one target."""
    path = manifest_path(target_flag, split_dir)
    if rebuild or not os.path.exists(path):
        man = build_manifest(target_flag, split_dir, scheme)
    else:
        man = read_manifest(target_flag, split_dir)
        if scheme_of(man) != scheme:
            raise SystemExit(
                f'{path} describes a {scheme_of(man)!r} draw but --scheme is '
                f'{scheme!r}. Redrawing would invalidate every result produced against '
                f'it, so it is not done implicitly: pass --rebuild to redraw, or '
                f'--scheme {scheme_of(man)} to keep this one.')
    parts = cut(target_flag, man, verbose)
    write_manifest(man, manifest_path(target_flag, split_dir))   # checksums now filled
    return man, write_bundle(target_flag, man, parts, dest, verbose)


def write_sha256sums(entries, dest=None):
    path = os.path.join(bundle_dir(dest), 'SHA256SUMS')
    have = {}
    if os.path.exists(path):
        with open(path) as f:
            have = {line.split('  ', 1)[1].strip(): line.split('  ', 1)[0]
                    for line in f if '  ' in line}
    have.update(entries)
    with open(path, 'w') as f:
        f.write(''.join(f'{d}  {n}\n' for n, d in sorted(have.items())))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--targets', nargs='+', help=f'default: all 11 ({", ".join(TARGETS)})')
    p.add_argument('--all', action='store_true', help='all 11 targets')
    p.add_argument('--split-dir', default=None, help='where the manifests go')
    p.add_argument('--dest', default=None, help=f'where the bundles go (default '
                                               f'{bundle_dir()})')
    p.add_argument('--rebuild', action='store_true',
                   help='redraw the manifest and rewrite the bundle')
    p.add_argument('--scheme', choices=['capped', 'legacy'], default=SCHEME,
                   help="'capped' (default) draws n_classes*%d images per fold at "
                        'natural prevalence and splits each draw into train and '
                        'validation; \'legacy\' reproduces the older two-pool draw'
                        % TOTAL_PER_CLASS)
    p.add_argument('--verify', action='store_true', help='report only, write nothing')
    args = p.parse_args(argv)

    targets = args.targets or (TARGETS if args.all else None)
    if not targets:
        p.error('give --targets or --all')

    status, sums = 0, {}
    for t in targets:
        path = bundle_path(t, args.dest)
        if args.verify:
            state = 'ok' if os.path.exists(path) else 'MISSING'
            size = f'{os.path.getsize(path) / 1e9:6.2f} GB' if state == 'ok' else ' ' * 9
            mpath = manifest_path(t, args.split_dir)
            drawn = scheme_of(read_manifest(t, args.split_dir)) if os.path.exists(mpath) \
                else '?'
            print(f'{t:16s} {state:8s} {size}  {drawn:7s} {path}')
            status |= state != 'ok'
            continue
        if os.path.exists(path) and not args.rebuild:
            print(f'{t:16s} exists   {os.path.getsize(path) / 1e9:6.2f} GB  {path}')
            continue
        print(f'{t}:', flush=True)
        _, (path, digest, size) = build(t, args.split_dir, args.dest, args.rebuild,
                                        scheme=args.scheme)
        sums[os.path.basename(path)] = digest
        print(f'{t:16s} built    {size / 1e9:6.2f} GB  sha256 {digest[:12]}')

    if sums:
        print(f'checksums -> {write_sha256sums(sums, args.dest)}')
    return int(status)


if __name__ == '__main__':
    sys.exit(main())
