"""Build the frozen train/validation subsets the fine-tuning benchmark runs on.

Run this once, before fine-tuning, if `data/splits/bundles/` is not already populated.
It needs the official MedMNIST files; `src/hpo_finetune.py` afterwards does not.

  1. draws the samples and writes `data/splits/<target>.json` -- the indices, per-class
     counts, checksums and the MD5 of the MedMNIST release they were cut from. A few tens
     of kB, tracked in git, and the definition of what the benchmark trains on.
  2. cuts those images out of `<target>_224.npz` and writes
     `data/splits/bundles/<target>_splits_224.npz`, plus a `SHA256SUMS` for the lot.

What gets drawn:

  Per fold, `n_classes*100` training images taken uniformly at random from the official
  training split -- not stratified, because the natural class imbalance is part of the
  task being studied -- and `n_classes*25` validation images taken from the official
  validation split. Nothing is removed from the training draw to build validation; the
  two come from the two pools MedMNIST already separates.

  Validation is allocated across classes in proportion to the class mix of the fold's
  own training draw, so it measures the quantity the test split does rather than a
  reweighted one, and a class that is rare in training is rare in validation too. It is
  drawn conditional on the training draw for that reason.

  A class the proportion leaves with fewer than 3 validation images takes half of its
  training count instead, and the largest classes pay for it. Half is what makes that
  self-limiting: a class holding 7 training images gets 3, one holding 5 gets 2, and no
  class is ever handed more validation images than it has training ones -- which is what
  a flat floor of 10 did, and why there is not one.

  Both parts are redrawn per fold, so their sampling error averages down over the folds
  of the final stage instead of sitting on every fold as one common offset.

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
from splits import (BUNDLE_SUFFIX, FOLDS, TARGETS, TRAIN_PER_CLASS,  # noqa: E402
                    VAL_PER_CLASS, VAL_SPARSE, VAL_SPARSE_SHARE, bundle_dir,
                    bundle_path, class_counts, entry_of, manifest_path, parts_of,
                    read_manifest, sha256)

MEDMNIST_ROOT = os.environ.get('MEDMNIST_ROOT', os.path.expanduser('~/.medmnist'))

SEED = 24                      # the seed src/data_split.py used, kept unchanged


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

def _allocate_val(train_counts, available, total):
    """How many validation images each class gets, given the training draw.

    `total` places handed out in proportion to the class mix of the training draw, by
    largest remainder, and never more of a class than the official validation split
    holds of it.

    On top of that, a class the proportion leaves with fewer than `VAL_SPARSE` images
    takes `VAL_SPARSE_SHARE` of its training count instead, and the largest classes pay
    for it. Below three positives a one-vs-rest AUC is not an estimate of anything, and
    that class still enters the macro average with the same weight as every other. Half
    of the training count is what makes the rescue self-limiting: it cannot hand a class
    more validation images than it has training ones, which is exactly what a flat floor
    of ten did to a class holding seven, and it leaves a class with five training images
    at two rather than pretending otherwise.

    A class the training draw misses entirely still gets no validation images, its AUC is
    undefined, and `auc_per_class` drops it from the macro average -- the honest outcome,
    since the draw genuinely cannot measure that class.
    """
    train_counts = np.asarray(train_counts, dtype=np.int64)
    available = np.asarray(available, dtype=np.int64)

    total = int(min(total, available.sum()))
    exact = total * train_counts / max(int(train_counts.sum()), 1)
    n = np.minimum(np.floor(exact).astype(np.int64), available)

    # hand out what rounding down left, by largest remainder among classes with room
    frac = exact - np.floor(exact)
    while int(n.sum()) < total:
        room = np.where(n < available, frac, -np.inf)
        if not np.isfinite(room).any():               # every class is at its pool size
            break
        c = int(np.argmax(room))
        n[c] += 1
        frac[c] -= 1.0                                # the next place goes elsewhere

    # lift the classes too sparse to estimate an AUC on, never above half their own
    # training count, and take the places back from whichever class is largest
    rescue = np.minimum((train_counts * VAL_SPARSE_SHARE).astype(np.int64), available)
    n = np.where(n < VAL_SPARSE, np.maximum(n, rescue), n)
    while int(n.sum()) > total:
        n[int(np.argmax(n))] -= 1
    return n


def draw_official(target_flag):
    """Training from the official training split, validation from the official one.

    Training is `n_classes*TRAIN_PER_CLASS` images drawn uniformly, so it carries the
    target's class imbalance exactly as the official split has it -- nothing is removed
    from it to build a validation set, which is the point of taking validation from the
    other pool.

    Validation is then drawn *conditional on that training draw*: `n_classes*VAL_PER_CLASS`
    images from the official validation split, allocated across classes in proportion to
    the training draw's own class mix, with `_allocate_val` lifting any class the
    proportion leaves too sparse to estimate an AUC on. That proportion is why the two
    are not drawn independently -- what a class is owed in validation is set by how much
    of it was drawn to train on, not by how much of it the official validation split
    happens to hold.

    Both are redrawn per fold, so their sampling error averages down over the folds of
    the final stage instead of resting on all of them as one common offset. Where the
    official validation split is barely larger than the budget the per-fold draws overlap
    and that averaging degrades gracefully towards none.
    """
    info = INFO[target_flag]
    n_classes = len(info['label'])
    n_train = info['n_samples']['train']
    labels_val = _labels(target_flag, 'val')
    y_train = _labels(target_flag, 'train')

    k_train = min(n_classes * TRAIN_PER_CLASS, n_train)
    k_val = n_classes * VAL_PER_CLASS

    train, val = {}, {}
    for fold in range(1, FOLDS + 1):
        rng = np.random.default_rng([SEED, fold])
        idx = np.sort(rng.choice(n_train, k_train, replace=False)).astype(np.int64)
        train[fold] = idx
        val[fold] = _draw_val_official(labels_val, y_train[idx], n_classes, k_val,
                                       info['task'], rng)
    return train, val


def _draw_val_official(labels, train_labels, n_classes, total, task, rng):
    """`total` images from the official validation split, given the training draw."""
    if task == 'multi-label, binary-class':
        # no single class per image to allocate on; keep the budget, draw uniformly
        return np.sort(rng.choice(len(labels), min(total, len(labels)),
                                  replace=False)).astype(np.int64)

    y = np.asarray(labels).reshape(-1)
    yt = np.asarray(train_labels).reshape(-1)
    members = [np.flatnonzero(y == c) for c in range(n_classes)]
    take = _allocate_val([int((yt == c).sum()) for c in range(n_classes)],
                         [len(m) for m in members], total)
    idx = np.concatenate([rng.choice(m, t, replace=False)
                          for m, t in zip(members, take) if t > 0])
    return np.sort(idx).astype(np.int64)


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


def build_manifest(target_flag, split_dir=None):
    """Draw the splits for one target and write `<target>.json`."""
    info = INFO[target_flag]
    common = {
        'target': target_flag,
        'n_classes': len(info['label']),
        'task': info['task'],
        'source': {'file': f'{target_flag}_224.npz', 'md5': info['MD5_224'],
                   'n_samples': info['n_samples']},
    }

    train, val = draw_official(target_flag)
    man = dict(common, draw={
        'train_per_class': TRAIN_PER_CLASS,
        'val_per_class': VAL_PER_CLASS,
        'val_sparse': VAL_SPARSE,
        'val_sparse_share': VAL_SPARSE_SHARE,
        'val_allocation': "in proportion to the class mix of the fold's own training "
                          'draw, by largest remainder, and never more of a class than '
                          'the official validation split holds of it; a class left with '
                          'fewer than %d images takes %g of its training count instead, '
                          'paid for out of the largest classes'
                          % (VAL_SPARSE, VAL_SPARSE_SHARE),
        'val_conditional_on_train': True,
        'folds': FOLDS, 'seed': SEED, 'stratified': False,
        'source_split': {'train': 'train', 'val': 'val'},
        'procedure': 'src/make_splits.py draw_official (PCG64, spawned per fold)',
    })
    man['train'] = {'folds_index': {str(f): _part_manifest(target_flag, 'train', idx)
                                    for f, idx in train.items()}}
    man['val'] = {'folds_index': {str(f): _part_manifest(target_flag, 'val', idx)
                                  for f, idx in val.items()}}

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
    """Which official split a part's indices point into, as the manifest records it."""
    return entry_of(man, part)['split']


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


def build(target_flag, split_dir=None, dest=None, rebuild=False, verbose=True):
    """Manifest and bundle for one target."""
    path = manifest_path(target_flag, split_dir)
    man = (build_manifest(target_flag, split_dir)
           if rebuild or not os.path.exists(path)
           else read_manifest(target_flag, split_dir))
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
            print(f'{t:16s} {state:8s} {size}  {path}')
            status |= state != 'ok'
            continue
        if os.path.exists(path) and not args.rebuild:
            print(f'{t:16s} exists   {os.path.getsize(path) / 1e9:6.2f} GB  {path}')
            continue
        print(f'{t}:', flush=True)
        _, (path, digest, size) = build(t, args.split_dir, args.dest, args.rebuild)
        sums[os.path.basename(path)] = digest
        print(f'{t:16s} built    {size / 1e9:6.2f} GB  sha256 {digest[:12]}')

    if sums:
        print(f'checksums -> {write_sha256sums(sums, args.dest)}')
    return int(status)


if __name__ == '__main__':
    sys.exit(main())
