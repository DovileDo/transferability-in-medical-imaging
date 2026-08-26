"""Reading the frozen train/validation subsets the fine-tuning benchmark runs on.

The subsets live in `data/splits/bundles/<target>_splits_224.npz`, one self-contained
archive per target holding the actual images for the validation split and all five
training folds, with the manifest that defines them embedded inside. Nothing here needs
MedMNIST, draws anything, or depends on a random stream: the pixels are read from the
bundle and checked against the checksums it carries.

Build the bundles first with `python src/make_splits.py --all` (that is the only step
that needs the official MedMNIST files), or fetch them from the archive they were
uploaded to. `data/splits/<target>.json` is the same manifest, tracked in git, and is
cross-checked against the bundle when it is present -- so a bundle holding a different
sample than the repository describes is refused rather than used.

A bundle is a plain compressed npz and can be read without this module:

    d = np.load('dermamnist_splits_224.npz')
    d['train_fold1_images'], d['train_fold1_labels']    # (700, 224, 224, 3), (700, 1)
    d['val_fold1_images'], d['val_fold1_labels']        # (175, 224, 224, 3), (175, 1)
    d['train_fold1_index']                              # rows of the official train split
    json.loads(str(d['manifest']))                      # checksums, counts, provenance

A fold is two draws from two pools: the training part indexes into the official
*training* split, the validation part into the official *validation* split. Both are
redrawn per fold, so every fold has its own `train_fold<k>` and `val_fold<k>`.
"""

import hashlib
import json
import os
import zipfile

import numpy as np
from medmnist import INFO

#: the 11 targets behind results/AUCs_model.csv
TARGETS = ['bloodmnist', 'breastmnist', 'dermamnist', 'octmnist', 'organamnist',
           'organcmnist', 'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist',
           'tissuemnist']

#: training images per class, drawn uniformly from the official training split, so the
#: training set carries the target's own class imbalance exactly as drawn.
TRAIN_PER_CLASS = 100
#: validation images per class of budget: `n_classes*25` drawn from the official
#: validation split, allocated in proportion to the training draw's own class mix.
VAL_PER_CLASS = 25
#: a class the proportional allocation leaves with fewer than this many validation
#: images takes VAL_SPARSE_SHARE of its training count instead: under three positives a
#: one-vs-rest AUC is not an estimate of anything.
VAL_SPARSE = 3
#: what such a class takes instead, as a share of what it holds in training. Half is
#: self-limiting -- it can never hand a class more validation images than training ones,
#: which is what a flat floor of three would do to a class holding four.
VAL_SPARSE_SHARE = 0.5
FOLDS = 5
BUNDLE_SUFFIX = '_splits_224.npz'

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_dir():
    """Where the manifests live."""
    return os.environ.get('SPLIT_DIR') or os.path.join(_REPO, 'data', 'splits')


def bundle_dir(dest=None):
    return dest or os.environ.get('BUNDLE_DIR') or os.path.join(default_dir(), 'bundles')


def bundle_path(target_flag, dest=None):
    return os.path.join(bundle_dir(dest), f'{target_flag}{BUNDLE_SUFFIX}')


def manifest_path(target_flag, split_dir=None):
    return os.path.join(split_dir or default_dir(), f'{target_flag}.json')


def parts_of(man, folds=None):
    """Every part one manifest describes: a training and a validation part per fold."""
    folds = sorted(int(f) for f in man['train']['folds_index']) if folds is None else folds
    return [f'train_fold{f}' for f in folds] + [f'val_fold{f}' for f in folds]


def sha256(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def class_counts(labels, target_flag):
    if INFO[target_flag]['task'] == 'multi-label, binary-class':
        return np.asarray(labels).sum(axis=0).astype(int).tolist()
    return np.bincount(np.asarray(labels).reshape(-1),
                       minlength=len(INFO[target_flag]['label'])).tolist()


def read_manifest(target_flag, split_dir=None):
    """The git-tracked manifest for one target."""
    path = manifest_path(target_flag, split_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. It is tracked in git, so pull it -- or rebuild it with '
            f'"python src/make_splits.py --targets {target_flag}".')
    with open(path) as f:
        return json.load(f)


def entry_of(man, part):
    """The manifest section for one part, 'train_fold<k>' or 'val_fold<k>'."""
    for kind in ('train', 'val'):
        prefix = f'{kind}_fold'
        if part.startswith(prefix):
            fold = part[len(prefix):]
            try:
                return man[kind]['folds_index'][fold]
            except KeyError:
                raise KeyError(f'{part}: fold {fold} is not in the manifest '
                               f'(it has {sorted(man[kind]["folds_index"])})') from None
    raise KeyError(f'unknown part {part!r}')


def part_sha256(target_flag, parts=None, split_dir=None):
    """`{part: images_sha256}` from the git-tracked manifest, without reading pixels.

    Lets a run record what every fold it will touch is supposed to be, at startup and
    for the price of reading one JSON file. `load_part` is what checks that the bundle
    actually holds those images, when the fold is used.
    """
    man = read_manifest(target_flag, split_dir)
    return {p: entry_of(man, p)['images_sha256'] for p in (parts or parts_of(man))}


def _open(target_flag, dest=None):
    path = bundle_path(target_flag, dest)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- the subsets have not been built on this machine. '
            f'Either copy the bundles in, or build them from MedMNIST with:\n'
            f'    python src/make_splits.py --targets {target_flag}')
    return path


def check_bundle(target_flag, parts=None, dest=None, split_dir=None):
    """Cheap up-front check: the bundle is there and holds the parts, without reading them.

    Opening the zip directory is instant, so a run can fail on a missing or truncated
    bundle before it spends anything, rather than four folds into a final stage.
    """
    path = _open(target_flag, dest)
    parts = parts or parts_of(read_manifest(target_flag, split_dir))
    with zipfile.ZipFile(path) as z:
        have = set(z.namelist())
    missing = [p for p in parts
               if f'{p}_images.npy' not in have or f'{p}_labels.npy' not in have]
    if missing:
        raise ValueError(f'{path} is missing {", ".join(missing)} -- rebuild it with '
                         f'"python src/make_splits.py --targets {target_flag} --rebuild"')
    return path


def load_part(target_flag, part, dest=None, split_dir=None, verbose=True):
    """`(images, labels, entry)` for one part, read out of the bundle.

    The images are hashed and compared with the manifest inside the bundle, and with the
    git-tracked manifest when there is one, so neither a damaged archive nor a bundle
    from a different draw can quietly change what is trained or selected on.
    """
    path = _open(target_flag, dest)
    with np.load(path, allow_pickle=False) as d:
        man = json.loads(str(d['manifest']))
        entry = entry_of(man, part)
        images, labels = d[f'{part}_images'], d[f'{part}_labels']

    got = sha256(images)
    if entry['images_sha256'] and got != entry['images_sha256']:
        raise ValueError(
            f'{path}: {part} images hash to {got[:12]}, the manifest inside it says '
            f'{entry["images_sha256"][:12]}. The archive is damaged; fetch it again or '
            f'rebuild it with "python src/make_splits.py --targets {target_flag} '
            f'--rebuild".')
    if sha256(labels) != entry['labels_sha256']:
        raise ValueError(f'{path}: {part} labels do not match the manifest inside it')

    tracked = manifest_path(target_flag, split_dir)
    if os.path.exists(tracked):
        mine = entry_of(read_manifest(target_flag, split_dir), part)
        if mine['index'] != entry['index'] or (
                mine['images_sha256'] and mine['images_sha256'] != got):
            raise ValueError(
                f'{path} holds a different {part} than {tracked} describes. The bundle '
                f'is from another draw -- use the bundle that goes with this checkout, '
                f'or move the manifest aside if the bundle is the one you mean.')

    if verbose:
        print(f'{target_flag} {part}: {entry["n_images"]} images, per-class '
              f'{entry["class_counts"]}, sha256 {got[:12]}', flush=True)
    return images, labels, dict(entry, images_sha256=got)


def load_val(target_flag, fold=1, dest=None, split_dir=None, verbose=True):
    """The validation part of one fold, checksum-verified on load."""
    return load_part(target_flag, f'val_fold{fold}', dest, split_dir, verbose)


def load_train(target_flag, fold=1, dest=None, split_dir=None, verbose=True):
    return load_part(target_flag, f'train_fold{fold}', dest, split_dir, verbose)
