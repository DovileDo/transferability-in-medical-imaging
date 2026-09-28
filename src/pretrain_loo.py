#!/usr/bin/env python
# coding: utf-8
"""Leave-target-out pre-training of a ResNet-50 on the MedMNIST collection.

For a target T, pre-trains from scratch on the training splits of every MedMNIST dataset
except T, and saves a torchvision resnet50 whose classifier covers the pooled label space.
It is the ResNet-50 counterpart of src/leave-one-out-pretrain.py and produces checkpoints
src/sources.py can load as the 'medmnist' source; that script is left untouched so the
ResNet-18 sources stay reproducible.

What is kept from the original, and why:

  The pooled label layout. Same source order and the same sharing rules -- the three
  organ datasets share their 11 organ labels, and pneumoniamnist maps onto chestmnist's
  'pneumonia' label plus a 'normal' slot -- so the classifier has exactly
  sources.n_source_classes('medmnist', T) outputs and LEEP reads the same label space.

  Input normalisation. Mean 0.5 / std 0.5, which is what src/sources.py assigns every
  MedMNIST source and what the fine-tuning benchmark therefore feeds it.

What changes, following current practice for training ResNets from scratch -- the
torchvision recipe update (Vryniotis 2021) and Wightman et al., "ResNet strikes back"
(2021) -- adapted where a pooled multi-dataset medical corpus differs from ImageNet:

  Loss. Cross-entropy over each sample's own dataset's classes, instead of BCE over all
  pooled classes. Under BCE a bloodmnist image is one positive and ~60 negatives from
  datasets it has nothing to do with, and most of its gradient comes from those. chestmnist
  is genuinely multi-label and keeps BCE over its own 14 labels. Label smoothing 0.1 on
  the cross-entropy part, spread over the sample's own classes only.

  Sampling. Datasets are drawn with probability proportional to n^0.5 (temperature
  sampling, standard for pooled multi-dataset training) instead of balancing all ~60
  classes. Class balancing across the pool made breastmnist's 546 images appear hundreds
  of times per epoch, while tissuemnist's 165k were undersampled; the square root keeps
  the small datasets visible without memorising them. Within a dataset its natural class
  mix is kept.

  Optimisation. SGD with Nesterov momentum 0.9, lr 0.1 per 256 images scaled linearly
  with batch size, 5-epoch linear warmup then cosine decay to zero, weight decay 1e-4 on
  weights only -- none on BatchNorm and biases -- and the last BatchNorm of every residual
  block initialised to zero. Trained for the full schedule, keeping an exponential moving
  average of the weights, which is what is saved. No early stopping: with a cosine
  schedule the last epochs are where the learning rate is lowest, and stopping on
  validation loss mostly cuts them off.

  Augmentation, per dataset rather than one pipeline for all, since what is harmless
  for one modality corrupts another. Random resized crops keeping 64-100% of the area,
  for every dataset. Horizontal flips, except on organamnist and organcmnist, whose
  left/right organ labels a flip swaps. Vertical flips and 90-degree rotations for
  bloodmnist, pathmnist and dermamnist, which have no up or down. Mild colour jitter for
  pathmnist, where stain varies between labs. Nothing else: chest X-ray, CT, OCT and
  ultrasound have a real up-down axis, and their intensity is often the signal.
  The floor is high on purpose: within a MedMNIST dataset the imaging scale is fixed, so
  size is often diagnostic -- cell size in blood smears, nuclear size in pathology,
  lesion size in ultrasound and dermoscopy -- and ImageNet's 8-100% range, a zoom of up
  to 3.5x, would teach the network to ignore it. 64% caps the zoom at 1.25x, enough
  variation to keep the small datasets from being memorised over 100 epochs.
  Mixup and CutMix are left out: they mix images across datasets, which has no meaning
  when each dataset has its own label space. The photometric operations of the original
  (equalise, autocontrast, sharpness) are dropped, since intensity carries diagnostic
  content in several of these modalities.

  Speed. Mixed precision (bfloat16 where the GPU has it), channels-last memory, cuDNN
  autotuning, and parallel data loading. The original disabled cuDNN, loaded data in one
  process, and re-scored the entire training set every epoch; none of that changed what
  was learned, only how long it took.

Resumable: a checkpoint is written every epoch and picked up on restart.

    python src/pretrain_loo.py --target dermamnist --out models/loo_resnet50
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sources                                                       # noqa: E402

#: the order the original script walks the collection in; the label layout depends on it
ORDER = ['chestmnist', 'pneumoniamnist', 'pathmnist', 'dermamnist', 'octmnist',
         'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
         'organcmnist', 'organsmnist']
ORGANS = ('organamnist', 'organcmnist', 'organsmnist')
#: datasets a horizontal flip corrupts: their labels include left and right organs
#: (femur, kidney, lung), and in axial and coronal slices a flip turns one into the other.
#: organsmnist is sagittal, where left and right are not in the plane.
NO_FLIP = ('organamnist', 'organcmnist')
#: datasets with no up or down -- blood smears, tissue patches, dermoscopy -- where every
#: rotation by 90 degrees and every reflection is an equally real image
ROTATION_FREE = ('bloodmnist', 'pathmnist', 'dermamnist')
#: mild stain variation for pathology, the main nuisance between labs and scanners
COLOUR_JITTER = {'pathmnist': dict(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.04)}
#: chestmnist's own label for pneumonia, and the slot after its 14 labels for 'normal'
CHEST_PNEUMONIA, CHEST_NORMAL = 6, 14
MEAN, STD = 0.5, 0.5
#: labels are padded to this width so a batch mixing chestmnist's 14 binary labels with
#: the others' single class index can be stacked; the loss reads only what it needs
LABEL_WIDTH = 14


def label_layout(target):
    """{dataset: (global class indices, 'multiclass' | 'multilabel')}, and the head size.

    Reproduces src/leave-one-out-pretrain.py exactly, so a checkpoint from either script
    has the same classifier geometry for the same held-out target.
    """
    pool = [d for d in ORDER if d != target]
    layout, at, chest_base = {}, 0, None
    for d in pool:
        n = len(INFO[d]['label'])
        if d == 'chestmnist':
            chest_base = at
            layout[d] = (list(range(at, at + n)), 'multilabel')
            at += n + 1                                  # plus the 'normal' slot
        elif d == 'pneumoniamnist' and chest_base is not None:
            layout[d] = ([chest_base + CHEST_NORMAL, chest_base + CHEST_PNEUMONIA],
                         'multiclass')
        elif d in ORGANS:
            layout[d] = (list(range(at, at + n)), 'multiclass')   # shared, not advanced
        else:
            layout[d] = (list(range(at, at + n)), 'multiclass')
            at += n
    n_out = at + (len(INFO['organamnist']['label']) if any(d in ORGANS for d in pool) else 0)
    return layout, n_out


class RandomRot90:
    """Rotate a square CHW image by a random multiple of 90 degrees."""

    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(4, ())), dims=(-2, -1))


def train_transform(d, crop_scale):
    """The augmentation for one dataset of the pool."""
    ops = [v2.RandomResizedCrop(224, scale=(crop_scale, 1.0), antialias=True)]
    if d not in NO_FLIP:
        ops.append(v2.RandomHorizontalFlip())
    if d in ROTATION_FREE:
        ops += [v2.RandomVerticalFlip(), RandomRot90()]
    if d in COLOUR_JITTER:
        ops.append(v2.ColorJitter(**COLOUR_JITTER[d]))
    return v2.Compose(ops)


class Pool(Dataset):
    """Every dataset of the pool, kept as its own uint8 array -- no 40 GB concatenation.

    Items are (image, dataset id, local label). Grayscale is augmented as one channel and
    only expanded to three at the end, which is a third of the work.
    """

    def __init__(self, arrays, transforms=None):
        self.arrays = arrays                      # list of (images, labels)
        self.transforms = transforms              # one per dataset, or None
        self.offsets = np.cumsum([0] + [len(a[0]) for a in arrays])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        d = int(np.searchsorted(self.offsets, i, side='right') - 1)
        imgs, labels = self.arrays[d]
        j = i - self.offsets[d]
        x = torch.from_numpy(imgs[j])
        x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)
        if self.transforms is not None:
            x = self.transforms[d](x)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        lab = np.zeros(LABEL_WIDTH, dtype=np.float32)
        v = np.asarray(labels[j], dtype=np.float32).reshape(-1)
        lab[:len(v)] = v
        return x.contiguous(), d, torch.from_numpy(lab)


def load_pool(target, root, split):
    """[(images, labels)] for the pool, in ORDER, from <root>/<dataset>_224.npz."""
    out = []
    for d in ORDER:
        if d == target:
            continue
        path = os.path.join(root, f'{d}_224.npz')
        if not os.path.exists(path):
            raise SystemExit(f'{path} not found -- the pool needs every MedMNIST dataset '
                             f'at 224, including chestmnist, which no benchmark target '
                             f'fetches')
        z = np.load(path)
        out.append((z[f'{split}_images'], z[f'{split}_labels']))
        print(f'  {split:5s} {d:15s} {len(out[-1][0]):7d}', flush=True)
    return out


class Loss(nn.Module):
    """Cross-entropy within each sample's own dataset; BCE for the multi-label one."""

    def __init__(self, layout, datasets, n_out, smoothing, device):
        super().__init__()
        self.smoothing = smoothing
        mask = torch.zeros(len(datasets), n_out, dtype=torch.bool)
        index = torch.full((len(datasets), 32), -1, dtype=torch.long)
        self.multilabel = torch.zeros(len(datasets), dtype=torch.bool)
        for i, d in enumerate(datasets):
            idx, kind = layout[d]
            mask[i, idx] = True
            index[i, :len(idx)] = torch.tensor(idx)
            self.multilabel[i] = kind == 'multilabel'
        self.mask, self.index = mask.to(device), index.to(device)
        self.multilabel = self.multilabel.to(device)
        self.chest_idx = None
        for i, d in enumerate(datasets):
            if layout[d][1] == 'multilabel':
                self.chest_idx = torch.tensor(layout[d][0], device=device)

    def forward(self, logits, ds, y):
        logits = logits.float()
        ml = self.multilabel[ds]
        total = logits.new_zeros(())
        if (~ml).any():
            lg, d, lab = logits[~ml], ds[~ml], y[~ml, 0].long()
            m = self.mask[d]
            logp = F.log_softmax(lg.masked_fill(~m, float('-inf')), dim=1)
            target = self.index[d, lab]
            nll = -logp.gather(1, target[:, None]).squeeze(1)
            smooth = -logp.masked_fill(~m, 0.0).sum(1) / m.sum(1)
            total = total + ((1 - self.smoothing) * nll + self.smoothing * smooth).sum()
        if ml.any():
            lg = logits[ml][:, self.chest_idx]
            total = total + F.binary_cross_entropy_with_logits(
                lg, y[ml][:, :len(self.chest_idx)], reduction='none').mean(1).sum()
        return total / len(ds)

    @torch.no_grad()
    def correct(self, logits, ds, y):
        """Per-sample top-1 within the sample's own classes; NaN for multi-label."""
        ml = self.multilabel[ds]
        out = torch.full((len(ds),), float('nan'), device=logits.device)
        if (~ml).any():
            m = self.mask[ds[~ml]]
            pred = logits[~ml].float().masked_fill(~m, float('-inf')).argmax(1)
            out[~ml] = (pred == self.index[ds[~ml], y[~ml, 0].long()]).float()
        return out


class EMA:
    """Exponential moving average of the weights; buffers are copied, as is standard."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        e = [p for p in self.model.parameters()]
        m = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(e, self.decay)
        torch._foreach_add_(e, m, alpha=1 - self.decay)
        for eb, mb in zip(self.model.buffers(), model.buffers()):
            eb.copy_(mb)


def param_groups(model, wd):
    """Weight decay on weight tensors only, none on BatchNorm parameters and biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


def schedule(step, total, warmup):
    """Linear warmup from 1% of the peak, then cosine to zero, per optimiser step."""
    if step < warmup:
        return 0.01 + 0.99 * step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, dtype, n_sets):
    model.eval()
    tot, n = 0.0, 0
    hit, cnt = torch.zeros(n_sets, device=device), torch.zeros(n_sets, device=device)
    for x, ds, y in loader:
        x = to_input(x, device)
        ds, y = ds.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=dtype):
            out = model(x)
        tot += loss_fn(out, ds, y).item() * len(ds)
        n += len(ds)
        c = loss_fn.correct(out, ds, y)
        ok = ~torch.isnan(c)
        hit.index_add_(0, ds[ok], c[ok])
        cnt.index_add_(0, ds[ok], torch.ones_like(c[ok]))
    model.train()
    acc = (hit / cnt.clamp(min=1)).tolist()
    return tot / n, [a if c > 0 else None for a, c in zip(acc, cnt.tolist())]


def to_input(x, device):
    x = x.to(device, non_blocking=True).float().div_(255).sub_(MEAN).div_(STD)
    return x.contiguous(memory_format=torch.channels_last)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, help='the MedMNIST dataset held out')
    ap.add_argument('--out', default='models/loo_resnet50')
    ap.add_argument('--data', default=os.environ.get('MEDMNIST_ROOT',
                                                     os.path.expanduser('~/.medmnist')))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=0.1, help='per 256 images; scaled linearly')
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--smoothing', type=float, default=0.1)
    ap.add_argument('--temperature', type=float, default=0.5,
                    help='datasets are sampled in proportion to n^temperature')
    ap.add_argument('--crop-scale', type=float, default=0.64,
                    help='smallest crop, as a share of the image area')
    ap.add_argument('--ema-decay', type=float, default=0.9998)
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--max-steps', type=int, default=0, help='stop early; for testing')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layout, n_out = label_layout(args.target)
    expected = sources.n_source_classes('medmnist', args.target)
    if n_out != expected:
        raise SystemExit(f'label layout gives {n_out} classes, sources.py expects {expected}')
    datasets = [d for d in ORDER if d != args.target]
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f'{args.target}.ckpt')
    print(f'{args.target}: pool of {len(datasets)} datasets, {n_out} classes', flush=True)

    # Data first, and the worker processes started before anything touches CUDA: they are
    # forked, so the pool's arrays are shared copy-on-write rather than duplicated per
    # worker, and no child inherits a CUDA context.
    train = Pool(load_pool(args.target, args.data, 'train'),
                 [train_transform(d, args.crop_scale) for d in datasets])
    val = Pool(load_pool(args.target, args.data, 'val'))
    sizes = np.array([len(a[0]) for a in train.arrays], dtype=np.float64)
    p = sizes ** args.temperature
    p /= p.sum()
    weights = np.concatenate([np.full(int(n), p[i] / n) for i, n in enumerate(sizes)])
    print('  sampling share: ' + ', '.join(f'{d[:-5]} {s:.3f}' for d, s in zip(datasets, p)),
          flush=True)
    sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(train),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    kw = dict(num_workers=args.workers, pin_memory=True,
              persistent_workers=args.workers > 0,
              multiprocessing_context='fork' if args.workers > 0 else None)
    if args.workers > 0:
        kw['prefetch_factor'] = 2
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, **kw)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, **kw)
    iter(train_loader)
    iter(val_loader)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)

    model = torchvision.models.resnet50(weights=None, num_classes=n_out,
                                        zero_init_residual=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay)
    lr = args.lr * args.batch_size / 256
    opt = torch.optim.SGD(param_groups(model, args.wd), lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: schedule(s, total, args.warmup_epochs * steps_per_epoch))
    loss_fn = Loss(layout, datasets, n_out, args.smoothing, device)

    start = 0
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c['model'])
        ema.model.load_state_dict(c['ema'])
        opt.load_state_dict(c['opt'])
        sched.load_state_dict(c['sched'])
        scaler.load_state_dict(c['scaler'])
        start = c['epoch'] + 1
        print(f'resumed from epoch {start}', flush=True)

    log_path = os.path.join(args.out, f'{args.target}_log.csv')
    print(f'{steps_per_epoch} steps/epoch, batch {args.batch_size}, peak lr {lr:.3f}, '
          f'{dtype}', flush=True)
    step = start * steps_per_epoch
    for epoch in range(start, args.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, ds, y in train_loader:
            x = to_input(x, device)
            ds, y = ds.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=dtype):
                out = model(x)
            loss = loss_fn(out, ds, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run += loss.item() * len(ds)
            seen += len(ds)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        dt = time.time() - t0
        row = {'epoch': epoch, 'train_loss': run / max(1, seen), 'lr': sched.get_last_lr()[0],
               'img_per_s': seen / dt, 'seconds': dt}
        last = epoch == args.epochs - 1 or bool(args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.val_every == 0 or last:
            vl, acc = evaluate(ema.model, val_loader, loss_fn, device, dtype, len(datasets))
            row['val_loss_ema'] = vl
            row.update({f'val_acc_{d}': a for d, a in zip(datasets, acc)})
        print('  '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}'
                        for k, v in row.items() if not k.startswith('val_acc_')), flush=True)
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        tmp = ckpt_path + '.tmp'
        torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch}, tmp)
        os.replace(tmp, ckpt_path)
        if last:
            break

    # the EMA weights are the model; the raw ones are kept for comparison. Plain
    # state dicts, which is what sources.load_backbone reads.
    torch.save(ema.model.state_dict(), os.path.join(args.out, f'{args.target}.pt'))
    torch.save(model.state_dict(), os.path.join(args.out, f'{args.target}_raw.pt'))
    with open(os.path.join(args.out, f'{args.target}.json'), 'w') as f:
        json.dump({'target': args.target, 'pool': datasets, 'n_classes': n_out,
                   'layout': {d: layout[d] for d in datasets}, 'args': vars(args)}, f, indent=1)
    print(f'saved {os.path.join(args.out, args.target)}.pt', flush=True)


if __name__ == '__main__':
    main()
