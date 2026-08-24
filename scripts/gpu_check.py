"""Can this torch actually run on this card?

An environment that imports torch and reports a GPU can still be unable to run a single
kernel on it: the wheel carries compiled code for a set of compute capabilities, and a
card outside that set fails with "no kernel image is available for execution on the
device". That failure arrives at the first trial, hours into a job, after the pretrained
weights have been fetched and the data staged, so it is worth a second up front.

The case that bites here is Volta. V100 is sm_70, and recent torch wheels start at
sm_80 -- `pip install torch` gives an environment that looks fine everywhere except on
the card it has to run on. The CUDA 11.8 build still carries sm_70.

Matching is by major version, not exact string. CUDA guarantees binary compatibility
within a major compute capability, so sm_80 code runs on an sm_86 card and sm_120 code
runs on the sm_121 GB10 this was written on -- an exact-match check calls both of those
broken. What does not work is the other direction: sm_90 code will not run on sm_86.

The second thing checked is size. A card that cannot hold the largest batch does not stop
the run: the trial raises, Optuna catches it and marks it failed, and the search carries
on over whatever is left. That is worse than stopping, because it is silent and it is
uneven -- vgg16 and densenet121 need 18 GiB at batch 128 where shufflenet needs 4, so a
small card removes the large batches from the heavy architectures only. The search space
then differs by architecture, which is a per-architecture perturbation of exactly the
quantity this benchmark measures.

    python scripts/gpu_check.py            # report and exit non-zero if unusable
    python scripts/gpu_check.py --run      # also do a forward and backward pass

Exit status is 0 when the card is usable, 1 when it is not, and 2 when there is no GPU.
A card that is usable but too small to search the full space warns and exits 0 -- picking
the card is the caller's business.
"""

import sys

#: The batch sizes the search draws from, smallest first.
BATCH_SIZES = (16, 32, 64, 128)

#: Peak reserved memory in GiB for one training step at 224px in fp32, measured per
#: architecture. Only the ones that set the requirement are listed; shufflenet, mnasnet,
#: mobilenet and googlenet all sit under half of vgg's. The two smallest batches are not
#: measured because nothing that can run the benchmark at all is troubled by them.
PEAK_GIB = {64: {'vgg': 9.1, 'densenet': 8.9, 'convnext': 8.4, 'resnet': 7.8},
            128: {'vgg': 17.9, 'densenet': 17.3, 'convnext': 15.1, 'resnet': 13.1}}

#: headroom over the measured peak: fragmentation, the workers' pinned buffers, and the
#: eval pass on a test split larger than the training fold
SLACK_GIB = 2.0


def capability(arch):
    """(major, minor) from an arch string like 'sm_70', 'sm_100' or 'sm_90a'."""
    _, _, tail = arch.partition('_')
    digits = ''.join(c for c in tail if c.isdigit())
    if not digits:
        return None
    return int(digits[:-1]), int(digits[-1])


def supports(built, device):
    """Whether any compiled arch in `built` runs on a card of capability `device`.

    Same major version and a minor no higher than the card's: that is the guarantee CUDA
    makes, and it is what lets one wheel cover a whole family.
    """
    for arch in built:
        cap = capability(arch)
        if cap and cap[0] == device[0] and cap[1] <= device[1]:
            return True
    return False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    import torch

    if not torch.cuda.is_available():
        print('no GPU visible to torch', file=sys.stderr)
        return 2

    device = torch.cuda.get_device_capability(0)
    built = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)
    print(f'[torch] {torch.__version__}, CUDA {torch.version.cuda}, '
          f'{name} (sm_{device[0]}{device[1]})')
    print(f'[torch] built for {", ".join(built)}')

    if not supports(built, device):
        print(f'\nthis torch has no compiled code for sm_{device[0]}{device[1]}. It will '
              f'either JIT from PTX,\nwhich is slow and not always possible, or fail on '
              f'the first kernel. Install a build\nthat covers this card and rebuild the '
              f'environment with scripts/env.sbatch.\nFor a V100 (sm_70) that is the CUDA '
              f'11.8 build:\n'
              f"  export HPO_TORCH_INSTALL='conda install -y pytorch torchvision "
              f"pytorch-cuda=11.8 -c pytorch -c nvidia'", file=sys.stderr)
        return 1

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    largest = max(bs for bs in PEAK_GIB)
    need = max(PEAK_GIB[largest].values()) + SLACK_GIB
    print(f'[torch] {total:.0f} GiB on the card, {need:.0f} GiB wanted for batch '
          f'{largest}')
    if total < need:
        # a batch with no measurement is one too small to be in question
        fits = [bs for bs in BATCH_SIZES
                if bs not in PEAK_GIB
                or max(PEAK_GIB[bs].values()) + SLACK_GIB <= total]
        worst = ', '.join(f'{a} {g:.0f}' for a, g in
                          sorted(PEAK_GIB[largest].items(), key=lambda kv: -kv[1]))
        print(f'\nthis card cannot hold batch {largest} for every architecture ({worst} '
              f'GiB).\nThose trials will be caught and marked failed rather than stopping '
              f'the run, so the\nsearch would continue with large batches available to the '
              f'light architectures and\nnot the heavy ones. Either ask for a bigger card, '
              f'or make the restriction explicit\nand equal for all of them:\n'
              f'  --batch-sizes {" ".join(str(b) for b in fits)}',
              file=sys.stderr)

    if '--run' in argv:
        import torchvision
        try:
            net = torchvision.models.resnet18(weights=None).cuda()
            net(torch.randn(2, 3, 224, 224, device='cuda')).sum().backward()
            torch.cuda.synchronize()
        except Exception as e:
            print(f'\na forward and backward pass on this card failed:\n  '
                  f'{type(e).__name__}: {e}', file=sys.stderr)
            return 1
        print('[torch] forward and backward pass on the GPU: ok')

    return 0


if __name__ == '__main__':
    sys.exit(main())
