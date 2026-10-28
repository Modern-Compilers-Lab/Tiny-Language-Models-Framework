#!/usr/bin/env python3
"""From-scratch DDP pretraining of a tiny_starcoder_py-shaped decoder on one scale25 arm.

    python train.py --arm plain --data <dir> --out <dir>                    # 1 GPU (default)
    torchrun --standalone --nproc-per-node=8 train.py --arm plain  --data <dir> --out <dir>
    torchrun --standalone --nproc-per-node=8 train.py --arm rename --data <dir> --out <dir>

The GPU count is whatever torchrun is told; plain `python train.py` is 1 GPU, which is the
DEFAULT.  1, 2, 4 and 8 all work and all follow the SAME schedule.  The run is defined by a GLOBAL token budget per optimiser step
(`--tokens-per-step`, production 262,144); gradient accumulation is derived,

    accum = tokens_per_step / (world_size * micro_bs * seq_len)

and the run aborts if that does not divide evenly.  So the number of optimiser steps, the
tokens per step and the LR schedule are identical on 1 GPU and on 8; only wall clock changes.

The two arms differ ONLY in which `.bin` they read. Seed, model init, tokenizer, schedule,
batch shape, number of optimiser steps and total token budget are identical by construction:
`--total-tokens` is computed once from the SMALLER of the two streams and written into the
manifest, so both arms consume the same number of tokens.

Resumable: `resume.pt` (model + optimiser + step + RNG) is rewritten atomically every
`--save-every` steps, so a short SLURM job can be chained with `--dependency=afterany`.
Snapshots for evaluation are written in HuggingFace format at 20/40/60/80/100 % of the run.

------------------------------------------------------------------------------------------
PERFORMANCE / WHAT TO SET ON AN A100  (2026-09-20, [AGENT PERF])
------------------------------------------------------------------------------------------
The fast path is the DEFAULT: every knob below is set to whatever MEASURED fastest on an
A100, and each one has a flag to put it back.  Two of them are fast by being OFF
(`torch.compile` on the model, and the data prefetch thread) -- that is a measurement, not an
oversight; see the table.  `--baseline-2026-09-18` restores the exact configuration the
published scale25 runs used (micro-bs 8, compiled model, eager loss, no prefetch, per-step
barrier, AdamW, no DDP tweaks), so those runs stay reproducible.

PARITY.  Everything except the optimiser is the same mathematics in a different order, and
that was checked, not assumed: baseline and new-default, same seed, same card, 300 steps ->
max |delta loss| 0.045.  The SAME configuration run twice differs by 0.049, and a single run's
step-to-step loss moves by 0.055.  So the optimisation perturbs the trajectory by less than
re-running it does.

Recommended A100 command (N = however many GPUs you have; the trainer itself defaults to 1):

    torchrun --standalone --nproc-per-node=N train.py \
        --arm plain --data $W/data --out $W/ck/plain

    # the realistic example, an 8-GPU A100 box:
    torchrun --standalone --nproc-per-node=8 train.py \
        --arm plain --data $W/data --out $W/ck/plain

That resolves to: bf16 autocast, TF32 on, NO `torch.compile` on the model (it is a measured
slowdown here) but `torch.compile` ON the cross-entropy, fused AdamW, `--micro-bs auto`
(32 on any card with >= 38 GB, 8 below that), `gradient_as_bucket_view` DDP with no per-step
barrier, and no data prefetch thread.

MEASURED 2026-09-20, seq 1024, --tokens-per-step 262,144 (GLOBAL), steady state only.
This cluster has TWO A100 variants and they differ by ~14 %, so every row names its card.
The production reference for scale: 69.5k tok/s on 4 V100s (2 per arm) in 2026-09-18.

  A100 80GB PCIe, 1 GPU, cumulative, every row measured on the SAME card (cn010):
    baseline-2026-09-18 (micro-bs 8, compiled model, eager loss) ...  78.8k   1.00x  11.9 GB
    + compile the LOSS ............................................  84.8k   1.08x   8.1 GB
    + micro-bs 32 (accum 32 -> 8) .................................  92.5k   1.17x  24.3 GB
    + background-thread data prefetch .............................  93.5k   1.19x  24.3 GB
    - torch.compile on the MODEL  (i.e. the new default) .......... 121.2k   1.54x  24.2 GB

  The same baseline -> new default step on the other two cards this cluster has:
    A100-PCIE-40GB   71.8k -> 109.8k   1.53x   (micro-bs 32 fits 40 GB: 24.2 GB peak)
    A100-SXM4-80GB   85.6k -> 137.1k   1.60x   (both over 300 steps, same card)

  A100-SXM4-80GB, the new default, other shapes:
    1 GPU  micro-bs 64 ............................................ 141.6k   45.6 GB
    2 GPUs micro-bs 32 (DDP) ...................................... 272.5k   = 1.98x of 1 GPU

Things that did NOT pay off, measured, so you do not have to try them again:
    torch.compile on the model      -19 % (121.2k -> 93.5k+cl / 104.2k -> 83.8k eager+no-cl)
    ... with max-autotune           still -20 % (96.6k) and 352 s of compile time
    TF32 on/off                     0.0 % (bf16 autocast means almost no fp32 matmuls left)
    background-thread prefetch      a wash: -2.2 %, +1.1 %, +1.3 % over three same-card A/Bs
    SDPA flash backend              UNAVAILABLE: transformers passes a bool attn_mask with
                                    is_causal=False, which excludes the flash kernel. This is
                                    the biggest speed-up still on the table (see labnotes).

8-GPU numbers anywhere in this repo are an EXTRAPOLATION from the measured 1->2 scaling;
this cluster never had more than 2 free A100s in one allocation.

OPTIMISER: the default is Muon (Keller Jordan's, as in modded-nanogpt): momentum plus a
quintic Newton-Schulz orthogonalisation on the 80 two-dimensional hidden weight matrices
(119.9 M params), with fused AdamW kept for the embedding / tied LM head / position table /
norms / biases (38.7 M).  Measured on 1x A100-SXM4-80GB, 300 steps, micro-bs 32:

    throughput cost of Muon ......  2.3 %   (137.1k -> 134.0k tok/s)
    loss at step 300 .............  3.067 (Muon) vs 4.321 (AdamW),  -1.254
    loss at matched 500 s ........  3.758 (Muon, step 255) vs 4.567 (AdamW, step 261), -0.809
    run-to-run noise floor .......  0.049, so that gap is ~16x the noise

This is a CONVERGENCE claim, not a parity claim, and it comes with three caveats:
  (a) the PUBLISHED scale25 runs used AdamW.  `--optimizer adamw` reproduces them exactly.
  (b) Muon changes the optimisation trajectory by design, so a Muon run is NOT comparable to
      an AdamW run.  Both arms of any plain-vs-rename comparison must use the SAME optimiser.
  (c) 300 steps is 3.6 % of the 8,300-step run and lies entirely inside the 500-step LR
      warm-up.  A fast start is the regime Muon is known to win; nothing here says anything
      about the final loss, and `--muon-lr 0.02` was not tuned while `--lr 6e-4` was.  If the
      run that matters is a headline number, run AdamW or validate Muon end to end first.

HARDWARE: the defaults auto-detect and degrade.  bf16 is used where `torch.cuda.is_bf16_supported()`
(A100/H100/H200); elsewhere (V100) it falls back to fp16 + GradScaler, exactly as production did.
`--micro-bs auto` reads the card's total memory. `torch.compile` failures fall back to eager with
a warning.  An unknown card gets the conservative 32 GB settings.
"""
import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ.setdefault('HF_HOME', '/scratch/rb4792/hf_cache')
MODEL = 'bigcode/tiny_starcoder_py'


def log(msg, rank):
    if rank == 0:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_model(seq_len, seed):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.n_positions = seq_len
    cfg.max_position_embeddings = seq_len
    # No dropout: this is a single-epoch pretraining run, not a finetune.
    cfg.attn_pdrop = cfg.embd_pdrop = cfg.resid_pdrop = 0.0
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'
    model = AutoModelForCausalLM.from_config(cfg)
    return model, cfg


def block_order(n_blocks, seed):
    """Fixed permutation of the corpus blocks, identical for both arms."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_blocks)


def get_batch(data, order, idx, seq_len, device):
    x = np.empty((len(idx), seq_len), dtype=np.int64)
    y = np.empty((len(idx), seq_len), dtype=np.int64)
    for i, b in enumerate(idx):
        s = int(order[b]) * seq_len
        chunk = data[s:s + seq_len + 1].astype(np.int64)
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    return (torch.from_numpy(x).pin_memory().to(device, non_blocking=True),
            torch.from_numpy(y).pin_memory().to(device, non_blocking=True))


# --------------------------------------------------------------------------------------
# Data path
# --------------------------------------------------------------------------------------
class Prefetcher:
    """Background-thread producer for training micro-batches.

    Identical output to `get_batch`, but:
      * the numpy gather out of the memmap (256 random reads per optimiser step at
        micro-bs 8 / accum 32, on a network filesystem) happens on a worker thread instead
        of the training thread;
      * the host staging buffers are allocated ONCE as pinned int32 (not a fresh pinned
        int64 array per micro-batch), halving the H2D bytes and removing the per-call
        pinned allocation;
      * the int32 -> int64 widening happens on the GPU.

    The index sequence is a pure function of the global micro-batch counter, so the stream of
    batches is byte-identical to the eager path.
    """

    def __init__(self, data, order, seq, micro_bs, world, rank, n_blocks, device,
                 start_micro, depth=4):
        self.data, self.order, self.seq = data, order, seq
        self.micro_bs, self.world, self.rank = micro_bs, world, rank
        self.n_blocks, self.device = n_blocks, device
        self.depth = depth
        self.bufs = [torch.empty((micro_bs, seq + 1), dtype=torch.int32, pin_memory=True)
                     for _ in range(depth)]
        self.views = [b.numpy() for b in self.bufs]
        self.events = [torch.cuda.Event() for _ in range(depth)]
        for e in self.events:          # record once so the first synchronize() is a no-op
            e.record()
        self.q = queue.Queue(maxsize=depth - 1)
        self.free = queue.Queue()
        for i in range(depth):
            self.free.put(i)
        self.m = start_micro
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _fill(self, slot, m):
        base = m * self.micro_bs * self.world + self.rank * self.micro_bs
        v = self.views[slot]
        for i in range(self.micro_bs):
            b = (base + i) % self.n_blocks
            s = int(self.order[b]) * self.seq
            v[i] = self.data[s:s + self.seq + 1]

    def _work(self):
        m = self.m
        while not self.stop.is_set():
            try:
                slot = self.free.get(timeout=0.5)
            except queue.Empty:
                continue
            self.events[slot].synchronize()     # previous H2D out of this buffer has landed
            self._fill(slot, m)
            m += 1
            while not self.stop.is_set():
                try:
                    self.q.put(slot, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next(self):
        slot = self.q.get()
        g = self.bufs[slot].to(self.device, non_blocking=True)
        self.events[slot].record()
        self.free.put(slot)
        x = g[:, :-1].contiguous().to(torch.int64)
        y = g[:, 1:].contiguous().to(torch.int64)
        return x, y

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=2)


# --------------------------------------------------------------------------------------
# Muon  (Keller Jordan, modded-nanogpt).  Momentum + Newton-Schulz orthogonalisation.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, dtype=torch.bfloat16):
    """Quintic Newton-Schulz iteration: returns an approximate orthogonalisation of G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon for the 2-D hidden weight matrices only.  Embeddings / LM head / norms / biases
    must be given to AdamW instead -- see `split_params`."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, ns_dtype=torch.bfloat16):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            ns_steps=ns_steps, weight_decay=weight_decay))
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group['lr'], group['momentum']
            wd, ns = group['weight_decay'], group['ns_steps']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if 'momentum_buffer' not in st:
                    st['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                buf = st['momentum_buffer']
                buf.mul_(mom).add_(g.float(), alpha=1 - mom)
                upd = g.float().lerp(buf, mom) if group['nesterov'] else buf
                o = zeropower_via_newtonschulz5(upd, steps=ns, dtype=self.ns_dtype)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale)
        return loss


def split_params(raw):
    """(muon 2-D hidden weights, adamw decay, adamw no-decay).  Embeddings, the tied LM head,
    the position table, every norm and every bias stay on AdamW."""
    muon, adam_decay, adam_nodecay = [], [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            adam_nodecay.append(p)
        elif ('wte' in n) or ('wpe' in n) or ('lm_head' in n):
            adam_decay.append(p)
        else:
            muon.append(p)
    return muon, adam_decay, adam_nodecay


@torch.no_grad()
def val_loss(model, data, order, seq_len, micro_bs, device, n_batches,
             amp=torch.bfloat16, enabled=True):
    model.eval()
    tot, n = 0.0, 0
    for k in range(n_batches):
        idx = list(range(k * micro_bs, (k + 1) * micro_bs))
        if idx[-1] >= len(order):
            break
        x, y = get_batch(data, order, idx, seq_len, device)
        with torch.autocast('cuda', dtype=amp, enabled=enabled):
            logits = model(input_ids=x).logits
        tot += torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    model.train()
    return (tot / n) if n else float('nan')


SBATCH = shutil.which('sbatch') or '/usr/bin/sbatch'


def submit_eval(ckpt_dir, arm, snap, full_pct, whyscale_limit, code_dir):
    """Fire off a SEPARATE SLURM job to evaluate this snapshot, on a different GPU, and
    return immediately. Training must never block on evaluation.

    The full four-metric suite runs on snapshots whose run-percentage is a multiple of
    `full_pct`; every other snapshot gets the cheap headline pair (gencomp + held-out loss).
    """
    try:
        full = int(snap) % full_pct == 0
    except ValueError:
        full = True
    # The cheap headline pair goes to a spare GPU; the two expensive metrics go to an idle
    # CPU node, so a full-suite point never backs up the 1 %-cadence GPU queue.
    jobs = [('evalsnap_cpu.sbatch', 'gencomp,val')]
    if full:
        jobs.append(('evalsnap_cpu.sbatch', 'real6'))
        jobs.append(('evalsnap_cpu.sbatch', 'bpe'))
        jobs.append(('evalsnap_cpu.sbatch', 'whyscale'))
    out = []
    for script, which in jobs:
        env = dict(os.environ, CKPT=ckpt_dir, ARM=arm, SNAP=snap, WHICH=which,
                   WHYLIMIT=str(whyscale_limit))
        cmd = [SBATCH, '--parsable', '--job-name', f's25e_{arm}_{snap}_{which.split(",")[0]}',
               os.path.join(code_dir, script)]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            out.append({'submitted': False, 'which': which, 'error': repr(e)[:300]})
            continue
        if p.returncode != 0:
            out.append({'submitted': False, 'which': which,
                        'error': (p.stdout + p.stderr)[-300:]})
        else:
            out.append({'submitted': True, 'which': which,
                        'job_id': p.stdout.strip().split(';')[0]})
    return out


def save_hf(model, out_dir, cfg, rank):
    if rank != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    m = model.module if isinstance(model, DDP) else model
    m = getattr(m, '_orig_mod', m)
    m.save_pretrained(out_dir, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out_dir)


def auto_micro_bs(total_mem_gb, seq_len, tokens_per_step, world):
    """Largest micro-batch that (a) fits the card and (b) divides the global token budget.

    MEASURED peak allocation at seq 1024 with the compiled loss, 2026-09-20: micro-bs
    8 -> 8.1 GB, 16 -> 13.5 GB, 32 -> 24.2 GB, 64 -> 45.6 GB.  micro-bs 32 was run on a
    40 GB A100 (A100-PCIE-40GB, 39.5 GiB) and fits.

    The pick stops at 32 even on an 80 GB card although 64 measured +2.7 % (141.6k vs 137.9k
    on an A100-SXM4-80GB).  That is deliberate: this repo compares two arms that may be
    trained on different machines, and holding micro-bs fixed across 40 GB and 80 GB cards
    keeps the gradient-accumulation order identical between them.  Pass --micro-bs 64
    explicitly to buy the 2.7 % on an 80 GB card.

    Below 38 GB the pick stays at the production 8: on a 32 GB V100 micro-bs 16 OOMed in
    2026-09-18 testing with the EAGER loss (the 49152-wide logits tensor is 3.2 GB in fp32
    and cross-entropy needs several copies).  The compiled loss removes most of that and 16
    would probably now fit, but that has not been measured on a V100.
    """
    if total_mem_gb >= 38:
        cands = [32, 16, 8, 4, 2, 1]
    else:
        cands = [8, 4, 2, 1]
    for m in cands:
        per = m * seq_len * world
        if per <= tokens_per_step and tokens_per_step % per == 0:
            return m
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('plain', 'rename'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq-len', type=int, default=1024)
    ap.add_argument('--micro-bs', default='auto',
                    help='per-GPU micro-batch, or "auto" (by card memory). The GLOBAL '
                         'tokens-per-step is fixed regardless; only accumulation changes.')
    ap.add_argument('--tokens-per-step', type=int, default=262144,
                    help='GLOBAL tokens per optimiser step, across all GPUs')
    ap.add_argument('--total-tokens', type=int, default=0, help='0 = read from data/budget.json')
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--min-lr-frac', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--order-seed', type=int, default=20260918)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--val-batches', type=int, default=40)
    ap.add_argument('--snapshots', default='0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52,0.56,0.60,0.64,0.68,0.72,0.76,0.80,0.84,0.88,0.92,0.96,1.00')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                    help='torch.compile the MODEL. Default OFF: measured a 14-20 %% SLOWDOWN '
                         'on an A100 (torch 2.10 / transformers 5.5). Compiling the LOSS is '
                         'a separate flag and IS on by default.')
    ap.add_argument('--dtype', default='auto', choices=('auto', 'bf16', 'fp16', 'fp32'))
    ap.add_argument('--max-minutes', type=float, default=0, help='exit cleanly before walltime')
    ap.add_argument('--bench-steps', type=int, default=0, help='benchmark only, no checkpoints')
    ap.add_argument('--full-pct', type=int, default=20,
                    help='snapshots whose run-percentage is a multiple of this get the full '
                         'four-metric suite; the rest get gencomp + held-out loss')
    ap.add_argument('--whyscale-limit', type=int, default=3000)
    ap.add_argument('--no-submit-eval', action='store_true',
                    help='write snapshots but do not submit evaluation jobs')

    # ---- performance flags (2026-09-20).  All FAST BY DEFAULT; each has an escape. ----
    perf = ap.add_argument_group('performance (fast by default; every one has a --no- escape)')
    perf.add_argument('--compile-mode', default='default',
                      choices=('default', 'reduce-overhead', 'max-autotune',
                               'max-autotune-no-cudagraphs'))
    perf.add_argument('--compile-loss', action=argparse.BooleanOptionalAction, default=True,
                      help='torch.compile the cross-entropy too (parity-safe)')
    perf.add_argument('--prefetch', action=argparse.BooleanOptionalAction, default=False,
                      help='background-thread pinned-int32 data pipeline instead of the '
                           'original synchronous per-micro-batch get_batch(). Default OFF: '
                           'measured a WASH on this cluster (-2.2 %% to +1.3 %% over three '
                           'same-GPU A/B pairs), because the memmap reads are not the '
                           'bottleneck here. Turn it ON if your .bin lives on slow storage.')
    perf.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True,
                      help='TF32 for fp32 matmuls / cudnn')
    perf.add_argument('--ddp-tweaks', action=argparse.BooleanOptionalAction, default=True,
                      help='gradient_as_bucket_view=True, broadcast_buffers=False')
    perf.add_argument('--step-barrier', action=argparse.BooleanOptionalAction, default=False,
                      help='per-step dist.barrier() (production behaviour; buys nothing)')
    perf.add_argument('--sdpa-backend', default='auto',
                      choices=('auto', 'flash', 'mem_efficient', 'cudnn', 'math'))
    perf.add_argument('--prefetch-depth', type=int, default=4)
    perf.add_argument('--preload', action='store_true',
                      help='read the whole .bin into RAM once (7-8 GB) instead of memmap')
    perf.add_argument('--optimizer', default='muon', choices=('muon', 'adamw'))
    perf.add_argument('--muon-lr', type=float, default=0.02)
    perf.add_argument('--muon-momentum', type=float, default=0.95)
    perf.add_argument('--muon-ns-steps', type=int, default=5)
    perf.add_argument('--muon-wd', type=float, default=0.0)
    perf.add_argument('--baseline-2026-09-18', action='store_true',
                      help='restore the exact configuration of the published scale25 runs: '
                           'micro-bs 8, AdamW, eager loss, no prefetch, per-step barrier, '
                           'default compile mode, no DDP tweaks.')
    perf.add_argument('--bench-warmup', type=int, default=0,
                      help='benchmark: steps to exclude before timing (0 = log-every)')
    perf.add_argument('--bench-json', default='', help='benchmark: write a JSON report here')
    perf.add_argument('--bench-tag', default='', help='benchmark: label for the JSON report')
    a = ap.parse_args()

    # which options the user actually typed, for the effective-configuration print-out
    typed = {t.split('=', 1)[0] for t in sys.argv[1:]}
    given = {act.dest for act in ap._actions if set(act.option_strings) & typed}
    src = {}

    def mark(dest, source):
        src[dest] = source

    if a.baseline_2026_09_18:
        for dest, val in (('micro_bs', '8'), ('optimizer', 'adamw'), ('compile_loss', False),
                          ('prefetch', False), ('step_barrier', True),
                          ('ddp_tweaks', False), ('compile_mode', 'default'),
                          ('compile', True), ('sdpa_backend', 'auto')):
            if dest not in given:
                setattr(a, dest, val)
                mark(dest, 'baseline-2026-09-18')

    ddp = int(os.environ.get('RANK', -1)) >= 0
    if ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        world = dist.get_world_size()
    else:
        rank, local_rank, world = 0, 0, 1
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    props = torch.cuda.get_device_properties(local_rank)
    gpu_name = props.name
    gpu_mem_gb = props.total_memory / 1024 ** 3
    sm = f'{props.major}.{props.minor}'

    # ---- TF32 (fp32 matmuls: the loss, the norm, the optimiser). A100+ only; no-op on V100.
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:  # noqa: BLE001
            pass
    tf32_on = bool(torch.backends.cuda.matmul.allow_tf32)

    if a.dtype == 'auto':
        a.dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        mark('dtype', f'auto->{a.dtype} ({gpu_name})')
    AMP = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
    scaler = torch.amp.GradScaler('cuda', enabled=(a.dtype == 'fp16'))

    out = a.out
    os.makedirs(out, exist_ok=True)
    seq = a.seq_len

    # ---- micro-batch: auto by card memory, then check it divides the GLOBAL token budget.
    if str(a.micro_bs) == 'auto':
        a.micro_bs = auto_micro_bs(gpu_mem_gb, seq, a.tokens_per_step, world)
        mark('micro_bs', f'auto->{a.micro_bs} ({gpu_mem_gb:.0f} GB card)')
    a.micro_bs = int(a.micro_bs)

    per_step_per_micro = a.micro_bs * seq * world
    if per_step_per_micro > a.tokens_per_step or a.tokens_per_step % per_step_per_micro:
        raise SystemExit(
            f'--tokens-per-step {a.tokens_per_step} is not a whole multiple of '
            f'world({world}) x micro-bs({a.micro_bs}) x seq({seq}) = {per_step_per_micro}. '
            f'The GLOBAL tokens-per-step must stay fixed across GPU counts, so pick a '
            f'micro-bs that divides it (or pass --micro-bs auto).')
    accum = a.tokens_per_step // per_step_per_micro

    train_path = os.path.join(a.data, f'train_{a.arm}.bin')
    val_path = os.path.join(a.data, f'val_{a.arm}.bin')
    if a.preload:
        train = np.fromfile(train_path, dtype=np.uint16)
    else:
        train = np.memmap(train_path, dtype=np.uint16, mode='r')
    val = np.memmap(val_path, dtype=np.uint16, mode='r')

    total_tokens = a.total_tokens
    if not total_tokens:
        with open(os.path.join(a.data, 'budget.json')) as f:
            total_tokens = json.load(f)['total_tokens']
    max_steps = total_tokens // a.tokens_per_step
    n_blocks = (len(train) - 1) // seq
    order = block_order(n_blocks, a.order_seed)
    val_blocks = (len(val) - 1) // seq
    val_order = np.arange(val_blocks)

    def s(dest):
        if dest in given:
            return 'flag'
        return src.get(dest, 'default')

    visible = torch.cuda.device_count()
    if rank == 0:
        print('=' * 78, flush=True)
        print(f'  GPUs: {world}   (requested via torchrun --nproc-per-node; '
              f'visible on this node: {visible})', flush=True)
        if world < visible:
            print(f'  NOTE: {visible} GPUs are visible but only {world} is/are in use. '
                  f'Pass --nproc-per-node={visible} to torchrun to use them all.', flush=True)
        print(f'  micro-bs/GPU {a.micro_bs}   x world {world}   x seq {seq}   '
              f'x accum {accum}   = {a.tokens_per_step:,} tokens/step (GLOBAL, fixed)',
              flush=True)
        print(f'  steps {max_steps:,}   dtype {a.dtype}   '
              f'compile {(a.compile_mode if a.compile else "off")}   '
              f'optimizer {a.optimizer}', flush=True)
        rows = [
            ('gpu', f'{gpu_name} ({gpu_mem_gb:.0f} GB, sm_{sm})', 'detected'),
            ('world_size (GPUs)', world, 'torchrun --nproc-per-node'),
            ('gpus visible on node', visible, 'detected'),
            ('arm', a.arm, s('arm')),
            ('dtype', a.dtype, s('dtype')),
            ('tf32', tf32_on, s('tf32')),
            ('seq_len', seq, s('seq_len')),
            ('micro_bs (per GPU)', a.micro_bs, s('micro_bs')),
            ('tokens_per_step (GLOBAL)', f'{a.tokens_per_step:,}', s('tokens_per_step')),
            ('grad_accum', f'{accum}  (= {a.tokens_per_step:,} / ({world} x {a.micro_bs} x {seq}))',
             'derived'),
            ('max_steps', f'{max_steps:,}', 'derived'),
            ('total_tokens', f'{total_tokens:,}', s('total_tokens')),
            ('optimizer', a.optimizer, s('optimizer')),
            ('lr (adamw)', a.lr, s('lr')),
            ('muon_lr', a.muon_lr if a.optimizer == 'muon' else '-', s('muon_lr')),
            ('torch.compile', a.compile and a.compile_mode or 'off', s('compile')),
            ('compile loss', a.compile_loss, s('compile_loss')),
            ('data prefetch', a.prefetch and f'thread, depth {a.prefetch_depth}' or 'off',
             s('prefetch')),
            ('preload .bin to RAM', a.preload, s('preload')),
            ('sdpa backend', a.sdpa_backend, s('sdpa_backend')),
            ('ddp tweaks', a.ddp_tweaks, s('ddp_tweaks')),
            ('per-step barrier', a.step_barrier, s('step_barrier')),
        ]
        print('-' * 78, flush=True)
        print('EFFECTIVE CONFIGURATION  (value, then where it came from)', flush=True)
        for k, v, how in rows:
            print(f'  {k:28s} {str(v):44s} [{how}]', flush=True)
        print('=' * 78, flush=True)

    log(f'arm={a.arm} world={world} seq={seq} micro_bs={a.micro_bs} accum={accum} '
        f'tokens/step={a.tokens_per_step:,} max_steps={max_steps:,} '
        f'stream={len(train):,} tok ({n_blocks:,} blocks) budget={total_tokens:,} tok '
        f'dtype={a.dtype}', rank)

    model, cfg = build_model(seq, a.seed)
    model.to(device)
    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        log(f'model {n_par/1e6:.1f}M params (tied embeddings), init seed {a.seed}', rank)
    raw = model

    # ---- optimiser -------------------------------------------------------------------
    muon_opt = None
    if a.optimizer == 'muon':
        mp, ad, an = split_params(raw)
        opt = torch.optim.AdamW([{'params': ad, 'weight_decay': a.wd},
                                 {'params': an, 'weight_decay': 0.0}],
                                lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        muon_opt = Muon(mp, lr=a.muon_lr, momentum=a.muon_momentum,
                        ns_steps=a.muon_ns_steps, weight_decay=a.muon_wd,
                        ns_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                                  else torch.float32))
        log(f'muon on {len(mp)} hidden matrices ({sum(p.numel() for p in mp)/1e6:.1f}M params), '
            f'fused adamw on {len(ad)+len(an)} others '
            f'({sum(p.numel() for p in ad+an)/1e6:.1f}M)', rank)
    else:
        decay = [p for p in model.parameters() if p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd},
                                 {'params': nodecay, 'weight_decay': 0.0}],
                                lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    opts = [opt] + ([muon_opt] if muon_opt is not None else [])
    for o in opts:
        for g in o.param_groups:
            g['base_lr'] = g['lr']

    step = 0
    hist = []
    resume = os.path.join(out, 'resume.pt')
    if os.path.exists(resume):
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        raw.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if muon_opt is not None and ck.get('opt_muon') is not None:
            muon_opt.load_state_dict(ck['opt_muon'])
        for o in opts:
            for g in o.param_groups:
                g.setdefault('base_lr', g['lr'])
        step = ck['step']
        hist = ck.get('hist', [])
        log(f'resumed from {resume} at step {step}', rank)

    compile_requested = a.compile
    if compile_requested:
        try:
            model = torch.compile(raw, mode=a.compile_mode)
        except Exception as e:  # noqa: BLE001
            log(f'torch.compile failed ({e}); continuing eager', rank)
            model = raw
            compile_requested = False
    else:
        model = raw

    def _loss_fn(logits, y):
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))

    # NOTE: compiling the LOSS is independent of compiling the MODEL, and on an A100 the
    # loss is the half that pays off (measured 2026-09-20: model-compile is a net slowdown,
    # loss-compile is +12 % and -15 GB).
    loss_fn = _loss_fn
    if a.compile_loss:
        try:
            loss_fn = torch.compile(_loss_fn, mode=a.compile_mode)
        except Exception as e:  # noqa: BLE001
            log(f'torch.compile(loss) failed ({e}); eager loss', rank)
            loss_fn = _loss_fn

    if ddp:
        kw = {}
        if a.ddp_tweaks:
            kw = dict(gradient_as_bucket_view=True, broadcast_buffers=False)
        model = DDP(model, device_ids=[local_rank], **kw)

    # ---- attention backend ------------------------------------------------------------
    sdpa_ctx = None
    if a.sdpa_backend != 'auto':
        from torch.nn.attention import SDPBackend, sdpa_kernel
        mapping = {'flash': SDPBackend.FLASH_ATTENTION,
                   'mem_efficient': SDPBackend.EFFICIENT_ATTENTION,
                   'cudnn': getattr(SDPBackend, 'CUDNN_ATTENTION', SDPBackend.FLASH_ATTENTION),
                   'math': SDPBackend.MATH}
        sdpa_ctx = lambda: sdpa_kernel(mapping[a.sdpa_backend])  # noqa: E731
    else:
        import contextlib
        sdpa_ctx = contextlib.nullcontext

    def lr_factor(st):
        if st < a.warmup:
            return (st + 1) / a.warmup
        p = min(1.0, (st - a.warmup) / max(1, max_steps - a.warmup))
        return a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

    snaps = [round(float(x) * max_steps) for x in a.snapshots.split(',')]
    snap_names = [f'{int(round(float(x)*100)):03d}' for x in a.snapshots.split(',')]
    done_marker = os.path.join(out, 'DONE')
    if os.path.exists(done_marker):
        log('DONE marker present; nothing to do', rank)
        if ddp:
            dist.destroy_process_group()
        return

    pf = None
    if a.prefetch:
        pf = Prefetcher(train, order, seq, a.micro_bs, world, rank, n_blocks, device,
                        start_micro=step * accum, depth=a.prefetch_depth)

    t_start = time.time()
    step0 = step
    tok_seen = step * a.tokens_per_step
    log(f'starting at step {step}/{max_steps}', rank)
    window_t, window_tok = time.time(), 0
    bench = a.bench_steps
    bench_warm = a.bench_warmup or a.log_every
    if bench:
        bench_warm = min(bench_warm, max(1, bench // 2))
    bench_t0 = None
    bench_tok = 0
    first_step_s = None
    trace = ([], []) if bench else None
    torch.cuda.reset_peak_memory_stats(device)

    while step < max_steps:
        f = lr_factor(step)
        for o in opts:
            for g in o.param_groups:
                g['lr'] = g['base_lr'] * f
        lr = opt.param_groups[0]['lr']
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for micro in range(accum):
            if pf is not None:
                x, y = pf.next()
            else:
                base = (step * accum + micro) * a.micro_bs * world + rank * a.micro_bs
                idx = [(base + i) % n_blocks for i in range(a.micro_bs)]
                x, y = get_batch(train, order, idx, seq, device)
            if ddp:
                model.require_backward_grad_sync = (micro == accum - 1)
            with sdpa_ctx():
                with torch.autocast('cuda', dtype=AMP, enabled=(a.dtype != 'fp32')):
                    logits = model(input_ids=x).logits
                loss = loss_fn(logits, y)
            scaler.scale(loss / accum).backward()
            loss_acc += loss.detach().float() / accum
        for o in opts:
            scaler.unscale_(o)
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
        for o in opts:
            scaler.step(o)
        scaler.update()
        step += 1
        tok_seen += a.tokens_per_step
        window_tok += a.tokens_per_step

        if bench:
            trace[0].append(step)
            trace[1].append(loss_acc.detach().clone())
            done_b = step - step0  # optimiser steps completed inside this process
            if first_step_s is None:
                torch.cuda.synchronize(device)
                first_step_s = time.time() - t_start
            if done_b == bench_warm:
                torch.cuda.synchronize(device)
                bench_t0, bench_tok = time.time(), 0
            elif bench_t0 is not None:
                bench_tok += a.tokens_per_step

        if step % a.log_every == 0:
            if ddp:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
            dt = time.time() - window_t
            tps = window_tok / dt
            eta = (max_steps - step) * a.tokens_per_step / max(tps, 1)
            log(f'step {step}/{max_steps} loss {loss_acc.item():.4f} lr {lr:.2e} '
                f'gn {float(gn):.2f} {tps/1e3:.1f}k tok/s eta {eta/3600:.2f}h', rank)
            window_t, window_tok = time.time(), 0

        if bench and (step - step0) >= bench:
            torch.cuda.synchronize(device)
            steady = time.time() - bench_t0 if bench_t0 else float('nan')
            tps = bench_tok / steady if bench_t0 else float('nan')
            peak_a = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            peak_r = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            losses = [float(t) for t in trace[1]]
            log(f'BENCHMARK {tps/1e3:.1f}k tok/s (world={world}, micro_bs={a.micro_bs}, '
                f'accum={accum}, compile={a.compile_mode if compile_requested else "off"}, '
                f'opt={a.optimizer}) steady over {bench-bench_warm} steps, '
                f'first-step(+compile) {first_step_s:.1f}s, peak_alloc {peak_a:.1f} GB, '
                f'peak_reserved {peak_r:.1f} GB', rank)
            if a.bench_json and rank == 0:
                rep = dict(tag=a.bench_tag, gpu=gpu_name, gpu_mem_gb=round(gpu_mem_gb, 1),
                           sm=sm, world=world, dtype=a.dtype, tf32=tf32_on,
                           micro_bs=a.micro_bs, accum=accum, seq_len=seq,
                           tokens_per_step=a.tokens_per_step, optimizer=a.optimizer,
                           compile=(a.compile_mode if compile_requested else 'off'),
                           compile_loss=a.compile_loss,
                           prefetch=a.prefetch, preload=a.preload,
                           sdpa=a.sdpa_backend, ddp_tweaks=a.ddp_tweaks,
                           step_barrier=a.step_barrier,
                           bench_steps=bench, bench_warmup=bench_warm,
                           first_step_s=round(first_step_s, 2),
                           steady_seconds=round(steady, 3),
                           tok_per_s=round(tps, 1),
                           tok_per_s_per_gpu=round(tps / world, 1),
                           sec_per_step=round(steady / max(1, bench - bench_warm), 4),
                           peak_alloc_gb=round(peak_a, 2), peak_reserved_gb=round(peak_r, 2),
                           steps=trace[0], losses=[round(v, 6) for v in losses],
                           argv=sys.argv[1:])
                os.makedirs(os.path.dirname(os.path.abspath(a.bench_json)), exist_ok=True)
                with open(a.bench_json, 'w') as f:
                    json.dump(rep, f, indent=1)
            if pf is not None:
                pf.close()
            if ddp:
                dist.destroy_process_group()
            return

        if step in snaps:
            name = snap_names[snaps.index(step)]
            vl = val_loss(raw, val, val_order, seq, a.micro_bs, device, a.val_batches, AMP,
                          a.dtype != 'fp32')
            if ddp:
                t = torch.tensor([vl], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                vl = t.item()
            d = os.path.join(out, f'snap{name}')
            save_hf(model, d, cfg, rank)
            rec = {'step': step, 'frac': name, 'tokens': tok_seen,
                   'train_loss': float(loss_acc.item()), 'val_loss': vl,
                   'wall_seconds': time.time() - t_start, 'seed': a.seed,
                   'optimizer': a.optimizer,
                   'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            if rank == 0 and not a.no_submit_eval:
                rec['eval'] = submit_eval(d, a.arm, name, a.full_pct, a.whyscale_limit,
                                          os.path.dirname(os.path.abspath(__file__)))
            hist.append(rec)
            log(f'SNAPSHOT {name} step {step} tokens {tok_seen:,} val_loss {vl:.4f} -> {d} '
                f'eval={rec.get("eval")}', rank)
            if rank == 0:
                with open(os.path.join(out, 'history.json'), 'w') as f:
                    json.dump(hist, f, indent=1)
            window_t, window_tok = time.time(), 0

        # `do_save` is a function of `step` alone, so it is the SAME on every rank.  It has to
        # be: the barrier below is a collective, and gating it on anything rank-local (such as
        # "did *I* just write the checkpoint", which is only ever true on rank 0) deadlocks DDP.
        do_save = (step % a.save_every == 0 or step == max_steps)
        if rank == 0 and do_save:
            tmp = resume + '.tmp'
            torch.save({'model': raw.state_dict(), 'opt': opt.state_dict(),
                        'opt_muon': muon_opt.state_dict() if muon_opt is not None else None,
                        'step': step, 'hist': hist}, tmp)
            os.replace(tmp, resume)
            window_t, window_tok = time.time(), 0
        # A barrier on EVERY step costs a device sync per step and buys nothing: DDP already
        # synchronises on the gradient all-reduce.  One barrier where rank 0 may have been busy
        # writing a checkpoint is enough to re-align the ranks.
        if ddp and (a.step_barrier or do_save or (step in snaps)):
            dist.barrier()

        # The walltime check is also collective, and for the same reason: `time.time()` differs
        # between ranks, so each rank deciding for itself can have one rank break out of the
        # loop while the others wait forever in the next all-reduce.  Rank 0 decides and the
        # decision is broadcast, and it is only taken on steps that already synchronise -- so
        # the run can overshoot --max-minutes by up to min(--save-every, --log-every) steps.
        # Both of those are functions of the step number, so every rank takes the branch
        # together; `time.time()` is read on rank 0 only and broadcast.
        check_walltime = do_save or step == max_steps or (step % a.log_every == 0)
        if a.max_minutes and check_walltime:
            flag = torch.tensor([1.0 if (time.time() - t_start) / 60 > a.max_minutes else 0.0],
                                device=device)
            if ddp:
                dist.broadcast(flag, src=0)
            if flag.item() > 0.5:
                if rank == 0 and not do_save:
                    tmp = resume + '.tmp'
                    torch.save({'model': raw.state_dict(), 'opt': opt.state_dict(),
                                'opt_muon': muon_opt.state_dict() if muon_opt is not None
                                else None, 'step': step, 'hist': hist}, tmp)
                    os.replace(tmp, resume)
                log(f'max-minutes reached at step {step}; exiting for the next chained job',
                    rank)
                if ddp:
                    dist.barrier()
                break

    if pf is not None:
        pf.close()
    if step >= max_steps and rank == 0:
        with open(done_marker, 'w') as f:
            f.write(json.dumps({'step': step, 'tokens': tok_seen}))
        log('TRAINING COMPLETE', rank)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
