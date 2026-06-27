"""
Train CodeBERT from scratch using MLM + RTD objectives on CodeSearchNet data.

Original CodeBERT hyperparameters (Feng et al., 2020):
  - Architecture: RoBERTa-base (12 layers, 768 hidden, 12 heads, 125M params)
  - Batch size: 2048 (effective, using gradient accumulation)
  - Learning rate: 5e-4
  - Optimizer: Adam
  - Warmup steps: 10,000
  - Max sequence length: 512
  - Max training steps: 100,000
  - Objectives: MLM + RTD
  - Precision: FP16

This script supports:
  - Single-GPU and multi-GPU training via PyTorch DDP
  - Gradient accumulation to achieve large effective batch sizes
  - MLM and RTD training objectives (RTD can be disabled with --mlm_only)
  - Per-epoch variable name diversification (--diversify_varnames)
  - Canonical variable naming (--canonical_varnames)
  - Identifier-aware MLM masking that skips identifier tokens (--identifier_aware_mlm)
  - Checkpointing, resume from checkpoint, and logging
  - Configurable data subset for small-scale testing

Experiment modes (controlled by flags):
  Exp 1: --diversify_varnames                (diverse variable names)
  Exp 2: --char_tokenizer                     (character/byte-level tokenizer)
  Exp 3: --identifier_aware_mlm --diversify_varnames (identifier-aware MLM)
  Exp 4: --canonical_varnames                 (canonical VAR_0, VAR_1 naming)
  Exp 5: --mlm_only --diversify_varnames     (MLM-only, no RTD)
"""

import os
import sys
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Wrap autocast for compatibility across PyTorch versions
_orig_autocast = autocast
def _compat_autocast(device_type="cuda", **kwargs):
    try:
        return _orig_autocast(device_type, **kwargs)
    except TypeError:
        return _orig_autocast(**kwargs)
autocast = _compat_autocast

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

LANGUAGES = ["python", "java", "javascript", "ruby", "go", "php"]

# Map from CodeSearchNet language name to the language string used by
# diversify_variable_names.py (they happen to be the same).
LANG_MAP = {lang: lang for lang in LANGUAGES}


class CodeSearchNetMLMDataset(Dataset):
    """Loads CodeSearchNet data and prepares it for MLM + RTD training.

    Supports optional per-epoch variable name diversification: when
    diversify=True, each __getitem__ call applies random variable renaming
    using a seed derived from (epoch, index) so that each epoch produces
    different renamings.

    When identifier_aware=True, each item also returns an identifier_mask
    tensor (1 for identifier tokens, 0 for structural tokens) computed by
    mapping tree-sitter variable spans to BPE token positions.

    When canonical=True, all variable names are replaced with a canonical
    scheme (VAR_0, VAR_1, ...) based on first-appearance order.  This is
    applied once at load time (the mapping is deterministic per snippet).
    """

    def __init__(self, tokenizer, max_length=512, data_pct=1.0,
                 cache_dir="data_cache", split="train", seed=42,
                 diversify=False, identifier_aware=False, canonical=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.diversify = diversify
        self.identifier_aware = identifier_aware
        self.canonical = canonical
        self.epoch = 0  # Updated externally for per-epoch diversity
        self.base_seed = seed
        self.examples = []
        self.languages = []  # Track language per example for diversification

        # Lazy-load canonicalize_code only when needed
        self._canonicalize_code = None
        if self.canonical:
            self._init_canonicalizer()

        logger.info(f"Loading CodeSearchNet ({split}, {data_pct*100:.1f}% of data)...")
        for lang in LANGUAGES:
            for _attempt in range(5):
                try:
                    ds = load_dataset("code_search_net", lang, cache_dir=cache_dir,
                                      split=split)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {lang} (attempt {_attempt+1}/5): {e}")
                    if _attempt == 4:
                        raise
                    import time as _t; _t.sleep(30 * (_attempt + 1))
            for item in ds:
                nl = item.get("func_documentation_string", "") or ""
                pl = item.get("func_code_string", "") or ""
                if pl.strip():
                    pl_clean = pl.strip()
                    # Apply canonical naming at load time (deterministic)
                    if self.canonical and self._canonicalize_code is not None:
                        try:
                            pl_clean = self._canonicalize_code(pl_clean, lang)
                        except Exception:
                            pass  # Keep original on failure
                    self.examples.append((nl.strip(), pl_clean))
                    self.languages.append(lang)

        # Subsample if data_pct < 1.0
        if data_pct < 1.0:
            rng = random.Random(seed)
            k = max(1, int(len(self.examples) * data_pct))
            indices = rng.sample(range(len(self.examples)), k)
            self.examples = [self.examples[i] for i in indices]
            self.languages = [self.languages[i] for i in indices]

        logger.info(f"  Loaded {len(self.examples)} examples "
                     f"(diversify={diversify}, identifier_aware={identifier_aware}, "
                     f"canonical={canonical})")

        # Lazy-load diversify_code and extract_variables only when needed
        self._diversify_code = None
        self._extract_variables = None
        if self.diversify:
            self._init_diversifier()
        if self.identifier_aware:
            self._init_identifier_extractor()

    def _init_diversifier(self):
        """Lazy import of diversify_code to avoid import overhead when not needed."""
        try:
            from diversify_variable_names import diversify_code
            self._diversify_code = diversify_code
            logger.info("  Variable name diversification enabled")
        except ImportError:
            logger.warning("  Could not import diversify_variable_names; "
                           "diversification disabled")
            self.diversify = False

    def _init_identifier_extractor(self):
        """Lazy import of extract_variables for identifier-aware MLM."""
        try:
            from diversify_variable_names import extract_variables
            self._extract_variables = extract_variables
            logger.info("  Identifier-aware MLM: variable extraction enabled")
        except ImportError:
            logger.warning("  Could not import extract_variables; "
                           "identifier-aware MLM disabled")
            self.identifier_aware = False

    def _init_canonicalizer(self):
        """Lazy import of canonicalize_code for canonical variable naming."""
        try:
            from canonicalize_variable_names import canonicalize_code
            self._canonicalize_code = canonicalize_code
            logger.info("  Canonical variable naming enabled (VAR_0, VAR_1, ...)")
        except ImportError:
            logger.warning("  Could not import canonicalize_variable_names; "
                           "canonical naming disabled")
            self.canonical = False

    def set_epoch(self, epoch):
        """Set the current epoch for per-epoch re-randomization."""
        self.epoch = epoch

    def __len__(self):
        return len(self.examples)

    def _compute_identifier_mask(self, pl, lang, encoding):
        """Compute a boolean mask marking which token positions are identifiers.

        Uses tree-sitter to extract variable names from the code, then maps
        variable name occurrences to token positions via offset mapping.
        Returns a tensor of shape (max_length,) with 1 for identifier tokens.
        """
        import re as _re
        ident_mask = torch.zeros(self.max_length, dtype=torch.long)

        if self._extract_variables is None:
            return ident_mask

        try:
            variables = self._extract_variables(pl, lang)
        except Exception:
            return ident_mask

        if not variables:
            return ident_mask

        # Find all character spans of variable occurrences in the code
        ident_char_spans = set()  # set of (start, end) in the PL string
        for var_name in variables:
            if lang == 'php':
                pattern = r'\$' + _re.escape(var_name) + r'(?=\b|\W|$)'
                for m in _re.finditer(pattern, pl):
                    # Include the $ in the span
                    ident_char_spans.add((m.start(), m.end()))
            else:
                pattern = r'\b' + _re.escape(var_name) + r'\b'
                for m in _re.finditer(pattern, pl):
                    ident_char_spans.add((m.start(), m.end()))

        if not ident_char_spans:
            return ident_mask

        # Get offset mapping for the PL segment.
        # When tokenizer encodes (NL, PL), the token sequence is:
        #   <s> NL_tokens </s></s> PL_tokens </s> <pad>...
        # offset_mapping gives character offsets within each segment.
        # We need to identify where the PL segment starts in the token sequence.
        offsets = encoding.get("offset_mapping")
        if offsets is None:
            return ident_mask

        # offsets is a list of (start, end) tuples, one per token
        # For pair encoding, we need to find the PL segment boundary.
        # Use token_type_ids if available, otherwise find the double </s>
        input_ids = encoding["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.squeeze(0).tolist()
        else:
            ids_list = input_ids if isinstance(input_ids[0], int) else input_ids[0]

        if isinstance(offsets, torch.Tensor):
            offsets_list = offsets.squeeze(0).tolist()
        else:
            offsets_list = offsets if isinstance(offsets[0], (list, tuple)) else offsets[0]

        # Find the PL segment start: after the second </s> (or first </s></s> pair)
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        pl_start_idx = None
        sep_count = 0
        for i, tid in enumerate(ids_list):
            if tid == sep_id:
                sep_count += 1
                if sep_count == 2:
                    pl_start_idx = i + 1
                    break

        if pl_start_idx is None:
            return ident_mask

        # For each token in the PL segment, check if its character span
        # overlaps with any identifier span
        for tok_idx in range(pl_start_idx, len(ids_list)):
            if tok_idx >= self.max_length:
                break
            if ids_list[tok_idx] == (self.tokenizer.pad_token_id or 1):
                break
            tok_start, tok_end = offsets_list[tok_idx]
            if tok_start == tok_end:
                continue  # Skip zero-width tokens (whitespace-only)
            # Check overlap with any identifier span
            for id_start, id_end in ident_char_spans:
                if tok_start < id_end and tok_end > id_start:
                    ident_mask[tok_idx] = 1
                    break

        return ident_mask

    def __getitem__(self, idx):
        nl, pl = self.examples[idx]
        lang = self.languages[idx]

        # Apply variable name diversification if enabled
        if self.diversify and self._diversify_code is not None:
            # Unique seed per (epoch, example) pair
            div_seed = self.base_seed + self.epoch * 1_000_000 + idx
            try:
                pl = self._diversify_code(pl, lang, seed=div_seed)
            except Exception:
                pass  # Keep original on failure

        # Encode as "[CLS] NL [SEP] PL [SEP]" (bimodal format)
        return_offsets = self.identifier_aware
        encoding = self.tokenizer(
            nl, pl,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=return_offsets,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if self.identifier_aware:
            ident_mask = self._compute_identifier_mask(pl, lang, encoding)
            return input_ids, attention_mask, ident_mask

        return input_ids, attention_mask


# ────────────────────────────────────────────────────────────────────
# MLM + RTD collator
# ────────────────────────────────────────────────────────────────────

class MLMRTDCollator:
    """Creates MLM and RTD targets from a batch of tokenized inputs.

    MLM: Mask 15% of tokens (80% [MASK], 10% random, 10% keep).
    RTD: Replace some tokens with random tokens and create binary labels.

    If identifier_aware=True, per-example identifier masks (computed by the
    dataset using tree-sitter) are used to skip identifier tokens during MLM.
    If mlm_only=True, RTD targets are still created (as zeros) but ignored
    in the model forward pass.
    """

    def __init__(self, tokenizer, mlm_prob=0.15, identifier_aware=False,
                 mlm_only=False):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.mlm_only = mlm_only
        self.identifier_aware = identifier_aware
        self.special_token_ids = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ])
        self.special_token_ids.discard(None)

        if identifier_aware:
            logger.info("Identifier-aware MLM: using per-example identifier masks "
                        "from tree-sitter AST parsing")

    def __call__(self, batch):
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])

        # Identifier masks (only present when identifier_aware is enabled)
        identifier_masks = None
        if self.identifier_aware and len(batch[0]) > 2:
            identifier_masks = torch.stack([b[2] for b in batch])

        # ── MLM targets ──
        mlm_input_ids, mlm_labels = self._mask_tokens(
            input_ids.clone(), identifier_masks)

        # ── RTD targets ──
        if self.mlm_only:
            # Still provide RTD tensors (zeros) for consistent forward signature
            rtd_input_ids = input_ids.clone()
            rtd_labels = torch.zeros_like(input_ids, dtype=torch.long)
        else:
            rtd_input_ids, rtd_labels = self._replace_tokens(
                input_ids.clone(), attention_mask)

        return {
            "mlm_input_ids": mlm_input_ids,
            "mlm_labels": mlm_labels,
            "rtd_input_ids": rtd_input_ids,
            "rtd_labels": rtd_labels,
            "attention_mask": attention_mask,
        }

    def _mask_tokens(self, input_ids, identifier_masks=None):
        """Standard MLM masking: 15% of tokens masked.

        If identifier_masks is provided, identifier tokens (value=1) are
        excluded from masking.
        """
        labels = input_ids.clone()
        prob_matrix = torch.full(input_ids.shape, self.mlm_prob)

        # Don't mask special tokens
        for sid in self.special_token_ids:
            prob_matrix.masked_fill_(input_ids == sid, 0.0)

        # Don't mask identifier tokens (identifier-aware MLM)
        if identifier_masks is not None:
            prob_matrix.masked_fill_(identifier_masks.bool(), 0.0)

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of masked: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of masked: replace with random token
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10%: keep original
        return input_ids, labels

    def _replace_tokens(self, input_ids, attention_mask):
        """RTD: Replace ~15% of tokens with random tokens, create binary labels."""
        labels = torch.zeros_like(input_ids, dtype=torch.long)
        prob_matrix = torch.full(input_ids.shape, self.mlm_prob)

        # Don't replace special tokens or padding
        for sid in self.special_token_ids:
            prob_matrix.masked_fill_(input_ids == sid, 0.0)
        prob_matrix.masked_fill_(attention_mask == 0, 0.0)

        replace_indices = torch.bernoulli(prob_matrix).bool()
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[replace_indices] = random_words[replace_indices]
        labels[replace_indices] = 1  # 1 = replaced

        return input_ids, labels


# ────────────────────────────────────────────────────────────────────
# RTD head (simple binary classifier on top of RoBERTa hidden states)
# ────────────────────────────────────────────────────────────────────

class RTDHead(nn.Module):
    """Binary classifier for Replaced Token Detection."""

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.classifier(x).squeeze(-1)
        return x


# ────────────────────────────────────────────────────────────────────
# Combined model: MLM + RTD
# ────────────────────────────────────────────────────────────────────

class CodeBERTPretrainModel(nn.Module):
    """CodeBERT pre-training model with MLM + RTD objectives.

    If mlm_only=True, only the MLM loss is computed (RTD head exists but
    is not used for the loss).
    """

    def __init__(self, config, mlm_only=False):
        super().__init__()
        self.roberta = RobertaForMaskedLM(config)
        self.rtd_head = RTDHead(config.hidden_size)
        self.mlm_only = mlm_only

    def forward(self, mlm_input_ids, mlm_labels, rtd_input_ids, rtd_labels,
                attention_mask):
        # MLM forward pass
        mlm_outputs = self.roberta(
            input_ids=mlm_input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels,
        )
        mlm_loss = mlm_outputs.loss

        if self.mlm_only:
            return {
                "loss": mlm_loss,
                "mlm_loss": mlm_loss.detach(),
                "rtd_loss": torch.tensor(0.0, device=mlm_loss.device),
            }

        # RTD forward pass (use the base model, not the MLM head)
        rtd_outputs = self.roberta.roberta(
            input_ids=rtd_input_ids,
            attention_mask=attention_mask,
        )
        rtd_logits = self.rtd_head(rtd_outputs.last_hidden_state)

        # RTD loss: binary cross entropy on non-padding tokens
        rtd_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        rtd_loss = rtd_loss_fn(rtd_logits, rtd_labels.float())
        rtd_loss = (rtd_loss * attention_mask).sum() / attention_mask.sum()

        total_loss = mlm_loss + rtd_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss.detach(),
            "rtd_loss": rtd_loss.detach(),
        }


# ────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CodeBERT from scratch")

    # Data
    p.add_argument("--data_pct", type=float, default=0.01,
                    help="Fraction of CodeSearchNet to use (default: 0.01 = 1%%)")
    p.add_argument("--data_cache", type=str, default="data_cache")
    p.add_argument("--max_length", type=int, default=512)

    # Model
    p.add_argument("--from_scratch", action="store_true", default=True,
                    help="Train from randomly initialized weights")
    p.add_argument("--model_name", type=str, default="microsoft/codebert-base",
                    help="Model name or path (used for config/tokenizer)")
    p.add_argument("--model_cache", type=str, default="models_cache")

    # Training hyperparameters (CodeBERT originals)
    p.add_argument("--per_gpu_batch_size", type=int, default=32,
                    help="Per-GPU batch size")
    p.add_argument("--effective_batch_size", type=int, default=2048,
                    help="Target effective batch size (gradient accumulation fills the gap)")
    p.add_argument("--lr", type=float, default=5e-4,
                    help="Learning rate (CodeBERT original: 5e-4)")
    p.add_argument("--warmup_steps", type=int, default=10000,
                    help="Warmup steps (CodeBERT original: 10000)")
    p.add_argument("--max_steps", type=int, default=100000,
                    help="Max training steps (CodeBERT original: 100000)")
    p.add_argument("--num_epochs", type=int, default=None,
                    help="Override max_steps with epoch-based training")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    # Experiment modes
    p.add_argument("--diversify_varnames", action="store_true", default=False,
                    help="Enable per-epoch variable name diversification")
    p.add_argument("--mlm_only", action="store_true", default=False,
                    help="Train with MLM loss only (disable RTD)")
    p.add_argument("--identifier_aware_mlm", action="store_true", default=False,
                    help="Never mask identifier tokens during MLM")
    p.add_argument("--char_tokenizer", action="store_true", default=False,
                    help="Use character-level (byte-level) tokenizer instead of BPE")
    p.add_argument("--canonical_varnames", action="store_true", default=False,
                    help="Replace variables with canonical names (VAR_0, VAR_1, ...)")

    # Output
    p.add_argument("--output_dir", type=str, default="checkpoints/codebert_scratch")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--save_every_n_epochs", type=int, default=1,
                    help="Save checkpoint every N epochs (default: 1 = every epoch). "
                         "Always saves the final checkpoint.")

    # Resume
    p.add_argument("--resume_from", type=str, default=None,
                    help="Path to checkpoint directory to resume from")
    p.add_argument("--constant_lr", action="store_true", default=False,
                    help="Use constant LR (no warmup/decay). Useful for continued training.")

    # Epoch evaluation
    p.add_argument("--eval_each_epoch", action="store_true", default=False,
                    help="Run zero-shot robustness evaluation after each epoch")
    p.add_argument("--eval_data_dir", type=str, default="data_cache/poj104",
                    help="Directory containing POJ-104 test.jsonl for evaluation")
    p.add_argument("--parallel_eval_gpus", type=str, default="",
                    help="Comma-separated GPU indices to use as a pool for parallel "
                         "evaluation subprocesses (e.g. '4,5,6,7'). Each eligible "
                         "checkpoint is launched on the first free GPU in the pool. "
                         "Empty string disables parallel eval.")
    p.add_argument("--parallel_eval_every_n", type=int, default=3,
                    help="Cadence (in epochs) for parallel eval launches.")

    # Distributed
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


def setup_distributed():
    """Set up distributed training if WORLD_SIZE > 1."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), dist.get_rank()
    elif torch.cuda.is_available():
        return 0, 1, 0
    else:
        return -1, 1, 0


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch,
                    args, loss_history):
    """Save model checkpoint with all state needed for resumption."""
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the model (unwrap DDP if needed)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.roberta.save_pretrained(ckpt_dir)

    # Save RTD head separately
    torch.save(model_to_save.rtd_head.state_dict(),
               os.path.join(ckpt_dir, "rtd_head.pt"))

    # Save optimizer, scheduler, scaler, and training state
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

    # Save loss history
    save_loss_history(loss_history, args.output_dir)

    logger.info(f"Checkpoint saved: {ckpt_dir}")


def save_loss_history(loss_history, output_dir):
    """Save loss history for analysis."""
    path = os.path.join(output_dir, "loss_history.json")
    with open(path, "w") as f:
        json.dump(loss_history, f, indent=2)


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, device,
                    skip_scheduler=False):
    """Load model checkpoint for resumption. Returns (step, epoch)."""
    logger.info(f"Resuming from checkpoint: {ckpt_dir}")

    # Load RoBERTa weights
    model_to_load = model.module if hasattr(model, "module") else model
    from transformers import RobertaForMaskedLM as _RFM
    model_to_load.roberta = _RFM.from_pretrained(ckpt_dir)
    model_to_load.roberta.to(device)

    # Load RTD head
    rtd_path = os.path.join(ckpt_dir, "rtd_head.pt")
    if os.path.exists(rtd_path):
        model_to_load.rtd_head.load_state_dict(
            torch.load(rtd_path, map_location=device))

    # Load training state
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    if not skip_scheduler:
        scheduler.load_state_dict(state["scheduler"])
    else:
        logger.info("Skipping scheduler restore (constant_lr mode)")
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    step = state["step"]
    epoch = state.get("epoch", 0)
    logger.info(f"Resumed from step {step}, epoch {epoch}")
    return step, epoch


def zeroshot_robustness_eval(model, tokenizer, eval_data_dir, device,
                             max_length=512, rename_counts=(0, 1, 4, 8)):
    """Quick zero-shot robustness evaluation on POJ-104 clone detection.

    Uses cosine similarity on CLS embeddings (no fine-tuning) to measure
    how well the model handles variable renaming. Fast (~5 min on GPU).
    """
    import re
    import numpy as np

    test_file = os.path.join(eval_data_dir, "test.jsonl")
    if not os.path.exists(test_file):
        logger.warning(f"Eval data not found: {test_file}, skipping evaluation")
        return None

    C_KEYWORDS = {
        'auto','break','case','char','const','continue','default','do',
        'double','else','enum','extern','float','for','goto','if','inline',
        'int','long','register','restrict','return','short','signed','sizeof',
        'static','struct','switch','typedef','union','unsigned','void',
        'volatile','while','bool','true','false','NULL','nullptr','class',
        'public','private','protected','virtual','template','typename',
        'namespace','using','new','delete','this','try','catch','throw',
    }
    C_STDLIB = {
        'printf','scanf','malloc','free','strlen','strcpy','strcmp','memset',
        'memcpy','abs','sqrt','pow','sort','reverse','max','min','swap',
        'push_back','begin','end','size','empty','find','insert','erase',
        'main','cin','cout','endl','string','vector','map','set','queue',
        'stack','pair','include','define','std','ios',
    }
    RESERVED = C_KEYWORDS | C_STDLIB

    def extract_vars(code):
        code_c = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
        code_c = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code_c)
        code_c = re.sub(r'//[^\n]*', '', code_c)
        code_c = re.sub(r'/\*.*?\*/', '', code_c, flags=re.DOTALL)
        code_c = re.sub(r'#\s*\w+[^\n]*', '', code_c)
        idents = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code_c))
        return [n for n in idents if n not in RESERVED and not n.startswith('__')]

    def rename_vars(code, n):
        variables = extract_vars(code)
        if not variables:
            return code, 0
        k = min(n, len(variables))
        selected = random.sample(variables, k)
        all_names = set(variables) | RESERVED
        rmap = {}
        for old in selected:
            chars = 'abcdefghijklmnopqrstuvwxyz'
            for _ in range(100):
                ln = random.randint(2, 8)
                nm = random.choice(chars) + ''.join(random.choice(chars + '0123456789_') for _ in range(ln - 1))
                if nm not in all_names:
                    rmap[old] = nm
                    all_names.add(nm)
                    break
        renamed = code
        for old in sorted(rmap, key=len, reverse=True):
            renamed = re.sub(r'\b' + re.escape(old) + r'\b', rmap[old], renamed)
        return renamed, k

    def tokenize_code(code, tok, block_size):
        if hasattr(tok, 'tokenize'):
            toks = tok.tokenize(code)[:block_size - 2]
            ids = tok.convert_tokens_to_ids([tok.cls_token] + toks + [tok.sep_token])
        else:
            enc = tok(code, truncation=True, max_length=block_size,
                      padding='max_length', return_tensors=None)
            ids = enc['input_ids']
            ids = ids[0] if isinstance(ids[0], list) else ids
            return ids[:block_size]
        ids += [tok.pad_token_id] * (block_size - len(ids))
        return ids[:block_size]

    # Load test data
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Tokenize all
    all_ids = []
    all_labels = []
    for js in test_data:
        code = ' '.join(js['code'].split())
        ids = tokenize_code(code, tokenizer, max_length)
        all_ids.append(ids)
        all_labels.append(int(js['label']))
    all_labels = np.array(all_labels)

    # Get CLS embeddings
    raw_model = model.module if hasattr(model, 'module') else model
    encoder = raw_model.roberta
    encoder.eval()
    all_vecs = []
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        for i in range(0, len(all_ids), 64):
            batch = torch.tensor(all_ids[i:i+64]).to(device)
            mask = batch.ne(pad_id)
            out = encoder(batch, attention_mask=mask)
            vecs = out[0][:, 0, :]  # CLS token
            all_vecs.append(vecs.cpu().numpy())
    vecs = np.concatenate(all_vecs, 0)

    # Cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vecs_n = vecs / norms
    scores = np.matmul(vecs_n, vecs_n.T)
    for i in range(len(all_labels)):
        scores[i, i] = -1e9

    # Correct at N=0
    sort_ids = np.argsort(scores, axis=-1)[:, ::-1]
    correct_0 = [i for i in range(len(all_labels))
                 if int(all_labels[sort_ids[i, 0]]) == int(all_labels[i])]

    results = {'num_test': len(all_labels), 'num_correct_0': len(correct_0),
               'accuracy_N0': 1.0}
    logger.info(f"  [Eval] Correct at N=0: {len(correct_0)}/{len(all_labels)}")

    random.seed(123456)
    np.random.seed(123456)
    for n_renames in rename_counts:
        if n_renames == 0:
            continue
        renamed_ids = []
        for idx in correct_0:
            rcode, _ = rename_vars(test_data[idx]['code'], n_renames)
            rcode = ' '.join(rcode.split())
            renamed_ids.append(tokenize_code(rcode, tokenizer, max_length))

        renamed_vecs = []
        with torch.no_grad():
            for i in range(0, len(renamed_ids), 64):
                batch = torch.tensor(renamed_ids[i:i+64]).to(device)
                mask = batch.ne(pad_id)
                out = encoder(batch, attention_mask=mask)
                v = out[0][:, 0, :]
                renamed_vecs.append(v.cpu().numpy())
        renamed_vecs = np.concatenate(renamed_vecs, 0)

        still_correct = 0
        for j, idx in enumerate(correct_0):
            rv = renamed_vecs[j]
            rn = np.linalg.norm(rv)
            rv_n = rv / max(rn, 1e-10)
            sim = np.dot(vecs_n, rv_n)
            sim[idx] = -1e9
            if int(all_labels[np.argmax(sim)]) == int(all_labels[idx]):
                still_correct += 1

        acc = still_correct / max(len(correct_0), 1)
        results[f'accuracy_N{n_renames}'] = acc
        logger.info(f"  [Eval] N={n_renames}: Accuracy = {acc:.4f} "
                     f"({still_correct}/{len(correct_0)})")

    encoder.train()
    return results


def main():
    args = parse_args()

    if args.no_fp16:
        args.fp16 = False

    local_rank, world_size, global_rank = setup_distributed()
    is_main = global_rank == 0

    if is_main:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Args: {vars(args)}")
        if args.diversify_varnames:
            logger.info("MODE: Per-epoch variable name diversification ENABLED")
        if args.char_tokenizer:
            logger.info("MODE: Character-level (byte) tokenizer ENABLED")
        if args.mlm_only:
            logger.info("MODE: MLM-only training (RTD disabled)")
        if args.identifier_aware_mlm:
            logger.info("MODE: Identifier-aware MLM (identifiers never masked)")
        if args.canonical_varnames:
            logger.info("MODE: Canonical variable naming (VAR_0, VAR_1, ...) ENABLED")

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")

    # ── Tokenizer ──
    if args.char_tokenizer:
        from char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer(model_max_length=args.max_length)
        if is_main:
            logger.info("MODE: Character-level (byte) tokenizer ENABLED")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, cache_dir=args.model_cache
        )
    if is_main:
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # ── Model ──
    config = RobertaConfig.from_pretrained(args.model_name, cache_dir=args.model_cache)
    if args.char_tokenizer:
        config.vocab_size = len(tokenizer)  # 261 for byte-level tokenizer
    if is_main:
        logger.info(f"Model config: {config.num_hidden_layers} layers, "
                     f"{config.hidden_size} hidden, {config.num_attention_heads} heads")

    model = CodeBERTPretrainModel(config, mlm_only=args.mlm_only)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Dataset ──
    dataset = CodeSearchNetMLMDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        data_pct=args.data_pct,
        cache_dir=args.data_cache,
        split="train",
        seed=args.seed,
        diversify=args.diversify_varnames,
        identifier_aware=args.identifier_aware_mlm,
        canonical=args.canonical_varnames,
    )

    collator = MLMRTDCollator(
        tokenizer,
        identifier_aware=args.identifier_aware_mlm,
        mlm_only=args.mlm_only,
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                      rank=global_rank, shuffle=True, seed=args.seed)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Gradient accumulation ──
    per_step_batch = args.per_gpu_batch_size * world_size
    grad_accum_steps = max(1, args.effective_batch_size // per_step_batch)
    actual_effective_batch = per_step_batch * grad_accum_steps
    if is_main:
        logger.info(f"Per-GPU batch: {args.per_gpu_batch_size}, "
                     f"World size: {world_size}, "
                     f"Grad accum: {grad_accum_steps}, "
                     f"Effective batch: {actual_effective_batch}")

    # ── Determine total steps ──
    if args.num_epochs is not None:
        steps_per_epoch = len(dataloader) // grad_accum_steps
        total_steps = steps_per_epoch * args.num_epochs
        if is_main:
            logger.info(f"Epoch-based training: {args.num_epochs} epochs, "
                         f"{steps_per_epoch} steps/epoch, {total_steps} total steps")
    else:
        total_steps = args.max_steps
        if is_main:
            logger.info(f"Step-based training: {total_steps} total steps")

    # ── Optimizer ──
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-6)

    # Scale warmup steps for small-scale runs
    if args.constant_lr:
        # Constant LR: no warmup, no decay — useful for continued training
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if is_main:
            logger.info(f"Using CONSTANT learning rate: {args.lr}")
    else:
        warmup = min(args.warmup_steps, total_steps // 5)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    if args.fp16 and device.type == "cuda":
        try:
            scaler = GradScaler("cuda")
        except TypeError:
            scaler = GradScaler()
    else:
        scaler = None

    # ── Resume from checkpoint ──
    start_step = 0
    start_epoch = 0
    loss_history = []

    if args.resume_from and os.path.isdir(args.resume_from):
        start_step, start_epoch = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, scaler, device,
            skip_scheduler=args.constant_lr)
        # Load existing loss history
        hist_path = os.path.join(args.output_dir, "loss_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                loss_history = json.load(f)
            if is_main:
                logger.info(f"Loaded {len(loss_history)} loss history entries")

    # ── Output dir ──
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Training ──
    if is_main:
        logger.info("=" * 60)
        logger.info(f"Starting training (from step {start_step})")
        logger.info("=" * 60)

    model.train()
    global_step = start_step
    accum_mlm_loss = 0.0
    accum_rtd_loss = 0.0
    accum_total_loss = 0.0
    accum_count = 0
    start_time = time.time()

    # Pool of (gpu_id -> Popen) slots. A slot's Popen is None when free.
    # When all slots are busy, this epoch's eval is skipped (don't stack).
    eval_pool_gpus = [int(g) for g in args.parallel_eval_gpus.split(",") if g.strip()] \
        if args.parallel_eval_gpus else []
    eval_pool = {gpu: None for gpu in eval_pool_gpus}

    def _maybe_launch_parallel_eval(step, epoch):
        """Launch evaluate.sh --only_step on the first free GPU in the pool.

        - rank-0 only (caller invokes this inside is_main blocks)
        - skipped if epoch isn't a multiple of parallel_eval_every_n
        - if every GPU in the pool is busy, the launch is skipped (no stacking)
        """
        if not eval_pool:
            return
        if epoch <= 0 or args.parallel_eval_every_n <= 0:
            return
        if epoch % args.parallel_eval_every_n != 0:
            return

        # Reap any finished evals so their slots become free again.
        free_gpu = None
        for gpu in eval_pool_gpus:
            proc = eval_pool[gpu]
            if proc is None or proc.poll() is not None:
                eval_pool[gpu] = None
                if free_gpu is None:
                    free_gpu = gpu

        if free_gpu is None:
            busy = [(g, eval_pool[g].pid) for g in eval_pool_gpus if eval_pool[g] is not None]
            logger.info(f"  [parallel_eval] all {len(eval_pool_gpus)} eval GPUs busy "
                        f"({busy}); skipping launch for epoch {epoch}.")
            return

        import subprocess as _sp
        # Strip the trailing '/poj104' that eval_data_dir carries.
        data_dir_for_eval = args.eval_data_dir.rstrip("/")
        if data_dir_for_eval.endswith("/poj104"):
            data_dir_for_eval = data_dir_for_eval[:-len("/poj104")]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(free_gpu)

        cmd = [
            "bash", "evaluate.sh",
            "--checkpoint_dir", args.output_dir,
            "--data_dir", data_dir_for_eval,
            "--only_step", str(step),
        ]
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"parallel_eval_epoch_{epoch}.log")
        log_fh = open(log_path, "a")
        proc = _sp.Popen(
            cmd, env=env, stdout=log_fh, stderr=_sp.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        eval_pool[free_gpu] = proc
        logger.info(f"  [parallel_eval] launched evaluate.sh for epoch {epoch} "
                    f"(step {step}) on GPU {free_gpu}, PID {proc.pid}, log {log_path}")

    epoch = start_epoch
    while global_step < total_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Update dataset epoch for per-epoch diversification
        dataset.set_epoch(epoch)

        if is_main:
            logger.info(f"Starting epoch {epoch}" +
                        (f" (diversify seed offset: {epoch})" if args.diversify_varnames else ""))

        for batch_idx, batch in enumerate(dataloader):
            # Skip batches already processed (for resume)
            effective_batch_idx = (batch_idx + 1) // grad_accum_steps
            if epoch == start_epoch + 1 and start_step > 0:
                # During the first epoch after resume, skip already-processed batches
                if effective_batch_idx <= (start_step % (len(dataloader) // grad_accum_steps)):
                    # But we still need to count through; skip full accum groups
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        if effective_batch_idx + (start_epoch * (len(dataloader) // grad_accum_steps)) < start_step:
                            continue

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            if args.fp16 and scaler is not None:
                with autocast("cuda"):
                    outputs = model(**batch)
                    loss = outputs["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()

            accum_mlm_loss += outputs["mlm_loss"].item()
            accum_rtd_loss += outputs["rtd_loss"].item()
            accum_total_loss += outputs["loss"].item()
            accum_count += 1

            # Optimizer step every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if args.fp16 and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if is_main and global_step % args.log_interval == 0:
                    avg_mlm = accum_mlm_loss / accum_count
                    avg_rtd = accum_rtd_loss / accum_count
                    avg_total = accum_total_loss / accum_count
                    elapsed = time.time() - start_time
                    steps_done = global_step - start_step
                    steps_per_sec = steps_done / max(elapsed, 1e-9)
                    eta = (total_steps - global_step) / max(steps_per_sec, 1e-9)

                    loss_history.append({
                        "step": global_step,
                        "epoch": epoch,
                        "mlm_loss": avg_mlm,
                        "rtd_loss": avg_rtd,
                        "total_loss": avg_total,
                        "lr": scheduler.get_last_lr()[0],
                    })

                    rtd_str = f", RTD: {avg_rtd:.4f}" if not args.mlm_only else ""
                    log_msg = (
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_total:.4f} (MLM: {avg_mlm:.4f}{rtd_str}) | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Speed: {steps_per_sec:.2f} steps/s | "
                        f"ETA: {eta/60:.1f} min"
                    )
                    logger.info(log_msg)
                    # Also write to a progress file with explicit flush
                    # to work around SLURM output buffering
                    progress_file = os.path.join(args.output_dir, "progress.log")
                    with open(progress_file, "a") as pf:
                        import datetime
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        pf.write(f"{ts} | INFO | {log_msg}\n")
                        pf.flush()

                    accum_mlm_loss = 0.0
                    accum_rtd_loss = 0.0
                    accum_total_loss = 0.0
                    accum_count = 0

                # Save checkpoint
                if is_main and (global_step % args.save_interval == 0 or
                                global_step == total_steps):
                    save_checkpoint(model, optimizer, scheduler, scaler,
                                    global_step, epoch, args, loss_history)

                if global_step >= total_steps:
                    break

        # ── End of epoch: conditionally save checkpoint + evaluate ──
        if is_main:
            logger.info(f"Epoch {epoch} complete. Global step: {global_step}")
            saved_this_epoch = False
            if epoch % args.save_every_n_epochs == 0 or global_step >= total_steps:
                save_checkpoint(model, optimizer, scheduler, scaler,
                                global_step, epoch, args, loss_history)
                saved_this_epoch = True
            else:
                logger.info(f"  (Skipping checkpoint save, next save at epoch "
                             f"{epoch + args.save_every_n_epochs - epoch % args.save_every_n_epochs})")

            # Fire off a parallel evaluation for this checkpoint, if eligible.
            if saved_this_epoch:
                _maybe_launch_parallel_eval(global_step, epoch)

        if is_main and args.eval_each_epoch:
            logger.info(f"{'='*60}")
            logger.info(f"Zero-shot evaluation after epoch {epoch} (step {global_step})")
            logger.info(f"{'='*60}")
            eval_results = zeroshot_robustness_eval(
                model, tokenizer, args.eval_data_dir, device,
                max_length=args.max_length)
            if eval_results:
                # Save eval results
                eval_path = os.path.join(args.output_dir, "epoch_eval_results.jsonl")
                entry = {"epoch": epoch, "step": global_step,
                         **eval_results}
                with open(eval_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                # Print summary table
                logger.info(f"  [SUMMARY] Epoch {epoch:>3d} | Step {global_step:>6d} | "
                             f"Correct@0: {eval_results['num_correct_0']:>5d} | "
                             f"N1: {eval_results.get('accuracy_N1', 0):.4f} | "
                             f"N4: {eval_results.get('accuracy_N4', 0):.4f} | "
                             f"N8: {eval_results.get('accuracy_N8', 0):.4f}")
            model.train()

    # ── Final save ──
    if is_main:
        save_checkpoint(model, optimizer, scheduler, scaler,
                        global_step, epoch, args, loss_history)
        save_loss_history(loss_history, args.output_dir)
        # Final-checkpoint eval (subject to the same don't-stack guard).
        _maybe_launch_parallel_eval(global_step, epoch)
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training complete! Total steps: {global_step}, "
                     f"Time: {elapsed/60:.1f} min")
        still_running = [(g, eval_pool[g].pid) for g in eval_pool_gpus
                         if eval_pool[g] is not None and eval_pool[g].poll() is None]
        if still_running:
            logger.info(f"  Parallel evals still running: {still_running}. "
                         f"They will continue after train.sh exits.")
        logger.info(f"Checkpoints saved to: {args.output_dir}")
        logger.info("=" * 60)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
