#!/usr/bin/env python3
"""
Zero-shot variable renaming robustness evaluation.

Evaluates a pre-trained CodeBERT checkpoint WITHOUT fine-tuning on any
downstream task. Uses mean-pooled cosine similarity on POJ-104 test set
to measure how well the raw pre-trained representations handle variable
renaming. This matches the ContraBERT paper's RQ1 zero-shot methodology.

Key difference from fine-tuned evaluation (evaluate_robustness.py):
  - Fine-tuned: model is fine-tuned on POJ-104 clone detection, then tested.
    Scores are high (~95%) because fine-tuning compensates for pre-training
    weaknesses.
  - Zero-shot (this script): raw pre-trained embeddings, no fine-tuning.
    Scores are much lower (~40-65%) and show larger differences between models.

Paper reference numbers (zero-shot, mean-pool cosine):
  CodeBERT:     N8 ~0.40
  ContraBERT_C: N8 ~0.65

Usage:
  python evaluate_zeroshot.py --checkpoint microsoft/codebert-base
  python evaluate_zeroshot.py --checkpoint checkpoints/exp1_diverse/checkpoint-29388
  python evaluate_zeroshot.py --checkpoint_dir checkpoints/exp1_diverse --every_n 10
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time

import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaModel, RobertaTokenizer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Variable renaming utilities ──

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
            nm = random.choice(chars) + ''.join(
                random.choice(chars + '0123456789_') for _ in range(ln - 1))
            if nm not in all_names:
                rmap[old] = nm
                all_names.add(nm)
                break
    renamed = code
    for old in sorted(rmap, key=len, reverse=True):
        renamed = re.sub(r'\b' + re.escape(old) + r'\b', rmap[old], renamed)
    return renamed, k


def tokenize_code(code, tokenizer, block_size):
    if hasattr(tokenizer, 'tokenize'):
        toks = tokenizer.tokenize(code)[:block_size - 2]
        ids = tokenizer.convert_tokens_to_ids(
            [tokenizer.cls_token] + toks + [tokenizer.sep_token])
    else:
        enc = tokenizer(code, truncation=True, max_length=block_size,
                        padding='max_length', return_tensors=None)
        ids = enc['input_ids']
        ids = ids[0] if isinstance(ids[0], list) else ids
        return ids[:block_size]
    ids += [tokenizer.pad_token_id] * (block_size - len(ids))
    return ids[:block_size]


def load_encoder(checkpoint_path, cache_dir='models_cache'):
    """Load encoder from checkpoint."""
    is_char = os.path.isdir(checkpoint_path) and os.path.exists(
        os.path.join(checkpoint_path, 'char_tokenizer_config.json'))

    if is_char:
        from char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer.from_pretrained(checkpoint_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(
            'microsoft/codebert-base', cache_dir=cache_dir)

    if os.path.isdir(checkpoint_path):
        try:
            mlm = RobertaForMaskedLM.from_pretrained(checkpoint_path)
            encoder = mlm.roberta
        except Exception:
            encoder = RobertaModel.from_pretrained(checkpoint_path)
    else:
        try:
            encoder = RobertaModel.from_pretrained(
                checkpoint_path, cache_dir=cache_dir)
        except Exception:
            mlm = RobertaForMaskedLM.from_pretrained(
                checkpoint_path, cache_dir=cache_dir)
            encoder = mlm.roberta

    return encoder, tokenizer


def evaluate_zeroshot(encoder, tokenizer, test_file, device,
                      block_size=400, rename_counts=(0, 1, 4, 8),
                      embedding_method='meanpool', renamed_dir=None):
    """Run zero-shot robustness evaluation.

    Uses mean-pooled embeddings + cosine similarity (matches ContraBERT
    paper methodology for zero-shot evaluation).

    If ``renamed_dir`` is provided and contains ``test_renamed_N{k}.jsonl``
    files (one per N>0 in ``rename_counts``), those shared, pre-generated
    var_xxx renamed test sets are used (so every model is scored on byte-for
    -byte identical perturbations). Otherwise the on-the-fly random-name
    renamer is used as a fallback.
    """
    # Load test data
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Optionally load shared, pre-generated renamed test sets (position-aligned
    # with test_data, same as the fine-tuned eval).
    renamed_sets = {}
    if renamed_dir:
        for n in rename_counts:
            if n == 0:
                continue
            path = os.path.join(renamed_dir, f'test_renamed_N{n}.jsonl')
            if not os.path.exists(path):
                logger.warning(f"Shared renamed set not found: {path}; "
                               f"will rename on the fly for N={n}")
                continue
            rlist = []
            with open(path) as f:
                for line in f:
                    rj = json.loads(line.strip())
                    rlist.append((rj['code'], rj.get('num_renamed', n)))
            if len(rlist) != len(test_data):
                logger.warning(f"Renamed set {path} length {len(rlist)} != "
                               f"test_data {len(test_data)}; ignoring")
                continue
            renamed_sets[n] = rlist
            logger.info(f"Loaded shared renamed test set N={n} "
                        f"({len(rlist)} examples) from {path}")

    # Tokenize
    all_ids = []
    all_labels = []
    for js in test_data:
        code = ' '.join(js['code'].split())
        ids = tokenize_code(code, tokenizer, block_size)
        all_ids.append(ids)
        all_labels.append(int(js['label']))
    all_labels = np.array(all_labels)

    # Get embeddings
    encoder.eval()
    pad_id = tokenizer.pad_token_id
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(all_ids), 64):
            batch = torch.tensor(all_ids[i:i+64]).to(device)
            mask = batch.ne(pad_id)
            out = encoder(batch, attention_mask=mask)

            if embedding_method == 'meanpool':
                # Mean pool over non-padding tokens
                hidden = out[0]  # [batch, seq_len, hidden]
                mask_f = mask.float().unsqueeze(-1)
                vecs = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            else:
                # CLS token
                vecs = out[0][:, 0, :]

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

    results = {
        'num_test': len(all_labels),
        'num_correct_0': len(correct_0),
        'accuracy_baseline': len(correct_0) / len(all_labels),
        'accuracy_N0': 1.0,
        'embedding_method': embedding_method,
    }
    logger.info(f"Correct at N=0: {len(correct_0)}/{len(all_labels)} "
                f"({len(correct_0)/len(all_labels)*100:.1f}%)")

    random.seed(123456)
    np.random.seed(123456)

    for n_renames in rename_counts:
        if n_renames == 0:
            continue

        renamed_ids = []
        for idx in correct_0:
            if n_renames in renamed_sets:
                # Use the shared, pre-generated var_xxx renaming for this example.
                rcode, _ = renamed_sets[n_renames][idx]
            else:
                rcode, _ = rename_vars(test_data[idx]['code'], n_renames)
            rcode = ' '.join(rcode.split())
            renamed_ids.append(tokenize_code(rcode, tokenizer, block_size))

        renamed_vecs = []
        with torch.no_grad():
            for i in range(0, len(renamed_ids), 64):
                batch = torch.tensor(renamed_ids[i:i+64]).to(device)
                mask = batch.ne(pad_id)
                out = encoder(batch, attention_mask=mask)
                if embedding_method == 'meanpool':
                    hidden = out[0]
                    mask_f = mask.float().unsqueeze(-1)
                    v = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                else:
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
        results[f'num_correct_N{n_renames}'] = still_correct
        logger.info(f"N={n_renames}: Accuracy = {acc:.4f} "
                     f"({still_correct}/{len(correct_0)})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot variable renaming robustness evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with multiple checkpoints')
    parser.add_argument('--every_n', type=int, default=1,
                        help='Evaluate every N-th checkpoint (with --checkpoint_dir)')
    parser.add_argument('--data_dir', type=str, default='data_cache/poj104')
    parser.add_argument('--renamed_dir', type=str, default=None,
                        help='Directory with pre-generated shared renamed test sets '
                             '(test_renamed_N{k}.jsonl). Defaults to --data_dir.')
    parser.add_argument('--cache_dir', type=str, default='models_cache')
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override the output filename stem (zeroshot_<NAME>.json).')
    parser.add_argument('--block_size', type=int, default=400)
    parser.add_argument('--embedding', type=str, default='meanpool',
                        choices=['meanpool', 'cls'],
                        help='Embedding method (default: meanpool, matches paper)')
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()
    if args.renamed_dir is None:
        args.renamed_dir = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = os.path.join(args.data_dir, 'test.jsonl')
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint:
        # Single checkpoint
        logger.info(f"Loading: {args.checkpoint}")
        encoder, tokenizer = load_encoder(args.checkpoint, args.cache_dir)
        encoder.to(device)
        results = evaluate_zeroshot(
            encoder, tokenizer, test_file, device,
            block_size=args.block_size,
            embedding_method=args.embedding,
            renamed_dir=args.renamed_dir)
        results['checkpoint'] = args.checkpoint
        name = args.experiment_name or os.path.basename(args.checkpoint.rstrip('/'))
        results['experiment_name'] = name

        # Save
        out_path = os.path.join(args.output_dir, f'zeroshot_{name}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {out_path}")

    elif args.checkpoint_dir:
        # Multiple checkpoints
        all_steps = sorted([
            int(d.replace('checkpoint-', ''))
            for d in os.listdir(args.checkpoint_dir)
            if d.startswith('checkpoint-')
        ])

        if not all_steps:
            logger.error(f"No checkpoints in {args.checkpoint_dir}")
            sys.exit(1)

        # Determine steps per epoch
        args_file = os.path.join(args.checkpoint_dir, 'training_args.json')
        steps_per_epoch = 918  # default for batch 2048
        if os.path.exists(args_file):
            with open(args_file) as f:
                ta = json.load(f)
            eff = ta.get('effective_batch_size', 2048)
            steps_per_epoch = 1880853 // eff

        results_all = []
        for step in all_steps:
            epoch = step // steps_per_epoch
            if args.every_n > 1 and epoch % args.every_n != 0 and step != all_steps[-1]:
                continue

            ckpt = os.path.join(args.checkpoint_dir, f'checkpoint-{step}')
            logger.info(f"\nEvaluating epoch {epoch} (step {step})...")

            encoder, tokenizer = load_encoder(ckpt, args.cache_dir)
            encoder.to(device)
            results = evaluate_zeroshot(
                encoder, tokenizer, test_file, device,
                block_size=args.block_size,
                embedding_method=args.embedding)
            results['checkpoint'] = ckpt
            results['epoch'] = epoch
            results['step'] = step
            results_all.append(results)

            # Save individual
            out_path = os.path.join(args.output_dir, f'zeroshot_epoch{epoch}.json')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Free memory
            del encoder
            torch.cuda.empty_cache()

        # Save combined
        out_path = os.path.join(args.output_dir, 'zeroshot_all.jsonl')
        with open(out_path, 'w') as f:
            for r in results_all:
                f.write(json.dumps(r) + '\n')
        logger.info(f"\nAll results saved to: {out_path}")
    else:
        parser.error("Provide --checkpoint or --checkpoint_dir")


if __name__ == '__main__':
    main()
