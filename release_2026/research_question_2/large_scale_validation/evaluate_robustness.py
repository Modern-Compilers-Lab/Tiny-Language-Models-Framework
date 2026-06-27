#!/usr/bin/env python3
"""
Unified evaluation script for variable renaming robustness.

Tests any trained CodeBERT checkpoint (including our 5 experiments) for
robustness to variable renaming, using the same benchmark and metrics as
the ContraBERT paper (Liu et al., ICSE 2023) for fair comparison.

Evaluation pipeline:
  1. Load a pre-trained encoder from one of:
       - HuggingFace model name (e.g. microsoft/codebert-base)
       - Our custom training checkpoint directory (RobertaForMaskedLM format)
  2. Fine-tune on POJ-104 clone detection (triplet loss)
  3. Evaluate MAP@R on original test set
  4. Apply variable renaming (N=1,4,8 edits) and measure accuracy degradation

Supported checkpoint formats:
  - HuggingFace hub models (RobertaModel or RobertaForMaskedLM)
  - Custom trained checkpoints from train_codebert.py
  - Character-level tokenizer checkpoints (auto-detected via char_tokenizer_config.json)

Metrics (matching ContraBERT paper):
  - MAP@R: Mean Average Precision at R on POJ-104 clone detection
  - Accuracy at N=0: baseline (always 1.0 by definition)
  - Accuracy at N=1,4,8: fraction of originally-correct predictions that
    remain correct after renaming N variables

Usage:
  # Evaluate original CodeBERT
  python evaluate_robustness.py --checkpoint microsoft/codebert-base

  # Evaluate our Experiment 1 (diverse names) checkpoint
  python evaluate_robustness.py --checkpoint checkpoints/exp1_diverse/checkpoint-XXXX

  # Evaluate Experiment 2 (character-level tokenizer)
  python evaluate_robustness.py --checkpoint checkpoints/exp2_charlevel/checkpoint-XXXX

  # Evaluate ContraBERT
  python evaluate_robustness.py --checkpoint claudios/ContraBERT_C

  # Compare all experiments
  python evaluate_robustness.py --compare_dir eval_results/
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (
    RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizer,
    AutoTokenizer, get_linear_schedule_with_warmup
)
from torch.optim import AdamW

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# Variable Renaming Utilities (shared with benchmark_variable_renaming.py)
# =============================================================================

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', 'bool', 'true', 'false',
    'NULL', 'nullptr', 'class', 'public', 'private', 'protected', 'virtual',
    'template', 'typename', 'namespace', 'using', 'new', 'delete', 'this',
    'try', 'catch', 'throw', 'const_cast', 'dynamic_cast', 'static_cast',
    'reinterpret_cast', 'operator', 'friend', 'explicit', 'mutable',
    'asm', 'wchar_t', 'typeid', 'and', 'or', 'not', 'xor', 'bitand',
    'bitor', 'compl', 'and_eq', 'or_eq', 'xor_eq', 'not_eq',
}

C_STDLIB = {
    'printf', 'scanf', 'fprintf', 'fscanf', 'sprintf', 'sscanf',
    'puts', 'gets', 'getchar', 'putchar', 'fgets', 'fputs',
    'malloc', 'calloc', 'realloc', 'free',
    'memset', 'memcpy', 'memmove', 'memcmp',
    'strlen', 'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp',
    'strstr', 'strchr', 'strrchr',
    'atoi', 'atof', 'atol', 'strtol', 'strtod',
    'abs', 'fabs', 'sqrt', 'pow', 'exp', 'log', 'log10',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'ceil', 'floor', 'round',
    'rand', 'srand', 'time',
    'exit', 'abort', 'assert',
    'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell', 'rewind',
    'qsort', 'bsearch',
    'isalpha', 'isdigit', 'isalnum', 'isspace', 'isupper', 'islower',
    'toupper', 'tolower',
    'main', 'cin', 'cout', 'cerr', 'endl', 'string', 'vector', 'map',
    'set', 'queue', 'stack', 'pair', 'sort', 'reverse', 'max', 'min',
    'swap', 'push_back', 'pop_back', 'begin', 'end', 'size', 'empty',
    'clear', 'find', 'insert', 'erase', 'front', 'back', 'top', 'push',
    'pop', 'first', 'second', 'make_pair', 'lower_bound', 'upper_bound',
    'next_permutation', 'prev_permutation', 'accumulate', 'fill',
    'count', 'unique', 'nth_element', 'partial_sort',
    'priority_queue', 'deque', 'list', 'bitset', 'algorithm',
    'iostream', 'cstdio', 'cstdlib', 'cstring', 'cmath', 'climits',
    'cfloat', 'cassert', 'cctype', 'ctime', 'fstream', 'sstream',
    'iomanip', 'numeric', 'functional', 'iterator', 'utility',
    'INT_MAX', 'INT_MIN', 'LONG_MAX', 'LONG_MIN', 'DBL_MAX', 'DBL_MIN',
    'LLONG_MAX', 'LLONG_MIN', 'UINT_MAX', 'SIZE_MAX',
    'std', 'ios', 'istream', 'ostream',
    'getline', 'to_string', 'stoi', 'stol', 'stof', 'stod',
    'memset', 'include', 'define', 'ifdef', 'ifndef', 'endif', 'pragma',
    'MAXN', 'MOD', 'INF', 'EPS', 'PI', 'N', 'M',
}

RESERVED = C_KEYWORDS | C_STDLIB


def extract_variables(code):
    """Extract variable/identifier names from C/C++ code using regex."""
    code_clean = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code_clean = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code_clean)
    code_clean = re.sub(r'//[^\n]*', '', code_clean)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
    code_clean = re.sub(r'#\s*\w+[^\n]*', '', code_clean)

    all_identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code_clean))

    variables = set()
    for name in all_identifiers:
        if name in RESERVED:
            continue
        if name == 'main':
            continue
        if name.startswith('__'):
            continue
        variables.add(name)

    return list(variables)


def generate_random_name(existing_names, length_range=(1, 8)):
    """Generate a unique 'var_<random number>' variable name.

    Every renamed variable becomes ``var_xxx`` where ``xxx`` is a random
    integer (e.g. ``var_48127``). The returned name is guaranteed not to
    collide with existing identifiers, reserved words, or names already
    assigned in this scope. ``length_range`` is kept for signature
    compatibility but is no longer used.
    """
    for _ in range(1000):
        name = 'var_' + str(random.randint(0, 999999))
        if name not in existing_names and name not in RESERVED:
            return name
    # Extremely unlikely fallback: widen the range until a unique name is found.
    while True:
        name = 'var_' + str(random.randint(0, 10 ** 9))
        if name not in existing_names and name not in RESERVED:
            return name


def rename_variables(code, num_renames):
    """Rename `num_renames` random variables in C/C++ code."""
    variables = extract_variables(code)
    if not variables:
        return code, 0

    num_to_rename = min(num_renames, len(variables))
    selected = random.sample(variables, num_to_rename)

    all_names = set(variables) | RESERVED
    rename_map = {}

    for old_name in selected:
        new_name = generate_random_name(all_names)
        all_names.add(new_name)
        rename_map[old_name] = new_name

    renamed_code = code
    for old_name in sorted(rename_map.keys(), key=len, reverse=True):
        new_name = rename_map[old_name]
        renamed_code = re.sub(
            r'\b' + re.escape(old_name) + r'\b', new_name, renamed_code)

    return renamed_code, num_to_rename


# =============================================================================
# Clone Detection Model (matches ContraBERT/CodeXGLUE)
# =============================================================================

class CloneModel(nn.Module):
    """Clone detection model using triplet loss with cosine/dot-product similarity."""

    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, p_input_ids=None, n_input_ids=None,
                labels=None):
        bs, _ = input_ids.size()
        input_ids = torch.cat((input_ids, p_input_ids, n_input_ids), 0)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(
            self.tokenizer.pad_token_id))[1]
        outputs = outputs.split(bs, 0)

        prob_1 = (outputs[0] * outputs[1]).sum(-1)
        prob_2 = (outputs[0] * outputs[2]).sum(-1)
        temp = torch.cat((outputs[0], outputs[1]), 0)
        temp_labels = torch.cat((labels, labels), 0)
        prob_3 = torch.mm(outputs[0], temp.t())
        mask = labels[:, None] == temp_labels[None, :]
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(
            torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()
        return loss, outputs[0]

    def get_embeddings(self, input_ids):
        """Get [CLS] embeddings for a batch of input_ids."""
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(
            self.tokenizer.pad_token_id))[1]
        return outputs


# =============================================================================
# Dataset
# =============================================================================

def tokenize_code(code, tokenizer, block_size):
    """Tokenize code into padded input_ids, supporting both BPE and char tokenizers.

    For BPE (RobertaTokenizer): uses tokenize() + convert_tokens_to_ids().
    For CharTokenizer: uses __call__() which returns input_ids directly.
    """
    if hasattr(tokenizer, 'tokenize'):
        # Standard HuggingFace tokenizer (BPE)
        code_tokens = tokenizer.tokenize(code)[:block_size - 2]
        source_tokens = ([tokenizer.cls_token] + code_tokens +
                         [tokenizer.sep_token])
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    else:
        # CharTokenizer: use __call__ directly
        encoding = tokenizer(
            code, truncation=True, max_length=block_size,
            padding='max_length', return_tensors=None)
        ids = encoding['input_ids']
        # Handle nested lists from CharTokenizer
        source_ids = ids[0] if isinstance(ids[0], list) else ids
        return source_ids[:block_size]

    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids[:block_size]


class CloneDetectionDataset(Dataset):
    """POJ-104 clone detection dataset."""

    def __init__(self, file_path, tokenizer, block_size=400, mode='train'):
        self.examples = []
        self.label_examples = defaultdict(list)

        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                code = ' '.join(js['code'].split())
                source_ids = tokenize_code(code, tokenizer, block_size)

                example = {
                    'input_ids': source_ids,
                    'index': int(js['index']),
                    'label': int(js['label']),
                    'code': js['code'],
                }
                self.examples.append(example)
                self.label_examples[int(js['label'])].append(
                    len(self.examples) - 1)

        logger.info(f"Loaded {len(self.examples)} examples from "
                    f"{file_path} ({mode})")
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        label = example['label']
        index = example['index']

        # Positive: same label, different index
        pos_candidates = [idx for idx in self.label_examples[label]
                          if self.examples[idx]['index'] != index]
        if pos_candidates:
            p_idx = random.choice(pos_candidates)
        else:
            p_idx = i
        p_example = self.examples[p_idx]

        # Negative: different label
        neg_labels = [l for l in self.label_examples if l != label]
        neg_label = random.choice(neg_labels)
        n_idx = random.choice(self.label_examples[neg_label])
        n_example = self.examples[n_idx]

        return (
            torch.tensor(example['input_ids']),
            torch.tensor(p_example['input_ids']),
            torch.tensor(n_example['input_ids']),
            torch.tensor(label)
        )


# =============================================================================
# Model Loading
# =============================================================================

def load_encoder(checkpoint_path, cache_dir='models_cache', device='cpu'):
    """Load encoder from any supported checkpoint format.

    Returns: (encoder, tokenizer, checkpoint_info)
    where checkpoint_info is a dict describing the loaded model.
    """
    info = {'checkpoint': checkpoint_path}

    # Check if this is a custom training checkpoint (has char_tokenizer_config.json)
    is_char_tokenizer = os.path.isdir(checkpoint_path) and os.path.exists(
        os.path.join(checkpoint_path, 'char_tokenizer_config.json'))

    # Load tokenizer
    if is_char_tokenizer:
        from char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer.from_pretrained(checkpoint_path)
        info['tokenizer'] = 'char_tokenizer'
        logger.info(f"Using character-level tokenizer "
                    f"(vocab_size={len(tokenizer)})")
    else:
        # Default: RobertaTokenizer from CodeBERT
        tokenizer = RobertaTokenizer.from_pretrained(
            'microsoft/codebert-base', cache_dir=cache_dir)
        info['tokenizer'] = 'roberta_bpe'

    # Load encoder
    if os.path.isdir(checkpoint_path):
        # Local checkpoint directory
        has_safetensors = os.path.exists(
            os.path.join(checkpoint_path, 'model.safetensors'))
        has_pytorch_bin = os.path.exists(
            os.path.join(checkpoint_path, 'pytorch_model.bin'))
        has_config = os.path.exists(
            os.path.join(checkpoint_path, 'config.json'))

        if not has_config:
            raise ValueError(
                f"No config.json found in {checkpoint_path}. "
                "Expected a HuggingFace model directory.")

        # Our train_codebert.py saves as RobertaForMaskedLM
        # Try loading as RobertaForMaskedLM first, extract .roberta encoder
        try:
            mlm_model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
            encoder = mlm_model.roberta
            info['format'] = 'RobertaForMaskedLM (custom checkpoint)'
            logger.info(f"Loaded encoder from RobertaForMaskedLM checkpoint")
        except Exception as e1:
            # Try as plain RobertaModel
            try:
                encoder = RobertaModel.from_pretrained(checkpoint_path)
                info['format'] = 'RobertaModel'
                logger.info(f"Loaded encoder from RobertaModel checkpoint")
            except Exception as e2:
                raise ValueError(
                    f"Could not load checkpoint from {checkpoint_path}.\n"
                    f"  As RobertaForMaskedLM: {e1}\n"
                    f"  As RobertaModel: {e2}")
    else:
        # HuggingFace model name/path
        try:
            encoder = RobertaModel.from_pretrained(
                checkpoint_path, cache_dir=cache_dir)
            info['format'] = 'RobertaModel (HuggingFace)'
        except Exception:
            # Try as RobertaForMaskedLM (e.g. ContraBERT)
            try:
                mlm_model = RobertaForMaskedLM.from_pretrained(
                    checkpoint_path, cache_dir=cache_dir)
                encoder = mlm_model.roberta
                info['format'] = 'RobertaForMaskedLM (HuggingFace)'
            except Exception as e:
                raise ValueError(
                    f"Could not load model '{checkpoint_path}': {e}")

    # Verify encoder has pooler for clone detection
    if not hasattr(encoder, 'pooler') or encoder.pooler is None:
        logger.warning("Encoder has no pooler layer. Adding a randomly "
                       "initialized pooler (will be trained during fine-tuning).")
        from transformers.models.roberta.modeling_roberta import RobertaPooler
        encoder.pooler = RobertaPooler(encoder.config)

    info['num_params'] = sum(p.numel() for p in encoder.parameters())
    info['hidden_size'] = encoder.config.hidden_size
    info['vocab_size'] = encoder.config.vocab_size

    logger.info(f"Encoder: {info['format']}, "
                f"{info['num_params']:,} params, "
                f"hidden_size={info['hidden_size']}, "
                f"vocab_size={info['vocab_size']}")

    return encoder, tokenizer, info


# =============================================================================
# Training (Fine-tuning on POJ-104)
# =============================================================================

def train_model(model, train_dataset, eval_dataset, args):
    """Fine-tune the clone detection model on POJ-104."""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=4, pin_memory=True
    )

    total_steps = args.num_epochs * len(train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    logger.info(f"***** Running fine-tuning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    best_map = 0.0
    model.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(
                tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            inputs = batch[0].to(args.device)
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            labels = batch[3].to(args.device)

            loss, _ = model(inputs, p_inputs, n_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_steps += 1

            if (step + 1) % 100 == 0:
                logger.info(
                    f"  Epoch {epoch+1}, Step {step+1}/"
                    f"{len(train_dataloader)}, "
                    f"Avg Loss: {epoch_loss/num_steps:.5f}")

        logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss/num_steps:.5f}")

        # Evaluate after each epoch
        eval_result = evaluate_map(model, eval_dataset, args)
        logger.info(f"Epoch {epoch+1} eval MAP@R: "
                    f"{eval_result['MAP@R']:.4f}")

        if eval_result['MAP@R'] > best_map:
            best_map = eval_result['MAP@R']
            os.makedirs(args.output_dir, exist_ok=True)
            best_path = os.path.join(
                args.output_dir, 'checkpoint-best-map', 'model.bin')
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            model_to_save = (model.module if hasattr(model, 'module')
                             else model)
            torch.save(model_to_save.state_dict(), best_path)
            logger.info(f"  New best MAP@R: {best_map:.4f}, "
                        f"saved to {best_path}")

    return best_map


# =============================================================================
# Evaluation
# =============================================================================

def get_embeddings(model, dataset, args):
    """Get [CLS] embeddings for all examples in the dataset."""
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler,
        batch_size=args.eval_batch_size, num_workers=4, pin_memory=True
    )

    model.eval()
    all_vecs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Computing embeddings"):
            inputs = batch[0].to(args.device)
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            labels = batch[3].to(args.device)

            _, vec = model(inputs, p_inputs, n_inputs, labels)
            all_vecs.append(vec.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    vecs = np.concatenate(all_vecs, 0)
    labels = np.concatenate(all_labels, 0)
    return vecs, labels


def compute_map_at_r(vecs, labels):
    """Compute MAP@R metric for clone detection."""
    scores = np.matmul(vecs, vecs.T)

    dic = {}
    for i in range(len(labels)):
        scores[i, i] = -1e9
        label = int(labels[i])
        if label not in dic:
            dic[label] = -1
        dic[label] += 1

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort')[:, ::-1]

    MAP = []
    for i in range(len(labels)):
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        if dic[label] > 0:
            MAP.append(sum(Avep) / dic[label])

    return float(np.mean(MAP))


def evaluate_map(model, dataset, args):
    """Evaluate MAP@R on a dataset."""
    vecs, labels = get_embeddings(model, dataset, args)
    map_score = compute_map_at_r(vecs, labels)
    return {'MAP@R': map_score}


def evaluate_robustness(model, tokenizer, test_file, args,
                        rename_counts=(0, 1, 4, 8)):
    """Evaluate variable renaming robustness on fine-tuned model.

    Following ContraBERT paper methodology:
      1. Get embeddings for original test samples
      2. Compute MAP@R at N=0
      3. Identify correctly-predicted samples (top-1 neighbor is true clone)
      4. For each N, rename N variables and measure accuracy retention
    """
    logger.info("=" * 60)
    logger.info("Variable Renaming Robustness Evaluation")
    logger.info("=" * 60)

    # Load test data
    test_data = []
    with open(test_file) as f:
        for line in f:
            js = json.loads(line.strip())
            test_data.append(js)

    # Optionally load shared, pre-generated renamed test sets so every model is
    # tested on identical var_xxx renamings. Keyed by position (file order),
    # which aligns with test_data / the embedding order below.
    renamed_sets = {}  # n -> list of (renamed_code, num_renamed), file-ordered
    renamed_dir = getattr(args, 'renamed_dir', None)
    if renamed_dir:
        for n_renames in rename_counts:
            if n_renames == 0:
                continue
            rf = os.path.join(renamed_dir, f'test_renamed_N{n_renames}.jsonl')
            if not os.path.exists(rf):
                logger.warning(f"Shared renamed set not found: {rf}; "
                               f"renaming on the fly for N={n_renames}")
                continue
            rlist = []
            with open(rf) as f:
                for line in f:
                    rj = json.loads(line.strip())
                    rlist.append((rj['code'], rj.get('num_renamed', n_renames)))
            if len(rlist) != len(test_data):
                logger.warning(f"Renamed set {rf} length {len(rlist)} != "
                               f"test_data {len(test_data)}; renaming on the "
                               f"fly for N={n_renames}")
                continue
            renamed_sets[n_renames] = rlist
            logger.info(f"Loaded shared renamed test set N={n_renames} "
                        f"({len(rlist)} examples) from {rf}")

    # Get original embeddings
    test_dataset = CloneDetectionDataset(
        test_file, tokenizer, args.block_size, mode='test')
    vecs, labels = get_embeddings(model, test_dataset, args)

    # Compute similarity matrix
    scores = np.matmul(vecs, vecs.T)
    for i in range(len(labels)):
        scores[i, i] = -1e9

    # Identify correctly predicted samples at N=0
    sort_ids = np.argsort(scores, axis=-1)[:, ::-1]
    correct_at_0 = []
    for i in range(len(labels)):
        top1_idx = sort_ids[i, 0]
        if int(labels[top1_idx]) == int(labels[i]):
            correct_at_0.append(i)

    logger.info(f"Correctly predicted at N=0: "
                f"{len(correct_at_0)} / {len(labels)}")

    # MAP@R at N=0
    map_0 = compute_map_at_r(vecs, labels)

    results = {
        'num_test_samples': len(labels),
        'num_correct_at_0': len(correct_at_0),
        'accuracy_at_0': len(correct_at_0) / len(labels),
        'MAP@R_N0': map_0,
    }
    logger.info(f"MAP@R at N=0: {map_0:.4f}")

    for n_renames in rename_counts:
        if n_renames == 0:
            results['accuracy_N0'] = 1.0
            results['num_correct_N0'] = len(correct_at_0)
            logger.info(f"N=0: Accuracy = 1.0000 "
                        f"({len(correct_at_0)}/{len(correct_at_0)})")
            continue

        logger.info(f"\nEvaluating with N={n_renames} variable renames...")

        renamed_ids_list = []
        num_actually_renamed = []

        for idx in tqdm(correct_at_0, desc=f"Renaming N={n_renames}"):
            if n_renames in renamed_sets:
                # Use the shared, pre-generated renaming for this example.
                renamed_code, actual_renamed = renamed_sets[n_renames][idx]
            else:
                code = test_data[idx]['code']
                renamed_code, actual_renamed = rename_variables(
                    code, n_renames)
            num_actually_renamed.append(actual_renamed)

            code_clean = ' '.join(renamed_code.split())
            source_ids = tokenize_code(
                code_clean, tokenizer, args.block_size)
            renamed_ids_list.append(source_ids)

        # Get embeddings for renamed samples
        renamed_vecs = []
        model.eval()
        with torch.no_grad():
            batch_size = args.eval_batch_size
            for i in range(0, len(renamed_ids_list), batch_size):
                batch_ids = torch.tensor(
                    renamed_ids_list[i:i+batch_size]).to(args.device)
                raw_model = (model.module if hasattr(model, 'module')
                             else model)
                batch_vecs = raw_model.get_embeddings(batch_ids)
                renamed_vecs.append(batch_vecs.cpu().numpy())

        renamed_vecs = np.concatenate(renamed_vecs, 0)

        # Check if nearest neighbor is still a true clone
        num_still_correct = 0
        for j, idx in enumerate(correct_at_0):
            renamed_vec = renamed_vecs[j]
            sim_scores = np.dot(vecs, renamed_vec)
            sim_scores[idx] = -1e9

            top1_idx = np.argmax(sim_scores)
            if int(labels[top1_idx]) == int(labels[idx]):
                num_still_correct += 1

        accuracy = (num_still_correct / len(correct_at_0)
                    if len(correct_at_0) > 0 else 0.0)
        avg_renamed = (np.mean(num_actually_renamed)
                       if num_actually_renamed else 0)

        results[f'accuracy_N{n_renames}'] = accuracy
        results[f'num_correct_N{n_renames}'] = num_still_correct
        results[f'avg_actually_renamed_N{n_renames}'] = float(avg_renamed)

        logger.info(f"N={n_renames}: Accuracy = {accuracy:.4f} "
                    f"({num_still_correct}/{len(correct_at_0)}), "
                    f"Avg vars actually renamed: {avg_renamed:.1f}")

    return results


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_results(results_dir):
    """Load all result files from a directory and print a comparison table."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json'):
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            name = fname.replace('.json', '')
            results[name] = data

    if not results:
        logger.info(f"No result files found in {results_dir}")
        return

    # Print comparison table
    print("\n" + "=" * 90)
    print("VARIABLE RENAMING ROBUSTNESS COMPARISON")
    print("=" * 90)
    print(f"{'Model':<35} {'MAP@R':>8} {'Acc N0':>8} {'Acc N1':>8} "
          f"{'Acc N4':>8} {'Acc N8':>8}")
    print("-" * 90)

    for name, data in sorted(results.items()):
        map_r = data.get('MAP@R_N0', 0)
        acc_0 = data.get('accuracy_N0', 0)
        acc_1 = data.get('accuracy_N1', 0)
        acc_4 = data.get('accuracy_N4', 0)
        acc_8 = data.get('accuracy_N8', 0)
        print(f"{name:<35} {map_r:>8.4f} {acc_0:>8.4f} {acc_1:>8.4f} "
              f"{acc_4:>8.4f} {acc_8:>8.4f}")

    print("=" * 90)

    # Print robustness drop (N8 - N0 accuracy loss)
    print(f"\n{'Model':<35} {'N8 Drop':>10} {'N4 Drop':>10} "
          f"{'N1 Drop':>10}")
    print("-" * 75)
    for name, data in sorted(results.items()):
        acc_0 = data.get('accuracy_N0', 1.0)
        acc_1 = data.get('accuracy_N1', 0)
        acc_4 = data.get('accuracy_N4', 0)
        acc_8 = data.get('accuracy_N8', 0)
        drop_1 = acc_0 - acc_1
        drop_4 = acc_0 - acc_4
        drop_8 = acc_0 - acc_8
        print(f"{name:<35} {drop_8:>10.4f} {drop_4:>10.4f} "
              f"{drop_1:>10.4f}")
    print("=" * 75)

    # Save comparison
    comparison_path = os.path.join(results_dir, 'comparison_table.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")


# =============================================================================
# Main
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='Unified Variable Renaming Robustness Evaluation')

    # Checkpoint
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Pre-trained model name or checkpoint directory to evaluate. '
             'Supports: HuggingFace model names (microsoft/codebert-base, '
             'claudios/ContraBERT_C), or local checkpoint dirs from '
             'train_codebert.py')

    # Data
    parser.add_argument(
        '--data_dir', type=str, default='data_cache/poj104',
        help='Directory containing POJ-104 train.jsonl, valid.jsonl, '
             'test.jsonl')
    parser.add_argument(
        '--cache_dir', type=str, default='models_cache',
        help='Directory for caching HuggingFace models')
    parser.add_argument(
        '--renamed_dir', type=str, default=None,
        help='Directory with pre-generated shared renamed test sets '
             '(test_renamed_N{k}.jsonl). Used so every model is tested on the '
             'exact same var_xxx renamings. Defaults to --data_dir. If a file '
             'is missing, falls back to on-the-fly renaming for that N.')

    # Fine-tuning hyperparameters (matching ContraBERT paper)
    parser.add_argument('--block_size', type=int, default=400,
                        help='Max sequence length (default: 400)')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Training batch size (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Fine-tuning epochs (default: 2)')
    parser.add_argument('--seed', type=int, default=123456,
                        help='Random seed (default: 123456)')

    # Evaluation options
    parser.add_argument('--rename_counts', type=str, default='0,1,4,8',
                        help='Comma-separated rename counts to test')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip fine-tuning (load existing fine-tuned '
                             'checkpoint from output_dir)')
    parser.add_argument('--skip_robustness', action='store_true',
                        help='Only do fine-tuning + MAP@R, skip '
                             'robustness evaluation')

    # Output
    parser.add_argument(
        '--output_dir', type=str, default='eval_results',
        help='Output directory for fine-tuned model and results')
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Name for this experiment (used in output filenames). '
             'Auto-detected from checkpoint path if not set.')

    # Comparison mode
    parser.add_argument(
        '--compare_dir', type=str, default=None,
        help='If set, load all .json result files from this directory '
             'and print a comparison table instead of running evaluation')

    # Misc
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    # Comparison mode
    if args.compare_dir:
        compare_results(args.compare_dir)
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required (or use --compare_dir)")

    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')
        args.n_gpu = 0
    else:
        args.device = torch.device('cuda')
        args.n_gpu = torch.cuda.device_count()

    logger.info(f"Device: {args.device}, N_GPU: {args.n_gpu}")

    set_seed(args.seed)

    args.rename_counts = [int(x) for x in args.rename_counts.split(',')]

    # Default the shared-renamed-set directory to the data directory.
    if args.renamed_dir is None:
        args.renamed_dir = args.data_dir

    # Auto-detect experiment name
    if args.experiment_name is None:
        if os.path.isdir(args.checkpoint):
            # Use parent directory name as experiment name
            parts = os.path.normpath(args.checkpoint).split(os.sep)
            # Find the experiment-level directory
            for p in reversed(parts):
                if p.startswith('exp') or p.startswith('baseline') or p.startswith('checkpoint'):
                    continue
                if p in ('checkpoints', 'models_cache', '.'):
                    continue
                args.experiment_name = p
                break
            if args.experiment_name is None:
                args.experiment_name = parts[-1]
        else:
            args.experiment_name = args.checkpoint.replace('/', '_')

    # Create experiment-specific output directory
    exp_output_dir = os.path.join(args.output_dir, args.experiment_name)
    args.output_dir = exp_output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # File paths
    train_file = os.path.join(args.data_dir, 'train.jsonl')
    eval_file = os.path.join(args.data_dir, 'valid.jsonl')
    test_file = os.path.join(args.data_dir, 'test.jsonl')

    for f_path in [train_file, eval_file, test_file]:
        if not os.path.exists(f_path):
            logger.error(f"Data file not found: {f_path}")
            logger.error("Please ensure POJ-104 data is in the data_dir.")
            sys.exit(1)

    # Load model
    logger.info("=" * 60)
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    encoder, tokenizer, checkpoint_info = load_encoder(
        args.checkpoint, cache_dir=args.cache_dir, device=str(args.device))

    config = encoder.config
    config.num_labels = 1
    model = CloneModel(encoder, config, tokenizer, args)
    model.to(args.device)

    # Single-GPU only: each evaluation process is pinned to exactly one GPU
    # (via CUDA_VISIBLE_DEVICES by the orchestrator), so we never wrap the model
    # in DataParallel and never split a single process across GPUs.

    # Fine-tune on POJ-104
    best_model_path = os.path.join(
        args.output_dir, 'checkpoint-best-map', 'model.bin')

    if args.skip_finetune and os.path.exists(best_model_path):
        logger.info(f"Loading existing fine-tuned checkpoint: "
                    f"{best_model_path}")
        model_to_load = (model.module if hasattr(model, 'module')
                         else model)
        model_to_load.load_state_dict(
            torch.load(best_model_path, map_location=args.device),
            strict=False)
    else:
        logger.info("=" * 60)
        logger.info("Fine-tuning on POJ-104 Clone Detection")
        logger.info("=" * 60)
        train_dataset = CloneDetectionDataset(
            train_file, tokenizer, args.block_size, mode='train')
        eval_dataset = CloneDetectionDataset(
            eval_file, tokenizer, args.block_size, mode='eval')
        best_map = train_model(model, train_dataset, eval_dataset, args)
        logger.info(f"Best validation MAP@R: {best_map:.4f}")

        # Load best checkpoint
        if os.path.exists(best_model_path):
            model_to_load = (model.module if hasattr(model, 'module')
                             else model)
            model_to_load.load_state_dict(
                torch.load(best_model_path, map_location=args.device),
                strict=False)
            logger.info(f"Loaded best fine-tuned checkpoint")

    # Evaluate MAP@R on test set
    logger.info("=" * 60)
    logger.info("Evaluating MAP@R on Test Set")
    logger.info("=" * 60)
    test_dataset = CloneDetectionDataset(
        test_file, tokenizer, args.block_size, mode='test')
    eval_result = evaluate_map(model, test_dataset, args)
    logger.info(f"Test MAP@R: {eval_result['MAP@R']:.4f}")

    # Robustness evaluation
    if not args.skip_robustness:
        raw_model = (model.module if hasattr(model, 'module') else model)
        robustness_results = evaluate_robustness(
            raw_model, tokenizer, test_file, args,
            rename_counts=args.rename_counts
        )

        # Add checkpoint info to results
        robustness_results['checkpoint_info'] = checkpoint_info
        robustness_results['experiment_name'] = args.experiment_name

        # Save results
        results_file = os.path.join(
            args.output_dir, f'{args.experiment_name}.json')
        with open(results_file, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Also save a copy to the parent eval_results dir for comparison
        parent_dir = os.path.dirname(args.output_dir)
        copy_path = os.path.join(parent_dir, f'{args.experiment_name}.json')
        with open(copy_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ROBUSTNESS RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Tokenizer: {checkpoint_info.get('tokenizer', 'N/A')}")
        logger.info(f"Num test samples: "
                    f"{robustness_results['num_test_samples']}")
        logger.info(f"Num correct at N=0: "
                    f"{robustness_results['num_correct_at_0']}")
        logger.info(f"MAP@R at N=0: "
                    f"{robustness_results.get('MAP@R_N0', 'N/A')}")
        for n in args.rename_counts:
            acc = robustness_results.get(f'accuracy_N{n}', 'N/A')
            if isinstance(acc, float):
                logger.info(f"  N={n}: Accuracy = {acc:.4f}")
            else:
                logger.info(f"  N={n}: Accuracy = {acc}")
    else:
        logger.info("Robustness evaluation skipped (--skip_robustness)")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
