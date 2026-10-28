#!/usr/bin/env python3
"""Rename N variables in a snippet to VAR_1 .. VAR_N.

Deterministic given (code, n, seed): the candidate list from extract.py is in
source order, and selection uses a private random.Random rather than the global
RNG, so a worker pool or a different PYTHONHASHSEED cannot perturb the result.

Substitution is longest-name-first so that renaming `i` cannot corrupt `idx`,
and every replacement is word-boundary anchored so it never fires inside a
longer identifier. Names already of the form VAR_<digits> in the source are
excluded from being *introduced* twice.
"""

import random
import re

from perturb.extract import (extract_int_literals, extract_variables,
                             identifier_spans)

PREFIX = 'VAR_'


def rename_variables(code, num_renames, language, seed):
    """Return (renamed_code, num_actually_renamed, mapping).

    Fewer than `num_renames` variables are renamed when the snippet does not
    have that many. The caller records the actual count, so a table can report
    the mean edits actually applied rather than the nominal budget.

    `mapping` ({old_name: VAR_k}) is returned because some datasets carry a
    pre-tokenized copy of the code alongside the source -- CodeSearchNet's
    `code_tokens` -- and that copy has to be renamed with the *same* mapping.
    Renaming only the source text while a stale `code_tokens` is what actually
    gets embedded is a silent no-op, and it is exactly the bug this signature
    exists to prevent.
    """
    if num_renames <= 0:
        return code, 0, {}

    candidates = extract_variables(code, language)
    if not candidates:
        return code, 0, {}

    rng = random.Random(seed)
    k = min(num_renames, len(candidates))
    # rng.sample over a source-ordered list -> reproducible selection.
    selected = rng.sample(candidates, k)

    # Rename in the order the variables appear in the source, so VAR_1 is the
    # first renamed variable a reader meets. Purely cosmetic, but it makes a
    # diff of the perturbed sets legible.
    selected.sort(key=candidates.index)

    taken = set(candidates)
    mapping = {}
    next_id = 1
    for old in selected:
        while f'{PREFIX}{next_id}' in taken:
            next_id += 1
        new = f'{PREFIX}{next_id}'
        taken.add(new)
        mapping[old] = new
        next_id += 1

    return substitute_identifiers(code, mapping), k, mapping


def rename_tokens(tokens, mapping):
    """Apply a rename mapping to a pre-tokenized copy of the same code.

    Exact whole-token equality, deliberately: an identifier is its own token in
    CodeSearchNet's `code_tokens`, whereas a string literal arrives as the token
    `"'count'"` (quotes included). Matching on equality renames the variable and
    leaves the string alone, which is what a word-boundary regex over the raw
    source also does.
    """
    return [mapping.get(token, token) for token in tokens]


# ---------------------------------------------------------------------------
# permute -- a derangement of the snippet's OWN variable names
# ---------------------------------------------------------------------------
#
# Why this view exists. Renaming to VAR_1..VAR_N confounds two things: the model
# loses the *binding* between a name and its role, AND it is handed identifier
# strings it has never seen. `permute` isolates the first. The k selected
# variables are shuffled among themselves with a derangement (no fixed points),
# so afterwards the snippet contains exactly the same set of identifier names it
# started with -- zero novel strings, no change to the identifier vocabulary --
# and only the name-to-role assignment has moved.
#
# The selection uses the SAME rng.sample draw as rename_variables for the same
# (code, n, seed), so `permute` and the VAR_k rename touch the same variables in
# the same snippet. The contrast is therefore controlled: only the substituted
# strings differ.
#
# k < 2 is a no-op: a set of one element has no derangement.


def _derangement(k, rng):
    """A random permutation of range(k) with no fixed point. k >= 2."""
    while True:
        order = list(range(k))
        rng.shuffle(order)
        if all(order[i] != i for i in range(k)):
            return order


def _substitute_simultaneously(code, mapping):
    """Apply {old: new} in ONE pass.

    Sequential re.sub would cascade: a->b followed by b->c turns the original
    a into c. One alternation with a callback rewrites every occurrence against
    the original text exactly once. Longest-first alternation so `i` can never
    shadow `idx`.
    """
    return substitute_identifiers(code, mapping)


def permute_variables(code, num_renames, language, seed):
    """Return (permuted_code, num_actually_permuted, mapping).

    `num_actually_permuted` is the size of the deranged set, 0 when the snippet
    has fewer than two renameable variables.
    """
    if num_renames <= 0:
        return code, 0, {}

    candidates = extract_variables(code, language)
    if len(candidates) < 2:
        return code, 0, {}

    rng = random.Random(seed)
    k = min(num_renames, len(candidates))
    selected = rng.sample(candidates, k)     # same draw as rename_variables
    selected.sort(key=candidates.index)
    if k < 2:
        return code, 0, {}

    order = _derangement(k, rng)
    mapping = {selected[i]: selected[order[i]] for i in range(k)}

    return _substitute_simultaneously(code, mapping), k, mapping


# ---------------------------------------------------------------------------
# constants -- integer literals changed, identifiers untouched
# ---------------------------------------------------------------------------
#
# The control view. Every touched integer literal becomes a DIFFERENT integer
# with the SAME number of digits, so the token count and the lexical shape of
# the snippet are preserved and not one identifier byte moves. If renaming
# damage were really "any surface edit costs the model something", this would
# cost about as much as a rename of the same budget. It does not, which is what
# makes the renaming result name-specific.
#
# Substitution is positional (byte offsets from the AST), not regex: `\b100\b`
# would also fire inside a string literal, and a string is not a constant.


def _different_same_width_int(text, rng):
    """A different integer with the same digit count (and no new leading zero)."""
    width = len(text)
    if width == 1:
        choices = [str(d) for d in range(10) if str(d) != text]
        return rng.choice(choices)
    if text[0] == '0':                       # e.g. octal-looking 0755: keep shape
        lo, hi = 0, 10 ** width - 1
        for _ in range(32):
            cand = str(rng.randint(lo, hi)).zfill(width)
            if cand != text:
                return cand
        return text
    lo, hi = 10 ** (width - 1), 10 ** width - 1
    for _ in range(32):
        cand = str(rng.randint(lo, hi))
        if cand != text:
            return cand
    return text


def perturb_constants(code, num_edits, language, seed):
    """Return (edited_code, num_distinct_values_changed, mapping).

    The budget counts DISTINCT literal values, matching the way a rename budget
    counts distinct variable names: `n=4` changes four different constants
    wherever they occur, exactly as `n=4` renames four different variables
    wherever they occur.
    """
    if num_edits <= 0:
        return code, 0, {}

    literals = extract_int_literals(code, language)
    if not literals:
        return code, 0, {}

    # distinct values in first-appearance order -> deterministic candidate list
    seen = set()
    values = []
    for _s, _e, text in literals:
        if text not in seen:
            seen.add(text)
            values.append(text)

    rng = random.Random(seed)
    k = min(num_edits, len(values))
    selected = rng.sample(values, k)
    selected.sort(key=values.index)

    taken = set(values)
    mapping = {}
    for old in selected:
        for _ in range(64):
            new = _different_same_width_int(old, rng)
            if new not in taken:
                break
        taken.add(new)
        mapping[old] = new

    raw = bytearray(code.encode('utf8'))
    for start, end, text in sorted(literals, key=lambda t: t[0], reverse=True):
        if text in mapping:
            raw[start:end] = mapping[text].encode('utf8')

    return raw.decode('utf8', 'replace'), k, mapping


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

SUBSTITUTION_FALLBACKS = {'n': 0}       # how often the AST path was unavailable


def substitute_identifiers(code, mapping, language='cpp'):
    """Apply {old_name: new_name} at identifier TOKEN positions, in one pass.

    Two properties, both of which a word-boundary `re.sub` fails to give:

    * It never fires inside a string, a character literal, a comment, or a
      preprocessor payload. `\\bs\\b` matches the `s` of "%s" and `\\bn\\b`
      matches the `n` of "\\n", so on C code the regex version quietly rewrites
      printf format strings while claiming to have renamed a variable.
    * It is simultaneous. Sequential substitution cascades -- a->b then b->c
      turns the original `a` into `c` -- which makes a derangement (`permute`)
      wrong in a way that is invisible unless you read the output.

    Falls back to the word-boundary regex only when the parser cannot produce
    spans, and counts that in SUBSTITUTION_FALLBACKS.
    """
    if not mapping:
        return code

    spans = identifier_spans(code, language)
    if spans is None:
        SUBSTITUTION_FALLBACKS['n'] += 1
        out = code
        for old in sorted(mapping, key=len, reverse=True):
            out = re.sub(rf'\b{re.escape(old)}\b', mapping[old], out)
        return out

    raw = bytearray(code.encode('utf8'))
    for start, end, text in reversed(spans):
        new = mapping.get(text)
        if new is not None:
            raw[start:end] = new.encode('utf8')
    return raw.decode('utf8', 'replace')


def rename_variables_wordboundary(code, num_renames, language, seed):
    """The substitution rule as shipped: word-boundary `re.sub` over raw source.

    Kept so the cost of the string-corruption bug can be MEASURED rather than
    assumed -- score a model on both and the difference is the bug's size.
    """
    if num_renames <= 0:
        return code, 0, {}
    candidates = extract_variables(code, language)
    if not candidates:
        return code, 0, {}
    rng = random.Random(seed)
    k = min(num_renames, len(candidates))
    selected = rng.sample(candidates, k)
    selected.sort(key=candidates.index)
    taken = set(candidates)
    mapping = {}
    next_id = 1
    for old in selected:
        while f'{PREFIX}{next_id}' in taken:
            next_id += 1
        mapping[old] = f'{PREFIX}{next_id}'
        taken.add(mapping[old])
        next_id += 1
    renamed = code
    for old in sorted(mapping, key=len, reverse=True):
        renamed = re.sub(rf'\b{re.escape(old)}\b', mapping[old], renamed)
    return renamed, k, mapping
