#!/usr/bin/env python3
"""Corpus construction for the scale25 experiment.

Two arms over the SAME documents in the SAME order:

    plain    a deterministic 25% subset of codeparrot/codeparrot-clean, as written
    rename   the same documents, every safely-renameable local identifier replaced by
             a fresh random name that is never reused anywhere else in the corpus

Everything except the identifier bytes is held byte-identical between the arms:
substitution happens at tree-sitter `identifier` TOKEN spans, so whitespace, comments,
string literals, attributes, imports, decorators and keyword-argument names are untouched.

Design notes
------------
* Name selection reuses the project's own extractor,
  `modernbert_scale/data/perturb/extract.py` (`_extract_python` skip rules + `_acceptable`),
  which is what `repro/repair.py --mode highdiv` uses. We reuse the SELECTION and replace the
  word-boundary `re.sub` SUBSTITUTION with a span substitution, because a `\\bs\\b` regex over
  a whole source file rewrites the `s` inside `"%s"` and inside comments -- that would make the
  two arms differ by more than identifiers, which is the one thing this experiment may not do.
  (`perturb/extract.py` makes exactly this argument in its own docstring for `identifier_spans`.)

* Corpus-wide name freshness is guaranteed structurally, not by a global `set`: for each name
  length L the space [0, 26**L) is walked through a bijection (a cycle-walking Feistel network),
  indexed by a globally unique counter, so two different global indices can never produce the
  same string.

* Documents whose tree-sitter parse contains an ERROR node are left untouched in BOTH arms and
  counted, so an unparseable file never differs between the arms.
"""
import hashlib
import os
import sys

_MBS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), 'modernbert_scale', 'data')
if _MBS not in sys.path:
    sys.path.insert(0, _MBS)

from perturb.extract import _get_parser, _acceptable, extract_variables  # noqa: E402

# ---------------------------------------------------------------------------
# Deterministic 25% subset
# ---------------------------------------------------------------------------
SUBSET_KEY = b'scale25-v1'          # recorded in the manifest; changing it changes the subset
SUBSET_MOD = 4                      # keep 1 document in 4  -> 25%
VAL_MOD = 512                       # of the kept documents, 1 in 512 is held out for validation


def doc_hash(content):
    """Stable 64-bit hash of the document text. Independent of PYTHONHASHSEED, of the
    file the document came from, and of its position in the stream."""
    h = hashlib.blake2b(content.encode('utf-8', 'surrogatepass'),
                        digest_size=8, key=SUBSET_KEY)
    return int.from_bytes(h.digest(), 'big')


def subset_role(content):
    """'skip' | 'train' | 'val' for one document. Pure function of the text."""
    h = doc_hash(content)
    if h % SUBSET_MOD != 0:
        return 'skip'
    return 'val' if (h // SUBSET_MOD) % VAL_MOD == 0 else 'train'


# ---------------------------------------------------------------------------
# Fresh-name generator: a bijection per name length, so names never repeat
# ---------------------------------------------------------------------------
_ALPHA = 'abcdefghijklmnopqrstuvwxyz'

# Lengths used for the fresh names, and their sampling weights. Lengths 1-4 are excluded
# because 26**4 = 456_976 is far smaller than the number of names this corpus needs, so
# including them would force reuse -- the opposite of the treatment.
NAME_LENGTHS = (5, 6, 7, 8)
NAME_WEIGHTS = (0.10, 0.30, 0.35, 0.25)


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


class _Feistel:
    """Bijection on [0, n) by a 4-round balanced Feistel network plus cycle walking."""

    def __init__(self, n, key):
        self.n = n
        bits = max(2, (n - 1).bit_length())
        self.half = (bits + 1) // 2
        self.mask = (1 << self.half) - 1
        self.dom = 1 << (2 * self.half)
        self.key = key

    def _f(self, r, rnd):
        return _splitmix64(((r << 8) | rnd) ^ self.key) & self.mask

    def __call__(self, x):
        while True:
            left, right = x >> self.half, x & self.mask
            for rnd in range(4):
                left, right = right, left ^ self._f(right, rnd)
            x = (left << self.half) | right
            if x < self.n:
                return x


class FreshNames:
    """Globally-unique identifier names.

    `shard` and `n_shards` partition the index space so that independent worker processes
    never collide: worker `shard` uses indices shard, shard+n_shards, shard+2*n_shards, ...
    """

    def __init__(self, shard=0, n_shards=1, key=0xC0FFEE):
        self.shard, self.n_shards = shard, n_shards
        self.counter = {L: 0 for L in NAME_LENGTHS}
        self.perm = {L: _Feistel(26 ** L, key ^ (L * 0x5DEECE66D)) for L in NAME_LENGTHS}
        self.exhausted = 0
        cum, acc = [], 0.0
        for w in NAME_WEIGHTS:
            acc += w
            cum.append(acc)
        self.cum = [c / acc for c in cum]

    def _int_to_str(self, x, L):
        out = []
        for _ in range(L):
            out.append(_ALPHA[x % 26])
            x //= 26
        return ''.join(reversed(out))

    def next(self, u):
        """`u` is a float in [0,1) that chooses the length (any deterministic source)."""
        for i, c in enumerate(self.cum):
            if u < c:
                L = NAME_LENGTHS[i]
                break
        else:
            L = NAME_LENGTHS[-1]
        for _ in range(len(NAME_LENGTHS)):
            idx = self.counter[L] * self.n_shards + self.shard
            if idx < 26 ** L:
                self.counter[L] += 1
                return self._int_to_str(self.perm[L](idx), L)
            self.exhausted += 1
            L = L + 1 if L + 1 in self.perm else NAME_LENGTHS[0]
        raise RuntimeError('every name-length space is exhausted')


# ---------------------------------------------------------------------------
# Renaming
# ---------------------------------------------------------------------------
# tree-sitter parents at which an `identifier` token must NOT be substituted even when its
# text is a renameable local elsewhere in the file: attribute access (`x.foo`), keyword
# argument names (`f(foo=1)`), and anything inside an import.
_NO_SUB_PARENTS = ('import_statement', 'import_from_statement', 'aliased_import',
                   'dotted_name', 'decorator')


def _scan(code_bytes, kw_out=None):
    """One tree-sitter pass -> (has_error, substitutable identifier spans, import-bound names).

    Returns (True, None, None) when the parse has an ERROR/MISSING node: such a document is
    left untouched in both arms.

    Import-bound names are collected so they can be removed from the rename map. Without this
    `import os` stays but every `os.path` use becomes `<newname>.path`: the arms would then
    differ by a broken binding, not only by identifier diversity.

    `kw_out`, when a set is passed, also collects the names used as KEYWORD-ARGUMENT names in
    this file (`f(width=1)`). Those spans are never substituted, so a parameter whose name is
    used that way anywhere in the file cannot be renamed either -- the call would stop matching
    the signature. Only the halfvar path passes it; the default behaviour is unchanged.
    """
    tree = _get_parser('python').parse(code_bytes)
    root = tree.root_node
    if root.has_error:
        return True, None, None
    spans, imported = [], set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == 'identifier':
            parent = node.parent
            ptype = parent.type if parent else ''
            bad = (ptype in _NO_SUB_PARENTS
                   or (ptype == 'attribute' and parent.child_by_field_name('attribute') == node)
                   or (ptype == 'keyword_argument'
                       and parent.child_by_field_name('name') == node))
            if bad:
                if ptype in ('import_statement', 'import_from_statement', 'aliased_import',
                             'dotted_name'):
                    imported.add(node.text.decode('utf8', 'replace'))
                elif kw_out is not None and ptype == 'keyword_argument':
                    kw_out.add(node.text.decode('utf8', 'replace'))
            else:
                spans.append((node.start_byte, node.end_byte,
                              node.text.decode('utf8', 'replace')))
        stack.extend(node.children)
    spans.sort(key=lambda t: t[0])
    return False, spans, imported


def rename_document(code, fresh, u_stream):
    """-> (new_code, status, n_names, n_sites)

    status is one of 'renamed', 'parse_error', 'no_names', 'unchanged'.
    `u_stream` is a callable returning a float in [0,1) (the name-length chooser).
    """
    code_bytes = code.encode('utf-8', 'surrogatepass')
    try:
        bad, spans, imported = _scan(code_bytes)
    except Exception:
        return code, 'parse_error', 0, 0
    if bad:
        return code, 'parse_error', 0, 0
    try:
        names = extract_variables(code, 'python')
    except Exception:
        return code, 'parse_error', 0, 0
    names = [n for n in names if _acceptable(n, 'python') and n not in imported]
    if not names:
        return code, 'no_names', 0, 0
    mapping = {}
    for n in names:
        mapping[n] = fresh.next(u_stream())
    out, pos, sites = [], 0, 0
    for a, b, text in spans:
        new = mapping.get(text)
        if new is None:
            continue
        out.append(code_bytes[pos:a])
        out.append(new.encode('ascii'))
        pos = b
        sites += 1
    if not sites:
        return code, 'no_names', len(mapping), 0
    out.append(code_bytes[pos:])
    new_code = b''.join(out).decode('utf-8', 'surrogatepass')
    return new_code, ('renamed' if new_code != code else 'unchanged'), len(mapping), sites


# ===========================================================================
# halfvar: rename a FRACTION of the LOCAL variables of EVERY function
# (added 2026-09-20 by [AGENT PKG] for work/scale25_pkg; additive -- nothing above changed)
#
# This is the treatment the user selected (labnotes "[AGENT DEC] FINAL decoder statement:
# halfvar at 50%"). `repro/repair.py --mode halfvar` defines the CHOICE:
#
#     keep = [n for n in names if rng.random() < mix_frac];  names = keep or names[:1]
#
# i.e. every renameable local of a function is selected independently with probability
# `mix_frac`, and a function that draws nothing renames its first candidate, so no function
# is left entirely untreated. repair.py operates on CodeSearchNet documents that hold exactly
# ONE function and substitutes with a word-boundary regex over the whole text. A codeparrot
# file holds many functions and a lot of module-level code, so two things must be added here:
#
#   scope    "the local variables of a function" is resolved with `symtable`, so a name is
#            renamed only inside the function that owns the binding. A module-level global, a
#            class attribute, an imported name, anything declared `global` in a nested scope,
#            and any name bound by a `class` body inside the function are excluded -- those are
#            the cases where renaming inside one function's span would change a binding that
#            something outside that span can see.
#   spans    substitution happens at tree-sitter `identifier` token spans (`_scan` above), never
#            by regex, so comments, strings, attributes, keyword-argument names and imports are
#            byte-identical between the arms.
#
# Uniform renaming of a name over the WHOLE byte span of its owning function is what makes this
# safe under shadowing: a nested scope that rebinds the same name is inside that span and is
# rewritten with it, so the shadowing structure is preserved and the fresh name (unique corpus-
# wide) can never capture anything.
# ===========================================================================
import ast          # noqa: E402
import symtable     # noqa: E402

# Scope kinds that own variable bindings and get a byte span of their own.
_FN_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _byte_line_starts(code_bytes):
    starts, pos = [0], 0
    for line in code_bytes.split(b'\n')[:-1]:
        pos += len(line) + 1
        starts.append(pos)
    return starts


def _byte_pos(starts, lineno, col):
    """ast line/col -> byte offset. `col_offset` is already a UTF-8 byte offset."""
    return starts[lineno - 1] + col


def _ast_scopes(tree, starts):
    """[(start_byte, end_byte, kind, name, lineno, parent_index)], outermost first."""
    out = []

    def visit(node, parent):
        me = parent
        if isinstance(node, _FN_NODES + (ast.ClassDef,)):
            kind = 'class' if isinstance(node, ast.ClassDef) else 'function'
            out.append([_byte_pos(starts, node.lineno, node.col_offset),
                        _byte_pos(starts, node.end_lineno, node.end_col_offset),
                        kind, node.name, node.lineno, parent])
            me = len(out) - 1
        for child in ast.iter_child_nodes(node):
            visit(child, me)

    for child in ast.iter_child_nodes(tree):
        visit(child, -1)
    return out


def _symtable_index(top):
    """(kind, name, lineno) -> table, for the function and class tables only.

    A key that is claimed by two tables (two `def`s of the same name on the same line) maps to
    None and is treated as unmatched, i.e. left untouched.
    """
    idx, children = {}, {}

    def walk(tb, parent_key):
        key = None
        if tb.get_type() in ('function', 'class'):
            key = (tb.get_type(), tb.get_name(), tb.get_lineno())
            idx[key] = None if key in idx else tb
            children.setdefault(key, [])
        for ch in tb.get_children():
            walk(ch, key if key is not None else parent_key)

    walk(top, None)
    return idx


def _subtree_unsafe(tb, acc):
    """Names anywhere below `tb` whose rename would reach outside the owning function:
    imported names, names declared `global`, and every name bound by a class body
    (a class attribute is read as `C.attr`, whose `.attr` half is not a substitutable span)."""
    is_class = tb.get_type() == 'class'
    for s in tb.get_symbols():
        try:
            if s.is_imported() or s.is_declared_global() or (is_class and s.is_local()):
                acc.add(s.get_name())
        except Exception:                                   # noqa: BLE001
            acc.add(s.get_name())
    for ch in tb.get_children():
        _subtree_unsafe(ch, acc)
    return acc


def _scope_candidates(tb, allowed, kwnames=()):
    """Renameable locals of ONE function table, intersected with the project's own
    renameable-name definition (`extract_variables` + `_acceptable`, computed file-wide).

    A PARAMETER whose name is used as a keyword-argument name anywhere in the file is dropped:
    `f(width=1)` is not a substitutable span, so renaming the parameter would leave a call that
    no longer matches its signature -- a difference between the arms that is not a name."""
    unsafe = set()
    for ch in tb.get_children():
        _subtree_unsafe(ch, unsafe)
    names = set()
    for s in tb.get_symbols():
        n = s.get_name()
        if n not in allowed or n in unsafe:
            continue
        try:
            ok = (s.is_local() and not s.is_global() and not s.is_free()
                  and not s.is_imported() and not s.is_namespace()
                  and not s.is_declared_global()
                  and not (s.is_parameter() and n in kwnames))
        except Exception:                                   # noqa: BLE001
            ok = False
        if ok:
            names.add(n)
    return names


def _innermost(scopes, a, b):
    """Index of the innermost scope containing [a,b), or -1 for module level."""
    best, best_len = -1, None
    for i, sc in enumerate(scopes):
        if sc[0] <= a and b <= sc[1]:
            ln = sc[1] - sc[0]
            if best_len is None or ln < best_len:
                best, best_len = i, ln
    return best


def rename_document_halfvar(code, fresh, u_stream, mix_frac=0.5):
    """-> (new_code, status, n_names, n_sites); same contract as `rename_document`.

    status: 'renamed' | 'parse_error' | 'no_names' | 'unchanged'.
    `n_names` counts (function, name) pairs renamed -- the same name in two functions is two.
    """
    code_bytes = code.encode('utf-8', 'surrogatepass')
    kwnames = set()
    try:
        bad, spans, imported = _scan(code_bytes, kw_out=kwnames)
    except Exception:                                       # noqa: BLE001
        return code, 'parse_error', 0, 0
    if bad:
        return code, 'parse_error', 0, 0
    try:
        tree = ast.parse(code)
        top = symtable.symtable(code, '<doc>', 'exec')
        allowed = {n for n in extract_variables(code, 'python')
                   if _acceptable(n, 'python') and n not in imported}
    except Exception:                                       # noqa: BLE001
        # A file tree-sitter parses but CPython will not (Python 2, mostly). Left untouched in
        # both arms, exactly like a tree-sitter parse error.
        return code, 'parse_error', 0, 0
    if not allowed:
        return code, 'no_names', 0, 0

    starts = _byte_line_starts(code_bytes)
    try:
        scopes = _ast_scopes(tree, starts)
    except Exception:                                       # noqa: BLE001
        return code, 'parse_error', 0, 0
    tbl = _symtable_index(top)

    # first-occurrence order of every substitutable identifier, per scope
    order = {}
    for a, b, text in spans:
        i = _innermost(scopes, a, b)
        order.setdefault(i, [])
        if text not in order[i]:
            order[i].append(text)

    # --- selection, per FUNCTION scope, in source order ---------------------
    chosen = {}                       # scope index -> {old: new}
    n_names = 0
    for i, sc in enumerate(scopes):
        if sc[2] != 'function':
            continue
        tb = tbl.get(('function', sc[3], sc[4]))
        if tb is None:
            continue
        cand = _scope_candidates(tb, allowed, kwnames)
        if not cand:
            continue
        ordered = [n for n in order.get(i, []) if n in cand]
        ordered += sorted(cand - set(ordered))
        keep = [n for n in ordered if u_stream() < mix_frac]
        keep = keep or ordered[:1]                    # repair.py: `keep or names[:1]`
        chosen[i] = {n: fresh.next(u_stream()) for n in keep}
        n_names += len(keep)
    if not n_names:
        return code, 'no_names', 0, 0

    # --- visibility, propagated down the scope tree -------------------------
    # vis[i] = what a span whose innermost scope is `i` resolves to. A class body sees the
    # enclosing FUNCTION's map minus its own attributes; the class's methods see the enclosing
    # function's map, NOT the class body's (Python does not close over class scope).
    vis = {-1: {}}
    inherit = {-1: {}}
    for i, sc in enumerate(scopes):
        base = inherit.get(sc[5], {})
        tb = tbl.get((sc[2], sc[3], sc[4]))
        own = set()
        if tb is not None:
            for s in tb.get_symbols():
                try:
                    if s.is_local() or s.is_imported() or s.is_declared_global():
                        own.add(s.get_name())
                except Exception:                           # noqa: BLE001
                    own.add(s.get_name())
        here = {k: v for k, v in base.items() if k not in own}
        here.update(chosen.get(i, {}))
        vis[i] = here
        # methods of a class do not see the class body; they see what the class saw
        inherit[i] = here if sc[2] == 'function' else base

    # --- substitution at identifier spans -----------------------------------
    out, pos, sites = [], 0, 0
    for a, b, text in spans:
        i = _innermost(scopes, a, b)
        new = vis.get(i, {}).get(text)
        if new is None:
            continue
        out.append(code_bytes[pos:a])
        out.append(new.encode('ascii'))
        pos = b
        sites += 1
    if not sites:
        return code, 'no_names', n_names, 0
    out.append(code_bytes[pos:])
    new_code = b''.join(out).decode('utf-8', 'surrogatepass')
    return new_code, ('renamed' if new_code != code else 'unchanged'), n_names, sites


# ---------------------------------------------------------------------------
# Certification: the two arms may differ ONLY at identifier tokens
# ---------------------------------------------------------------------------
def identifier_skeleton(code):
    """Every substitutable identifier replaced by \\x00, so two texts with the same skeleton
    are byte-identical everywhere except at identifier tokens. Returns None on a parse error.

    Uses the SAME span scan as the renamer, so this is a check of the substitution, not a
    re-implementation of it."""
    code_bytes = code.encode('utf-8', 'surrogatepass')
    bad, spans, _ = _scan(code_bytes)
    if bad:
        return None
    out, pos = [], 0
    for a, b, _t in spans:
        out.append(code_bytes[pos:a])
        out.append(b'\x00')
        pos = b
    out.append(code_bytes[pos:])
    return b''.join(out)


def invert_check(original, treated):
    """Re-derive `original` from `treated` by substituting every changed identifier span back.
    Returns (ok, n_changed_sites, n_distinct_new_names). A True means the treatment touched
    identifier tokens only, and did so bijectively."""
    ob, tb = (original.encode('utf-8', 'surrogatepass'),
              treated.encode('utf-8', 'surrogatepass'))
    obad, ospans, _ = _scan(ob)
    tbad, tspans, _ = _scan(tb)
    if obad or tbad or len(ospans) != len(tspans):
        return False, 0, 0
    back, pos, changed, new_names = [], 0, 0, set()
    for (oa, obk, otext), (ta, tbk, ttext) in zip(ospans, tspans):
        if otext != ttext:
            changed += 1
            new_names.add(ttext)
        back.append(tb[pos:ta])
        back.append(otext.encode('utf-8', 'surrogatepass'))
        pos = tbk
    back.append(tb[pos:])
    return b''.join(back) == ob, changed, len(new_names)
