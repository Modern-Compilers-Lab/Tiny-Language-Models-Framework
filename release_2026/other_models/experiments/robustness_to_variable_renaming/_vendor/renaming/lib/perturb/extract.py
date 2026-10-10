#!/usr/bin/env python3
"""Find the renameable variables in a snippet, in first-appearance order.

Two properties this module guarantees, both of which the old project got wrong:

Determinism. The old `extract_variables` ended in `list(set(...))`, so the order
of the candidate list depended on Python's string-hash seed. `random.sample` on
that list therefore picked *different* variables on every fresh process, and the
documented `--seed 123456` did not reproduce anything unless PYTHONHASHSEED was
pinned. Here every extractor returns variables in source order, deduplicated
stably, so a seed means what it says.

Only actual variables. Renaming a function name, a type, a struct field, or a
library call would change the program's meaning (or break it). C/C++ extraction
walks declarator nodes in the tree-sitter AST rather than matching every
identifier with a regex; Python skips function names, attributes, and imports.
A regex fallback runs only when the parser fails outright.
"""

import re

import tree_sitter

# C and C++ share an extractor: tree-sitter-cpp parses C, and the declarator
# node types we care about are identical in both grammars.
_LANG_ALIASES = {'c': 'cpp', 'c++': 'cpp', 'cpp': 'cpp', 'python': 'python',
                 'java': 'java'}

_PARSERS = {}


def _get_parser(language):
    if language not in _PARSERS:
        if language == 'cpp':
            import tree_sitter_cpp
            lang = tree_sitter.Language(tree_sitter_cpp.language())
        elif language == 'python':
            import tree_sitter_python
            lang = tree_sitter.Language(tree_sitter_python.language())
        elif language == 'java':
            import tree_sitter_java
            lang = tree_sitter.Language(tree_sitter_java.language())
        else:
            raise ValueError(f"no parser for language {language!r}")
        _PARSERS[language] = tree_sitter.Parser(lang)
    return _PARSERS[language]


# ---------------------------------------------------------------------------
# Reserved identifiers
# ---------------------------------------------------------------------------

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'inline',
    'int', 'long', 'register', 'restrict', 'return', 'short', 'signed',
    'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned',
    'void', 'volatile', 'while', 'bool', 'true', 'false', 'NULL', 'nullptr',
    'class', 'public', 'private', 'protected', 'virtual', 'template',
    'typename', 'namespace', 'using', 'new', 'delete', 'this', 'try', 'catch',
    'throw', 'operator', 'friend', 'explicit', 'mutable', 'asm', 'wchar_t',
    'typeid', 'const_cast', 'dynamic_cast', 'static_cast', 'reinterpret_cast',
    'and', 'or', 'not', 'xor', 'size_t', 'main',
}

# Not exhaustive, and it does not need to be: the AST walk already restricts
# candidates to declared variables, so this is a second line of defence against
# grammar quirks (e.g. a macro parsed as a declarator).
C_COMMON = {
    'printf', 'scanf', 'malloc', 'free', 'memset', 'memcpy', 'strlen', 'strcpy',
    'strcmp', 'std', 'cin', 'cout', 'cerr', 'endl', 'string', 'vector', 'map',
    'set', 'queue', 'stack', 'pair', 'sort', 'swap', 'max', 'min', 'abs',
    'INT_MAX', 'INT_MIN', 'LLONG_MAX', 'LLONG_MIN',
}

PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
    'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
    'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
    'self', 'cls', 'print', 'len', 'range', 'int', 'str', 'float', 'bool',
    'list', 'dict', 'set', 'tuple', 'type', 'isinstance', 'super', 'object',
    'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'min', 'max',
    'sum', 'abs', 'round', 'all', 'any', 'open', 'Exception', 'ValueError',
    'TypeError', 'KeyError', 'IndexError',
}

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
    'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null',
    'var', 'record', 'yield', 'sealed', 'permits',
}

# Types and library names that are declarator-shaped but must not be renamed.
JAVA_COMMON = {
    'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Character',
    'Byte', 'Short', 'Object', 'Math', 'System', 'List', 'ArrayList', 'Map',
    'HashMap', 'Set', 'HashSet', 'Arrays', 'Collections', 'Scanner', 'File',
    'Exception', 'RuntimeException', 'Thread', 'StringBuilder', 'StringBuffer',
    'Iterator', 'Comparable', 'Comparator', 'Number', 'Class', 'Void',
    'BufferedReader', 'InputStreamReader', 'PrintWriter', 'main', 'args',
}

RESERVED = {
    'cpp': C_KEYWORDS | C_COMMON,
    'python': PYTHON_KEYWORDS,
    'java': JAVA_KEYWORDS | JAVA_COMMON,
}


def _acceptable(name, language):
    return (name
            and not name.startswith('__')
            and name not in RESERVED[language]
            and re.fullmatch(r'[A-Za-z_]\w*', name) is not None)


# ---------------------------------------------------------------------------
# AST walks
# ---------------------------------------------------------------------------

# In the tree-sitter C/C++ grammar, a declared name sits under a chain of
# declarator nodes. Peel the chain to reach the identifier: `int *a[10]` is
# array_declarator(pointer_declarator(identifier)).
_CPP_DECLARATOR_CHAIN = {
    'pointer_declarator', 'array_declarator', 'reference_declarator',
    'parenthesized_declarator', 'init_declarator',
}

# `int a, b;` and `int *p[4] = {0};` both hang several declarators off one
# declaration node, so we peel each child rather than just the `declarator` field.
_CPP_MULTI_DECLARATOR_NODES = {'declaration', 'field_declaration'}

# These declare exactly one name, reachable through the `declarator` field.
_CPP_SINGLE_DECLARATOR_NODES = {
    'parameter_declaration', 'optional_parameter_declaration',
    'range_based_for_statement',
}


def _peel(node):
    """Follow a declarator chain to the identifier it declares, if any.

    `int *a[10]` parses as array_declarator(pointer_declarator(identifier)).
    A function_declarator means we found a function, not a variable -- bail out
    so we never rename a callable.
    """
    current = node
    for _ in range(16):                     # guard against a pathological AST
        if current is None:
            return None
        if current.type == 'identifier':
            return current
        if current.type == 'function_declarator':
            return None
        if current.type in _CPP_DECLARATOR_CHAIN:
            current = current.child_by_field_name('declarator')
            continue
        return None
    return None


def _extract_cpp(root):
    names = []
    stack = [root]
    while stack:
        node = stack.pop()

        if node.type in _CPP_MULTI_DECLARATOR_NODES:
            for child in node.children:
                ident = _peel(child)
                if ident is not None:
                    names.append((ident.start_byte,
                                  ident.text.decode('utf8', 'replace')))
        elif node.type in _CPP_SINGLE_DECLARATOR_NODES:
            ident = _peel(node.child_by_field_name('declarator'))
            if ident is not None:
                names.append((ident.start_byte,
                              ident.text.decode('utf8', 'replace')))

        stack.extend(node.children)
    return names


def _extract_python(root):
    """Identifiers bound as variables: assignment targets, parameters, loop vars."""
    names = []

    def visit(node):
        if node.type == 'identifier':
            parent = node.parent
            ptype = parent.type if parent else ''
            skip = (
                # the function's or class's own name
                (ptype in ('function_definition', 'class_definition')
                 and parent.child_by_field_name('name') == node)
                # the `.attr` half of `obj.attr`
                or (ptype == 'attribute'
                    and parent.child_by_field_name('attribute') == node)
                # the `f` in `f(x)`
                or (ptype == 'call' and parent.child_by_field_name('function') == node)
                # the `k` in `f(k=1)`
                or (ptype == 'keyword_argument'
                    and parent.child_by_field_name('name') == node)
                or ptype in ('import_statement', 'import_from_statement',
                             'aliased_import', 'dotted_name', 'decorator')
            )
            if not skip:
                names.append((node.start_byte, node.text.decode('utf8', 'replace')))
        for child in node.children:
            visit(child)

    visit(root)
    return names


def _extract_java(root):
    """Java local variables and parameters, in source order.

    Restricted to `local_variable_declaration` and `formal_parameter` on
    purpose. Fields, method names and class names are all deliberately excluded:
    renaming a field or a method would change the program's interface, and in
    BigCloneBench the two functions of a pair are independent compilation units
    whose *shared* API names are part of what makes them clones. Renaming those
    would not be a semantics-preserving intervention.
    """
    names = []
    stack = [root]
    while stack:
        node = stack.pop()

        if node.type == 'local_variable_declaration':
            for child in node.children:
                if child.type != 'variable_declarator':
                    continue
                ident = child.child_by_field_name('name')
                if ident is not None and ident.type == 'identifier':
                    names.append((ident.start_byte,
                                  ident.text.decode('utf8', 'replace')))
        elif node.type in ('formal_parameter', 'catch_formal_parameter',
                           'enhanced_for_statement'):
            ident = node.child_by_field_name('name')
            if ident is not None and ident.type == 'identifier':
                names.append((ident.start_byte,
                              ident.text.decode('utf8', 'replace')))

        stack.extend(node.children)
    return names


_EXTRACTORS = {'cpp': _extract_cpp, 'python': _extract_python,
               'java': _extract_java}


# ---------------------------------------------------------------------------
# Regex fallback (only when the parser fails)
# ---------------------------------------------------------------------------

def _extract_regex(code, language):
    stripped = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    stripped = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", stripped)
    stripped = re.sub(r'//[^\n]*', '', stripped)
    stripped = re.sub(r'#[^\n]*', '', stripped)
    stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL)

    names = []
    for match in re.finditer(r'\b([A-Za-z_]\w*)\b', stripped):
        # An identifier immediately followed by '(' is a call, not a variable.
        if stripped[match.end():match.end() + 1] == '(':
            continue
        names.append((match.start(), match.group(1)))
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_variables(code, language):
    """Renameable variable names, deduplicated, in order of first appearance.

    The return order is a pure function of the source text -- no set iteration,
    no hash seed.
    """
    language = _LANG_ALIASES.get(language.lower())
    if language is None:
        raise ValueError(f"unsupported language; expected one of {sorted(set(_LANG_ALIASES))}")

    try:
        tree = _get_parser(language).parse(bytes(code, 'utf8'))
        occurrences = _EXTRACTORS[language](tree.root_node)
        if not occurrences and len(code.strip()) > 20:
            occurrences = _extract_regex(code, language)
    except Exception:
        occurrences = _extract_regex(code, language)

    seen = set()
    ordered = []
    for _offset, name in sorted(occurrences, key=lambda pair: pair[0]):
        if name in seen or not _acceptable(name, language):
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


# ---------------------------------------------------------------------------
# Integer literals (the `constants` control view)
# ---------------------------------------------------------------------------
#
# The `constants` view swaps integer literals for different integers with the
# same digit count, leaving every identifier byte-identical. It is the control
# that says whether renaming damage is name-specific or is just "any surface
# edit of the same size".
#
# Offsets are returned, not names, because substitution must be positional: a
# word-boundary regex for `100` would also fire inside a string literal or a
# `#include` line, and those are not program constants.

_CPP_INT_LITERAL_NODES = {'number_literal'}
_JAVA_INT_LITERAL_NODES = {'decimal_integer_literal'}
_PY_INT_LITERAL_NODES = {'integer'}

_LITERAL_NODES = {'cpp': _CPP_INT_LITERAL_NODES,
                  'java': _JAVA_INT_LITERAL_NODES,
                  'python': _PY_INT_LITERAL_NODES}

_PLAIN_INT = re.compile(r'\A[0-9]+\Z')


def extract_int_literals(code, language):
    """Plain decimal integer literals as (start_byte, end_byte, text) triples.

    Source order, no deduplication. Hex, octal-with-0x, floats, and anything
    carrying a suffix (`1e9`, `3.5f`, `10UL`) are excluded: they are a different
    lexical class and rewriting their digits can change the token count, which
    would confound a control whose whole point is to hold everything but the
    digits fixed.
    """
    language = _LANG_ALIASES.get(language.lower())
    if language is None or language not in _LITERAL_NODES:
        raise ValueError(f"no integer-literal extractor for {language!r}")

    try:
        tree = _get_parser(language).parse(bytes(code, 'utf8'))
    except Exception:
        return []

    wanted = _LITERAL_NODES[language]
    out = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in wanted:
            text = node.text.decode('utf8', 'replace')
            if _PLAIN_INT.fullmatch(text):
                out.append((node.start_byte, node.end_byte, text))
        stack.extend(node.children)
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# Identifier token positions (safe substitution)
# ---------------------------------------------------------------------------
#
# The shipped substituter in rename.py is a word-boundary `re.sub` over the raw
# source. On POJ-104 that is not safe: C programs are full of format strings and
# escapes whose payload is a single letter, and `\bs\b` matches the `s` of
# "%s" and `\bn\b` matches the `n` of "\n". Renaming a variable called `s` then
# silently rewrites `scanf("%s", s)` into `scanf("%VAR_1", VAR_1)` -- a change to
# the program's OUTPUT FORMAT, not to its identifiers, and a much larger surface
# edit than the rename it was supposed to be.
#
# Substituting only at identifier node positions removes that. Comments, string
# and character literals, and preprocessor payloads are all left alone because
# none of them contain an `identifier` node.

_IDENT_NODE_TYPES = {'identifier'}


def identifier_spans(code, language):
    """(start_byte, end_byte, text) for every identifier TOKEN, in source order.

    Only the plain `identifier` node type. `field_identifier` (`p.x`),
    `type_identifier`, and friends are excluded on purpose: a local named `x`
    must not drag a struct field called `x` along with it.

    Returns None if the parser is unavailable for the language, so the caller can
    fall back to the word-boundary regex and record that it did.
    """
    language = _LANG_ALIASES.get(language.lower())
    if language is None:
        return None
    try:
        tree = _get_parser(language).parse(bytes(code, 'utf8'))
    except Exception:
        return None

    spans = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _IDENT_NODE_TYPES:
            spans.append((node.start_byte, node.end_byte,
                          node.text.decode('utf8', 'replace')))
        stack.extend(node.children)
    spans.sort(key=lambda t: t[0])
    return spans
