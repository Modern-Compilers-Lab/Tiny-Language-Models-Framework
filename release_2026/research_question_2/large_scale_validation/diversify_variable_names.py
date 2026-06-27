#!/usr/bin/env python3
"""
Diversify variable names in the CodeSearchNet dataset.

Replaces variable names in code with random unique names to increase diversity.
Works across all 6 CodeSearchNet languages: Python, Java, JavaScript, Ruby, Go, PHP.

Design:
  - Uses tree-sitter AST parsing to accurately identify variable/parameter names,
    distinguishing them from function declarations, type names, and keywords.
  - Falls back to regex-based extraction if tree-sitter parsing fails.
  - Accepts a random seed so that different seeds produce different renamings,
    enabling per-epoch re-randomization during training.
  - Can be used as a library (call diversify_code()) or as a standalone script
    to produce a diversified copy of the full dataset.

Usage as standalone script:
    python diversify_variable_names.py --seed 42 --output_dir data_cache/diversified
    python diversify_variable_names.py --seed 123 --output_dir data_cache/diversified_e2

Usage as a library during training:
    from diversify_variable_names import diversify_code
    new_code = diversify_code(code, language="python", seed=epoch * 1000 + idx)
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import logging
from typing import List, Set, Tuple, Optional, Dict

import tree_sitter
import tree_sitter_python
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_go
import tree_sitter_ruby
import tree_sitter_php

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Language-specific keyword sets (identifiers that must not be renamed)
# ────────────────────────────────────────────────────────────────────

PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield',
    # Common builtins that should not be renamed
    'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list', 'dict',
    'set', 'tuple', 'type', 'isinstance', 'issubclass', 'hasattr', 'getattr',
    'setattr', 'delattr', 'super', 'object', 'property', 'staticmethod',
    'classmethod', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
    'min', 'max', 'sum', 'abs', 'round', 'all', 'any', 'open', 'input',
    'iter', 'next', 'id', 'hash', 'hex', 'oct', 'bin', 'chr', 'ord',
    'repr', 'format', 'vars', 'dir', 'globals', 'locals', 'exec', 'eval',
    'compile', 'callable', 'Exception', 'ValueError', 'TypeError',
    'KeyError', 'IndexError', 'AttributeError', 'RuntimeError',
    'StopIteration', 'NotImplementedError', 'OSError', 'IOError',
    'self', 'cls', '__init__', '__name__', '__main__', '__all__',
    '__doc__', '__file__', '__class__',
}

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
    'char', 'class', 'const', 'continue', 'default', 'do', 'double',
    'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
    'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface',
    'long', 'native', 'new', 'package', 'private', 'protected', 'public',
    'return', 'short', 'static', 'strictfp', 'super', 'switch',
    'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void',
    'volatile', 'while', 'true', 'false', 'null',
    # Common classes/methods that should not be renamed
    'System', 'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean',
    'Object', 'Class', 'Math', 'Arrays', 'Collections', 'List', 'Map',
    'Set', 'ArrayList', 'HashMap', 'HashSet', 'Iterator', 'Iterable',
    'Comparable', 'Comparator', 'Override', 'Deprecated', 'SuppressWarnings',
    'out', 'err', 'in', 'println', 'printf', 'print', 'main', 'args',
    'toString', 'equals', 'hashCode', 'length', 'size', 'get', 'put',
    'add', 'remove', 'contains', 'isEmpty', 'toArray', 'valueOf',
    'parseInt', 'parseDouble', 'parseLong',
}

JS_KEYWORDS = {
    'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case',
    'catch', 'char', 'class', 'const', 'continue', 'debugger', 'default',
    'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends',
    'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if',
    'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'let',
    'long', 'native', 'new', 'null', 'of', 'package', 'private',
    'protected', 'public', 'return', 'short', 'static', 'super', 'switch',
    'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try',
    'typeof', 'undefined', 'var', 'void', 'volatile', 'while', 'with',
    'yield',
    'console', 'log', 'error', 'warn', 'Math', 'JSON', 'Array', 'Object',
    'String', 'Number', 'Boolean', 'Date', 'RegExp', 'Error', 'Map', 'Set',
    'Promise', 'Symbol', 'Proxy', 'Reflect', 'parseInt', 'parseFloat',
    'isNaN', 'isFinite', 'require', 'module', 'exports', 'process',
    'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'prototype', 'constructor', 'length', 'push', 'pop', 'shift',
    'unshift', 'splice', 'slice', 'concat', 'join', 'indexOf',
    'forEach', 'map', 'filter', 'reduce', 'find', 'some', 'every',
    'keys', 'values', 'entries', 'then', 'catch', 'resolve', 'reject',
}

GO_KEYWORDS = {
    'break', 'case', 'chan', 'const', 'continue', 'default', 'defer',
    'else', 'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import',
    'interface', 'map', 'package', 'range', 'return', 'select', 'struct',
    'switch', 'type', 'var',
    # Built-in types and functions
    'bool', 'byte', 'complex64', 'complex128', 'error', 'float32',
    'float64', 'int', 'int8', 'int16', 'int32', 'int64', 'rune',
    'string', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr',
    'true', 'false', 'nil', 'iota',
    'append', 'cap', 'close', 'complex', 'copy', 'delete', 'imag',
    'len', 'make', 'new', 'panic', 'print', 'println', 'real', 'recover',
    'main', 'init', '_',
    # Common packages
    'fmt', 'os', 'io', 'net', 'http', 'json', 'log', 'math', 'sort',
    'strings', 'strconv', 'sync', 'time', 'context', 'errors', 'bytes',
    'bufio', 'regexp', 'path', 'filepath',
}

RUBY_KEYWORDS = {
    'BEGIN', 'END', 'alias', 'and', 'begin', 'break', 'case', 'class',
    'def', 'defined?', 'do', 'else', 'elsif', 'end', 'ensure', 'false',
    'for', 'if', 'in', 'module', 'next', 'nil', 'not', 'or', 'redo',
    'rescue', 'retry', 'return', 'self', 'super', 'then', 'true',
    'undef', 'unless', 'until', 'when', 'while', 'yield',
    '__FILE__', '__LINE__', '__ENCODING__',
    'puts', 'print', 'p', 'raise', 'require', 'require_relative',
    'include', 'extend', 'attr_reader', 'attr_writer', 'attr_accessor',
    'initialize', 'new', 'to_s', 'to_i', 'to_f', 'to_a', 'to_h',
    'each', 'map', 'select', 'reject', 'reduce', 'inject', 'collect',
    'detect', 'find', 'sort', 'sort_by', 'length', 'size', 'empty?',
    'nil?', 'is_a?', 'kind_of?', 'respond_to?', 'send', 'method',
}

PHP_KEYWORDS = {
    'abstract', 'and', 'array', 'as', 'break', 'callable', 'case',
    'catch', 'class', 'clone', 'const', 'continue', 'declare', 'default',
    'die', 'do', 'echo', 'else', 'elseif', 'empty', 'enddeclare',
    'endfor', 'endforeach', 'endif', 'endswitch', 'endwhile', 'eval',
    'exit', 'extends', 'final', 'finally', 'fn', 'for', 'foreach',
    'function', 'global', 'goto', 'if', 'implements', 'include',
    'include_once', 'instanceof', 'insteadof', 'interface', 'isset',
    'list', 'match', 'namespace', 'new', 'or', 'print', 'private',
    'protected', 'public', 'readonly', 'require', 'require_once',
    'return', 'static', 'switch', 'throw', 'trait', 'try', 'unset',
    'use', 'var', 'while', 'xor', 'yield',
    'true', 'false', 'null', 'TRUE', 'FALSE', 'NULL',
    'this', 'self', 'parent', 'static',
    '__construct', '__destruct', '__call', '__callStatic', '__get',
    '__set', '__isset', '__unset', '__sleep', '__wakeup', '__toString',
    '__invoke', '__set_state', '__clone', '__debugInfo',
    'string', 'int', 'float', 'bool', 'void', 'object', 'mixed',
}

KEYWORDS_BY_LANG = {
    'python': PYTHON_KEYWORDS,
    'java': JAVA_KEYWORDS,
    'javascript': JS_KEYWORDS,
    'go': GO_KEYWORDS,
    'ruby': RUBY_KEYWORDS,
    'php': PHP_KEYWORDS,
}

# ────────────────────────────────────────────────────────────────────
# Tree-sitter parsers (lazy-initialized)
# ────────────────────────────────────────────────────────────────────

_PARSERS: Dict[str, tree_sitter.Parser] = {}

def _get_parser(language: str) -> tree_sitter.Parser:
    """Get or create a tree-sitter parser for the given language."""
    if language not in _PARSERS:
        lang_map = {
            'python': tree_sitter.Language(tree_sitter_python.language()),
            'java': tree_sitter.Language(tree_sitter_java.language()),
            'javascript': tree_sitter.Language(tree_sitter_javascript.language()),
            'go': tree_sitter.Language(tree_sitter_go.language()),
            'ruby': tree_sitter.Language(tree_sitter_ruby.language()),
            # language_php_only() handles PHP without <?php ?> tags,
            # which is how CodeSearchNet stores PHP snippets.
            'php': tree_sitter.Language(tree_sitter_php.language_php_only()),
        }
        _PARSERS[language] = tree_sitter.Parser(lang_map[language])
    return _PARSERS[language]


# ────────────────────────────────────────────────────────────────────
# AST-based variable extraction per language
# ────────────────────────────────────────────────────────────────────

def _extract_variables_python(root_node) -> Set[str]:
    """Extract renameable variable/parameter names from Python AST."""
    variables = set()
    def visit(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip function/class definitions (the function/class name itself)
            if ptype in ('function_definition', 'class_definition'):
                # Only skip if this is the name child (first identifier child)
                if parent.children and parent.child_by_field_name('name') == node:
                    for child in node.children:
                        visit(child)
                    return
            # Skip decorator names
            if ptype == 'decorator':
                for child in node.children:
                    visit(child)
                return
            # Skip import names
            if ptype in ('import_statement', 'import_from_statement',
                         'aliased_import', 'dotted_name'):
                for child in node.children:
                    visit(child)
                return
            # Skip attribute access (the attribute name after the dot)
            if ptype == 'attribute' and node == parent.child_by_field_name('attribute'):
                for child in node.children:
                    visit(child)
                return
            variables.add(name)
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


def _extract_variables_java(root_node) -> Set[str]:
    """Extract renameable variable/parameter names from Java AST."""
    variables = set()
    def visit(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip method/class/interface declarations (the name itself)
            if ptype in ('method_declaration', 'class_declaration',
                         'interface_declaration', 'enum_declaration',
                         'annotation_type_declaration', 'constructor_declaration'):
                if parent.child_by_field_name('name') == node:
                    for child in node.children:
                        visit(child)
                    return
            # Skip type identifiers (used as types in declarations)
            if ptype in ('type_identifier', 'generic_type', 'scoped_type_identifier'):
                for child in node.children:
                    visit(child)
                return
            # Skip package/import names
            if ptype in ('package_declaration', 'import_declaration',
                         'scoped_identifier'):
                for child in node.children:
                    visit(child)
                return
            # Skip method invocation name (the method being called)
            if ptype == 'method_invocation' and parent.child_by_field_name('name') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip field access name
            if ptype == 'field_access' and parent.child_by_field_name('field') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip annotation names
            if ptype in ('marker_annotation', 'annotation'):
                for child in node.children:
                    visit(child)
                return
            variables.add(name)
        elif node.type == 'type_identifier':
            # Skip type identifiers entirely
            return
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


def _extract_variables_javascript(root_node) -> Set[str]:
    """Extract renameable variable/parameter names from JavaScript AST."""
    variables = set()
    def visit(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip function/class declarations (the name itself)
            if ptype in ('function_declaration', 'class_declaration',
                         'method_definition', 'generator_function_declaration'):
                if parent.child_by_field_name('name') == node:
                    for child in node.children:
                        visit(child)
                    return
            # Skip property access (the property name after the dot)
            if ptype == 'member_expression' and parent.child_by_field_name('property') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip import specifiers
            if ptype in ('import_specifier', 'namespace_import'):
                for child in node.children:
                    visit(child)
                return
            variables.add(name)
        elif node.type == 'property_identifier':
            # Skip property identifiers (object property names)
            return
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


def _is_keyed_element_key(node) -> bool:
    """Check if a node is the key (first child) of a keyed_element (Go struct literal field)."""
    parent = node.parent
    if parent is None:
        return False
    # literal_element -> keyed_element: the first literal_element child is the key
    if parent.type == 'literal_element':
        grandparent = parent.parent
        if grandparent and grandparent.type == 'keyed_element':
            children = [c for c in grandparent.children if c.type == 'literal_element']
            if children and children[0] == parent:
                return True
    return False


def _extract_variables_go(root_node) -> Set[str]:
    """Extract renameable variable/parameter names from Go AST."""
    variables = set()
    def visit(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            if name == '_':
                for child in node.children:
                    visit(child)
                return
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip function declarations (the name itself)
            if ptype == 'function_declaration' and parent.child_by_field_name('name') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip method declarations
            if ptype == 'method_declaration' and parent.child_by_field_name('name') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip package clause
            if ptype == 'package_clause':
                for child in node.children:
                    visit(child)
                return
            # Skip import paths
            if ptype in ('import_spec', 'import_declaration'):
                for child in node.children:
                    visit(child)
                return
            # Skip selector expression field name
            if ptype == 'selector_expression' and parent.child_by_field_name('field') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip struct literal field keys (e.g., Name: in Op{Name: val})
            if _is_keyed_element_key(node):
                for child in node.children:
                    visit(child)
                return
            variables.add(name)
        elif node.type in ('type_identifier', 'package_identifier', 'field_identifier'):
            # Skip type names, package names, field names
            return
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


def _extract_variables_ruby(root_node) -> Set[str]:
    """Extract renameable variable/parameter names from Ruby AST."""
    variables = set()
    def visit(node):
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip method definitions (the name itself)
            if ptype == 'method' and parent.child_by_field_name('name') == node:
                for child in node.children:
                    visit(child)
                return
            # Skip class/module definitions
            if ptype in ('class', 'module'):
                if parent.child_by_field_name('name') == node:
                    for child in node.children:
                        visit(child)
                    return
            # Skip method calls (the method name) -- but keep the receiver
            if ptype == 'call':
                if parent.child_by_field_name('method') == node:
                    for child in node.children:
                        visit(child)
                    return
            variables.add(name)
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


def _extract_variables_php(root_node) -> Set[str]:
    """Extract renameable variable names from PHP AST.

    PHP variables always start with $, represented as variable_name nodes.
    """
    variables = set()
    def visit(node):
        if node.type == 'variable_name':
            # Get the inner name (without $)
            text = node.text.decode('utf-8')
            name = text.lstrip('$')
            parent = node.parent
            ptype = parent.type if parent else ''
            # Skip $this
            if name == 'this':
                for child in node.children:
                    visit(child)
                return
            # Skip member access name (the property/method after ->)
            if ptype == 'member_access_expression':
                if parent.child_by_field_name('name') == node:
                    for child in node.children:
                        visit(child)
                    return
            variables.add(name)
        for child in node.children:
            visit(child)
    visit(root_node)
    return variables


_EXTRACTORS = {
    'python': _extract_variables_python,
    'java': _extract_variables_java,
    'javascript': _extract_variables_javascript,
    'go': _extract_variables_go,
    'ruby': _extract_variables_ruby,
    'php': _extract_variables_php,
}


# ────────────────────────────────────────────────────────────────────
# Regex-based fallback extraction
# ────────────────────────────────────────────────────────────────────

def _extract_variables_regex(code: str, language: str) -> Set[str]:
    """Fallback regex-based variable extraction when tree-sitter fails."""
    # Remove strings and comments
    code_clean = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code_clean = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code_clean)
    code_clean = re.sub(r'//[^\n]*', '', code_clean)
    code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
    code_clean = re.sub(r'#[^\n]*', '', code_clean)

    if language == 'php':
        # PHP: extract $variable_name patterns
        identifiers = set(re.findall(r'\$([a-zA-Z_]\w*)', code_clean))
        identifiers.discard('this')
    else:
        identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code_clean))

    keywords = KEYWORDS_BY_LANG.get(language, set())
    return {name for name in identifiers
            if name not in keywords and not name.startswith('__')}


# ────────────────────────────────────────────────────────────────────
# Core: extract variables from code
# ────────────────────────────────────────────────────────────────────

def extract_variables(code: str, language: str) -> List[str]:
    """Extract renameable variable names from source code.

    Uses tree-sitter AST parsing with regex fallback.
    Returns a sorted list of unique variable names.
    """
    try:
        parser = _get_parser(language)
        code_bytes = code.encode('utf-8')
        tree = parser.parse(code_bytes)
        # For PHP, CodeSearchNet snippets often start with "public function ..."
        # which causes a root-level error, but the function body is still
        # correctly parsed. So we use the AST even with errors for PHP.
        # For other languages, fall back to regex on parse errors.
        use_ast = not tree.root_node.has_error or language == 'php'
        if use_ast:
            extractor = _EXTRACTORS[language]
            variables = extractor(tree.root_node)
            # If AST extraction found nothing on a non-trivial snippet, try regex
            if not variables and len(code.strip()) > 20:
                variables = _extract_variables_regex(code, language)
        else:
            variables = _extract_variables_regex(code, language)
    except Exception:
        variables = _extract_variables_regex(code, language)

    keywords = KEYWORDS_BY_LANG.get(language, set())
    variables = {v for v in variables if v not in keywords and len(v) > 0}
    return sorted(variables)


# ────────────────────────────────────────────────────────────────────
# Random name generation
# ────────────────────────────────────────────────────────────────────

# Pool of characters for random names
_ALPHA = string.ascii_lowercase
_ALNUM = string.ascii_lowercase + string.digits + '_'


def generate_random_name(rng: random.Random, existing: Set[str],
                         keywords: Set[str], min_len: int = 2,
                         max_len: int = 10) -> str:
    """Generate a random variable name that doesn't collide with existing names or keywords."""
    forbidden = existing | keywords
    for _ in range(200):
        length = rng.randint(min_len, max_len)
        name = rng.choice(_ALPHA) + ''.join(rng.choice(_ALNUM) for _ in range(length - 1))
        # Ensure it doesn't end with underscore and isn't a keyword
        if name not in forbidden and not name.startswith('__'):
            return name
    # Extreme fallback
    return 'v_' + str(rng.randint(100000, 999999))


# ────────────────────────────────────────────────────────────────────
# Core: diversify variable names in a single code snippet
# ────────────────────────────────────────────────────────────────────

def diversify_code(code: str, language: str, seed: int = 42,
                   rename_fraction: float = 1.0) -> str:
    """Replace variable names in code with random unique names.

    Args:
        code: Source code string.
        language: One of 'python', 'java', 'javascript', 'ruby', 'go', 'php'.
        seed: Random seed for reproducible renaming. Different seeds produce
              different random names, enabling per-epoch diversity.
        rename_fraction: Fraction of variables to rename (1.0 = all).

    Returns:
        Code with variable names replaced by random unique names.
    """
    variables = extract_variables(code, language)
    if not variables:
        return code

    rng = random.Random(seed)
    keywords = KEYWORDS_BY_LANG.get(language, set())

    # Select which variables to rename
    num_to_rename = max(1, int(len(variables) * rename_fraction))
    if rename_fraction < 1.0:
        to_rename = rng.sample(variables, min(num_to_rename, len(variables)))
    else:
        to_rename = variables

    # Generate new names
    all_names = set(variables) | keywords
    rename_map = {}
    for old_name in to_rename:
        new_name = generate_random_name(rng, all_names, keywords)
        all_names.add(new_name)
        rename_map[old_name] = new_name

    # Apply renames: sort by length descending to avoid partial matches
    result = code
    if language == 'php':
        # PHP: variables have $ prefix in code but not in our extracted names
        for old_name in sorted(rename_map.keys(), key=len, reverse=True):
            new_name = rename_map[old_name]
            # Replace $old_name with $new_name (word boundary after name)
            result = re.sub(
                r'\$' + re.escape(old_name) + r'(?=\b|\W|$)',
                '$' + new_name,
                result
            )
    else:
        for old_name in sorted(rename_map.keys(), key=len, reverse=True):
            new_name = rename_map[old_name]
            result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, result)

    return result


# ────────────────────────────────────────────────────────────────────
# Batch processing for the full dataset
# ────────────────────────────────────────────────────────────────────

LANGUAGES = ["python", "java", "javascript", "ruby", "go", "php"]


def diversify_dataset(seed: int = 42, data_cache: str = "data_cache",
                      output_dir: Optional[str] = None,
                      split: str = "train",
                      rename_fraction: float = 1.0) -> dict:
    """Diversify variable names across the entire CodeSearchNet dataset.

    Args:
        seed: Base random seed. Each example gets seed = base_seed * 1000000 + index
              so that different base seeds produce completely different renamings.
        data_cache: Path to HuggingFace dataset cache.
        output_dir: If provided, save diversified data as JSONL files.
        split: Dataset split ('train', 'validation', 'test').
        rename_fraction: Fraction of variables to rename per example.

    Returns:
        Dictionary mapping language -> list of (original_nl, diversified_code) tuples.
    """
    from datasets import load_dataset

    results = {}
    total_examples = 0
    total_renamed = 0

    for lang in LANGUAGES:
        logger.info(f"Processing {lang} ({split})...")
        ds = load_dataset("code_search_net", lang, cache_dir=data_cache, split=split)

        diversified = []
        lang_renamed = 0

        for idx, item in enumerate(ds):
            nl = item.get("func_documentation_string", "") or ""
            code = item.get("func_code_string", "") or ""

            if not code.strip():
                diversified.append((nl.strip(), code.strip()))
                continue

            # Unique seed per example: ensures different examples get different names
            # even within the same epoch, and different epochs (different base seed)
            # produce different names for the same example.
            example_seed = seed * 1_000_000 + total_examples + idx

            try:
                new_code = diversify_code(code, lang, seed=example_seed,
                                          rename_fraction=rename_fraction)
                if new_code != code:
                    lang_renamed += 1
            except Exception as e:
                new_code = code  # Keep original on failure

            diversified.append((nl.strip(), new_code.strip()))

        total_examples += len(ds)
        total_renamed += lang_renamed
        results[lang] = diversified

        logger.info(f"  {lang}: {len(diversified)} examples, "
                     f"{lang_renamed} modified ({lang_renamed/max(len(diversified),1)*100:.1f}%)")

    logger.info(f"Total: {total_examples} examples, {total_renamed} modified "
                f"({total_renamed/max(total_examples,1)*100:.1f}%)")

    # Save to JSONL if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for lang in LANGUAGES:
            out_path = os.path.join(output_dir, f"{lang}_{split}.jsonl")
            with open(out_path, 'w') as f:
                for nl, code in results[lang]:
                    json.dump({"func_documentation_string": nl,
                               "func_code_string": code,
                               "language": lang}, f)
                    f.write('\n')
            logger.info(f"Saved {out_path}")

    return results


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diversify variable names in CodeSearchNet dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (use different seeds for per-epoch diversity)")
    parser.add_argument("--data_cache", type=str, default="data_cache",
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for diversified JSONL files")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process")
    parser.add_argument("--rename_fraction", type=float, default=1.0,
                        help="Fraction of variables to rename (0.0-1.0)")
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo on a few examples per language")
    args = parser.parse_args()

    if args.demo:
        # Quick demo: show before/after for one example per language
        from datasets import load_dataset
        for lang in LANGUAGES:
            ds = load_dataset("code_search_net", lang, cache_dir=args.data_cache,
                              split="train")
            code = ds[0]["func_code_string"]
            variables = extract_variables(code, lang)
            new_code = diversify_code(code, lang, seed=args.seed)

            print(f"\n{'='*70}")
            print(f"Language: {lang}")
            print(f"Variables found: {variables}")
            print(f"--- Original (first 300 chars) ---")
            print(code[:300])
            print(f"--- Diversified (first 300 chars) ---")
            print(new_code[:300])
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_cache, f"diversified_seed{args.seed}")

    diversify_dataset(
        seed=args.seed,
        data_cache=args.data_cache,
        output_dir=args.output_dir,
        split=args.split,
        rename_fraction=args.rename_fraction,
    )


if __name__ == "__main__":
    main()
