"""Process set-up shared by every entry point.

The evaluation library (renaming/lib) has three requirements that must hold BEFORE it is
imported, and this module is the one place that enforces them:

1. PYTHONHASHSEED=0 when the interpreter starts. The renamed test views are seeded with hashes,
   so another hash seed scores different renamed texts. `setup()` re-executes the current
   process with PYTHONHASHSEED=0 if it is not already set.
2. HF_HOME, XDG_CACHE_HOME and WHYSCALE_CACHE must point inside the work directory. The library
   has its own fallback defaults (paths on the machine it was written on); setting the variables
   first means those fallbacks are never used.
3. renaming/lib must be on sys.path, because its modules import each other by bare name
   (`import whyscale`, `from perturb.extract import ...`).

Paths are taken from the environment, which config.sh exports:
    RENAMING_WORK   the work directory (data, models, results, caches); default ./work
"""
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(PKG_ROOT, 'renaming', 'lib')


def work_dir():
    return os.path.abspath(os.environ.get('RENAMING_WORK') or os.path.join(PKG_ROOT, 'work'))


def paths():
    """Every location the package reads or writes, derived from the work directory."""
    w = work_dir()
    return {
        'work': w,
        'hf_home': os.path.join(w, 'cache', 'hf'),
        'xdg_cache': os.path.join(w, 'cache', 'xdg'),
        'eval_corpus': os.path.join(w, 'eval_corpus'),     # = WHYSCALE_CACHE
        'data': os.path.join(w, 'data'),
        'models': os.path.join(w, 'models'),
        'results': os.path.join(w, 'results'),
        'report': os.path.join(w, 'report'),
        'logs': os.path.join(w, 'logs'),
        'pilot': os.path.join(w, 'pilot'),
    }


def setup(reexec=True):
    """Call first thing in every entry point (before importing anything from renaming.lib)."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        if not reexec:
            raise SystemExit('PYTHONHASHSEED must be 0 before Python starts (config.sh sets it)')
        os.environ['PYTHONHASHSEED'] = '0'
        # sys.orig_argv keeps "-m renaming.x.y", which sys.argv has already replaced by a path
        argv = list(getattr(sys, 'orig_argv', [sys.executable] + sys.argv))[1:]
        os.execv(sys.executable, [sys.executable] + argv)
    p = paths()
    os.environ.setdefault('HF_HOME', p['hf_home'])
    os.environ.setdefault('XDG_CACHE_HOME', p['xdg_cache'])
    os.environ.setdefault('WHYSCALE_CACHE', p['eval_corpus'])
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    import warnings
    # CodeSearchNet functions contain regex strings with invalid escapes; parsing them raises
    # thousands of harmless SyntaxWarnings that would bury the logs.
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    return p


def lib_module(name):
    """Import a library module and check that it really comes from renaming/lib."""
    import importlib
    m = importlib.import_module(name)
    if os.path.dirname(os.path.abspath(m.__file__)) != LIB:
        raise SystemExit('%s was imported from %s, not from %s. Unset PYTHONPATH and re-run.'
                         % (name, m.__file__, LIB))
    return m


def device(requested='auto'):
    import torch
    if requested in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return requested


def write_json(path, doc):
    """Atomic write, so an interrupted run never leaves half a result file behind."""
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def last_snapshot(model_dir):
    """The final HuggingFace snapshot written by the trainer (snapNNN with the largest NNN)."""
    import glob
    snaps = sorted(d for d in glob.glob(os.path.join(model_dir, 'snap*'))
                   if os.path.isfile(os.path.join(d, 'config.json')))
    return snaps[-1] if snaps else None
