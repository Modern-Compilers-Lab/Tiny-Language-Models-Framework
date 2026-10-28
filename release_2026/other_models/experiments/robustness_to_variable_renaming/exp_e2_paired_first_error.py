"""Appendix E.2: where the first error falls in names that renaming breaks.

Writes results/firsterror.json and results/metrics.json.
"""

import sys

from run import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], experiment="firsterror"))
