"""Table 6 and Appendix E.2: force the copy heads onto earlier occurrences of a name.

Lists the selected heads and measures repairs against random-head and
other-variable controls. Writes results/force.json and results/metrics.json.
"""

import sys

from run import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], experiment="force"))
