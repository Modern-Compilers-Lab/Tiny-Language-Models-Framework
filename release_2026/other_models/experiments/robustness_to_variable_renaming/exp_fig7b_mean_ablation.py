"""Figure 7B: mean-ablate the k highest-ranked copy heads and random controls.

Head means are computed per view on half A; accuracy is measured on half B.
Writes results/meanablate.json and results/metrics.json.
"""

import sys

from run import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], experiment="meanablate"))
