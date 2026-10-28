"""Figure 7A: copy score of the heads ranked on half A, measured on half B.

Writes results/copyscore.json and results/metrics.json. The ablation sweep is
disabled (--kmax 0); exp_zero_ablation.py runs it.
"""

import sys

from run import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] + ["--kmax", "0"], experiment="copyscore"))
