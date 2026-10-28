"""Section 4.5: zero-ablate the k highest-ranked copy heads and random controls.

Writes results/copyscore.json (with the k = 1..kmax curve) and results/metrics.json.
"""

import sys

from run import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], experiment="copyscore"))
