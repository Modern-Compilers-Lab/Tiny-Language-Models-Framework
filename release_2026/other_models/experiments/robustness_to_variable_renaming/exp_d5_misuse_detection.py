"""Appendix D.5: two-way variable-misuse detection and its retention.

Writes results/pilot_vm.json. This is not the localization task of Figure 6.
"""

from pilot import main

if __name__ == "__main__":
    raise SystemExit(main(what="vm"))
