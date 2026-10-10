"""Every experiment run on one trained 158M model. Each module scores ONE model directory and
writes ONE JSON; scripts/04_evaluate.sh runs all of them for an (arm, seed).

    tasks.py       Figure 6, Table 4, Figure 16  (identifier reproduction, variable-misuse
                   localization, cloze test; N = 0, 1, 2, 4, 8 variables renamed)
    firsterror.py  Section 4.5 "Localising the failure" and Appendix E.2 (where errors fall)
    copyscore.py   Figure 7A (copy score) and the zero-ablation check
    meanablate.py  Figure 7B (mean ablation of k copy heads vs k random heads)
    force.py       forced attention: repairs by the copy heads, random heads, other variables;
                   Table 6 (the selected heads)
"""
