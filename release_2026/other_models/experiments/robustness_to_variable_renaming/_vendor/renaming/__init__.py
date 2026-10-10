"""Variable-renaming experiments on a 158M tiny_starcoder_py decoder (ICLR submission, Sec. 4.5).

Sub-packages
    renaming.lib          the project's evaluation and training code, kept byte-identical to the
                          version that produced the paper's numbers (see lib/README.md)
    renaming.data         download the data and build the three training corpora
    renaming.experiments  every experiment run on a trained model
    renaming.report       figures, tables and the paper-vs-run comparison
    renaming.pilot        the 18.9M pilot of Appendix D.5

Every entry point is a module with a command line: `python -m renaming.<sub>.<module> --help`.
"""
