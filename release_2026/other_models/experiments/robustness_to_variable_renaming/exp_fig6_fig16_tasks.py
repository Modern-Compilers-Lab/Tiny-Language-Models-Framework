"""Figure 6, Table 4 and Figure 16: the three behavioural robustness tasks.

Run twice, in separate processes: --group idrep (identifier reproduction) and
--group bench (variable-misuse localization and ClozeTest). Writes
results/tasks_idrep.json or results/tasks_bench.json.
"""

from tasks import main

if __name__ == "__main__":
    raise SystemExit(main())
