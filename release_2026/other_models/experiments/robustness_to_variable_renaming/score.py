"""Summary metrics recomputed from saved result files; needs no model."""

import json
from pathlib import Path


def rate(k, n):
    if not 0 <= k <= n:
        raise ValueError("Invalid success count or denominator")
    return k / n if n else None


def summarize(run):
    r = Path(run) / "results"
    out = {}

    def read(name):
        return json.loads((r / (name + ".json")).read_text())

    if (r / "copyscore.json").exists():
        c = read("copyscore")
        for v in ("id", "ood"):
            b = c["splitB"][v]
            if b["n"] == 0 or b["n_score_queries"] == 0:
                raise ValueError("Empty measured split")
            out["copy_score_" + v] = b["highest_copy_score"]
            out["first_piece_" + v] = b["first_piece_acc"]
        out["splitB_items"] = c["splitB"]["ood"]["n"]
    if (r / "firsterror.json").exists():
        f = read("firsterror")
        b = f["paired"]["all"]
        if sum(b["c2i_at"].values()) != b["c2i"]:
            raise ValueError("Paired counts are inconsistent")
        out.update(
            paired_items=b["items"],
            c2i=b["c2i"],
            first_error_start_share=rate(b["c2i_at"]["start"], b["c2i"]),
        )
        # Every bin is recounted from the saved per-item decisions.
        for key, cell in f["paired"].items():
            ds = [
                d
                for d in f["decisions"]
                if key == "all"
                or ("4+" if d["pieces"] >= 4 else str(d["pieces"])) == key
            ]
            assert len(ds) == cell["items"]
            assert sum(d["id"] is None for d in ds) == cell["correct_before"]
            for loc in ("start", "middle", "end"):
                found = sum(d["id"] is None and d["ood"] == loc for d in ds)
                assert found == cell["c2i_at"][loc]
    if (r / "force.json").exists():
        f = read("force")
        out["force_failures"] = f["n_failures"]
        sc = f["selfcheck"]
        if (
            sc["max_abs_diff"] is None
            or sc["max_abs_diff"] > 5e-5
            or sc["pred_unchanged"][0] != sc["pred_unchanged"][1]
        ):
            raise ValueError("Forced-attention self-check failed")
        for k, b in f["by_k"].items():
            for field in (
                "copy_this_rescue",
                "random_this_rescue",
                "copy_other_rescue",
                "copy_other_writes_other",
            ):
                x = b[field]
                out[f"force_{k}_{field}"] = rate(x["k"], x["n"])
    if (r / "meanablate.json").exists():
        f = read("meanablate")
        for b in f["curve"]:
            out[f"mean_k{b['k']}_ood"] = b["acc_ood"]
            out[f"mean_k{b['k']}_random_ood"] = b["random_ood_mean"]
    return out
