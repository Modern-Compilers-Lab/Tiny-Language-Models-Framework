"""Complete benchmark release downloads, reusable by offline preparation."""

from ..source_downloads import download_files


def source_files(name):
    from .data import BIFI_DATA, HF, MT4PY, RAW

    if name == "codexglue_py150":
        base = HF + "/google/code_x_glue_cc_code_completion_token/resolve/main/python/"
        return [
            ("python/" + file, base + file)
            for file in (
                "train-00000-of-00002.parquet",
                "train-00001-of-00002.parquet",
                "test-00000-of-00001.parquet",
            )
        ]
    if name.startswith("codexglue_cloze"):
        variant = "all" if name.endswith("_all") else "maxmin"
        base = (
            RAW
            + f"/microsoft/CodeXGLUE/main/Code-Code/ClozeTesting-{variant}/data/cloze-{variant}/"
        )
        files = [("clozeTest.json", base + "python/clozeTest.json")]
        if variant == "all":
            files.append(("cloze_test_words.txt", base + "cloze_test_words.txt"))
        for split in ("test", "validation"):
            file = f"{split}-00000-of-00001.parquet"
            files.append(
                (
                    "csn/" + file,
                    HF + "/code-search-net/code_search_net/resolve/main/python/" + file,
                )
            )
        return files
    if name == "bifi":
        return [("data.zip", BIFI_DATA + "data.zip")]
    if name == "manytypes4py":
        return [("ManyTypes4PyDataset-v0.7.tar.gz", MT4PY)]
    if name.startswith("cruxeval_"):
        return [
            (
                "cruxeval.jsonl",
                RAW + "/facebookresearch/cruxeval/main/data/cruxeval.jsonl",
            )
        ]
    raise ValueError(name)


def download(name, run, budget=None):
    return download_files(name, run, source_files(name), budget)
