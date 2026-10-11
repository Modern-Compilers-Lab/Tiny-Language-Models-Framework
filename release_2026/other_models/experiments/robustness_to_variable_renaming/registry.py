"""Drop-in modules expose register(registries); duplicate names fail loudly."""

import importlib.util
from pathlib import Path


class Registry(dict):
    def add(self, name, value):
        if name in self:
            raise ValueError(f"Duplicate plugin: {name}")
        self[name] = value
        return value


MODELS, DATASETS, TRANSFORMS, TASKS = (Registry() for _ in range(4))


def discover(folder):
    for path in sorted(Path(folder).glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(
            "other_models_plugin_" + path.stem, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.register(
            dict(
                models=MODELS,
                datasets=DATASETS,
                transformations=TRANSFORMS,
                tasks=TASKS,
            )
        )


def builtins():
    from .data_transformations import renaming
    from .tasks import (
        completion,
        cloze,
        bugs,
        paper,
        variable_misuse,
        wrong_operator,
        swapped_operands,
        exception,
    )
