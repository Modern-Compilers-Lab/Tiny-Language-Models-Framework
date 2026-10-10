from ..registry import TASKS
from .bugs import classify

TASKS.add("cubert_variable_misuse", classify)
