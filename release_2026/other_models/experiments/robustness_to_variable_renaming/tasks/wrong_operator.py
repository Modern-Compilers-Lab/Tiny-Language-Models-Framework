from ..registry import TASKS
from .bugs import classify

TASKS.add("cubert_wrong_binary_operator", classify)
