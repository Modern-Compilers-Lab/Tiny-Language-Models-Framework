from ..registry import TASKS
from .bugs import classify

TASKS.add("cubert_swapped_operands", classify)
