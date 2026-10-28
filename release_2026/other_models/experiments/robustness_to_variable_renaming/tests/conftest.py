import sys
from pathlib import Path

# The experiment modules import each other by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
