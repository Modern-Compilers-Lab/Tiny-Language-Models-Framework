"""Launch from any working directory without installing the package."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from other_models.cli import main

    main()
