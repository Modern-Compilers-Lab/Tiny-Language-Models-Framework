"""Launch from any working directory without installing the package."""

import sys
from pathlib import Path

if __name__ == "__main__":
    package = Path(__file__).resolve().parent
    # Direct script execution must not shadow third-party datasets with datasets.py.
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != package]
    sys.path.insert(0, str(package.parent))
    from other_models.cli import main

    main()
