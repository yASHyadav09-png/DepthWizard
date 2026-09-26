"""DepthWizard backend application package."""

import sys
from pathlib import Path

__version__ = "0.3.0"

# The trained model and its inference code live in the repository's `depthwizard`
# package (one level up); make it importable when the server runs from backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
