import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

for path in (ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
