"""Step 05 tests: the repository root and the step folder on the import path, and the screening script loaded under
its own name (``step05_screen``), so it never collides with the ``run.py`` of another step in one pytest session."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent.parent, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if "step05_screen" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("step05_screen", HERE.parent / "screen.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["step05_screen"] = _mod
    _spec.loader.exec_module(_mod)
