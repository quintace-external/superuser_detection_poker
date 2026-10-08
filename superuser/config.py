"""Default input for the CLI and notebook. Replace data/sample_hands.csv with your own export,
or point INPUT_FILE / --input at another file."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "sample_hands.csv"
DEFAULT_PLAYER = 1001          # player id in the sample file; change to the player in your own file
DEFAULT_OUTPUT = PACKAGE_ROOT / "output"   # overwritten on every run
