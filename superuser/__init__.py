"""Superuser (card-visibility) forensics for hand-replay records."""
from .parser import parse_hand, Hand
from .pipeline import analyse_player
from .cli import read_hands
from .config import DEFAULT_INPUT, DEFAULT_PLAYER

__all__ = ["parse_hand", "Hand", "analyse_player", "read_hands", "DEFAULT_INPUT", "DEFAULT_PLAYER"]
__version__ = "0.1.0"
