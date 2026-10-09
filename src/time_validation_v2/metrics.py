"""Compatibility import; shared WRMSSE calculations have one implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m5_shared.metrics import *
