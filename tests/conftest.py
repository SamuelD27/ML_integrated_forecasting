"""Pytest configuration for crypto_trading tests."""
import sys
from pathlib import Path

# Add parent directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
