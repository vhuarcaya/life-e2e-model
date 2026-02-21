"""Pytest configuration — add src/life_e2e to sys.path for bare imports."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'life_e2e'))
