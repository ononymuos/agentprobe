import pytest

def pytest_collection_modifyitems(config, items):
    try:
        import torch
        import z3
    except ImportError:
        skip_ml = pytest.mark.skip(reason="ML dependencies (torch, z3) not installed locally to respect 2GB RAM limit. Install with `pip install -e .[ml]` in prod.")
        for item in items:
            item.add_marker(skip_ml)
