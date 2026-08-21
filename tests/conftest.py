import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import importlib.util
        torch_spec = importlib.util.find_spec("torch")
        if not torch_spec:
            raise ImportError
        z3_spec = importlib.util.find_spec("z3")
        if not z3_spec:
            raise ImportError
    except ImportError:
        skip_ml = pytest.mark.skip(reason="ML dependencies (torch, z3) not installed locally to respect 2GB RAM limit. Install with `pip install -e .[ml]` in prod.")
        for item in items:
            item.add_marker(skip_ml)
