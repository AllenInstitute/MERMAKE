import pytest


def _has_cuda_device():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``gpu`` when no CUDA device is available."""
    if _has_cuda_device():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
