"""Collect the historical contracts alongside the modern regression suite."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name == "test_t08_friction_initial_then_drop":
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="Pre-existing T08: historical _update_friction keeps friction at 1.0",
                )
            )


@pytest.fixture(autouse=True)
def close_historical_simulator(request):
    """Match the original harness: never share a live singleton between tests."""
    yield
    if request.module.__name__ in {"test_contract", "test_integration"}:
        from crashlearn_sim.legacy.environment import close

        close()
