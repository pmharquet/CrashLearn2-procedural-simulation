"""Regression checks for installed commands, resource paths and unchanged physics."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from crashlearn_sim.simulation.world import ProceduralSimulation


@pytest.mark.parametrize(
    "scenario",
    json.loads(
        (Path(__file__).parent / "fixtures/simulation-baseline.json").read_text(encoding="utf-8")
    ),
    ids=lambda scenario: f"{scenario['weather']}-{scenario['profile']}",
)
def test_simulation_matches_pre_refactor_baseline(scenario):
    sim = ProceduralSimulation(
        scenario["seed"], num_cars=2, weather=scenario["weather"], profile=scenario["profile"]
    )
    for tick in range(40):
        sim.step([(3.0, 0.02 * np.sin(tick / 8)), (2.5, -0.01)])
    np.testing.assert_allclose(sim.states, scenario["states"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sim.observation()["lidar"], scenario["lidar"], atol=1e-6)
    np.testing.assert_array_equal(sim.status, scenario["status"])


def test_resources_and_legacy_cli_outside_repository(tmp_path, monkeypatch):
    result = subprocess.run(
        [sys.executable, "-m", "crashlearn_sim", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    monkeypatch.chdir(tmp_path)
    from crashlearn_sim.competition.models import tracks

    assert len(tracks()) == 23


def test_core_import_does_not_load_ui_or_training():
    code = (
        "import sys; import crashlearn_sim.simulation.world; "
        "assert not {'pygame', 'moderngl', 'torch', 'stable_baselines3'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=60)


@pytest.mark.parametrize("module", ["crashlearn_sim.legacy.record", "crashlearn_sim.legacy.replay"])
def test_legacy_commands_are_safe_to_import(module, tmp_path):
    subprocess.run([sys.executable, "-c", f"import {module}"], cwd=tmp_path, check=True, timeout=30)
    assert not list(tmp_path.iterdir())


def test_no_application_modules_at_repository_root():
    assert not list(ROOT.glob("*.py"))
