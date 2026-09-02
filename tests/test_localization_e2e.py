"""End-to-end regression test: PET ES update applies AUTOADALOC/SIGM localization.

Runs the real chain used by the app per drilling step:
read-config-equivalent keys -> configure_autoadaloc -> pipt_init.init_da ->
Assimilate.run() -> posterior_state_estimate.npz. The DA is re-initialized
fresh for every strength, matching the app's reset-after-each-action logic.

cv2 is stubbed because PET's QA plotting imports it and this environment lacks
libGL; cv2 is not used in this code path.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import numpy as np
import pandas as pd
import pytest

from pipt import pipt_init
from pipt.loop.assimilation import Assimilate

from wf_demo.localization import configure_autoadaloc

NX, NE, ND = 12, 40, 5


class LinearSim:
    def __init__(self, G):
        self.input_dict = {"parallel": 1}
        self.G = G

    def setup_fwd_run(self, **kwargs):
        pass

    def run_fwd_sim(self, state, member_i):
        return [{"d1": self.G @ np.asarray(state["x"])}]


def _write_inputs(tmp_path, G, x_true):
    rng = np.random.default_rng(3)
    dist = np.abs(np.subtract.outer(np.arange(NX), np.arange(NX)))
    L = np.linalg.cholesky(np.exp(-(dist / 3.0) ** 2) + 1e-10 * np.eye(NX))
    prior = L @ rng.standard_normal((NX, NE))
    np.savez(tmp_path / "prior.npz", **{"x": prior})

    y_obs = G @ x_true + 0.001 * rng.standard_normal(ND)
    pd.DataFrame({"d1": [y_obs]}, index=[0]).to_pickle(tmp_path / "data.pkl")
    pd.DataFrame({"d1": [["ABS", [0.01] * ND]]}, index=[0]).to_pickle(
        tmp_path / "var.pkl"
    )
    return prior


def _run_da(G, enabled, strength):
    keys = {
        "daalg": ["es", "es"],
        "analysis": "approx",
        "energy": 98,
        "obsname": "TVD",
        "truedata": "data.pkl",
        "truedataindex": [1.0],
        "assimindex": [[0]],
        "datatype": ["d1"],
        "staticvar": ["x"],
        "state": "x",
        "prior_x": {
            "vario": "sph", "mean": 0, "var": 1, "range": 1,
            "aniso": 1, "angle": 0, "grid": [1, 1, 1],
        },
        "importstaticvar": "prior.npz",
        "datavar": "var.pkl",
        "restart": "no",
    }
    keys = configure_autoadaloc(keys, enabled=enabled, strength=strength, state_size=NX)

    np.random.seed(11)
    analysis = pipt_init.init_da(keys, keys, LinearSim(G))
    if enabled:
        assert analysis.localization.loc_info["autoadaloc"] is True
        assert analysis.localization.loc_info["type"] == "sigm"
        assert analysis.localization.loc_info["nstd"] == strength
    else:
        assert not hasattr(analysis, "localization")

    Assimilate(analysis).run()
    post = np.load("SaveOutputs/posterior_state_estimate.npz")["x"]
    return post


@pytest.fixture
def e2e_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "SaveOutputs").mkdir()
    rng = np.random.default_rng(5)
    G = rng.standard_normal((ND, NX))
    x_true = rng.standard_normal(NX)
    prior = _write_inputs(tmp_path, G, x_true)
    return G, prior


def test_es_update_applies_autoadaloc_strength_after_fresh_reinit(e2e_inputs):
    G, prior = e2e_inputs
    norm = np.linalg.norm(prior)

    rel = {}
    for label, enabled, strength in (
        ("off", False, 0.9),
        ("high", True, 0.99),
        ("mid", True, 0.5),
        ("low", True, 0.01),
    ):
        rel[label] = np.linalg.norm(_run_da(G, enabled, strength) - prior) / norm

    assert rel["low"] < 0.1 * rel["off"]
    assert rel["low"] < rel["mid"] < rel["off"] * 1.05
    assert abs(rel["high"] - rel["off"]) < 0.05 * rel["off"]
