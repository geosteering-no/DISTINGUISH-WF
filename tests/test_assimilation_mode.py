import numpy as np
import pytest

from wf_demo.assimilation_mode import (
    ordered_assimilation_steps,
    run_assimilation_sequence,
)


def test_assimilation_steps_always_order_zero_d_before_one_d():
    assert ordered_assimilation_steps(["1D", "0D"]) == ("0D", "1D")
    assert ordered_assimilation_steps(["1D"]) == ("1D",)
    assert ordered_assimilation_steps([]) == ()


def test_assimilation_steps_reject_unknown_simulator():
    with pytest.raises(ValueError, match="Unknown assimilation"):
        ordered_assimilation_steps(["2D"])


def test_assimilation_sequence_passes_first_posterior_to_second_step():
    calls = []

    def assimilate(state, simulator):
        calls.append((simulator, state.copy()))
        return state + (1 if simulator == "0D" else 10)

    posterior = run_assimilation_sequence(
        np.array([0.0]), ["1D", "0D"], assimilate
    )

    assert [name for name, _ in calls] == ["0D", "1D"]
    np.testing.assert_array_equal(calls[1][1], np.array([1.0]))
    np.testing.assert_array_equal(posterior, np.array([11.0]))
