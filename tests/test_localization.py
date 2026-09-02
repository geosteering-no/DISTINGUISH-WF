import pytest

from wf_demo.localization import configure_autoadaloc


def test_localization_disabled_removes_existing_configuration():
    keys = {"analysis": "approx", "localization": {"type": "other"}}

    configured = configure_autoadaloc(keys, enabled=False, strength=0.9, state_size=60)

    assert "localization" not in configured
    assert "localization" in keys


def test_localization_enabled_uses_only_autoadaloc_sigmoid():
    configured = configure_autoadaloc(
        {"analysis": "approx"},
        enabled=True,
        strength=0.75,
        state_size=4096,
    )

    assert configured["localization"] == {
        "field": [4096, 1, 1],
        "autoadaloc": 0.75,
        "type": "sigm",
    }


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_localization_rejects_strength_outside_correlation_range(strength):
    with pytest.raises(ValueError, match="between 0 and 1"):
        configure_autoadaloc({}, enabled=True, strength=strength, state_size=60)


def test_localization_rejects_empty_state():
    with pytest.raises(ValueError, match="State size must be positive"):
        configure_autoadaloc({}, enabled=True, strength=0.9, state_size=0)
