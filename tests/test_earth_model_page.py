from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_pixel_model_controls_build_matching_prior_ensemble():
    page = Path(__file__).resolve().parents[1] / "wf_demo" / "earth_model_page.py"
    app = AppTest.from_file(page).run(timeout=30)

    app.radio[0].set_value("Pixel-based").run(timeout=30)

    assert not app.exception
    assert app.radio[1].label == "Pixel prior source"
    assert app.radio[1].options == ["Geostatistical", "GAN samples"]
    assert app.selectbox[0].options == ["Gaussian", "Spherical", "Exponential", "Cubic"]
    assert [control.label for control in app.number_input] == [
        "Horizontal correlation length (cells)",
        "Vertical correlation length (cells)",
        "Prior mean",
        "Prior standard deviation",
        "Localization strength",
    ]
    assert app.toggle[0].label == "Use auto-adaptive localization"
    assert app.text_input[0].label == "Custom well start row (0-63)"
    assert app.text_input[0].value == "31"
    assert app.number_input[-1].disabled

    app.toggle[0].set_value(True).run(timeout=30)

    assert app.session_state["localization_enabled"] is True
    assert not app.number_input[-1].disabled

    app.number_input[-1].set_value(0.25).run(timeout=30)
    assert app.session_state["localization_strength"] == 0.25

    # Streamlit drops widget state when the widget is not rendered (page
    # switch). A returning session therefore only carries the plain session
    # keys; a fresh run must restore the widgets from them.
    restored = AppTest.from_file(page)
    restored.session_state["localization_enabled"] = True
    restored.session_state["localization_strength"] = 0.25
    restored.run(timeout=30)

    assert not restored.exception
    assert restored.toggle[0].value is True
    assert not restored.number_input[-1].disabled
    assert restored.number_input[-1].value == 0.25
    assert restored.session_state["localization_enabled"] is True

    app.text_input[0].set_value("17").run(timeout=30)
    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert app.session_state["earth_model_type"] == "pixel"
    assert app.session_state["earth_model_config"]["pixel_prior_source"] == "Geostatistical"
    assert app.session_state["earth_model_config"]["start_row"] == 17
    assert app.session_state["earth_model_prior"].shape == (4096, 250)
