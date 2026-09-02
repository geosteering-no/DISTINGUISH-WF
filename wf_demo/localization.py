def configure_autoadaloc(keys_data, enabled, strength, state_size):
    configured_keys = keys_data.copy()
    if not enabled:
        configured_keys.pop("localization", None)
        return configured_keys
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Localization strength must be between 0 and 1")
    if state_size < 1:
        raise ValueError("State size must be positive")

    configured_keys["localization"] = {
        "field": [int(state_size), 1, 1],
        "autoadaloc": float(strength),
        "type": "sigm",
    }
    return configured_keys
