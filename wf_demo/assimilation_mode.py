ASSIMILATION_ORDER = ("0D", "1D")


def ordered_assimilation_steps(selected):
    selected = set(selected or ())
    unknown = selected.difference(ASSIMILATION_ORDER)
    if unknown:
        raise ValueError(f"Unknown assimilation simulator(s): {sorted(unknown)}")
    return tuple(step for step in ASSIMILATION_ORDER if step in selected)


def run_assimilation_sequence(state, steps, assimilate):
    for step in ordered_assimilation_steps(steps):
        state = assimilate(state, step)
    return state
