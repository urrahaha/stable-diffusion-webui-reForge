import numpy as np
import re

# Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp


def normalise_steps(step, n_steps):
    if step >= 1:
        return int(step)
    if step < 0:
        return 0
    return int(n_steps * step)

def sorted_positions(raw_steps, n_steps):
    steps = [[float(s.strip()) for s in re.split("[@~]", x)]
             for x in re.split("[,;]", str(raw_steps))]
    # If we just got a single number, just return it
    if len(steps[0]) == 1:
        return steps[0][0]

    # Add implicit 1s to any steps which don't have a weight
    
    steps = [[normalise_steps(s[1] if len(s) == 2 else 1, n_steps), s[0]] for s in steps]
    step_triggers = {}
    for [step, weight] in steps:
        step_triggers[step] = weight
    return step_triggers


def calculate_weight(m, step, max_steps, step_offset=2):
    if isinstance(m, list):
        if m[1][-1] <= 1.0:
            if max_steps > 0:
                step = (step) / (max_steps - step_offset)
            else:
                step = 1.0
        else:
            step = step
        v = np.interp(step, m[1], m[0])
        return v
    else:
        return m


def params_to_weights(params, steps):
    weights = {"unet": None, "te": 1.0, "hrunet": None, "hrte": None}

    if len(params.positional) > 1:
        weights["te"] = sorted_positions(params.positional[1], steps)

    if len(params.positional) > 2:
        weights["unet"] = sorted_positions(params.positional[2], steps)

    if params.named.get("te"):
        weights["te"] = sorted_positions(params.named.get("te"), steps)

    if params.named.get("unet"):
        weights["unet"] = sorted_positions(params.named.get("unet"), steps)

    if params.named.get("hr"):
        weights["hrunet"] = sorted_positions(params.named.get("hr"), steps)
        weights["hrte"] = sorted_positions(params.named.get("hr"), steps)

    if params.named.get("hrunet"):
        weights["hrunet"] = sorted_positions(params.named.get("hrunet"), steps)

    if params.named.get("hrte"):
        weights["hrte"] = sorted_positions(params.named.get("hrte"), steps)

    # If unet ended up unset, then use the te value
    weights["unet"] = weights["unet"] if weights["unet"] is not None else weights["te"]
    # If hrunet ended up unset, use unet value
    weights["hrunet"] = weights["hrunet"] if weights["hrunet"] is not None else weights["unet"]
    # If hrte ended up unset, use te value
    weights["hrte"] = weights["hrte"] if weights["hrte"] is not None else weights["te"]

    weights_return = {}
    for (key, value) in weights.items():
        for (step, weight) in value.items():
            if step not in weights_return:
                weights_return[step] = {}
            weights_return[step][key] = weight
    return weights_return


hires = False
loractl_active = True

def is_hires():
    return hires


def set_hires(value):
    global hires
    hires = value


def set_active(value):
    global loractl_active
    loractl_active = value

def is_active():
    global loractl_active
    return loractl_active
