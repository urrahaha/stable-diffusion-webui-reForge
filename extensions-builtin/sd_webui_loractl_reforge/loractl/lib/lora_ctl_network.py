from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils

import sys, importlib
from pathlib import Path

# extensions-builtin isn't normally referencable due to the dash; this hacks around that
lora_path = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin" / "Lora")
sys.path.insert(0, lora_path)
import network, networks, extra_networks_lora
sys.path.remove(lora_path)

lora_weights = {}


def reset_weights():
    lora_weights.clear()


class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    def activate(self, p, params_list):
        if not utils.is_active():
            return super().activate(p, params_list)

        for params in params_list:
            assert params.items
            name = params.positional[0]
            
            # Get the initial weight if it exists
            initial_weight = 1.0
            if len(params.positional) > 1:
                try:
                    # First convert to string to safely check for special characters
                    weight_str = str(params.positional[1])
                    # Check if it's a simple weight value
                    if '@' not in weight_str and ',' not in weight_str:
                        initial_weight = float(params.positional[1])
                except ValueError:
                    pass

            if lora_weights.get(name, None) is None:
                weights = utils.params_to_weights(params)
                # If we have a simple initial weight, use it as base
                if initial_weight != 1.0:
                    for key in weights:
                        if not isinstance(weights[key], list):
                            weights[key] = initial_weight
                lora_weights[name] = weights

            # Use the initial weight instead of hardcoded 1
            params.positional = [name, initial_weight]
            params.named = {}
            
        return super().activate(p, params_list)
