from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils

import sys, importlib
from pathlib import Path

# extensions-builtin isn't normally referencable due to the dash; this hacks around that
lora_path = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin" / "Lora")
sys.path.insert(0, lora_path)
import network, networks, extra_networks_lora
sys.path.remove(lora_path)

from modules.processing import StableDiffusionProcessing
from modules.extra_networks import ExtraNetworkParams

lora_weights = {}

params_map = {}

step = 0

def reset_weights():
    lora_weights.clear()
    params_map.clear()

class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    def __init__(self):
        self.initialised = False
        super().__init__()

    def clear(self):
        global step
        step = 0
        self.initialised = False
        reset_weights()

    def activate(self, p: StableDiffusionProcessing, params_list: list[ExtraNetworkParams]):
        if not utils.is_active():
            return super().activate(p, params_list)
        if self.initialised == False:
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
                weights = utils.params_to_weights(params, p.steps)
                for (key, value) in weights.items():
                    if key not in lora_weights:
                        lora_weights[key] = {}
                    lora_weights[key][name] = value
                # Use the initial weight instead of hardcoded 1
                params.positional = [name, initial_weight]
                params.named = {}
                params_map[name] = params
            self.initialised = True
        if step in lora_weights:
            for (name, weights) in lora_weights[step].items():
                params_map[name].positional = [name, str(weights['unet'])]
                params_map[name].named = {}
        super().activate(p, params_list)
        