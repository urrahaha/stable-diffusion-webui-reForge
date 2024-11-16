from modules import extra_networks, shared
import networks

class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')
        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""

    remove_symbols = str.maketrans('', '', ":,")

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        self.errors.clear()

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        block_weights = []

        for params in params_list:
            assert params.items

            names.append(params.positional[0])

            # Handle TE multiplier
            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))
            te_multipliers.append(te_multiplier)

            # Handle weights/unet multiplier
            if len(params.positional) > 2:
                # Check if the third parameter contains commas - indicating it's a weight string
                if ',' in str(params.positional[2]):
                    unet_multipliers.append(te_multiplier)  # Use TE multiplier as default
                    block_weights.append(params.positional[2])
                else:
                    # It's a regular unet multiplier
                    unet_multiplier = float(params.positional[2])
                    unet_multipliers.append(unet_multiplier)
                    block_weights.append(None)
            else:
                unet_multipliers.append(te_multiplier)
                block_weights.append(None)

            # Handle dynamic dimension
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            dyn_dims.append(dyn_dim)

        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)

        # Process block weights if any exist
        for i, weights in enumerate(block_weights):
            if weights is not None:
                # Add block weight to prompt parameters
                p.extra_generation_params[f"lora_block_weight_{names[i]}"] = weights
                # Remove the lora syntax from prompt to avoid double-processing
                original_syntax = f"<lora:{names[i]}:{te_multipliers[i]}:{weights}>"
                p.prompt = p.prompt.replace(original_syntax, "")

        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    p.lora_hashes[item.mentioned_name.translate(self.remove_symbols)] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ', '.join(f'{k}: {v}' for k, v in p.lora_hashes.items())

    def deactivate(self, p):
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            self.errors.clear()
            