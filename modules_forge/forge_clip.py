from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from ldm_patched.modules import model_management
from modules import sd_models
from modules.shared import opts


def move_clip_to_gpu():
    if sd_models.model_data.sd_model is None:
        print('Error: CLIP called before SD is loaded!')
        return

    model_management.load_model_gpu(sd_models.model_data.sd_model.forge_objects.clip.patcher)
    return


class CLIP_SD_15_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.minimal_clip_skip = 1

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        return z


class CLIP_SD_21_H(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)

        return z


class CLIP_SD_XL_L(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]

        return z


class CLIP_SD_XL_G(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        if self.wrapped.layer == "penultimate":
            self.wrapped.layer = "hidden"
            self.wrapped.layer_idx = -2

        self.id_start = 49406
        self.id_end = 49407
        self.id_pad = 0
        self.minimal_clip_skip = 2

    def encode_with_transformers(self, tokens):
        self.wrapped.transformer.text_model.embeddings.to(tokens.device)
        if hasattr(self.wrapped.transformer, 'text_model'):
            outputs = self.wrapped.transformer(tokens, output_hidden_states=self.wrapped.layer == "hidden")
        else:
            outputs = self.wrapped.transformer(tokens, intermediate_output=self.wrapped.layer_idx if self.wrapped.layer == "hidden" else None)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers] if hasattr(outputs, 'hidden_states') else outputs[1]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx] if hasattr(outputs, 'hidden_states') else outputs[1]

        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[2]
        text_projection = self.wrapped.text_projection
        pooled_output = pooled_output.float().to(text_projection.device) @ text_projection.float()
        z.pooled = pooled_output
        return z
    
import torch
import torch.nn as nn

class CLIP_SD3(nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.transformer = wrapped.cond_stage_model
        self.minimal_clip_skip = 2

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        
        tokens = self.tokenize(text)
        cond, pooled = self.encode_from_tokens(tokens, return_pooled=True)
        
        return {
            'crossattn': cond,
            'vector': pooled,
        }

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        result = {}
        for key in ['l', 'g', 't5xxl']:
            if hasattr(self.tokenizer, key):
                tokenizer = getattr(self.tokenizer, key)
                result[key] = [tokenizer.tokenize_with_weights(text, return_word_ids=False) for text in texts]
        
        return result

    def encode_from_tokens(self, tokens, return_pooled=True):
        l_out, l_pooled = self.encode_with_transformers(self.transformer.clip_l, tokens['l'][0])
        g_out, g_pooled = self.encode_with_transformers(self.transformer.clip_g, tokens['g'][0])
        
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        
        if hasattr(self.transformer, 't5xxl'):
            t5_out, _ = self.encode_with_transformers(self.transformer.t5xxl, tokens['t5xxl'][0])
            lg_out = torch.cat([lg_out, t5_out], dim=-2)
        
        if return_pooled:
            pooled = torch.cat([l_pooled, g_pooled], dim=-1)
            return lg_out, pooled
        else:
            return lg_out

    def encode_with_transformers(self, clip_model, tokens):
        tokens = torch.tensor(tokens[0]).to(self.device)
        clip_model.transformer.text_model.embeddings.to(tokens.device)
        
        # SD3 models use SDClipModel which doesn't accept output_hidden_states
        outputs = clip_model(tokens)
        
        # Unpack the outputs
        z, pooled = outputs

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            # For SD3, we might need to adjust this logic
            # This is a placeholder and may need further adjustment
            z = z[:, -opts.CLIP_stop_at_last_layers:]
        
        return z, pooled

    def encode_embedding_init_text(self, init_text, num_vectors_per_token):
        tokens = self.tokenize(init_text)
        cond, _ = self.encode_from_tokens(tokens, return_pooled=False)
        return cond[0, -1].unsqueeze(0).repeat(num_vectors_per_token, 1)

    def encode(self, tokens):
        return self.encode_from_tokens(tokens)

    def get_target_prompt_token_count(self, token_count):
        return token_count

    def tokenize_line(self, line):
        return self.tokenize([line])

    def process_text(self, texts):
        return self.tokenize(texts)

    @property
    def device(self):
        return next(self.parameters()).device

    # Add these methods to make it compatible with FrozenCLIPEmbedderWithCustomWords
    def hijack_add_custom_word(self, *args, **kwargs):
        pass

    def hijack_del_custom_word(self, *args, **kwargs):
        pass

    def hijack_get_prompt_lengths(self, text):
        tokens = self.tokenize([text])
        return {k: len(v[0]) for k, v in tokens.items()}

    def hijack_get_word_ids(self, text, idxs):
        tokens = self.tokenize([text])
        return {k: [t[idxs] for t in v[0]] for k, v in tokens.items()}

    def hijack_reconstruct_cond_batch(self, tokens):
        return tokens
