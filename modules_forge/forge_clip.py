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
        self.clip_l = wrapped.cond_stage_model.clip_l
        self.clip_g = wrapped.cond_stage_model.clip_g
        self.t5xxl = getattr(wrapped.cond_stage_model, 't5xxl', None)  # This is actually another CLIP model

        # Set special tokens
        self.id_start = getattr(self.tokenizer.clip_l, 'start_token', 0)
        self.id_end = getattr(self.tokenizer.clip_l, 'end_token', 0)
        self.id_pad = self.id_end
        self.comma_token = self.tokenizer.clip_l.tokenizer.get_vocab().get(',</w>', None)

        self.token_mults = {}
        self.minimal_clip_skip = 1

    def tokenize(self, texts):
        if isinstance(texts, str):
            return self.tokenizer.tokenize_with_weights(texts)
        elif isinstance(texts, list):
            return [self.tokenizer.tokenize_with_weights(text) for text in texts]
        else:
            raise ValueError(f"Unsupported input type for tokenization: {type(texts)}")

    def encode_with_transformers(self, tokens):
        move_clip_to_gpu()
        outputs = {}

        # Process CLIP-L
        l_tokens = torch.tensor([t[0] for t in tokens["l"][0]]).unsqueeze(0)
        self.clip_l.transformer.text_model.embeddings.to(l_tokens.device)
        outputs_l = self.clip_l.transformer(l_tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        # Process CLIP-G
        g_tokens = torch.tensor([t[0] for t in tokens["g"][0]]).unsqueeze(0)
        self.clip_g.transformer.text_model.embeddings.to(g_tokens.device)
        outputs_g = self.clip_g.transformer(g_tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)

        if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
            z_l = outputs_l.hidden_states[-opts.CLIP_stop_at_last_layers]
            z_g = outputs_g.hidden_states[-opts.CLIP_stop_at_last_layers]
        else:
            z_l = outputs_l.last_hidden_state
            z_g = outputs_g.last_hidden_state

        combined_output = torch.cat([z_l, z_g], dim=-1)

        # Process T5XXL (which is actually another CLIP model)
        if self.t5xxl and "t5xxl" in tokens:
            t5_tokens = torch.tensor([t[0] for t in tokens["t5xxl"][0]]).unsqueeze(0)
            self.t5xxl.transformer.text_model.embeddings.to(t5_tokens.device)
            outputs_t5 = self.t5xxl.transformer(t5_tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)
            if opts.CLIP_stop_at_last_layers > self.minimal_clip_skip:
                z_t5 = outputs_t5.hidden_states[-opts.CLIP_stop_at_last_layers]
            else:
                z_t5 = outputs_t5.last_hidden_state
            combined_output = torch.cat([combined_output, z_t5], dim=-1)

        return combined_output

    def encode(self, texts):
        tokens = self.tokenize(texts)
        if isinstance(tokens, list):
            return torch.stack([self.encode_with_transformers(t) for t in tokens])
        else:
            return self.encode_with_transformers(tokens)

    def encode_embedding_init_text(self, init_text, nvpt):
        tokens_l = self.tokenizer.clip_l.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        tokens_g = self.tokenizer.clip_g.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]

        embedded_l = self.clip_l.transformer.text_model.embeddings.token_embedding(tokens_l.to(self.clip_l.transformer.text_model.embeddings.token_embedding.weight.device)).squeeze(0)
        embedded_g = self.clip_g.transformer.text_model.embeddings.token_embedding(tokens_g.to(self.clip_g.transformer.text_model.embeddings.token_embedding.weight.device)).squeeze(0)

        # Pad or truncate to nvpt
        embedded_l = self._pad_or_truncate(embedded_l, nvpt)
        embedded_g = self._pad_or_truncate(embedded_g, nvpt)

        combined_embedding = torch.cat([embedded_l, embedded_g], dim=-1)

        # Add T5XXL (CLIP) embedding if available
        if self.t5xxl:
            tokens_t5 = self.tokenizer.t5xxl.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            embedded_t5 = self.t5xxl.transformer.text_model.embeddings.token_embedding(tokens_t5.to(self.t5xxl.transformer.text_model.embeddings.token_embedding.weight.device)).squeeze(0)
            embedded_t5 = self._pad_or_truncate(embedded_t5, nvpt)
            combined_embedding = torch.cat([combined_embedding, embedded_t5], dim=-1)

        return combined_embedding

    def _pad_or_truncate(self, tensor, target_length):
        if tensor.shape[0] < target_length:
            return torch.cat([tensor, torch.zeros(target_length - tensor.shape[0], tensor.shape[1], device=tensor.device)], dim=0)
        else:
            return tensor[:target_length]

    def forward(self, text):
        return self.encode(text)