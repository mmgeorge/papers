"""PP-FormulaNet Plus-L decoder: custom PyTorch module matching PaddleOCR architecture.

Architecture (from PaddleOCR source):
- Pre-norm MBart decoder: 8 layers, d_model=512, 16 heads, head_dim=32, FFN=2048
- enc_to_dec_proj: Linear(1024, 512) projects encoder output
- Embedding scaling: sqrt(d_model) = sqrt(512)
- Positional embedding offset: 2
- Q pre-scaling: Q * head_dim^(-0.5)
- Activation: standard GELU (erf-based, not tanh approximation)
- Final LayerNorm after all layers, before lm_head
- lm_head: Linear(512, 50000, bias=False)

KV cache: fixed-size pre-allocated buffers with step-based scatter writes.
All shapes are static (batch and max_seq fixed), enabling CUDA graphs.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture constants
D_MODEL = 512
N_HEADS = 16
HEAD_DIM = D_MODEL // N_HEADS  # 32
FFN_DIM = 2048
N_LAYERS = 8
VOCAB_SIZE = 50000
MAX_POS = 2560
POS_OFFSET = 2
ENCODER_DIM = 1024
EMBED_SCALE = math.sqrt(D_MODEL)  # ~22.63
MAX_SEQ = 512  # max decode steps (practical limit for formula recognition)


class MBartAttention(nn.Module):
    """Multi-head attention with fixed-size KV cache (scatter-based)."""

    def __init__(self, d_model, n_heads, is_cross_attention=False, kdim=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(kdim or d_model, d_model)
        self.v_proj = nn.Linear(kdim or d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.is_cross_attention = is_cross_attention

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                step=None):
        bsz, tgt_len, _ = hidden_states.shape

        # Q projection with pre-scaling
        query = self.q_proj(hidden_states) * self.scaling
        query = query.view(bsz, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.is_cross_attention:
            # Always project from encoder hidden states (no caching branch).
            kv_input = key_value_states
            key = self.k_proj(kv_input).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(kv_input).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
            present_key_value = (key, value)

            # Cross-attention: attend to all encoder positions, no mask needed
            attn_weights = torch.matmul(query, key.transpose(2, 3))
        else:
            # Self-attention: project current token, write into fixed-size buffer
            key_new = self.k_proj(hidden_states).view(bsz, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
            value_new = self.v_proj(hidden_states).view(bsz, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)

            key_buffer, value_buffer = past_key_value
            max_seq = key_buffer.shape[2]

            # Write new KV at position `step` using Where (avoids ScatterElements
            # which breaks CUDA graphs). Write mask is 1 at position step, 0 elsewhere.
            positions = torch.arange(max_seq, device=key_buffer.device)
            write_mask = (positions == step).view(1, 1, max_seq, 1)
            key_buffer = torch.where(write_mask, key_new.expand_as(key_buffer), key_buffer)
            value_buffer = torch.where(write_mask, value_new.expand_as(value_buffer), value_buffer)
            present_key_value = (key_buffer, value_buffer)

            # Attend to full buffer, masking positions > step with -inf
            attn_mask = torch.where(positions <= step, 0.0, float('-inf'))
            attn_mask = attn_mask.view(1, 1, 1, max_seq)

            attn_weights = torch.matmul(query, key_buffer.transpose(2, 3)) + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value if self.is_cross_attention else value_buffer)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, present_key_value


class MBartDecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.self_attn = MBartAttention(d_model, n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = MBartAttention(d_model, n_heads, is_cross_attention=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, encoder_hidden_states, past_key_value=None,
                step=None):
        self_attn_past = (past_key_value[0], past_key_value[1])

        # Self-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_present = self.self_attn(
            hidden_states, past_key_value=self_attn_past, step=step
        )
        hidden_states = residual + hidden_states

        # Cross-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_present = self.encoder_attn(
            hidden_states, key_value_states=encoder_hidden_states
        )
        hidden_states = residual + hidden_states

        # FFN (pre-norm)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        present_key_value = self_present + cross_present  # tuple of 4 tensors
        return hidden_states, present_key_value


class PPFormulaNetDecoder(nn.Module):
    """Full PP-FormulaNet Plus-L decoder with autoregressive generation."""

    def __init__(self):
        super().__init__()
        self.enc_to_dec_proj = nn.Linear(ENCODER_DIM, D_MODEL)
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.embed_positions = nn.Embedding(MAX_POS + POS_OFFSET, D_MODEL)
        self.layernorm_embedding = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList([
            MBartDecoderLayer(D_MODEL, N_HEADS, FFN_DIM) for _ in range(N_LAYERS)
        ])
        self.layer_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, encoder_hidden_states, past_key_values, step):
        """Single decode step with fixed-size KV cache.

        Args:
            input_ids: [B, 1] current token ID
            encoder_hidden_states: [B, enc_seq, 1024] from encoder
            past_key_values: list of 8 tuples, each (self_K, self_V, cross_K, cross_V)
                where self_K/V are [B, 16, max_seq, 32] fixed-size buffers
            step: int64 tensor [1], current decode position (0 = BOS)

        Returns:
            logits: [B, 1, 50000]
            present_key_values: updated KV cache (same shapes)
        """
        # Project encoder output (1024 -> 512)
        enc_hidden = self.enc_to_dec_proj(encoder_hidden_states)

        # Embed tokens (scaled by sqrt(d_model))
        hidden = self.embed_tokens(input_ids) * EMBED_SCALE

        # Positional embedding at position `step` with offset=2
        hidden = hidden + self.embed_positions(step + POS_OFFSET)

        # LayerNorm on combined embeddings
        hidden = self.layernorm_embedding(hidden)

        # Decoder layers
        present_key_values = []
        for i, layer in enumerate(self.layers):
            hidden, present_kv = layer(hidden, enc_hidden, past_key_values[i], step=step)
            present_key_values.append(present_kv)

        # Final LayerNorm + LM head
        hidden = self.layer_norm(hidden)
        logits = self.lm_head(hidden)

        return logits, present_key_values

    @torch.no_grad()
    def generate(self, encoder_hidden_states, max_length=MAX_SEQ,
                 bos_token_id=0, eos_token_id=2):
        """Autoregressive generation with fixed-size KV buffers."""
        bsz = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        # Pre-allocate fixed-size KV buffers (zeros)
        past_key_values = []
        for _ in range(N_LAYERS):
            k_buf = torch.zeros(bsz, N_HEADS, max_length, HEAD_DIM, device=device)
            v_buf = torch.zeros(bsz, N_HEADS, max_length, HEAD_DIM, device=device)
            past_key_values.append((k_buf, v_buf, None, None))

        input_ids = torch.full((bsz, 1), bos_token_id, dtype=torch.long, device=device)
        generated = [bos_token_id]

        for s in range(max_length):
            step = torch.tensor([s], dtype=torch.long, device=device)
            logits, past_key_values = self(input_ids, encoder_hidden_states,
                                           past_key_values, step)
            next_token = logits[:, -1, :].argmax(dim=-1)
            token_id = next_token.item()
            generated.append(token_id)

            if token_id == eos_token_id:
                break

            input_ids = next_token.unsqueeze(1)

        return generated


def load_decoder(weights_path=None):
    """Load decoder with converted PaddlePaddle weights.

    Args:
        weights_path: Path to decoder_weights.npz. If None, looks for
            'decoder_weights.npz' in the current directory.
    """
    if weights_path is None:
        weights_path = "decoder_weights.npz"
    model = PPFormulaNetDecoder()

    # Load weights from npz.
    # NPZ keys have prefix "model.decoder." which maps to the module root.
    data = np.load(weights_path)
    state_dict = model.state_dict()

    # Build mapping: strip "model.decoder." prefix from npz keys
    npz_to_pytorch = {}
    for npz_key in data.files:
        pt_key = npz_key
        if pt_key.startswith("model.decoder."):
            pt_key = pt_key[len("model.decoder."):]
        npz_to_pytorch[pt_key] = npz_key

    loaded = 0
    for name, param in state_dict.items():
        npz_key = npz_to_pytorch.get(name)
        if npz_key is not None:
            tensor = torch.from_numpy(data[npz_key].copy())
            if tensor.shape != param.shape:
                raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {tensor.shape}")
            state_dict[name] = tensor
            loaded += 1
        else:
            print(f"  WARNING: {name} not found in weights file")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {loaded}/{len(state_dict)} parameters from {weights_path}")
    return model
