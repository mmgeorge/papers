"""TableFormer decoder with fixed-size KV cache for CUDA-graph-friendly ONNX export.

All shapes are hardcoded (batch=1, known dims) to avoid Shape/Gather/Expand
ops in the ONNX graph. This is required for CUDA graph capture where all
nodes must partition to CUDA EP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import D_MODEL, N_HEADS, HEAD_DIM, N_LAYERS, VOCAB_SIZE

FFN_DIM = 1024
MAX_SEQ = 512
MAX_PE = 1024  # positional encoding buffer size (from checkpoint)


class TableFormerAttention(nn.Module):
    """Multi-head attention with fixed-size KV cache and torch.where scatter."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(
        self,
        query: torch.Tensor,
        key_value_source: torch.Tensor,
        step: torch.Tensor,
        key_buffer: torch.Tensor | None = None,
        value_buffer: torch.Tensor | None = None,
    ):
        """
        For self-attention: query == key_value_source == new token hidden state.
            key_buffer/value_buffer are the fixed-size KV cache.
        For cross-attention: query = decoder hidden, key_value_source = encoder_memory.
            key_buffer/value_buffer are None (recomputed every step).

        Args:
            query: [B, 1, D_MODEL]
            key_value_source: [B, 1, D_MODEL] (self) or [B, 784, D_MODEL] (cross)
            step: [1] int64
            key_buffer: [B, N_HEADS, MAX_SEQ, HEAD_DIM] or None
            value_buffer: [B, N_HEADS, MAX_SEQ, HEAD_DIM] or None

        Returns:
            output: [B, 1, D_MODEL]
            new_key_buffer: same shape as key_buffer (or None)
            new_value_buffer: same shape as value_buffer (or None)
        """
        # All shapes hardcoded (batch=1) to avoid Shape/Gather/Expand ONNX ops
        # that would prevent CUDA graph capture.

        # Project Q
        q = self.q_proj(query)  # [1, 1, D]
        q = q.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)  # [1, H, 1, Hd]

        if key_buffer is not None:
            # Self-attention with KV cache
            k_new = self.k_proj(key_value_source)  # [1, 1, D]
            v_new = self.v_proj(key_value_source)  # [1, 1, D]
            k_new = k_new.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
            v_new = v_new.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
            # [1, H, 1, Hd]

            # Scatter write at position `step` using torch.where
            positions = torch.arange(MAX_SEQ, device=key_buffer.device)
            write_mask = (positions == step).view(1, 1, MAX_SEQ, 1)
            # Hardcode expand shape to avoid Shape(key_buffer) → Expand chain
            key_buffer = torch.where(
                write_mask,
                k_new.expand(1, N_HEADS, MAX_SEQ, HEAD_DIM),
                key_buffer,
            )
            value_buffer = torch.where(
                write_mask,
                v_new.expand(1, N_HEADS, MAX_SEQ, HEAD_DIM),
                value_buffer,
            )

            # Causal mask: attend to positions 0..step, mask step+1..MAX_SEQ-1
            attn_mask = torch.where(
                positions <= step, 0.0, float("-inf")
            )  # [MAX_SEQ]
            attn_mask = attn_mask.view(1, 1, 1, MAX_SEQ)  # [1, 1, 1, MAX_SEQ]

            # Attention: Q [1,H,1,Hd] @ K^T [1,H,Hd,MAX_SEQ] → [1,H,1,MAX_SEQ]
            scale = 1.0 / math.sqrt(HEAD_DIM)
            attn_weights = torch.matmul(q, key_buffer.transpose(2, 3)) * scale
            attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)

            # [1,H,1,MAX_SEQ] @ [1,H,MAX_SEQ,Hd] → [1,H,1,Hd]
            attn_output = torch.matmul(attn_weights, value_buffer)
        else:
            # Cross-attention: no cache, full K/V from encoder_memory
            # Use -1 to infer seq_len from input, avoiding Shape ops
            k = self.k_proj(key_value_source)  # [1, seq_len, D]
            v = self.v_proj(key_value_source)  # [1, seq_len, D]
            k = k.view(1, -1, N_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, -1, N_HEADS, HEAD_DIM).transpose(1, 2)

            scale = 1.0 / math.sqrt(HEAD_DIM)
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # [1, H, 1, Hd] → [1, 1, D]
        attn_output = attn_output.transpose(1, 2).reshape(1, 1, D_MODEL)
        output = self.out_proj(attn_output)

        return output, key_buffer, value_buffer


class TableFormerDecoderLayer(nn.Module):
    """Single decoder layer with self-attn (KV cache) + cross-attn + FFN."""

    def __init__(self):
        super().__init__()
        self.self_attn = TableFormerAttention()
        self.cross_attn = TableFormerAttention()
        self.linear1 = nn.Linear(D_MODEL, FFN_DIM)
        self.linear2 = nn.Linear(FFN_DIM, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        step: torch.Tensor,
        self_key_buffer: torch.Tensor,
        self_value_buffer: torch.Tensor,
    ):
        """
        Args:
            tgt: [B, 1, D_MODEL] — current token hidden state
            memory: [B, 784, D_MODEL] — encoder output
            step: [1] int64
            self_key_buffer: [B, H, MAX_SEQ, Hd]
            self_value_buffer: [B, H, MAX_SEQ, Hd]
        Returns:
            output: [B, 1, D_MODEL]
            updated self_key_buffer, self_value_buffer
        """
        # Self-attention with KV cache
        residual = tgt
        sa_out, self_key_buffer, self_value_buffer = self.self_attn(
            tgt, tgt, step, self_key_buffer, self_value_buffer
        )
        tgt = self.norm1(residual + sa_out)

        # Cross-attention (no cache)
        residual = tgt
        ca_out, _, _ = self.cross_attn(tgt, memory, step)
        tgt = self.norm2(residual + ca_out)

        # FFN
        residual = tgt
        ff_out = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(residual + ff_out)

        return tgt, self_key_buffer, self_value_buffer


class TableFormerDecoder(nn.Module):
    """Full decoder: embedding + PE + N layers + output projection."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # PE buffer loaded from checkpoint (1024 positions, only MAX_SEQ used)
        self.register_buffer("pe", torch.zeros(MAX_PE, 1, D_MODEL))
        self.layers = nn.ModuleList(
            [TableFormerDecoderLayer() for _ in range(N_LAYERS)]
        )
        self.fc = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        step: torch.Tensor,
        *past_self_kv_flat: torch.Tensor,
    ):
        """
        Args:
            input_ids: [1, 1] int64
            encoder_memory: [1, 784, D_MODEL]
            step: [1] int64
            past_self_kv_flat: 12 tensors of [1, H, MAX_SEQ, Hd]
        Returns:
            logits: [1, 1, VOCAB_SIZE]
            hidden_state: [1, 1, D_MODEL]
            present_kv_flat: 12 tensors of [1, H, MAX_SEQ, Hd]
        """
        # Embed current token
        x = self.embedding(input_ids)  # [1, 1, D_MODEL]

        # Add positional encoding for current step
        # pe is [MAX_PE, 1, D_MODEL], index_select avoids dynamic Slice op
        pe_slice = torch.index_select(self.pe, 0, step)  # [1, 1, D_MODEL]
        x = x + pe_slice

        # Unpack flat KV into per-layer pairs
        past_kv_pairs = []
        for layer_idx in range(N_LAYERS):
            base = layer_idx * 2
            past_kv_pairs.append(
                (past_self_kv_flat[base], past_self_kv_flat[base + 1])
            )

        # Run through decoder layers
        present_kv_flat = []
        for layer_idx, layer in enumerate(self.layers):
            k_buf, v_buf = past_kv_pairs[layer_idx]
            x, k_buf, v_buf = layer(x, encoder_memory, step, k_buf, v_buf)
            present_kv_flat.append(k_buf)
            present_kv_flat.append(v_buf)

        # Output
        hidden_state = x  # [1, 1, D_MODEL]
        logits = self.fc(x)  # [1, 1, VOCAB_SIZE]

        return (logits, hidden_state, *present_kv_flat)
