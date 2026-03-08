"""PP-FormulaNet Plus-L encoder: Vary-VIT-B-Formula ported from PaddleOCR.

Architecture (from PaddleOCR ppocr/modeling/backbones/rec_vary_vit.py):
- Input: [B, 1, 768, 768] grayscale -> repeat to 3 channels
- PatchEmbed: Conv2D(3->768, k=16, s=16) -> [B, 48, 48, 768]
- Absolute positional embedding [1, 48, 48, 768]
- 12 ViTDet transformer blocks:
    - Blocks 0,1,3,4,6,7,9,10: window attention (14x14)
    - Blocks 2,5,8,11: global attention
    - All: decomposed relative position embeddings (h + w)
- Neck: Conv2D(768->256, 1x1) -> LN2d -> Conv2D(256->256, 3x3) -> LN2d
- net_2: Conv2D(256->512, 3x3, stride=2) -> [B, 512, 24, 24]
- net_3: Conv2D(512->1024, 3x3, stride=2) -> [B, 1024, 12, 12]
- Flatten + transpose -> [B, 144, 1024]
- mm_projector_vary: Linear(1024->1024)
-> encoder_hidden_states: [B, 144, 1024]
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture constants
IMG_SIZE = 768
PATCH_SIZE = 16
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64
MLP_DIM = EMBED_DIM * 4  # 3072
OUT_CHANS = 256
WINDOW_SIZE = 14
GLOBAL_ATTN_INDEXES = (2, 5, 8, 11)
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 48
PADDED_SIZE = ((GRID_SIZE + WINDOW_SIZE - 1) // WINDOW_SIZE) * WINDOW_SIZE  # 56
PAD_AMOUNT = PADDED_SIZE - GRID_SIZE  # 8


# --- Window attention helpers ---

def window_partition(x: torch.Tensor, window_size: int):
    """Partition [B, H, W, C] into non-overlapping windows with padding.

    Returns:
        windows: [B * num_windows, window_size, window_size, C]
        (Hp, Wp): padded height and width
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    """Reverse window partition and remove padding.

    Args:
        windows: [B * num_windows, window_size, window_size, C]
        pad_hw: (Hp, Wp) padded dimensions
        hw: (H, W) original dimensions
    Returns:
        x: [B, H, W, C]
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size,
                        window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor):
    """Get relative positional embeddings for (q_size, k_size).

    Args:
        q_size: query spatial size
        k_size: key spatial size
        rel_pos: learned relative positions [L, head_dim]

    Returns:
        Positional embeddings [q_size, k_size, head_dim]
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate if sizes don't match (shouldn't happen with fixed 768x768 input)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, dtype=torch.float32, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=torch.float32, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]):
    """Add decomposed relative position embeddings to attention weights.

    Args:
        attn: [B*num_heads, q_h*q_w, k_h*k_w]
        q: [B*num_heads, q_h*q_w, head_dim]
        rel_pos_h/w: learned relative position embeddings
        q_size: (q_h, q_w)
        k_size: (k_h, k_w)
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # [q_h, k_h, head_dim]
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # [q_w, k_w, head_dim]

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.reshape(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).reshape(B, q_h * q_w, k_h * k_w)
    return attn


# --- Model components ---

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (manual mean/var over channel dim)."""

    def __init__(self, num_channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.epsilon)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """Conv2D patch embedding: [B, 3, 768, 768] -> [B, 48, 48, 768]."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)            # [B, 768, 48, 48]
        x = x.permute(0, 2, 3, 1)   # [B, 48, 48, 768]
        return x


class Attention(nn.Module):
    """Multi-head attention with decomposed relative position embeddings."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 use_rel_pos: bool = True,
                 input_size: Tuple[int, int] = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and input_size is not None:
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor):
        B, H, W, _ = x.shape

        # Fused QKV projection
        qkv = (self.qkv(x)
               .reshape(B, H * W, 3, self.num_heads, -1)
               .permute(2, 0, 3, 1, 4))
        # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # q, k, v: [B*num_heads, H*W, head_dim]

        attn = (q * self.scale) @ k.transpose(1, 2)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = F.softmax(attn, dim=-1)

        x = (attn @ v)
        x = (x.reshape(B, self.num_heads, H, W, -1)
              .permute(0, 2, 3, 1, 4)
              .reshape(B, H, W, -1))
        x = self.proj(x)
        return x


class Block(nn.Module):
    """ViTDet transformer block with optional window attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, use_rel_pos: bool = True,
                 window_size: int = 0,
                 input_size: Tuple[int, int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    """Full ViTDet encoder with neck and downsampling convolutions."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, GRID_SIZE, GRID_SIZE, EMBED_DIM))

        self.blocks = nn.ModuleList()
        for i in range(DEPTH):
            ws = WINDOW_SIZE if i not in GLOBAL_ATTN_INDEXES else 0
            self.blocks.append(Block(
                dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=4.0,
                qkv_bias=True, use_rel_pos=True, window_size=ws,
                input_size=(GRID_SIZE, GRID_SIZE),
            ))

        self.neck = nn.Sequential(
            nn.Conv2d(EMBED_DIM, OUT_CHANS, kernel_size=1, bias=False),
            LayerNorm2d(OUT_CHANS),
            nn.Conv2d(OUT_CHANS, OUT_CHANS, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(OUT_CHANS),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

        # Mask to zero padding after norm1: ensures window blocks see same
        # zero-padded input as the original (where F.pad adds zeros AFTER norm)
        pad_mask = torch.ones(1, PADDED_SIZE, PADDED_SIZE, 1)
        pad_mask[:, GRID_SIZE:, :, :] = 0
        pad_mask[:, :, GRID_SIZE:, :] = 0
        self.register_buffer('pad_mask', pad_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)                              # [B, 48, 48, 768]
        # Pre-pad to 56x56 so window_partition (window_size=14) needs no
        # per-block padding (56 % 14 == 0). Global blocks still see 48x48.
        pos = F.pad(self.pos_embed, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))
        x = F.pad(x, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))  # [B, 56, 56, 768]
        x = x + pos
        for i, blk in enumerate(self.blocks):
            if blk.window_size == 0:
                # Global attention: needs original 48x48 spatial size
                x = x[:, :GRID_SIZE, :GRID_SIZE, :].contiguous()
                x = blk(x)
                if i < DEPTH - 1:
                    x = F.pad(x, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))
            else:
                # Inline window block with mask after norm1 to zero padding
                # (matches original where F.pad adds zeros after norm)
                shortcut = x
                x = blk.norm1(x)
                x = x * self.pad_mask
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, WINDOW_SIZE)
                x = blk.attn(x)
                x = window_unpartition(x, WINDOW_SIZE, pad_hw, (H, W))
                x = shortcut + x
                x = x + blk.mlp(blk.norm2(x))
        # Last block is global (block 11), output is already 48x48
        x = self.neck(x.permute(0, 3, 1, 2))                 # [B, 256, 48, 48]
        x = self.net_2(x)                                     # [B, 512, 24, 24]
        x = self.net_3(x)                                     # [B, 1024, 12, 12]
        return x


class VaryVitBFormula(nn.Module):
    """Top-level encoder wrapper: 1->3 channel repeat, ViT, flatten, projection."""

    def __init__(self):
        super().__init__()
        self.vision_tower_high = ImageEncoderViT()
        self.mm_projector_vary = nn.Linear(1024, 1024)

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: [B, 1, 768, 768] grayscale float32

        Returns:
            encoder_hidden_states: [B, 144, 1024]
        """
        x = pixel_values.repeat(1, 3, 1, 1)        # [B, 3, 768, 768]
        x = self.vision_tower_high(x)               # [B, 1024, 12, 12]
        x = x.flatten(2).transpose(1, 2)            # [B, 144, 1024]
        x = self.mm_projector_vary(x)               # [B, 144, 1024]
        return x


def load_encoder(weights_path=None):
    """Load encoder with converted PaddlePaddle weights.

    Args:
        weights_path: Path to encoder_weights.npz. If None, looks for
            'encoder_weights.npz' in the current directory.
    """
    if weights_path is None:
        weights_path = "encoder_weights.npz"
    model = VaryVitBFormula()
    data = np.load(weights_path)
    state_dict = model.state_dict()

    loaded = 0
    for name, param in state_dict.items():
        if name in data:
            tensor = torch.from_numpy(data[name].copy())
            if tensor.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {param.shape}, got {tensor.shape}")
            state_dict[name] = tensor
            loaded += 1
        else:
            print(f"  WARNING: {name} not found in weights file")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {loaded}/{len(state_dict)} parameters from {weights_path}")
    return model
