"""TableFormer encoder: ResNet-18 + resnet_block + TransformerEncoder."""

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from .weights import D_MODEL, N_HEADS, N_LAYERS

ENC_IMAGE_SIZE = 28
NUM_POSITIONS = ENC_IMAGE_SIZE * ENC_IMAGE_SIZE  # 784
FFN_DIM = 1024


def _resnet_block():
    """Two BasicBlocks projecting 256→512 channels (stride=1)."""
    downsample = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(512),
    )
    return nn.Sequential(
        BasicBlock(256, 512, stride=1, downsample=downsample),
        BasicBlock(512, 512, stride=1),
    )


class TableFormerEncoder(nn.Module):
    """Encoder: ResNet-18(layer0..3) → resnet_block(256→512) → TransformerEncoder."""

    def __init__(self):
        super().__init__()
        # ResNet-18 truncated at layer3 (children 0..6 inclusive)
        import torchvision

        resnet = torchvision.models.resnet18()
        modules = list(resnet.children())[:-3]  # drop layer4, avgpool, fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((ENC_IMAGE_SIZE, ENC_IMAGE_SIZE))

        # Project 256→512
        self.input_filter = _resnet_block()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS, enable_nested_tensor=False
        )

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: [1, 3, 448, 448] float32
        Returns:
            encoder_memory: [1, 784, 512] — for decoder cross-attention
            enc_out_raw: [1, 256, 28, 28] — raw ResNet features for bbox decoder
        """
        # ResNet backbone → [1, 256, 28, 28]
        x = self.resnet(pixel_values)
        x = self.adaptive_pool(x)
        enc_out_raw = x  # save before projection

        # Project 256→512 → [1, 512, 28, 28]
        x = self.input_filter(x)

        # Flatten spatial → [1, 784, 512]
        x = x.permute(0, 2, 3, 1).reshape(1, NUM_POSITIONS, D_MODEL)

        # Seq-first for TransformerEncoder → [784, 1, 512]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        # Back to batch-first → [1, 784, 512]
        encoder_memory = x.permute(1, 0, 2)

        return encoder_memory, enc_out_raw
