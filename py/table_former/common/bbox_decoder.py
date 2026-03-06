"""TableFormer bbox decoder: batched attention-gated MLP over N cells."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from .weights import D_MODEL

BBOX_ATTENTION_DIM = 512  # from checkpoint shapes
BBOX_CLASSES = 2  # num_classes (output is num_classes + 1 = 3)


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


class MLP(nn.Module):
    """Multi-layer perceptron: Linear(in→hid) → ReLU → ... → Linear(hid→out)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BBoxDecoder(nn.Module):
    """Batched bbox prediction over N cells.

    Takes raw encoder features + per-cell hidden states from the tag decoder.
    """

    def __init__(self):
        super().__init__()
        self._input_filter = _resnet_block()
        self._init_h = nn.Linear(D_MODEL, D_MODEL)

        # CellAttention components
        self._attention_encoder_att = nn.Linear(D_MODEL, BBOX_ATTENTION_DIM)
        self._attention_tag_decoder_att = nn.Linear(D_MODEL, BBOX_ATTENTION_DIM)
        self._attention_language_att = nn.Linear(D_MODEL, BBOX_ATTENTION_DIM)
        self._attention_full_att = nn.Linear(BBOX_ATTENTION_DIM, 1)

        # Gate
        self._f_beta = nn.Linear(D_MODEL, D_MODEL)

        # Output heads
        self._bbox_embed = MLP(D_MODEL, 256, 4, 3)  # Linear(512→256)→ReLU→Linear(256→256)→ReLU→Linear(256→4)
        self._class_embed = nn.Linear(D_MODEL, BBOX_CLASSES + 1)  # 3 classes

    def forward(
        self,
        enc_out_raw: torch.Tensor,
        cell_hidden_states: torch.Tensor,
    ):
        """
        Args:
            enc_out_raw: [1, 256, 28, 28] — raw ResNet features
            cell_hidden_states: [N, D_MODEL] — decoder hidden states for each cell
        Returns:
            bboxes: [N, 4] — cxcywh normalized 0-1
            classes: [N, 3] — raw logits
        """
        # Project 256→512
        x = self._input_filter(enc_out_raw)  # [1, 512, 28, 28]
        x = x.permute(0, 2, 3, 1).reshape(1, 784, D_MODEL)  # [1, 784, 512]

        n_cells = cell_hidden_states.shape[0]

        # Init hidden state from mean-pooled encoder features
        mean_enc = x.mean(dim=1)  # [1, 512]
        h = self._init_h(mean_enc)  # [1, 512]
        h = h.expand(n_cells, -1)  # [N, 512]

        # Batched attention
        # att1 from encoder features: [1, 784, att_dim] — broadcasts to [N, 784, att_dim]
        att1 = self._attention_encoder_att(x)  # [1, 784, 512]
        # att2 from cell hidden states: [N, att_dim]
        att2 = self._attention_tag_decoder_att(cell_hidden_states)  # [N, 512]
        # att3 from init hidden: [N, att_dim]
        att3 = self._attention_language_att(h)  # [N, 512]

        # Combine: [N, 784, 1]
        att = self._attention_full_att(
            F.relu(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))
        )  # [N, 784, 1]
        alpha = F.softmax(att.squeeze(2), dim=1)  # [N, 784]

        # Weighted sum via matmul: [N, 784] @ [784, 512] → [N, 512]
        awe = torch.matmul(alpha, x.squeeze(0))  # [N, 512]

        # Gate
        gate = torch.sigmoid(self._f_beta(h))  # [N, 512]
        awe = gate * awe  # [N, 512]

        # Final hidden
        h_final = awe * h  # [N, 512]

        # Predict
        bboxes = self._bbox_embed(h_final).sigmoid()  # [N, 4]
        classes = self._class_embed(h_final)  # [N, 3]

        return bboxes, classes

    def load_weights(self, bbox_sd: dict):
        """Load weights from bbox state dict with name remapping."""
        sd = {}
        for name, tensor in bbox_sd.items():
            if name.startswith("_attention._encoder_att."):
                new = name.replace("_attention._encoder_att.", "_attention_encoder_att.")
            elif name.startswith("_attention._tag_decoder_att."):
                new = name.replace("_attention._tag_decoder_att.", "_attention_tag_decoder_att.")
            elif name.startswith("_attention._language_att."):
                new = name.replace("_attention._language_att.", "_attention_language_att.")
            elif name.startswith("_attention._full_att."):
                new = name.replace("_attention._full_att.", "_attention_full_att.")
            elif name.startswith("_bbox_embed."):
                new = name
            elif name.startswith("_class_embed."):
                new = name
            elif name.startswith("_f_beta."):
                new = name
            elif name.startswith("_init_h."):
                new = name
            elif name.startswith("_input_filter."):
                new = name
            else:
                new = name
            sd[new] = tensor
        self.load_state_dict(sd)
