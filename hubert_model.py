import copy
from typing import Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor().to(device)
        self.feature_projection = FeatureProjection().to(device)
        self.positional_embedding = PositionalConvEmbedding().to(device)
        self.norm = nn.LayerNorm(768).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        ).to(device)
        self.proj = nn.Linear(768, 256).to(device)

        self.masked_spec_embed = nn.Parameter(torch.cuda.FloatTensor(768).uniform_().to(device))
        self.label_embedding = nn.Embedding(num_label_embeddings, 256).to(device)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
        self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x.to(device))
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.encode(x.to(device))
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    @torch.jit.script
    @torch.jit.unused
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav.to(device))
        return self.proj(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False).to(device)
        self.norm0 = nn.GroupNorm(512, 512).to(device)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False).to(device)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False).to(device)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False).to(device)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False).to(device)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False).to(device)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x.to(device))))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512).to(device)
        self.projection = nn.Linear(512, 768).to(device)
        self.dropout = nn.Dropout(0.1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.to(device))
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv.to(device), name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src.to(device)
        if mask is not None:
            mask = mask.to(device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(device)
        for layer in self.layers[:output_layer]:
            output = layer(
                output.to(device), src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


def _compute_mask(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask.to(torch.bool)


def hubert_soft(path: str) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = torch.load(path, map_location=device)
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.load_state_dict(checkpoint)
    hubert.to(device)
    hubert.eval()
    return hubert
