#!/usr/bin/env python
"""
Auto-converted from Generation_Finetuned_Model_9-18length.ipynb.
Review configuration variables below before running.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from loralib.layers import Conv1d, LoRALayer
import pkg_resources

from sequence_models.constants import MSA_ALPHABET
from sequence_models.convolutional import MaskedCausalConv1d
from evodiff.utils import Tokenizer, download_model
from evodiff.generate import generate_oaardm, generate_d3pm


USE_LORA_UPBEDDER = False
USE_LORA_BYTNET_SEQ1 = False
USE_LORA_BYTNET_SEQ2 = False
USE_LORA_DECODER = False
USE_LORA_MASKEDCONV1 = False
USE_LORA_MASKEDCONV2 = False
LORA_R = 8
LORA_ALPHA = 2
LORA_DROPOUT = 0.0


def ensure_pretrained_weights(model_name: str, local_path: str) -> None:
    if Path(local_path).exists():
        return
    print(f"Pretrained weights not found. Downloading {model_name} to {local_path} ...")
    state = download_model(model_name)
    torch.save(state, local_path)


def replace_key(key: str) -> str:
    key_replacements = {
        r'(layers\.\d+\.conv)(\.weight)': r'\1.conv\2',
        r'(layers\.\d+\.conv)(\.bias)': r'\1.conv\2',
    }
    for pattern, replacement in key_replacements.items():
        new_key = re.sub(pattern, replacement, key)
        if new_key != key:
            return new_key
    return key


class PositionFeedForwardForFinetuning(nn.Module):
    def __init__(self, d_in, d_out, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True):
        super().__init__()
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        self.conv = Conv1d(d_in, d_out, kernel_size=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionFeedForward(nn.Module):
    def __init__(self, d_in, d_out, rank=None):
        super().__init__()
        if rank is None:
            self.conv = nn.Conv1d(d_in, d_out, 1)
            self.factorized = False
        else:
            layer = nn.Linear(d_in, d_out)
            w = layer.weight.data
            self.bias = layer.bias
            u, s, v = torch.svd(w)
            s = torch.diag(s[:rank].sqrt())
            u = u[:, :rank]
            v = v.t()[:rank]
            self.u = nn.Parameter(u @ s)
            self.v = nn.Parameter(s @ v)
            self.factorized = True

    def forward(self, x):
        if self.factorized:
            w = self.u @ self.v
            return x @ w.t() + self.bias
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class ModConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True, **kwargs):
        super(ModConvLoRA, self).__init__()
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((kernel_size, in_channels, r)))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros((kernel_size, r, out_channels // self.conv.groups)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ModConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    lora_weights = self.bmm_lora_weights()
                    self.conv.weight.data -= lora_weights.view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    lora_weights = self.bmm_lora_weights()
                    self.conv.weight.data += lora_weights.view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def bmm_lora_weights(self):
        lora_weight = torch.bmm(self.lora_A, self.lora_B)
        lora_weight = torch.permute(lora_weight, (2, 1, 0))
        return lora_weight

    def forward(self, x):
        if self.r > 0 and not self.merged:
            lora_weight_combined = self.bmm_lora_weights()
            weight = self.conv.weight + lora_weight_combined * self.scaling
            return F.conv1d(x, weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return self.conv(x)


class ModConvLoRA2(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True, **kwargs):
        super(ModConvLoRA2, self).__init__()
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros((out_channels // self.conv.groups, r)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ModConvLoRA2, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias,
            )
        return self.conv(x)


class ModConv1d1(ModConvLoRA):
    def __init__(self, *args, **kwargs):
        super(ModConv1d1, self).__init__(nn.Conv1d, *args, **kwargs)


class ModConv1d2(ModConvLoRA2):
    def __init__(self, *args, **kwargs):
        super(ModConv1d2, self).__init__(nn.Conv1d, *args, **kwargs)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1dForFinetuning1(ModConv1d1):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, padding=padding, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1dForFinetuning2(ModConv1d2):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, padding=padding, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, causal=False, activation='relu', rank=None):
        super().__init__()
        if causal:
            self.conv = MaskedCausalConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        else:
            if USE_LORA_MASKEDCONV1:
                self.conv = MaskedConv1dForFinetuning1(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
            elif USE_LORA_MASKEDCONV2:
                self.conv = MaskedConv1dForFinetuning2(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
            else:
                self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)

        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        else:
            act = nn.ReLU

        seq1_cls = PositionFeedForwardForFinetuning if USE_LORA_BYTNET_SEQ1 else PositionFeedForward
        seq2_cls = PositionFeedForwardForFinetuning if USE_LORA_BYTNET_SEQ2 else PositionFeedForward

        layers1 = [nn.LayerNorm(d_in), act(), seq1_cls(d_in, d_h), nn.LayerNorm(d_h), act()]
        layers2 = [nn.LayerNorm(d_h), act(), seq2_cls(d_h, d_out)]

        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        return x + self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        if self.d_model % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={self.d_model:d})")
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.to(x.device)
        return pe[x]


class ByteNetTime(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None, padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True, timesteps=None):
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps)
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs, d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            if USE_LORA_UPBEDDER:
                self.up_embedder = PositionFeedForwardForFinetuning(d_embedding, d_model)
            else:
                self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model // 2 if slim else d_model
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank, activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1])
            e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])
            e = torch.add(e2, e)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None, padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu', tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.embedder = ByteNetTime(
            n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
            padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
            slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs, timesteps=timesteps,
        )
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            if USE_LORA_DECODER:
                self.decoder = PositionFeedForwardForFinetuning(d_model, n_tokens)
            else:
                self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model) if final_ln else nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)


class Config:
    def __init__(self, model_scale: str):
        if model_scale == '38M':
            self.d_embed = 8
            self.d_model = 1024
            self.activation = 'gelu'
            self.slim = True
            self.n_layers = 16
            self.kernel_size = 5
        else:
            self.d_embed = 8
            self.d_model = 1280
            self.activation = 'gelu'
            self.slim = False
            self.n_layers = 56
            self.kernel_size = 5
        self.r = 128
        self.diffusion_timesteps = 500
        self.n_tokens = len(MSA_ALPHABET)
        self.causal = False
        self.padding_idx = 28
        self.dropout = 0.0
        self.final_norm = True
        self.tie_weights = False
        self.rank = None
        self.n_frozen_embs = None


def get_generation_components(model_type: str, model_scale: str):
    if model_type == 'OADM':
        tokenizer = Tokenizer()
        return tokenizer, None, None, None
    timesteps = 500
    tokenizer = Tokenizer(sequences=True)
    q_prod, q_t = tokenizer.q_random_schedule(timesteps=timesteps)
    return tokenizer, q_t, q_prod, timesteps


def load_base_weights(model, model_type: str, model_scale: str, downloads_dir: Path, use_lora_flags: bool):
    if model_type == 'OADM':
        model_name = f"oaar-{model_scale}"
    else:
        model_name = f"d3pm-uniform-{model_scale}"
    saved_state_dict_path = str(downloads_dir / f"{model_name}.tar")
    ensure_pretrained_weights(model_name, saved_state_dict_path)
    state_dict = torch.load(saved_state_dict_path, map_location=torch.device('cpu'))
    model_state = state_dict["model_state_dict"]
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    if USE_LORA_UPBEDDER:
        model_state = {k.replace("embedder.up_embedder.conv.", "embedder.up_embedder.conv.conv."): v for k, v in model_state.items()}
    if USE_LORA_BYTNET_SEQ1:
        model_state = {k.replace("sequence1.2.conv.", "sequence1.2.conv.conv."): v for k, v in model_state.items()}
    if USE_LORA_BYTNET_SEQ2:
        model_state = {k.replace("sequence2.2.conv.", "sequence2.2.conv.conv."): v for k, v in model_state.items()}
    if USE_LORA_DECODER:
        model_state = {k.replace("decoder.conv.", "decoder.conv.conv."): v for k, v in model_state.items()}

    if USE_LORA_MASKEDCONV1 or USE_LORA_MASKEDCONV2:
        model_state = {replace_key(k): v for k, v in model_state.items()}

    model.load_state_dict(model_state, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sequences from a fine-tuned EvoDiff model.")
    parser.add_argument("--model-path", default="models/paper/EvoDiff_Finetuning_640M_state_dict.pt", help="Path to the fine-tuned .pt state_dict file.")
    parser.add_argument("--output-dir", default="data/sequences", help="Directory to write generated sequences.")
    parser.add_argument("--model-type", choices=["OADM", "D3PM"], default="OADM")
    parser.add_argument("--model-scale", choices=["38M", "640M"], default="640M")
    parser.add_argument("--seq-len-min", type=int, default=9)
    parser.add_argument("--seq-len-max", type=int, default=18)
    parser.add_argument("--generate-number", type=int, default=1000)
    parser.add_argument("--reweighting-term", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=2)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-lora-upembedder", action="store_true")
    parser.add_argument("--use-lora-bn-seq1", action="store_true")
    parser.add_argument("--use-lora-bn-seq2", action="store_true")
    parser.add_argument("--use-lora-decoder", action="store_true")
    parser.add_argument("--use-lora-maskedconv1", action="store_true")
    parser.add_argument("--use-lora-maskedconv2", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.use_lora_maskedconv1 and args.use_lora_maskedconv2:
        raise ValueError("use-lora-maskedconv1 and use-lora-maskedconv2 cannot both be true")

    global USE_LORA_UPBEDDER, USE_LORA_BYTNET_SEQ1, USE_LORA_BYTNET_SEQ2
    global USE_LORA_DECODER, USE_LORA_MASKEDCONV1, USE_LORA_MASKEDCONV2
    global LORA_R, LORA_ALPHA, LORA_DROPOUT

    USE_LORA_UPBEDDER = args.use_lora_upembedder
    USE_LORA_BYTNET_SEQ1 = args.use_lora_bn_seq1
    USE_LORA_BYTNET_SEQ2 = args.use_lora_bn_seq2
    USE_LORA_DECODER = args.use_lora_decoder
    USE_LORA_MASKEDCONV1 = args.use_lora_maskedconv1
    USE_LORA_MASKEDCONV2 = args.use_lora_maskedconv2
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    downloads_dir = models_dir / "evodiff"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    jst = timezone(timedelta(hours=+9))
    current_datetime = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_filename = str(Path(args.output_dir) / f"{current_datetime}_{args.seq_len_min}-{args.seq_len_max}length.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)

    config = Config(args.model_scale)
    model = ByteNetLMTime(
        n_tokens=config.n_tokens,
        d_embedding=config.d_embed,
        d_model=config.d_model,
        n_layers=config.n_layers,
        kernel_size=config.kernel_size,
        r=config.r,
        causal=config.causal,
        padding_idx=config.padding_idx,
        rank=None,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
        final_ln=config.final_norm,
        slim=config.slim,
        activation=config.activation,
        timesteps=config.diffusion_timesteps,
    )

    load_base_weights(model, args.model_type, args.model_scale, downloads_dir, True)

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)

    tokenizer, q_t, q_bar, timesteps = get_generation_components(args.model_type, args.model_scale)

    model.to(device)
    model.eval()

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence'])

    sequence_lengths = range(args.seq_len_min, args.seq_len_max + 1)
    generate_number = args.generate_number

    for seq_len in sequence_lengths:
        unique_sequences = set()
        while len(unique_sequences) < generate_number:
            batch_size = min(generate_number - len(unique_sequences), 10000)
            if args.model_type == "OADM":
                _, sequences = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device=device)
            else:
                _, sequences = generate_d3pm(model, tokenizer, q_t, q_bar, timesteps, seq_len, batch_size=batch_size, device=device)
            unique_sequences.update(sequences)
            if len(unique_sequences) >= generate_number:
                break

        with open(output_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for seq in list(unique_sequences)[:generate_number]:
                writer.writerow([seq])

    print(f"Total unique sequences generated for each length: {generate_number}")
    print(f"Saved: {output_filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
