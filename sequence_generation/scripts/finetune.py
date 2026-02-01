#!/usr/bin/env python
"""
Auto-converted from EvoDiff_Finetuning_v5.ipynb.
Review configuration variables below before running.
"""

from __future__ import annotations

import os
import sys
import math
import csv
import re
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.request
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from loralib.layers import Conv1d, LoRALayer
from sequence_models.constants import MSA_ALPHABET
from sequence_models.convolutional import MaskedConv1d as SM_MaskedConv1d, MaskedCausalConv1d
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sequence_models.metrics import MaskedAccuracy
import pkg_resources
from evodiff.utils import Tokenizer, download_model
from evodiff.collaters import OAMaskCollater, D3PMCollater
from evodiff.generate import generate_oaardm, generate_d3pm
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss

def download_file(url: str, dest: str) -> None:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    urllib.request.urlretrieve(url, str(dest_path))

def load_cpp_dataframe(cellppd_dir: Path, pd) -> "pd.DataFrame":
    cpp_df_1 = pd.read_csv(cellppd_dir / "cpp_data_1.txt", header=None)
    cpp_df_2 = pd.read_csv(cellppd_dir / "cpp_data_2.txt", header=None)
    cpp_df_3 = pd.read_csv(cellppd_dir / "cpp_data_3.txt", header=None)
    cpp_df = pd.concat([cpp_df_1, cpp_df_2, cpp_df_3], axis=0).reset_index(drop=True)
    cpp_df = pd.DataFrame(cpp_df[0].str.upper())
    cpp_df.columns = ['sequence']
    return cpp_df

def run_training_loop(trainer, model, dl_train, dl_valid, optimizer, scheduler, epochs, scaler):
    train_losses = []
    valid_losses = []
    for e in range(epochs):
        train_loss, _, _ = trainer(model, dl_train, optimizer, scheduler, epoch_num=e+1, train=True, scaler=scaler)
        train_losses.append(train_loss)
        valid_loss, _, _ = trainer(model, dl_valid, None, None, epoch_num=e+1, train=False)
        valid_losses.append(valid_loss)
    return train_losses, valid_losses

def generate_sequences_and_save(model, tokenizer, model_type, seq_len, generate_num, output_csv, device, Q=None, Q_bar=None, timesteps=None):
    model.eval()
    if model_type == "OADM":
        _, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=generate_num, device=device)
    else:
        _, generated_sequence = generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=generate_num, device=device)
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence'])
        for seq in generated_sequence:
            writer.writerow([seq])

def save_qc(qc_path, graph_path, start_time, model_type, reweighting_term, lora_r, lora_alpha, lora_dropout, train_losses, valid_losses, state_dict_keys):
    jst = timezone(timedelta(hours=+9))
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, valid_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    execution_time = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")
    with open(qc_path, 'w') as file:
        file.write(f"File Name: {os.path.basename(qc_path)}\n")
        file.write(f"\nStart Time: {start_time}\n")
        file.write(f"End Time: {execution_time}\n")
        file.write(f"\nAlgorithm : {model_type}\n")
        if model_type == "D3PM":
            file.write(f"_lamda:{reweighting_term} ==> loss = (lvb_loss + ({reweighting_term} * ce_loss)) * n_tokens")
        file.write(f"\nLoRA_r: {lora_r}\n")
        file.write(f"LoRA_alpha: {lora_alpha}\n")
        file.write(f"LoRA_dropout: {lora_dropout}\n")
        file.write(f"\nTraining and Validation Losses:\n")
        file.write(f"  - Train Losses: {train_losses}\n")
        file.write(f"  - Valid Losses: {valid_losses}\n")
        file.write("State Dict Keys:\n")
        for key in state_dict_keys:
            file.write(f"  - {key}\n")

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    downloads_dir = models_dir / "evodiff"
    results_dir = repo_root / "results"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        return v.lower() in ("1", "true", "yes", "y")

    parser = argparse.ArgumentParser(description="Finetune EvoDiff with LoRA.")
    parser.add_argument("--tag", default="EvoDiff_Finetuning_640M")
    parser.add_argument("--experiment-index", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--generate-num", type=int, default=10000)
    parser.add_argument("--model-type", choices=["OADM", "D3PM"], default="OADM")
    parser.add_argument("--reweighting-term", type=float, default=0.0)
    parser.add_argument("--use-lora-upembedder", type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq1", type=str2bool, default=False)
    parser.add_argument("--use-lora-bn-seq2", type=str2bool, default=False)
    parser.add_argument("--use-lora-decoder", type=str2bool, default=False)
    parser.add_argument("--use-lora-maskedconv1", type=str2bool, default=False)
    parser.add_argument("--use-lora-maskedconv2", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=2)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--initial-lr", type=float, default=1e-7)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=20)
    args = parser.parse_args()
    return run_pipeline(args, repo_root, downloads_dir, results_dir)



# Globals used by module-level helpers (set in run_pipeline)
USE_LORA_UPBEDDER = False
USE_LORA_BYTNET_SEQ1 = False
USE_LORA_BYTNET_SEQ2 = False
USE_LORA_DECODER = False
USE_LORA_MASKEDCONV1 = False
USE_LORA_MASKEDCONV2 = False
LORA_R = 8
LORA_ALPHA = 2
LORA_DROPOUT = 0.0
REWEIGHTING_TERM = 0.0
DEVICE = None

class PositionFeedForward_for_finetuning(nn.Module):
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
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)



class mod_ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True, **kwargs):
        super(mod_ConvLoRA, self).__init__()
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((kernel_size , in_channels, r))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((kernel_size , r, out_channels//self.conv.groups))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True): 
        super(mod_ConvLoRA, self).train(mode)
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




class mod_ConvLoRA2(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True, **kwargs):
        super(mod_ConvLoRA2, self).__init__()
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r , in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(mod_ConvLoRA2, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class mod_Conv1d_1(mod_ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(mod_Conv1d_1, self).__init__(nn.Conv1d, *args, **kwargs)


class mod_Conv1d_2(mod_ConvLoRA2):
    def __init__(self, *args, **kwargs):
        super(mod_Conv1d_2, self).__init__(nn.Conv1d, *args, **kwargs)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):

        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1d_for_finetuning_1(mod_Conv1d_1):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True):

        padding = dilation * (kernel_size - 1) // 2
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                               groups=groups, bias=bias, padding=padding, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)



class MaskedConv1d_for_finetuning_2(mod_Conv1d_2):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True, r=None, lora_alpha=None, lora_dropout=None, merge_weights=True):

        padding = dilation * (kernel_size - 1) // 2
        if r is None:
            r = LORA_R
        if lora_alpha is None:
            lora_alpha = LORA_ALPHA
        if lora_dropout is None:
            lora_dropout = LORA_DROPOUT
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                               groups=groups, bias=bias, padding=padding, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

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
                    self.conv = MaskedConv1d_for_finetuning_1(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
            elif (not USE_LORA_MASKEDCONV1) and USE_LORA_MASKEDCONV2:
                    self.conv = MaskedConv1d_for_finetuning_2(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
            else:
                    self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)

        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU

        PositionFeedForward_seq1 = PositionFeedForward_for_finetuning if USE_LORA_BYTNET_SEQ1 else PositionFeedForward
        PositionFeedForward_seq2 = PositionFeedForward_for_finetuning if USE_LORA_BYTNET_SEQ2 else PositionFeedForward

        layers1 = [nn.LayerNorm(d_in),act(),PositionFeedForward_seq1(d_in, d_h), nn.LayerNorm(d_h),act()]
        layers2 = [nn.LayerNorm(d_h),act(),PositionFeedForward_seq2(d_h, d_out), ]

        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x]



class ByteNetTime(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=True,
                 timesteps=None):

        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps)
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()

        if down_embed:
            if USE_LORA_UPBEDDER:
                self.up_embedder = PositionFeedForward_for_finetuning(d_embedding, d_model)
            else:
                self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            # expand dim of e2 to match e1
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

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True, timesteps=None):
        super().__init__()
        self.embedder = ByteNetTime(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
                if USE_LORA_DECODER:
                    self.decoder = PositionFeedForward_for_finetuning(d_model, n_tokens)
                else:
                    self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)




class Config:
    def __init__(self):
        self.experiment = "pretrain"
        self.task = "mlm"
        self.dataset = "uniref50"
        self.d_embed = 8
        self.d_model = 1280
        self.activation = "gelu"
        self.slim = False
        self.epochs = 100
        self.n_layers = 56
        self.kernel_size = 5
        self.r = 128
        self.max_tokens = 6000
        self.max_batch_size = 800
        self.bucket_size = 1000
        self.opt_level = "O2"
        self.lr = 1e-4
        self.warmup_steps = 16000
        self.train_steps = 8000
        self.diffusion_timesteps = 500
        self.accumulate = 1
        self.n_tokens = len(MSA_ALPHABET)
        self.causal = False
        self.padding_idx = 28
        self.dropout = 0.0
        self.final_norm = True
        self.tie_weights = False
        self.rank = None
        self.n_frozen_embs = None


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


def load_local_model_checkpoint(download_weight_path: str, device: torch.device):
    return torch.load(download_weight_path, map_location=device)


def load_sequence_checkpoint(model_name: str, config_path: str, diffusion_timesteps, tokenizer, downloads_dir: Path, device: torch.device):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if model_name == "oaar-38M":
        local_path = str(downloads_dir / 'oaar-640M.tar')
        ensure_pretrained_weights("oaar-640M", local_path)
        _ = load_local_model_checkpoint(download_weight_path=local_path, device=device)
    elif model_name == "d3pm-uniform-38M":
        local_path = str(downloads_dir / 'd3pm-uniform-640M.tar')
        ensure_pretrained_weights("d3pm-uniform-640M", local_path)
        _ = load_local_model_checkpoint(download_weight_path=local_path, device=device)
    else:
        local_path = str(downloads_dir / f"{model_name}.tar")
        ensure_pretrained_weights(model_name, local_path)
        _ = load_local_model_checkpoint(download_weight_path=local_path, device=device)
    return tokenizer


def OA_DM_640M(downloads_dir: Path, device: torch.device):
    tokenizer = Tokenizer()
    collater = OAMaskCollater(tokenizer=tokenizer)
    file_path = pkg_resources.resource_filename('config', 'config640M.json')
    tokenizer = load_sequence_checkpoint("oaar-640M", file_path, diffusion_timesteps=None, tokenizer=tokenizer, downloads_dir=downloads_dir, device=device)
    scheme = 'mask'
    return collater, tokenizer, scheme


def D3PM_UNIFORM_640M(downloads_dir: Path, device: torch.device, return_all=False):
    dt = 500
    tokenizer = Tokenizer(sequences=True)
    Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=dt)
    collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=dt, Q=Q_t, Q_bar=Q_prod)
    file_path = pkg_resources.resource_filename('config', 'config640M.json')
    tokenizer = load_sequence_checkpoint("d3pm-uniform-640M", file_path, diffusion_timesteps=dt, tokenizer=tokenizer, downloads_dir=downloads_dir, device=device)
    scheme = 'd3pm'
    if return_all:
        return collater, tokenizer, scheme, dt, Q_prod, Q_t
    return collater, tokenizer, scheme

class CPPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len=2048):
        self.sequences = df['sequence'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        consensus = self.sequences[idx]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return (consensus,)

def lr_schedule(epoch, initial_lr, max_lr, ramp_up_epochs, hold_epochs, gamma):
    if epoch < ramp_up_epochs:
        return (max_lr / initial_lr) ** (epoch / ramp_up_epochs)
    if epoch < hold_epochs:
        return max_lr / initial_lr
    return max_lr / initial_lr * (gamma ** (epoch - hold_epochs))



def epoch_oadm(model, dataloader, optimizer, scheduler, epoch_num, train=True, scaler=None, device=None, padding_idx=None, loss_func=None, accu_func=None):

    total_loss = 0.0
    total_nll_loss = 0.0
    total_accu = 0.0
    total_tokens = 0
    total_seqs = 0

    if train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(dataloader, 1):
        if train:
            optimizer.zero_grad()

        with autocast(dtype=torch.float32):
            src, timestep, tgt, mask = batch
            mask = mask.to(device)
            timestep = timestep.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            input_mask = (src != padding_idx).float()
            n_tokens = input_mask.sum()
            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))
            ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)
            loss = ce_loss
            accu = accu_func(outputs, tgt, mask) * n_tokens

        total_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_accu += accu.item()
        total_tokens += n_tokens.item()
        total_seqs += len(src)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    avg_loss = total_loss / idx
    avg_nll_loss = total_nll_loss / idx
    avg_accu = total_accu / total_tokens

    if train:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_num} Train Summary - Loss: {avg_loss:.4f}, NLL Loss: {avg_nll_loss:.4f}, Accuracy: {avg_accu:.4f}, Current Learning Rate: {current_lr:.4e}")
    else:
        print(f"Epoch {epoch_num} Valid Summary - Loss: {avg_loss:.4f}, NLL Loss: {avg_nll_loss:.4f}, Accuracy: {avg_accu:.4f}")

    return avg_loss, avg_nll_loss, avg_accu


def epoch_d3pm(model, dataloader, optimizer, epoch_num, train=True, scaler=None, device=None, padding_idx=None, loss_func1=None, loss_func2=None, reweighting_term=0.0, accu_func=None):

    total_loss = 0.0
    total_nll_loss = 0.0
    total_accu = 0.0
    total_tokens = 0
    total_seqs = 0

    if train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(dataloader, 1):
        if train:
            optimizer.zero_grad()

        with autocast(dtype=torch.float32):
            src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
            q = q.to(device)
            Q = Q.to(device)
            Q_bar = Q_bar.to(device)
            src_onehot = src_onehot.to(device)
            tgt_onehot = tgt_onehot.to(device)

            timestep = timestep.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            input_mask = (src != padding_idx).float()
            n_tokens = input_mask.sum()
            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))

            lvb_loss = loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar)
            ce_loss = loss_func2(outputs, tgt, input_mask)
            lvb_loss = lvb_loss.to(torch.float32)
            ce_loss = ce_loss.to(torch.float32)
            loss = (lvb_loss + (reweighting_term * ce_loss)) * n_tokens
            nll_loss = ce_loss * n_tokens
            accu = accu_func(outputs, tgt, input_mask) * n_tokens

        total_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_accu += accu.item()
        total_tokens += n_tokens.item()
        total_seqs += len(src)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    avg_loss = total_loss / idx
    avg_nll_loss = total_nll_loss / idx
    avg_accu = total_accu / total_tokens

    print(f"Epoch {epoch_num} Summary - Loss: {avg_loss:.4f}, NLL Loss: {avg_nll_loss:.4f}, Accuracy: {avg_accu:.4f}")

    return avg_loss, avg_nll_loss, avg_accu




def epoch_d3pm_warmup(model, dataloader, optimizer, scheduler, epoch_num, train=True, scaler=None, device=None, padding_idx=None, loss_func1=None, loss_func2=None, reweighting_term=0.0, accu_func=None):

    total_loss = 0.0
    total_nll_loss = 0.0
    total_accu = 0.0
    total_tokens = 0
    total_seqs = 0

    if train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(dataloader, 1):
        if train:
            optimizer.zero_grad()

        with autocast(dtype=torch.float32):
            src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
            q = q.to(device)
            Q = Q.to(device)
            Q_bar = Q_bar.to(device)
            src_onehot = src_onehot.to(device)
            tgt_onehot = tgt_onehot.to(device)

            timestep = timestep.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            input_mask = (src != padding_idx).float()
            n_tokens = input_mask.sum()

            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))

            lvb_loss = loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar)
            ce_loss = loss_func2(outputs, tgt, input_mask)
            lvb_loss = lvb_loss.to(torch.float32)
            ce_loss = ce_loss.to(torch.float32)
            loss = (lvb_loss + (reweighting_term * ce_loss)) * n_tokens
            nll_loss = ce_loss * n_tokens
            accu = accu_func(outputs, tgt, input_mask) * n_tokens

        total_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_accu += accu.item()
        total_tokens += n_tokens.item()
        total_seqs += len(src)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    avg_loss = total_loss / idx
    avg_nll_loss = total_nll_loss / idx
    avg_accu = total_accu / total_tokens

    if train:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_num} Train Summary - Loss: {avg_loss:.4f}, NLL Loss: {avg_nll_loss:.4f}, Accuracy: {avg_accu:.4f}, Current Learning Rate: {current_lr:.4e}")
    else:
        print(f"Epoch {epoch_num} Valid Summary - Loss: {avg_loss:.4f}, NLL Loss: {avg_nll_loss:.4f}, Accuracy: {avg_accu:.4f}")

    return avg_loss, avg_nll_loss, avg_accu


def display_trainable_parameters_with_count(model: nn.Module) -> None:
    trainable_params = []
    non_trainable_params = []
    total_trainable_params = 0
    total_non_trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params.append((name, num_params))
            total_trainable_params += num_params
        else:
            non_trainable_params.append((name, num_params))
            total_non_trainable_params += num_params

    print("Trainable Parameters:")
    for name, count in trainable_params:
        print(f"  - {name} : {count}")

    print(f"\nTotal Trainable Parameters: {total_trainable_params}")
    print(f"Total Non-Trainable Parameters: {total_non_trainable_params}")


def count_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_state_dict_keys(model: nn.Module):
    return list(model.state_dict().keys())


def run_pipeline(args, repo_root, downloads_dir, results_dir):
    global USE_LORA_UPBEDDER, USE_LORA_BYTNET_SEQ1, USE_LORA_BYTNET_SEQ2
    global USE_LORA_DECODER, USE_LORA_MASKEDCONV1, USE_LORA_MASKEDCONV2
    global LORA_R, LORA_ALPHA, LORA_DROPOUT, REWEIGHTING_TERM, DEVICE

    jst = timezone(timedelta(hours=+9))
    start_time = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = device
    print("device :", device)
    tag = args.tag
    experiment_index = args.experiment_index

    epochs = args.epochs
    generate_num = args.generate_num
    model_type = args.model_type

    reweighting_term = args.reweighting_term
    use_lora_upembedder = args.use_lora_upembedder
    use_lora_ByteNetBlock_seq1 = args.use_lora_bn_seq1
    use_lora_ByteNetBlock_seq2 = args.use_lora_bn_seq2
    use_lora_decoder = args.use_lora_decoder

    use_lora_MaskedConv1d_No1 = args.use_lora_maskedconv1
    use_lora_MaskedConv1d_No2 = args.use_lora_maskedconv2

    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout

    REWEIGHTING_TERM = reweighting_term
    USE_LORA_UPBEDDER = use_lora_upembedder
    USE_LORA_BYTNET_SEQ1 = use_lora_ByteNetBlock_seq1
    USE_LORA_BYTNET_SEQ2 = use_lora_ByteNetBlock_seq2
    USE_LORA_DECODER = use_lora_decoder
    USE_LORA_MASKEDCONV1 = use_lora_MaskedConv1d_No1
    USE_LORA_MASKEDCONV2 = use_lora_MaskedConv1d_No2
    LORA_R = lora_r
    LORA_ALPHA = lora_alpha
    LORA_DROPOUT = lora_dropout

    """
        model = ByteNetLMTime(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                          causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=args.dropout,
                          tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation,
                          timesteps=diffusion_timesteps)
    """





    config = Config()

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
        timesteps=config.diffusion_timesteps
    )
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key)
    downloaded_file_path = str(downloads_dir) + "/"

    if model_type == "OADM":
        saved_state_dict_path = downloaded_file_path + 'oaar-640M.tar'
        ensure_pretrained_weights("oaar-640M", saved_state_dict_path)
    if model_type == "D3PM":
        saved_state_dict_path = downloaded_file_path + 'd3pm-uniform-640M.tar'
        ensure_pretrained_weights("d3pm-uniform-640M", saved_state_dict_path)

    state_dict = torch.load(saved_state_dict_path, map_location=torch.device('cpu'))
    model_state = state_dict["model_state_dict"]
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}



    if use_lora_upembedder:
        model_state = {k.replace("embedder.up_embedder.conv.", "embedder.up_embedder.conv.conv."): v for k, v in model_state.items()}
    if use_lora_ByteNetBlock_seq1:
        model_state = {k.replace("sequence1.2.conv.", "sequence1.2.conv.conv."): v for k, v in model_state.items()}
    if use_lora_ByteNetBlock_seq2:
        model_state = {k.replace("sequence2.2.conv.", "sequence2.2.conv.conv."): v for k, v in model_state.items()}
    if use_lora_decoder:
        model_state = {k.replace("decoder.conv.", "decoder.conv.conv."): v for k, v in model_state.items()}

    if use_lora_MaskedConv1d_No1 or use_lora_MaskedConv1d_No2:
        model_state = {replace_key(k): v for k, v in model_state.items()}

    model.load_state_dict(model_state, strict=False)

    if model_type == "OADM":
        collater, tokenizer, scheme = OA_DM_640M(downloads_dir, device)
    elif model_type == "D3PM":
        collater, tokenizer, scheme, timesteps, Q_bar, Q = D3PM_UNIFORM_640M(downloads_dir, device, return_all=True)
    cellppd_dir = Path("data/cellppd")
    cpp_df = load_cpp_dataframe(cellppd_dir, pd)
    train_df, valid_df = train_test_split(cpp_df, test_size=0.1, random_state=42)
    ds_train = CPPDataset(train_df)
    ds_valid = CPPDataset(valid_df)
    batch_size = args.batch_size
    dl_train = DataLoader(dataset=ds_train, shuffle=True, batch_size=batch_size, num_workers=4, collate_fn=collater)
    dl_valid = DataLoader(dataset=ds_valid, shuffle=False, batch_size=batch_size, num_workers=4, collate_fn=collater)


    if model_type == "OADM":
        loss_func = OAMaskedCrossEntropyLoss(reweight=True)
    if model_type == "D3PM":
        loss_func1 = D3PMLVBLoss(tmax=config.diffusion_timesteps, tokenizer=tokenizer)
        loss_func2 = D3PMCELoss(tokenizer=tokenizer)

    accu_func = MaskedAccuracy()

    initial_lr = args.initial_lr
    max_lr = args.max_lr
    ramp_up_epochs = epochs / 10
    hold_epochs = epochs / 2
    final_lr = args.final_lr
    decay_epochs = epochs - hold_epochs
    gamma = (final_lr / max_lr) ** (1 / decay_epochs)

    weight_decay = args.weight_decay

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: lr_schedule(e, initial_lr, max_lr, ramp_up_epochs, hold_epochs, gamma))

    padding_idx = tokenizer.pad_id
    masking_idx = tokenizer.mask_id
    print('Using {} as padding index'.format(padding_idx))
    print('Using {} as masking index'.format(masking_idx))
    
    scaler = GradScaler()

    lora.mark_only_lora_as_trainable(model)
    display_trainable_parameters_with_count(model)
    model.to(device)

    train_losses = []
    valid_losses = []

    if model_type == "OADM":
        trainer = partial(
            epoch_oadm,
            device=device,
            padding_idx=padding_idx,
            loss_func=loss_func,
            accu_func=accu_func,
        )
    elif model_type == "D3PM":
        trainer = partial(
            epoch_d3pm_warmup,
            device=device,
            padding_idx=padding_idx,
            loss_func1=loss_func1,
            loss_func2=loss_func2,
            reweighting_term=reweighting_term,
            accu_func=accu_func,
        )



    train_losses, valid_losses = run_training_loop(
        trainer=trainer,
        model=model,
        dl_train=dl_train,
        dl_valid=dl_valid,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        scaler=scaler,
    )

    seq_len = args.seq_len
    current_date_jst = datetime.now(jst).strftime("%Y%m%d")

    base_dir = str(results_dir)

    base_file_name = f"{current_date_jst}_{tag}_run{experiment_index}"

    experiment_dir = f"{base_dir}/{current_date_jst}_{tag}_run{experiment_index}"
    os.makedirs(experiment_dir, exist_ok=True)

    state_dict_path = os.path.join(experiment_dir, f"{base_file_name}_state_dict.pt")
    model_save_path = os.path.join(experiment_dir, f"{base_file_name}_model.pth")
    csv_path = os.path.join(experiment_dir, f"{base_file_name}_generation.csv")
    qc_path = os.path.join(experiment_dir, f"{base_file_name}_QC.txt")
    graph_path = os.path.join(experiment_dir, f"{base_file_name}_QC_graph.png")

    torch.save(lora.lora_state_dict(model), state_dict_path)

    generate_sequences_and_save(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        seq_len=seq_len,
        generate_num=generate_num,
        output_csv=csv_path,
        device=device,
        Q=Q if model_type == "D3PM" else None,
        Q_bar=Q_bar if model_type == "D3PM" else None,
        timesteps=timesteps if model_type == "D3PM" else None,
    )

    state_dict_keys = get_state_dict_keys(model)
    save_qc(
        qc_path=qc_path,
        graph_path=graph_path,
        start_time=start_time,
        model_type=model_type,
        reweighting_term=reweighting_term,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_losses=train_losses,
        valid_losses=valid_losses,
        state_dict_keys=state_dict_keys,
    )



if __name__ == "__main__":
    sys.exit(main())
