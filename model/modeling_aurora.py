#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import PiscesConfig
from .moe import MoELayer
from .multimodal import VisionEncoder, AudioEncoder, DocEncoder


class RMSNorm(nn.Module):
    """RMS normalization layer"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding"""
    def __init__(self, dim, max_seq_len=8192, base=1e6):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
    
    def forward(self, x, seq_len):
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """Multi-head attention with grouped-query attention"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
    
    def forward(self, x, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, t), self.rope(k, t)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MoE MLP"""
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MoELayer(cfg)
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
    
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PiscesModel(nn.Module):
    """Pisces L1 multimodal MoE model"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Core components
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden_size)
        
        # Multimodal encoders
        self.vision = VisionEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.doc = DocEncoder(cfg)
        
        # Output heads
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims)
    
    def forward(self, input_ids, images=None, audio=None, docs=None, labels=None):
        b, t = input_ids.shape
        x = self.embed(input_ids)
        
        # Add multimodal tokens
        if images is not None:
            x = torch.cat([self.vision(images), x], dim=1)
            t += 1
        if audio is not None:
            x = torch.cat([self.audio(audio), x], dim=1)
            t += 1
        if docs is not None:
            x = torch.cat([self.doc(docs), x], dim=1)
            t += 1
        
        # Create causal mask
        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        
        # Forward through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        task_logits = self.task_head(x[:, 0])
        eval_score = self.eval_head(x.mean(1))
        
        return logits, loss, task_logits, eval_score