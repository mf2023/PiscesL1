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

import torch
from torch import nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    """Expert routing gate for MoE"""
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x):
        logits = self.gate(x)
        scores, idx = torch.topk(logits, 2, dim=-1)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        return scores, idx


class MoELayer(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg.hidden_size, cfg.moe_num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
            ) for _ in range(cfg.moe_num_experts)
        ])
    
    def forward(self, x):
        b, t, d = x.shape
        h = x.view(-1, d)
        scores, idx = self.gate(h)
        y = torch.zeros_like(h)
        
        for i in range(2):
            mask = idx[:, i]
            y += scores[:, i:i+1] * torch.stack([self.experts[m](h) for m in mask])
        
        return y.view(b, t, d)