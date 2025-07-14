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
from transformers import CLIPImageProcessor, ASTFeatureExtractor


class VisionEncoder(nn.Module):
    """Vision encoder using CLIP"""
    def __init__(self, cfg):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
            self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.proj = nn.Linear(768, cfg.hidden_size)
            self.enabled = True
        except Exception as e:
            print(f"❌ Warning: Failed to load CLIP model: {e}")
            print("❌ Vision encoding will be disabled")
            self.enabled = False
            self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
    
    def forward(self, pixel_values):
        if not self.enabled:
            # Return zero tensor as placeholder
            return torch.zeros(pixel_values.shape[0], 1, self.proj.out_features, device=pixel_values.device)
        h = self.clip(pixel_values=pixel_values).pooler_output
        return self.proj(h).unsqueeze(1)


class AudioEncoder(nn.Module):
    """Audio encoder using AST"""
    def __init__(self, cfg):
        super().__init__()
        try:
            from transformers import ASTModel
            self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self.proj = nn.Linear(768, cfg.hidden_size)
            self.enabled = True
        except Exception as e:
            print(f"❌ Warning: Failed to load AST model: {e}")
            print("❌ Audio encoding will be disabled")
            self.enabled = False
            self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
    
    def forward(self, audio_input):
        if not self.enabled:
            # Return zero tensor as placeholder
            return torch.zeros(audio_input['input_values'].shape[0], 1, self.proj.out_features, device=audio_input['input_values'].device)
        h = self.ast(**audio_input).last_hidden_state.mean(1)
        return self.proj(h).unsqueeze(1)


class DocEncoder(nn.Module):
    """Document encoder using LayoutLMv3"""
    def __init__(self, cfg):
        super().__init__()
        try:
            from transformers import LayoutLMv3Model
            self.lm = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
            self.proj = nn.Linear(768, cfg.hidden_size)
            self.enabled = True
        except Exception as e:
            print(f"❌ Warning: Failed to load LayoutLMv3 model: {e}")
            print("❌ Document encoding will be disabled")
            self.enabled = False
            self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
    
    def forward(self, doc_input):
        if not self.enabled:
            # Return zero tensor as placeholder
            return torch.zeros(doc_input['input_ids'].shape[0], 1, self.proj.out_features, device=doc_input['input_ids'].device)
        h = self.lm(**doc_input).last_hidden_state.mean(1)
        return self.proj(h).unsqueeze(1)