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

import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from model.tokenizer import get_tokenizer
from modelscope.msdatasets import MsDataset
from model.multimodal import VisionEncoder, AudioEncoder


class PiscesDataset(Dataset):
    """Pisces dataset with multimodal support"""
    def __init__(self, subset="tiny", split="train", config=None):
        # Try loading from local cache first
        cache_path = os.path.join("data_cache", subset)
        
        try:
            if os.path.exists(cache_path):
                print(f"✅ Loading dataset from local cache: {cache_path}")
                self.ds = load_from_disk(cache_path)
                if split == "train" and "train" in self.ds:
                    self.ds = self.ds["train"]
                elif split == "test" and "test" in self.ds:
                    self.ds = self.ds["test"]
                print(f"✅ Local dataset loaded successfully: {len(self.ds)} samples")
            else:
                print(f"❌ Local cache not found, trying online download: {subset}")
                msds = MsDataset.load(subset, split=split)
                if hasattr(msds, 'to_hf_dataset'):
                    self.ds = msds.to_hf_dataset()
                else:
                    self.ds = msds
                print(f"✅ Online dataset loaded successfully: {len(self.ds)} samples")
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            print("❌ Creating test dataset...")
            # Create simple test dataset
            self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        
        self.tokenizer = get_tokenizer()
        self.config = config

        # Initialize preprocessors
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        text = item.get("text", "")
        input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]

        pixel_values = None
        if "image" in item and self.vision_encoder:
            try:
                pixel_values = self.vision_encoder.process_image(item["image"])
            except Exception as e:
                print(f"❌ Image processing error: {e}")

        audio_input = None
        if "audio" in item and self.audio_encoder:
            try:
                audio_input = self.audio_encoder.process_audio(item["audio"])
            except Exception as e:
                print(f"❌ Audio processing error: {e}")

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input
        }