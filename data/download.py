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
import shutil
from modelscope.msdatasets import MsDataset

ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "..", "data_cache")
os.makedirs(DATA, exist_ok=True)


def save(ds, name):
    """Save dataset to local cache"""
    try:
        save_path = os.path.join(DATA, name)
        print(f"✅ Saving {name} to {save_path}...")
        ds.save_to_disk(save_path)
        print(f"✅ {name} saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {name}: {e}")
        return False


def download_datasets(max_samples_per_dataset=50000):
    """Download all datasets with size control"""
    print("✅ Starting ModelScope dataset download...")
    print(f"Max samples per dataset: {max_samples_per_dataset:,}")
    
            # Core datasets for Pisces L1 training
    datasets = [
        # === Text Datasets (Essential) ===
        ("AI-ModelScope/TinyStories", "tiny_stories", "High-quality text stories"),
        ("AI-ModelScope/alpaca-gpt4-data-zh", "alpaca_zh", "Chinese instruction data"),
        ("AI-ModelScope/alpaca-gpt4-data-en", "alpaca_en", "English instruction data"),
        
        # === Image-Text Datasets (Core) ===
        ("AI-ModelScope/coco-2014-val-30", "coco_val", "COCO validation set"),
        ("AI-ModelScope/LLaVA-Instruct-150K", "llava_instruct", "LLaVA instruction data"),
        
        # === Audio Datasets (Core) ===
        ("AI-ModelScope/audioset-mini-100", "audioset_mini", "AudioSet mini subset"),
        ("lmms-lab/AudioSetCaps_350k_converted", "audioset_caps", "Audio caption data"),
        
        # === Advanced Datasets (Optional) ===
        ("AI-ModelScope/ultrachat-200k", "ultrachat", "UltraChat dialogue data"),
        ("AI-ModelScope/ShareGPT-90k", "sharegpt", "ShareGPT dialogue data"),
        ("AI-ModelScope/coco-2014-train-100", "coco_train", "COCO training subset"),
        ("AI-ModelScope/ESC-50", "esc50", "Environmental sound classification"),
        ("AI-ModelScope/MMC4", "mmc4", "Multimodal C4 dataset"),
        ("AI-ModelScope/WebVid-10M", "webvid", "Web video dataset")
    ]
    
    success_count = 0
    for dataset_name, save_name, description in datasets:
        try:
            print(f"✅ Downloading {description} ({dataset_name})...")
            ds = MsDataset.load(dataset_name, split='train')
            original_size = len(ds)
            print(f"✅ Dataset loaded successfully, original samples: {original_size:,}")
            
            # Limit dataset size if needed
            if original_size > max_samples_per_dataset:
                print(f"✅ Limiting to {max_samples_per_dataset:,} samples...")
                ds = ds.select(range(min(max_samples_per_dataset, original_size)))
            
            if save(ds, save_name):
                success_count += 1
                
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
    
    print(f"✅ Download completed! Success: {success_count}/{len(datasets)}")
    return success_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=50000, 
                       help="Maximum samples per dataset (default: 50000)")
    args = parser.parse_args()
    
    download_datasets(args.max_samples)