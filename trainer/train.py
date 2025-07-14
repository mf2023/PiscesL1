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
import sys
import torch
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from data.dataset import PiscesDataset
from model import PiscesModel, PiscesConfig
from trainer.checkpoint import save_ckpt, load_ckpt
from transformers import get_linear_schedule_with_warmup


def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/0.5B.json")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save_dir", default="ckpt")
    p.add_argument("--resume", default="")
    p.add_argument("--dataset", default="tiny_stories", choices=[
        # Essential datasets
        "tiny_stories", "alpaca_zh", "alpaca_en",
        # Core multimodal datasets  
        "coco_val", "llava_instruct", "audioset_mini", "audioset_caps",
        # Advanced datasets
        "ultrachat", "sharegpt", "coco_train", "esc50", "mmc4", "webvid"
    ])
    return p.parse_args()


def collate_fn(batch):
    """Custom batch processing for variable length data"""
    # Text data
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    # Image data
    pixel_values = None
    if any(item["pixel_values"] is not None for item in batch):
        pixel_values = torch.cat([item["pixel_values"] for item in batch if item["pixel_values"] is not None])

    # Audio data
    audio_input = None
    if any(item["audio_input"] is not None for item in batch):
        audio_input = torch.cat([item["audio_input"] for item in batch if item["audio_input"] is not None])

    labels = input_ids.clone()  # Autoregressive labels

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "audio_input": audio_input
    }


def main():
    """Main training function"""
    print("✅ Starting training...")
    args = parse()
    cfg = PiscesConfig.from_json(args.config)
    model = PiscesModel(cfg).cuda()

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=100000
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_ckpt(args.resume, model, optimizer)

    # Load dataset
    print("✅ Loading dataset...")
    train_ds = PiscesDataset(subset=args.dataset, split="train", config=cfg)
    print(f"✅ Dataset loaded successfully, size: {len(train_ds)}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,  # Use custom batch processing
        pin_memory=True
    )
    print("✅ DataLoader created successfully")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        for step, batch in enumerate(loader):
            # Move to GPU
            device_batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            _, loss, _, _ = model(**device_batch)

            # Multi-GPU handling
            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {step} | Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"{args.save_dir}/pisces_{args.dataset}_epoch{epoch + 1}.pt"
        save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")

    print("✅ Training completed!")


if __name__ == "__main__":
    main()