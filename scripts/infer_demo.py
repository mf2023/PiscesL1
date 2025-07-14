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
import argparse
from PIL import Image
from model import PiscesModel, PiscesConfig
from transformers import LlamaTokenizerFast


def main():
    """Inference demo for Pisces L1 model"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--prompt", required=True, help="Input prompt")
    ap.add_argument("--image", help="Optional image path")
    args = ap.parse_args()
    
    # Load model
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    model = PiscesModel(cfg).eval().cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    
    # Load tokenizer
    tok = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    ids = tok.encode(args.prompt, return_tensors="pt").cuda()
    
    # Process image if provided
    images = None
    if args.image:
        from torchvision.transforms import functional as TF
        img = Image.open(args.image).convert("RGB").resize((224, 224))
        images = TF.to_tensor(img).unsqueeze(0).cuda()
    
    # Generate
    with torch.no_grad():
        logits, _, _, _ = model(ids, images=images)
    
    print(tok.decode(logits.argmax(-1)[0]))


if __name__ == "__main__":
    main()