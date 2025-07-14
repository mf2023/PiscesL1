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
import sys
import os

def check_gpu():
    """Check GPU availability and status"""
    print("🔍 GPU Status Check")
    print("=" * 50)
    
    # Check PyTorch CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Check current memory usage
            if i == 0:  # Only check first GPU
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Cached: {cached:.2f} GB")
    else:
        print("❌ No CUDA-capable GPU found")
        print("💡 Training will use CPU (slower but functional)")
    
    print("\n" + "=" * 50)
    
    # Test basic tensor operations
    print("🧪 Testing tensor operations...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations successful on {device}")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1) 