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
import platform
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
TRAIN_PY   = os.path.join(ROOT_DIR, "trainer",  "train.py")
INFER_PY   = os.path.join(ROOT_DIR, "scripts", "infer_demo.py")

SYSTEM    = platform.system()            # Windows / Linux / Darwin
HAS_GPU   = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if HAS_GPU else 0

def ask(prompt, choices):
    while True:
        val = input(prompt).strip().lower()
        if val in [str(c).lower() for c in choices]:
            return val
        print("Invalid choice, please try again.")

def run_cmd(cmd):
    print(">>>", cmd)
    subprocess.call(cmd, shell=True)

def build_cmd(mode, size):
    cfg = os.path.join(CONFIG_DIR, f"{size}.json")
    if not os.path.exists(cfg):
        raise FileNotFoundError(cfg)

    py = "python" if SYSTEM == "Windows" else "python3"

    if mode == "train":
        base_args = f'"{TRAIN_PY}" --config "{cfg}"'
        if HAS_GPU and GPU_COUNT > 1:
            return f'torchrun --nproc_per_node={GPU_COUNT} {base_args}'
        elif HAS_GPU:
            return f'set CUDA_VISIBLE_DEVICES=0 && {py} {base_args}' if SYSTEM == "Windows" else \
                   f'CUDA_VISIBLE_DEVICES=0 {py} {base_args}'
        else:
            threads = os.cpu_count()
            return f'set OMP_NUM_THREADS={threads} && {py} {base_args}' if SYSTEM == "Windows" else \
                   f'OMP_NUM_THREADS={threads} {py} {base_args}'

    prompt = input("Prompt: ").strip()
    img    = input("Image path (optional, press Enter to skip): ").strip()
    base_args = f'"{INFER_PY}" --config "{cfg}" --prompt "{prompt}"'
    if img:
        base_args += f' --image "{img}"'

    if HAS_GPU:
        return f'set CUDA_VISIBLE_DEVICES=0 && {py} {base_args}' if SYSTEM == "Windows" else \
               f'CUDA_VISIBLE_DEVICES=0 {py} {base_args}'
    else:
        threads = os.cpu_count()
        return f'set OMP_NUM_THREADS={threads} && {py} {base_args}' if SYSTEM == "Windows" else \
               f'OMP_NUM_THREADS={threads} {py} {base_args}'

def main():
    print("Pisces L1 Universal Launcher")
    print("-" * 40)
    print(f"OS  : {SYSTEM}")
    print(f"GPU : {'Yes (' + str(GPU_COUNT) + ')' if HAS_GPU else 'No'}")
    print("-" * 40)

    mode = ask("Mode? [train / infer]: ", ["train", "infer"])
    size = ask("Size? [0.5B / 70B]: ", ["0.5B", "70B"])

    cmd = build_cmd(mode, size)
    run_cmd(cmd)

if __name__ == "__main__":
    main()