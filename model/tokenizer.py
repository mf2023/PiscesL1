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

from transformers import LlamaTokenizerFast


def get_tokenizer():
    """Get tokenizer with fallback to dummy tokenizer"""
    try:
        print("✅ Loading tokenizer...")
        tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        print("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("❌ Creating dummy tokenizer...")
        # Create simple tokenizer for testing
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=None,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            vocab_size=1000
        )
        return tokenizer