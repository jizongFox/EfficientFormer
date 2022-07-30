from unittest import TestCase

import torch

from models.efficientformer import Attention as EAttention


class TestAttention(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._input_x = torch.randn(8, 128, 16, 16)
        self._input_x_flatten = self._input_x.flatten(2).transpose(1, 2)

    def test_efficientformer_attention(self):
        attention = EAttention(dim=128, key_dim=16, num_heads=4, attn_ratio=4, resolution=16)
        attention(self._input_x_flatten)
