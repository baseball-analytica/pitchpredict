# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import torch


class PitchPredictDeepNN(torch.nn.RNN):
    """
    A deep neural network for pitch prediction.
    """

    def __init__(self) -> None:
        super().__init__()