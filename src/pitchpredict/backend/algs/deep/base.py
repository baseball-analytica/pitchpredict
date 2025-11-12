# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os
from typing import Any

import torch

from pitchpredict.backend.algs.base import PitchPredictAlgorithm


class DeepPitchPredictAlgorithm(PitchPredictAlgorithm):
    """
    A deep learning-based pitch prediction algorithm.
    """

    def __init__(
        self,
        name: str = "deep",
        use_existing: bool = True,
        model_path: str = ".pitchpredict_models/weights/deep.pth",
        # build parameters (for when use_existing is False)
        date_start: str = "2015-04-01",
        date_end: str = "2024-12-31",
        
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            **kwargs,
        )
        self.use_existing = use_existing
        self.model_path = model_path

    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        if self.use_existing:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"model path {self.model_path} does not exist")
            self.load_model()
        else:
            self.build_model()

    def build_model(self) -> None:
        """
        Build a new model from scratch using the given parameters.
        """
        raise NotImplementedError("Not implemented")

    def load_model(self) -> None:
        """
        Load a pre-trained model from the given path.
        """
        try:
            self.model = torch.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"error loading model from {self.model_path}: {e}")