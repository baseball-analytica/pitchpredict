# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any

import numpy as np
from pydantic import BaseModel


class SimilarityWeights(BaseModel):
    """
    Weights for the similarity algorithm.
    Since this will be softmaxed, the weights do not need to sum to 1.
    """

    batter_id: float = 1.0
    pitcher_age: float = 0.6
    pitcher_throws: float = 0.4
    batter_age: float = 0.4
    batter_hits: float = 0.4
    count_balls: float = 0.5
    count_strikes: float = 0.5
    outs: float = 0.2
    bases_state: float = 0.3
    score_bat: float = 0.1
    score_fld: float = 0.1
    inning: float = 0.1
    pitch_number: float = 0.1
    number_through_order: float = 0.2
    game_date: float = 0.05
    fielder_2_id: float = 0.3
    fielder_3_id: float = 0.05
    fielder_4_id: float = 0.05
    fielder_5_id: float = 0.05
    fielder_6_id: float = 0.05
    fielder_7_id: float = 0.05
    fielder_8_id: float = 0.05
    fielder_9_id: float = 0.05
    batter_days_since_prev_game: float = 0.05
    pitcher_days_since_prev_game: float = 0.05
    strike_zone_top: float = 0.1
    strike_zone_bottom: float = 0.1

    def softmax(self) -> dict[str, float]:
        """
        Apply softmax to the weights and return a dictionary of said weights.
        """
        weights = self.model_dump()
        weights = {k: np.exp(v) for k, v in weights.items()}
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        return weights
