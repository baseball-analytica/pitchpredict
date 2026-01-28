# SPDX-License-Identifier: MIT

import torch

from pitchpredict.backend.algs.xlstm.predictor import create_grammar_mask
from pitchpredict.backend.algs.xlstm.tokens import (
    PitchToken,
    TokenCategory,
    get_tokens_in_category,
    VOCAB_SIZE,
)


def test_create_grammar_mask_forced_category() -> None:
    device = torch.device("cpu")
    mask = create_grammar_mask(
        PitchToken.RESULT_IS_BALL.value,
        VOCAB_SIZE,
        device,
        force_category=TokenCategory.PITCH_TYPE,
    )

    assert not mask[PitchToken.PA_END.value]
    assert not mask[PitchToken.SPEED_IS_65.value]
    for token in get_tokens_in_category(TokenCategory.PITCH_TYPE):
        assert mask[token.value]
