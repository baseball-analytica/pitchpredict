#!/usr/bin/env python3
"""End-to-end test of the xLSTM algorithm through the PitchPredict API.

Uses real MLBAM player IDs so embeddings are meaningful.

Usage:
    uv run python -m tools.test_xlstm_e2e
"""

from __future__ import annotations

import asyncio

from pitchpredict.api import PitchPredict
import pitchpredict.types.api as api_types

# Real MLBAM IDs
GERRIT_COLE = 543037
AARON_JUDGE = 592450
JOSE_TREVINO = 624431  # catcher


async def test_cold_start():
    """Test prediction with no pitch history (cold start)."""
    print("=== Cold Start Test ===")
    api = PitchPredict()

    result = await api.predict_pitcher(
        pitcher_id=GERRIT_COLE,
        batter_id=AARON_JUDGE,
        prev_pitches=[],
        algorithm="xlstm",
        sample_size=5,
        pitcher_throws="R",
        batter_hits="R",
        count_balls=0,
        count_strikes=0,
        outs=0,
        inning=1,
        fielder_2_id=JOSE_TREVINO,
    )

    _print_result(result)


async def test_with_history():
    """Test prediction with one previous pitch."""
    print("\n=== With History Test ===")
    api = PitchPredict()

    prev = api_types.Pitch(
        pitch_type="FF",
        speed=97.0,
        spin_rate=2350.0,
        spin_axis=210.0,
        release_pos_x=-1.5,
        release_pos_z=6.2,
        release_extension=6.7,
        vx0=-5.0,
        vy0=-140.0,
        vz0=1.5,
        ax=-10.0,
        ay=30.0,
        az=-28.0,
        plate_pos_x=0.2,
        plate_pos_z=2.5,
        result="called_strike",
        pa_id=1,
    )

    result = await api.predict_pitcher(
        pitcher_id=GERRIT_COLE,
        batter_id=AARON_JUDGE,
        prev_pitches=[prev],
        algorithm="xlstm",
        sample_size=10,
        pitcher_throws="R",
        batter_hits="R",
        count_balls=0,
        count_strikes=1,
        outs=0,
        inning=1,
        fielder_2_id=JOSE_TREVINO,
    )

    _print_result(result)


def _print_result(result):
    print("\nPitch type probabilities:")
    probs = result.basic_pitch_data.get("pitch_type_probs", {})
    for pt, p in sorted(probs.items(), key=lambda x: -x[1]):
        if p > 0.01:
            print(f"  {pt}: {p:.1%}")

    print(f"\nSpeed: {result.basic_pitch_data.get('pitch_speed_mean', 0):.1f} mph "
          f"(std: {result.basic_pitch_data.get('pitch_speed_std', 0):.1f})")
    print(f"Plate X: {result.basic_pitch_data.get('pitch_x_mean', 0):.2f} "
          f"(std: {result.basic_pitch_data.get('pitch_x_std', 0):.2f})")
    print(f"Plate Z: {result.basic_pitch_data.get('pitch_z_mean', 0):.2f} "
          f"(std: {result.basic_pitch_data.get('pitch_z_std', 0):.2f})")

    meta = result.prediction_metadata
    print(f"\nMetadata: {meta.get('n_samples_generated', '?')} samples generated, "
          f"{meta.get('n_history_tokens', '?')} history tokens, "
          f"{meta.get('duration', '?'):.2f}s")


async def main():
    await test_cold_start()
    await test_with_history()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
