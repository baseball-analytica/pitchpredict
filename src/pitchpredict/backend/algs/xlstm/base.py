# SPDX-License-Identifier: MIT
"""xLSTM algorithm for pitch prediction."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

import torch
from fastapi import HTTPException

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.algs.xlstm.checkpoint import load_model
from pitchpredict.backend.algs.xlstm.sequence import (
    ContextDefaults,
    build_history_sequence,
    contexts_to_packed,
)
from pitchpredict.backend.algs.xlstm.predictor import (
    GenerationConfig,
    generate_pitches,
)
from pitchpredict.backend.algs.xlstm.decoding import aggregate_pitch_stats
from pitchpredict.backend.algs.xlstm.tokens import PitchToken, TokenCategory
from pitchpredict.backend.algs.xlstm.model import BaseballxLSTM, ModelConfig
import pitchpredict.types.api as api_types


class XlstmAlgorithm(PitchPredictAlgorithm):
    """xLSTM-based pitch prediction algorithm using autoregressive generation."""

    def __init__(
        self,
        name: str = "xlstm",
        checkpoint_path: str | None = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the xLSTM algorithm.

        Args:
            name: Algorithm instance name
            checkpoint_path: Path to checkpoint file, or None to download from HuggingFace
            device: Device to run inference on (default: cuda if available, else cpu)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(name, **kwargs)
        self.logger = logging.getLogger("pitchpredict.backend.algs.xlstm")
        self._checkpoint_path = checkpoint_path
        self._device_arg = device
        self._model: BaseballxLSTM | None = None
        self._config: ModelConfig | None = None
        self._device: torch.device | None = None

    def _ensure_model_loaded(self) -> tuple[BaseballxLSTM, torch.device]:
        """Lazily load the model on first use."""
        if self._model is None:
            self.logger.info("Loading xLSTM model...")
            self._model, self._config = load_model(
                checkpoint_path=self._checkpoint_path,
                device=self._device_arg,
            )
            self._device = next(self._model.parameters()).device
            self.logger.info(f"xLSTM model loaded on {self._device}")
        return self._model, self._device

    def _validate_xlstm_request(
        self,
        request: api_types.PredictPitcherRequest,
    ) -> None:
        """Validate request for xLSTM-specific requirements."""
        if request.prev_pitches is None:
            raise HTTPException(
                status_code=400,
                detail="prev_pitches is required for xLSTM algorithm (empty list allowed for cold-start)",
            )
        missing_pa = [
            idx
            for idx, pitch in enumerate(request.prev_pitches)
            if pitch.pa_id is None or pitch.pa_id <= 0
        ]
        if missing_pa:
            raise HTTPException(
                status_code=400,
                detail=f"pa_id is required for all prev_pitches (invalid at indices: {missing_pa})",
            )

    def _build_context_defaults(
        self,
        request: api_types.PredictPitcherRequest,
    ) -> ContextDefaults:
        """Build context defaults from request."""
        def _default(value: Any, fallback: Any) -> Any:
            return fallback if value is None else value

        game_year = 2024
        if request.game_date is not None:
            year_str = request.game_date[:4]
            if len(year_str) != 4 or not year_str.isdigit():
                raise HTTPException(
                    status_code=400,
                    detail="game_date must start with a 4-digit year (YYYY or YYYY-MM-DD)",
                )
            game_year = int(year_str)

        return ContextDefaults(
            pitcher_id=request.pitcher_id,
            batter_id=request.batter_id,
            pitcher_age=_default(request.pitcher_age, 28),
            pitcher_throws=_default(request.pitcher_throws, "R"),
            batter_age=_default(request.batter_age, 28),
            batter_hits=_default(request.batter_hits, "R"),
            count_balls=_default(request.count_balls, 0),
            count_strikes=_default(request.count_strikes, 0),
            outs=_default(request.outs, 0),
            bases_state=_default(request.bases_state, 0),
            score_bat=_default(request.score_bat, 0),
            score_fld=_default(request.score_fld, 0),
            inning=_default(request.inning, 1),
            pitch_number=_default(request.pitch_number, 1),
            number_through_order=_default(request.number_through_order, 1),
            game_date=game_year,
            fielder_2_id=_default(request.fielder_2_id, 0),
            fielder_3_id=_default(request.fielder_3_id, 0),
            fielder_4_id=_default(request.fielder_4_id, 0),
            fielder_5_id=_default(request.fielder_5_id, 0),
            fielder_6_id=_default(request.fielder_6_id, 0),
            fielder_7_id=_default(request.fielder_7_id, 0),
            fielder_8_id=_default(request.fielder_8_id, 0),
            fielder_9_id=_default(request.fielder_9_id, 0),
            batter_days_since_prev_game=_default(request.batter_days_since_prev_game, 1),
            pitcher_days_since_prev_game=_default(request.pitcher_days_since_prev_game, 1),
            strike_zone_top=_default(request.strike_zone_top, 3.5),
            strike_zone_bottom=_default(request.strike_zone_bottom, 1.5),
        )

    def _convert_pitches_to_dicts(
        self,
        pitches: list[api_types.Pitch],
    ) -> list[dict[str, Any]]:
        """Convert Pitch objects to dictionaries for sequence building.

        Note: This expects pitches to have a pa_id attribute.
        """
        return [pitch.model_dump() for pitch in pitches]

    def _build_basic_pitch_data(
        self,
        pitch_type_probs: dict[str, float],
        decoded_pitches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build basic_pitch_data from logits and decoded samples."""
        if not decoded_pitches:
            return {
                "pitch_type_probs": pitch_type_probs,
                "pitch_speed_mean": 0.0,
                "pitch_speed_std": 0.0,
                "pitch_x_mean": 0.0,
                "pitch_x_std": 0.0,
                "pitch_z_mean": 0.0,
                "pitch_z_std": 0.0,
            }

        speeds = [p["speed"] for p in decoded_pitches]
        plate_x = [p["plate_pos_x"] for p in decoded_pitches]
        plate_z = [p["plate_pos_z"] for p in decoded_pitches]

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            mean = _mean(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            return variance ** 0.5

        return {
            "pitch_type_probs": pitch_type_probs,
            "pitch_speed_mean": _mean(speeds),
            "pitch_speed_std": _std(speeds),
            "pitch_x_mean": _mean(plate_x),
            "pitch_x_std": _std(plate_x),
            "pitch_z_mean": _mean(plate_z),
            "pitch_z_std": _std(plate_z),
        }

    def _build_detailed_pitch_data(
        self,
        decoded_pitches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build detailed_pitch_data from decoded samples."""
        if not decoded_pitches:
            return {}

        # Categorize pitches
        fastball_types = {"FF", "FC", "SI", "FA"}
        fastballs = [p for p in decoded_pitches if p["pitch_type"] in fastball_types]
        offspeed = [p for p in decoded_pitches if p["pitch_type"] not in fastball_types]

        total = len(decoded_pitches)
        prob_fastball = len(fastballs) / total if total > 0 else 0.0
        prob_offspeed = len(offspeed) / total if total > 0 else 0.0

        def _compute_stats(pitches: list[dict[str, Any]], field: str) -> dict[str, float]:
            values = [p[field] for p in pitches]
            if not values:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "p05": 0.0,
                    "p25": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p95": 0.0,
                }
            values = sorted(values)
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance ** 0.5

            def _percentile(pct: float) -> float:
                idx = int(pct * (n - 1))
                return values[min(idx, n - 1)]

            return {
                "mean": mean,
                "std": std,
                "p05": _percentile(0.05),
                "p25": _percentile(0.25),
                "p50": _percentile(0.50),
                "p75": _percentile(0.75),
                "p95": _percentile(0.95),
            }

        def _pitch_type_probs(pitches: list[dict[str, Any]]) -> dict[str, float]:
            if not pitches:
                return {}
            counts: dict[str, int] = {}
            for p in pitches:
                pt = p["pitch_type"]
                counts[pt] = counts.get(pt, 0) + 1
            total = len(pitches)
            return {pt: count / total for pt, count in counts.items()}

        fb_speed = _compute_stats(fastballs, "speed")
        fb_x = _compute_stats(fastballs, "plate_pos_x")
        fb_z = _compute_stats(fastballs, "plate_pos_z")

        os_speed = _compute_stats(offspeed, "speed")
        os_x = _compute_stats(offspeed, "plate_pos_x")
        os_z = _compute_stats(offspeed, "plate_pos_z")

        all_speed = _compute_stats(decoded_pitches, "speed")
        all_x = _compute_stats(decoded_pitches, "plate_pos_x")
        all_z = _compute_stats(decoded_pitches, "plate_pos_z")

        return {
            "pitch_prob_fastball": prob_fastball,
            "pitch_prob_offspeed": prob_offspeed,
            "pitch_data_fastballs": {
                "pitch_type_probs": _pitch_type_probs(fastballs),
                "pitch_speed_mean": fb_speed["mean"],
                "pitch_speed_std": fb_speed["std"],
                "pitch_speed_p05": fb_speed["p05"],
                "pitch_speed_p25": fb_speed["p25"],
                "pitch_speed_p50": fb_speed["p50"],
                "pitch_speed_p75": fb_speed["p75"],
                "pitch_speed_p95": fb_speed["p95"],
                "pitch_x_mean": fb_x["mean"],
                "pitch_x_std": fb_x["std"],
                "pitch_x_p05": fb_x["p05"],
                "pitch_x_p25": fb_x["p25"],
                "pitch_x_p50": fb_x["p50"],
                "pitch_x_p75": fb_x["p75"],
                "pitch_x_p95": fb_x["p95"],
                "pitch_z_mean": fb_z["mean"],
                "pitch_z_std": fb_z["std"],
                "pitch_z_p05": fb_z["p05"],
                "pitch_z_p25": fb_z["p25"],
                "pitch_z_p50": fb_z["p50"],
                "pitch_z_p75": fb_z["p75"],
                "pitch_z_p95": fb_z["p95"],
            },
            "pitch_data_offspeed": {
                "pitch_type_probs": _pitch_type_probs(offspeed),
                "pitch_speed_mean": os_speed["mean"],
                "pitch_speed_std": os_speed["std"],
                "pitch_speed_p05": os_speed["p05"],
                "pitch_speed_p25": os_speed["p25"],
                "pitch_speed_p50": os_speed["p50"],
                "pitch_speed_p75": os_speed["p75"],
                "pitch_speed_p95": os_speed["p95"],
                "pitch_x_mean": os_x["mean"],
                "pitch_x_std": os_x["std"],
                "pitch_x_p05": os_x["p05"],
                "pitch_x_p25": os_x["p25"],
                "pitch_x_p50": os_x["p50"],
                "pitch_x_p75": os_x["p75"],
                "pitch_x_p95": os_x["p95"],
                "pitch_z_mean": os_z["mean"],
                "pitch_z_std": os_z["std"],
                "pitch_z_p05": os_z["p05"],
                "pitch_z_p25": os_z["p25"],
                "pitch_z_p50": os_z["p50"],
                "pitch_z_p75": os_z["p75"],
                "pitch_z_p95": os_z["p95"],
            },
            "pitch_data_overall": {
                "pitch_type_probs": _pitch_type_probs(decoded_pitches),
                "pitch_speed_mean": all_speed["mean"],
                "pitch_speed_std": all_speed["std"],
                "pitch_speed_p05": all_speed["p05"],
                "pitch_speed_p25": all_speed["p25"],
                "pitch_speed_p50": all_speed["p50"],
                "pitch_speed_p75": all_speed["p75"],
                "pitch_speed_p95": all_speed["p95"],
                "pitch_x_mean": all_x["mean"],
                "pitch_x_std": all_x["std"],
                "pitch_x_p05": all_x["p05"],
                "pitch_x_p25": all_x["p25"],
                "pitch_x_p50": all_x["p50"],
                "pitch_x_p75": all_x["p75"],
                "pitch_x_p95": all_x["p95"],
                "pitch_z_mean": all_z["mean"],
                "pitch_z_std": all_z["std"],
                "pitch_z_p05": all_z["p05"],
                "pitch_z_p25": all_z["p25"],
                "pitch_z_p50": all_z["p50"],
                "pitch_z_p75": all_z["p75"],
                "pitch_z_p95": all_z["p95"],
            },
        }

    def _build_basic_outcome_data(
        self,
        decoded_pitches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build basic_outcome_data from decoded samples."""
        if not decoded_pitches:
            return {
                "outcome_probs": {},
                "swing_probability": 0.0,
                "swing_event_probs": {},
                "contact_probability": 0.0,
                "contact_event_probs": {},
            }

        # Count results
        result_counts: dict[str, int] = {}
        for p in decoded_pitches:
            r = p["result"]
            result_counts[r] = result_counts.get(r, 0) + 1

        total = len(decoded_pitches)

        # Map to outcome categories (S=strike, B=ball, X=contact)
        ball_results = {
            "ball", "ball_in_dirt", "blocked_ball", "automatic_ball",
            "intentional_ball", "pitchout", "hit_by_pitch",
        }
        strike_results = {
            "called_strike", "swinging_strike", "swinging_strike_blocked",
            "swinging_pitchout", "foul_tip", "automatic_strike",
            "foul", "foul_bunt", "bunt_foul_tip", "foul_pitchout", "missed_bunt",
        }
        contact_results = {"hit_into_play"}

        ball_count = sum(result_counts.get(r, 0) for r in ball_results)
        strike_count = sum(result_counts.get(r, 0) for r in strike_results)
        contact_count = sum(result_counts.get(r, 0) for r in contact_results)

        outcome_probs = {}
        if ball_count > 0:
            outcome_probs["ball"] = ball_count / total
        if strike_count > 0:
            outcome_probs["strike"] = strike_count / total
        if contact_count > 0:
            outcome_probs["contact"] = contact_count / total

        return {
            "outcome_probs": outcome_probs,
            "swing_probability": 0.0,  # Not available from token-level prediction
            "swing_event_probs": {},
            "contact_probability": contact_count / total if total > 0 else 0.0,
            "contact_event_probs": {},
        }

    def _build_detailed_outcome_data(
        self,
        decoded_pitches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build detailed_outcome_data from decoded samples."""
        # xLSTM doesn't predict swing mechanics or batted ball data
        return {}

    def _decoded_to_pitch_objects(
        self,
        decoded_pitches: list[dict[str, Any]],
    ) -> list[api_types.Pitch]:
        """Convert decoded pitch dictionaries to Pitch objects."""
        result = []
        for p in decoded_pitches:
            try:
                pitch = api_types.Pitch(
                    pitch_type=p["pitch_type"],
                    speed=p["speed"],
                    spin_rate=p["spin_rate"],
                    spin_axis=p["spin_axis"],
                    release_pos_x=p["release_pos_x"],
                    release_pos_z=p["release_pos_z"],
                    release_extension=p["release_extension"],
                    vx0=p["vx0"],
                    vy0=p["vy0"],
                    vz0=p["vz0"],
                    ax=p["ax"],
                    ay=p["ay"],
                    az=p["az"],
                    plate_pos_x=p["plate_pos_x"],
                    plate_pos_z=p["plate_pos_z"],
                    result=p["result"],
                )
                result.append(pitch)
            except Exception as e:
                self.logger.warning(f"Failed to convert decoded pitch: {e}")
                continue
        return result

    async def predict_pitcher(
        self,
        request: api_types.PredictPitcherRequest,
        **kwargs: Any,
    ) -> api_types.PredictPitcherResponse:
        """Predict the pitcher's next pitch using xLSTM generation."""
        self.logger.debug("predict_pitcher called")

        try:
            start_time = datetime.now()

            # Validate request
            self._validate_xlstm_request(request)

            # Load model if needed
            model, device = self._ensure_model_loaded()

            # Build context defaults
            defaults = self._build_context_defaults(request)

            # Convert prev_pitches to dicts
            prev_pitches = request.prev_pitches or []
            pitch_dicts = self._convert_pitches_to_dicts(prev_pitches)

            # Build history sequence
            seq_result = build_history_sequence(pitch_dicts, defaults)

            # Append structural tokens for generation based on sequence state
            tokens = seq_result.tokens
            contexts = list(seq_result.contexts)
            if seq_result.pa_open:
                # PA still in progress — generate the next pitch directly
                pass
            elif tokens and tokens[-1] == PitchToken.SESSION_END.value:
                # Session ended — start new session + PA for generation
                tokens = tokens + [PitchToken.SESSION_START.value, PitchToken.PA_START.value]
                last_ctx = contexts[-1]
                contexts.append(last_ctx)
                contexts.append(last_ctx)
            elif tokens and tokens[-1] == PitchToken.PA_END.value:
                # PA ended but session still open — start new PA
                tokens = tokens + [PitchToken.PA_START.value]
                contexts.append(contexts[-1])

            # Convert contexts to packed tensors
            packed_ctx = contexts_to_packed(contexts, device)

            # Configure generation
            sample_size = request.sample_size if request.sample_size else 10
            gen_config = GenerationConfig(
                sample_size=sample_size,
                temperature=1.0,
                top_k=5,
            )

            # Generate pitches
            gen_result = generate_pitches(
                model=model,
                history_tokens=tokens,
                history_context=packed_ctx,
                config=gen_config,
                device=device,
                force_first_category=TokenCategory.PITCH_TYPE if seq_result.pa_open else None,
            )

            # Build response
            decoded_pitches = gen_result.decoded_pitches
            pitch_type_probs = gen_result.pitch_type_probs

            basic_pitch_data = self._build_basic_pitch_data(pitch_type_probs, decoded_pitches)
            detailed_pitch_data = self._build_detailed_pitch_data(decoded_pitches)
            basic_outcome_data = self._build_basic_outcome_data(decoded_pitches)
            detailed_outcome_data = self._build_detailed_outcome_data(decoded_pitches)
            pitches = self._decoded_to_pitch_objects(decoded_pitches)

            end_time = datetime.now()
            prediction_metadata = self.get_pitcher_prediction_metadata(
                start_time=start_time,
                end_time=end_time,
                n_history_tokens=len(seq_result.tokens),
                n_samples_generated=len(decoded_pitches),
                sample_size=sample_size,
            )

            self.logger.info("predict_pitcher completed successfully")

            return api_types.PredictPitcherResponse(
                basic_pitch_data=basic_pitch_data,
                detailed_pitch_data=detailed_pitch_data,
                basic_outcome_data=basic_outcome_data,
                detailed_outcome_data=detailed_outcome_data,
                pitches=pitches,
                prediction_metadata=prediction_metadata,
            )

        except HTTPException:
            raise
        except (ValueError, KeyError) as e:
            self.logger.error(f"predict_pitcher invalid input: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.logger.error(f"predict_pitcher failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_batter(
        self,
        batter_name: str,
        pitcher_name: str,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
        pitch_type: str,
        pitch_speed: float,
        pitch_x: float,
        pitch_z: float,
    ) -> dict[str, Any]:
        """Predict the batter's next outcome (not implemented for xLSTM)."""
        raise HTTPException(
            status_code=501,
            detail="Batter prediction is not implemented for xLSTM algorithm",
        )

    async def predict_batted_ball(
        self,
        launch_speed: float,
        launch_angle: float,
        spray_angle: float | None = None,
        bb_type: str | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        batter_id: int | None = None,
        game_date: str | None = None,
    ) -> dict[str, Any]:
        """Predict batted ball outcomes (not implemented for xLSTM)."""
        raise HTTPException(
            status_code=501,
            detail="Batted ball prediction is not implemented for xLSTM algorithm",
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get algorithm metadata."""
        return {
            "algorithm_name": "xlstm",
            "instance_name": self.name,
            "model_loaded": self._model is not None,
            "device": str(self._device) if self._device else None,
        }

    def get_pitcher_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get pitcher prediction metadata."""
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        n_history_tokens = kwargs.get("n_history_tokens", 0)
        n_samples_generated = kwargs.get("n_samples_generated", 0)
        sample_size = kwargs.get("sample_size", 0)

        result: dict[str, Any] = {
            "algorithm": "xlstm",
            "n_history_tokens": n_history_tokens,
            "n_samples_generated": n_samples_generated,
            "sample_size": sample_size,
        }

        if start_time and end_time:
            result["start_time"] = start_time.isoformat()
            result["end_time"] = end_time.isoformat()
            result["duration"] = (end_time - start_time).total_seconds()

        return result

    def get_batter_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get batter prediction metadata (not implemented for xLSTM)."""
        return {"error": "Batter prediction not implemented for xLSTM"}

    def get_batted_ball_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get batted ball prediction metadata (not implemented for xLSTM)."""
        return {"error": "Batted ball prediction not implemented for xLSTM"}
