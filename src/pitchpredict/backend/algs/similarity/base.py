# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime
import logging
from typing import Any

from fastapi import HTTPException
import pandas as pd

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.fetching import get_pitches_from_pitcher, get_pitches_to_batter, get_player_id_from_name, get_all_batted_balls
import pitchpredict.types.api as api_types
import pitchpredict.backend.algs.similarity.types as similarity_types

class SimilarityAlgorithm(PitchPredictAlgorithm):
    """
    The PitchPredict algorithm that uses similarity and nearest-neighbor analysis to predict the next pitch and outcome.
    """

    def __init__(
        self,
        name: str = "similarity",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.logger = logging.getLogger("pitchpredict.backend.algs.similarity")

    async def predict_pitcher(
        self,
        request: api_types.PredictPitcherRequest,
        sample_pctg: float = 0.05,
        sample_size: int = 1,
        **kwargs: Any,
    ) -> api_types.PredictPitcherResponse:
        """
        Predict the pitcher's next pitch and its outcome.
        """
        self.logger.debug("predict_pitcher called")

        try:
            start_time = datetime.now()
            self.logger.debug(f"start time: {start_time}")

            pitcher_id = request.pitcher_id

            request_values = request.model_dump()
            request_sample_pctg = request_values.get("sample_pctg")
            if request_sample_pctg is not None:
                try:
                    sample_pctg = float(request_sample_pctg)
                except (TypeError, ValueError):
                    self.logger.warning("invalid sample_pctg in request; using default")

            effective_sample_size = request.sample_size if request.sample_size is not None else sample_size
            weights = similarity_types.SimilarityWeights().softmax()

            pitches = await get_pitches_from_pitcher(
                pitcher_id=pitcher_id,
                start_date="2015-01-01",
                end_date=request.game_date,
            )
            self.logger.debug(f"successfully fetched {pitches.shape[0]} pitches")

            similar_pitches = await self._get_similar_pitches_for_pitcher(
                pitches=pitches,
                context=request,
                weights=weights,
                sample_pctg=sample_pctg,
            )
            self.logger.debug(f"successfully fetched {similar_pitches.shape[0]} similar pitches")

            basic_pitch_data, detailed_pitch_data = await self._digest_pitch_data(similar_pitches)
            self.logger.debug("successfully digested pitch data")

            basic_outcome_data, detailed_outcome_data = await self._digest_outcome_data(
                pitches=similar_pitches
            )
            self.logger.debug("successfully digested outcome data")

            sampled_pitches = self._sample_pitches(pitches=similar_pitches, n=effective_sample_size)
            self.logger.debug(f"successfully sampled {len(sampled_pitches)} pitches")

            prediction_metadata = self.get_pitcher_prediction_metadata(
                start_time=start_time,
                end_time=datetime.now(),
                n_pitches_total=len(pitches),
                n_pitches_sampled=len(similar_pitches),
                sample_pctg=sample_pctg,
            )

            self.logger.info("predict_pitcher completed successfully")

            return api_types.PredictPitcherResponse(
                basic_pitch_data=basic_pitch_data,
                detailed_pitch_data=detailed_pitch_data,
                basic_outcome_data=basic_outcome_data,
                detailed_outcome_data=detailed_outcome_data,
                pitches=sampled_pitches,
                prediction_metadata=prediction_metadata,
            )

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
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
        """
        Predict the batter's next outcome.
        """
        self.logger.debug("predict_batter called")

        try:
            start_time = datetime.now()
            self.logger.debug(f"start time: {start_time}")

            pitcher_id = await get_player_id_from_name(pitcher_name)
            batter_id = await get_player_id_from_name(batter_name)

            pitches = await get_pitches_to_batter(
                batter_id=batter_id,
                start_date="2015-01-01",
                end_date=game_date,
            )
            self.logger.debug(f"successfully fetched {pitches.shape[0]} pitches")

            sample_pctg = 0.05
            similar_pitches = await self._get_similar_pitches_for_batter(
                pitches=pitches,
                pitcher_id=pitcher_id,
                balls=balls,
                strikes=strikes,
                score_bat=score_bat,
                score_fld=score_fld,
                game_date=game_date,
                pitch_type=pitch_type,
                pitch_speed=pitch_speed,
                pitch_x=pitch_x,
                pitch_z=pitch_z,
                sample_pctg=sample_pctg,
            )
            self.logger.debug(f"successfully fetched {similar_pitches.shape[0]} similar pitches")

            basic_outcome_data, detailed_outcome_data = await self._digest_outcome_data(
                pitches=similar_pitches,
            )
            self.logger.debug("successfully digested outcome data")
            
            prediction_metadata = self.get_batter_prediction_metadata(
                start_time=start_time,
                end_time=datetime.now(),
                n_pitches_total=len(pitches),
                n_pitches_sampled=len(similar_pitches),
                sample_pctg=sample_pctg,
            )

            self.logger.info("predict_batter completed successfully")

            return {
                "algorithm_metadata": self.get_metadata(),
                "basic_outcome_data": basic_outcome_data,
                "detailed_outcome_data": detailed_outcome_data,
                "prediction_metadata": prediction_metadata,
            }
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_similar_pitches_for_pitcher(
        self,
        pitches: pd.DataFrame,
        context: api_types.PredictPitcherRequest,
        weights: dict[str, float],
        sample_pctg: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get the pitches most similar to the given context for this pitcher.
        """
        self.logger.debug("get_similar_pitches_for_pitcher called")

        try:
            context_values = context.model_dump()
            similarity_score = pd.Series(0.0, index=pitches.index)

            def add_weighted_score(field: str, scores: pd.Series) -> None:
                weight = weights.get(field, 0.0)
                if weight <= 0.0:
                    return
                nonlocal similarity_score
                similarity_score = similarity_score.add(scores * weight, fill_value=0.0)
                pitches[f"score_{field}"] = scores

            def score_equal(series: pd.Series, target: Any) -> pd.Series:
                return series.eq(target).astype(float).fillna(0.0)

            def score_numeric(series: pd.Series, target: float, max_diff: float | None = None) -> pd.Series:
                values = pd.to_numeric(series, errors="coerce")
                if max_diff is None or max_diff <= 0 or pd.isna(max_diff):
                    return values.eq(target).astype(float).fillna(0.0)
                diff = (values - target).abs()
                score = 1 - (diff / max_diff)
                return score.clip(lower=0).fillna(0.0)

            eq_fields = {
                "batter_id": "batter",
                "pitcher_throws": "p_throws",
                "batter_hits": "stand",
                "count_balls": "balls",
                "count_strikes": "strikes",
                "outs": "outs_when_up",
                "inning": "inning",
                "number_through_order": "n_thruorder_pitcher",
                "fielder_2_id": "fielder_2",
                "fielder_3_id": "fielder_3",
                "fielder_4_id": "fielder_4",
                "fielder_5_id": "fielder_5",
                "fielder_6_id": "fielder_6",
                "fielder_7_id": "fielder_7",
                "fielder_8_id": "fielder_8",
                "fielder_9_id": "fielder_9",
            }

            numeric_fields: dict[str, tuple[str, float | None]] = {
                "pitcher_age": ("age_pit", None),
                "batter_age": ("age_bat", None),
                "score_bat": ("bat_score", None),
                "score_fld": ("fld_score", None),
                "pitch_number": ("pitch_number", None),
                "batter_days_since_prev_game": ("batter_days_since_prev_game", None),
                "pitcher_days_since_prev_game": ("pitcher_days_since_prev_game", None),
                "strike_zone_top": ("sz_top", 0.2),
                "strike_zone_bottom": ("sz_bot", 0.1),
            }

            for field, column in eq_fields.items():
                value = context_values.get(field)
                if value is None or column not in pitches.columns:
                    continue
                add_weighted_score(field, score_equal(pitches[column], value))

            for field, (column, max_diff) in numeric_fields.items():
                value = context_values.get(field)
                if value is None or column not in pitches.columns:
                    continue
                tolerance = max_diff
                if tolerance is None:
                    tolerance = pd.to_numeric(pitches[column], errors="coerce").std()
                add_weighted_score(field, score_numeric(pitches[column], float(value), tolerance))

            bases_state = context_values.get("bases_state")
            if bases_state is not None and {"on_1b", "on_2b", "on_3b"}.issubset(pitches.columns):
                runner_on_1b = pitches["on_1b"].notna()
                runner_on_2b = pitches["on_2b"].notna()
                runner_on_3b = pitches["on_3b"].notna()
                target_on_1b = bool(bases_state & 1)
                target_on_2b = bool(bases_state & 2)
                target_on_3b = bool(bases_state & 4)
                mismatches = (
                    (runner_on_1b != target_on_1b).astype(int)
                    + (runner_on_2b != target_on_2b).astype(int)
                    + (runner_on_3b != target_on_3b).astype(int)
                )
                scores = (1 - (mismatches / 3)).clip(lower=0)
                add_weighted_score("bases_state", scores)

            if context.game_date is not None and "game_date" in pitches.columns:
                target_date = datetime.strptime(context.game_date, "%Y-%m-%d")
                dates = pd.to_datetime(pitches["game_date"], errors="coerce")
                day_diff = (dates - target_date).dt.days.abs()
                date_scores = (1 - (day_diff / 365.0)).clip(lower=0).fillna(0.0)
                add_weighted_score("game_date", date_scores)

            pitches["similarity_score"] = similarity_score

            pitches = pitches.sort_values(by="similarity_score", ascending=False)
            n_samples = max(100, int(len(pitches) * sample_pctg))
            pitches = pitches.head(n_samples)

            self.logger.info(f"successfully fetched {pitches.shape[0]} similar pitches")
            return pitches

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_similar_pitches_for_batter(
        self,
        pitches: pd.DataFrame,
        pitcher_id: int,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
        pitch_type: str,
        pitch_speed: float,
        pitch_x: float,
        pitch_z: float,
        sample_pctg: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get the pitches most similar to the given context for this batter.
        """
        self.logger.debug("get_similar_pitches_for_batter called")

        try:
            # append "similarity score" column to pitches for each parameter
            # 'score_pitcher_name': 1 if pitcher_name is the same as the pitcher in the pitch, 0 otherwise
            pitches["score_pitcher_name"] = pitches["pitcher"].apply(lambda x: 1 if x == pitcher_id else 0)
            # 'score_balls': 1 if balls is the same as the balls in the pitch, 0 otherwise
            pitches["score_balls"] = pitches["balls"].apply(lambda x: 1 if x == balls else 0)
            # 'score_strikes': 1 if strikes is the same as the strikes in the pitch, 0 otherwise
            pitches["score_strikes"] = pitches["strikes"].apply(lambda x: 1 if x == strikes else 0)
            # 'score_score_bat': 1 if score_bat is the same as the score_bat in the pitch, 0 otherwise
            pitches["score_score_bat"] = pitches["bat_score"].apply(lambda x: 1 if x == score_bat else 0)
            # 'score_score_fld': 1 if score_fld is the same as the score_fld in the pitch, 0 otherwise
            pitches["score_score_fld"] = pitches["fld_score"].apply(lambda x: 1 if x == score_fld else 0)
            # 'score_game_date': 1 if game_date is the same as the game_date in the pitch, 0 otherwise
            pitches["score_game_date"] = pitches["game_date"].apply(lambda x: 1 if x == game_date else 0)
            # 'score_pitch_type': 1 if pitch_type is the same as the pitch_type in the pitch, 0 otherwise
            pitches["score_pitch_type"] = pitches["pitch_type"].apply(lambda x: 1 if x == pitch_type else 0)
            # 'score_pitch_speed': 1 if pitch_speed is the same as the pitch_speed in the pitch, 0 otherwise
            pitches["score_pitch_speed"] = pitches["release_speed"].apply(lambda x: 1 if x == pitch_speed else 0)
            # 'score_pitch_x': 1 if pitch_x is the same as the pitch_x in the pitch, 0 otherwise
            pitches["score_pitch_x"] = pitches["plate_x"].apply(lambda x: 1 if x == pitch_x else 0)
            # 'score_pitch_z': 1 if pitch_z is the same as the pitch_z in the pitch, 0 otherwise
            pitches["score_pitch_z"] = pitches["plate_z"].apply(lambda x: 1 if x == pitch_z else 0)

            # create a single similarity score: a weighted average of the individual similarity scores above
            pitches["similarity_score"] = (
                pitches["score_pitcher_name"] * 0.25 +
                pitches["score_balls"] * 0.15 +
                pitches["score_strikes"] * 0.15 +
                pitches["score_score_bat"] * 0.05 +
                pitches["score_score_fld"] * 0.05 +
                pitches["score_game_date"] * 0.1 +
                pitches["score_pitch_type"] * 0.05 +
                pitches["score_pitch_speed"] * 0.05 +
                pitches["score_pitch_x"] * 0.05 +
                pitches["score_pitch_z"] * 0.05
            )

            # sort pitches by similarity score and return the top N pitches
            pitches = pitches.sort_values(by="similarity_score", ascending=False)
            pitches = pitches.head(int(len(pitches) * sample_pctg))

            self.logger.info(f"successfully fetched {pitches.shape[0]} similar pitches")
            return pitches

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _digest_pitch_data(
        self,
        pitches: pd.DataFrame,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Create a final summary of the pitch data.

        Args:
            pitches: A list of pitch dictionaries.

        Returns:
            A tuple containing the basic pitch data and the detailed pitch data.
        """
        self.logger.debug("digest_pitch_data called")

        try:
            pitch_type_value_counts = pitches["pitch_type"].value_counts()
            pitch_type_probs = pitch_type_value_counts / pitch_type_value_counts.sum()
            pitch_speed_mean = pitches["release_speed"].mean()
            pitch_speed_std = pitches["release_speed"].std()
            pitch_x_mean = pitches["plate_x"].mean()
            pitch_x_std = pitches["plate_x"].std()
            pitch_z_mean = pitches["plate_z"].mean()
            pitch_z_std = pitches["plate_z"].std()

            basic = {
                "pitch_type_probs": pitch_type_probs.to_dict(),
                "pitch_speed_mean": pitch_speed_mean,
                "pitch_speed_std": pitch_speed_std,
                "pitch_x_mean": pitch_x_mean,
                "pitch_x_std": pitch_x_std,
                "pitch_z_mean": pitch_z_mean,
                "pitch_z_std": pitch_z_std,
            }

            fastballs = pitches.loc[(pitches["pitch_type"] == "FF") | (pitches["pitch_type"] == "FC") | (pitches["pitch_type"] == "SI")]
            prob_fastballs = fastballs.shape[0] / pitches.shape[0]
            offspeed = pitches.loc[(pitches["pitch_type"] != "FF") & (pitches["pitch_type"] != "FC") & (pitches["pitch_type"] != "SI")]
            prob_offspeed = offspeed.shape[0] / pitches.shape[0]

            fastball_type_value_counts = fastballs["pitch_type"].value_counts()
            fastball_type_probs = fastball_type_value_counts / fastball_type_value_counts.sum()
            fastball_speed_mean = fastballs["release_speed"].mean()
            fastball_speed_std = fastballs["release_speed"].std()
            fastball_speed_p05 = fastballs["release_speed"].quantile(0.05)
            fastball_speed_p25 = fastballs["release_speed"].quantile(0.25)
            fastball_speed_p50 = fastballs["release_speed"].quantile(0.50)
            fastball_speed_p75 = fastballs["release_speed"].quantile(0.75)
            fastball_speed_p95 = fastballs["release_speed"].quantile(0.95)
            fastball_x_mean = fastballs["plate_x"].mean()
            fastball_x_std = fastballs["plate_x"].std()
            fastball_x_p05 = fastballs["plate_x"].quantile(0.05)
            fastball_x_p25 = fastballs["plate_x"].quantile(0.25)
            fastball_x_p50 = fastballs["plate_x"].quantile(0.50)
            fastball_x_p75 = fastballs["plate_x"].quantile(0.75)
            fastball_x_p95 = fastballs["plate_x"].quantile(0.95)
            fastball_z_mean = fastballs["plate_z"].mean()
            fastball_z_std = fastballs["plate_z"].std()
            fastball_z_p05 = fastballs["plate_z"].quantile(0.05)
            fastball_z_p25 = fastballs["plate_z"].quantile(0.25)
            fastball_z_p50 = fastballs["plate_z"].quantile(0.50)
            fastball_z_p75 = fastballs["plate_z"].quantile(0.75)
            fastball_z_p95 = fastballs["plate_z"].quantile(0.95)

            offspeed_type_value_counts = offspeed["pitch_type"].value_counts()
            offspeed_type_probs = offspeed_type_value_counts / offspeed_type_value_counts.sum()
            offspeed_speed_mean = offspeed["release_speed"].mean()
            offspeed_speed_std = offspeed["release_speed"].std()
            offspeed_speed_p05 = offspeed["release_speed"].quantile(0.05)
            offspeed_speed_p25 = offspeed["release_speed"].quantile(0.25)
            offspeed_speed_p50 = offspeed["release_speed"].quantile(0.50)
            offspeed_speed_p75 = offspeed["release_speed"].quantile(0.75)
            offspeed_speed_p95 = offspeed["release_speed"].quantile(0.95)
            offspeed_x_mean = offspeed["plate_x"].mean()
            offspeed_x_std = offspeed["plate_x"].std()
            offspeed_x_p05 = offspeed["plate_x"].quantile(0.05)
            offspeed_x_p25 = offspeed["plate_x"].quantile(0.25)
            offspeed_x_p50 = offspeed["plate_x"].quantile(0.50)
            offspeed_x_p75 = offspeed["plate_x"].quantile(0.75)
            offspeed_x_p95 = offspeed["plate_x"].quantile(0.95)
            offspeed_z_mean = offspeed["plate_z"].mean()
            offspeed_z_std = offspeed["plate_z"].std()
            offspeed_z_p05 = offspeed["plate_z"].quantile(0.05)
            offspeed_z_p25 = offspeed["plate_z"].quantile(0.25)
            offspeed_z_p50 = offspeed["plate_z"].quantile(0.50)
            offspeed_z_p75 = offspeed["plate_z"].quantile(0.75)
            offspeed_z_p95 = offspeed["plate_z"].quantile(0.95)

            pitch_speed_p05 = pitches["release_speed"].quantile(0.05)
            pitch_speed_p25 = pitches["release_speed"].quantile(0.25)
            pitch_speed_p50 = pitches["release_speed"].quantile(0.50)
            pitch_speed_p75 = pitches["release_speed"].quantile(0.75)
            pitch_speed_p95 = pitches["release_speed"].quantile(0.95)
            pitch_x_p05 = pitches["plate_x"].quantile(0.05)
            pitch_x_p25 = pitches["plate_x"].quantile(0.25)
            pitch_x_p50 = pitches["plate_x"].quantile(0.50)
            pitch_x_p75 = pitches["plate_x"].quantile(0.75)
            pitch_x_p95 = pitches["plate_x"].quantile(0.95)
            pitch_z_p05 = pitches["plate_z"].quantile(0.05)
            pitch_z_p25 = pitches["plate_z"].quantile(0.25)
            pitch_z_p50 = pitches["plate_z"].quantile(0.50)
            pitch_z_p75 = pitches["plate_z"].quantile(0.75)
            pitch_z_p95 = pitches["plate_z"].quantile(0.95)

            detailed = {
                "pitch_prob_fastball": prob_fastballs,
                "pitch_prob_offspeed": prob_offspeed,
                "pitch_data_fastballs": {
                    "pitch_type_probs": fastball_type_probs.to_dict(),
                    "pitch_speed_mean": fastball_speed_mean,
                    "pitch_speed_std": fastball_speed_std,
                    "pitch_speed_p05": fastball_speed_p05,
                    "pitch_speed_p25": fastball_speed_p25,
                    "pitch_speed_p50": fastball_speed_p50,
                    "pitch_speed_p75": fastball_speed_p75,
                    "pitch_speed_p95": fastball_speed_p95,
                    "pitch_x_mean": fastball_x_mean,
                    "pitch_x_std": fastball_x_std,
                    "pitch_x_p05": fastball_x_p05,
                    "pitch_x_p25": fastball_x_p25,
                    "pitch_x_p50": fastball_x_p50,
                    "pitch_x_p75": fastball_x_p75,
                    "pitch_x_p95": fastball_x_p95,
                    "pitch_z_mean": fastball_z_mean,
                    "pitch_z_std": fastball_z_std,
                    "pitch_z_p05": fastball_z_p05,
                    "pitch_z_p25": fastball_z_p25,
                    "pitch_z_p50": fastball_z_p50,
                    "pitch_z_p75": fastball_z_p75,
                    "pitch_z_p95": fastball_z_p95,
                },
                "pitch_data_offspeed": {
                    "pitch_type_probs": offspeed_type_probs.to_dict(),
                    "pitch_speed_mean": offspeed_speed_mean,
                    "pitch_speed_std": offspeed_speed_std,
                    "pitch_speed_p05": offspeed_speed_p05,
                    "pitch_speed_p25": offspeed_speed_p25,
                    "pitch_speed_p50": offspeed_speed_p50,
                    "pitch_speed_p75": offspeed_speed_p75,
                    "pitch_speed_p95": offspeed_speed_p95,
                    "pitch_x_mean": offspeed_x_mean,
                    "pitch_x_std": offspeed_x_std,
                    "pitch_x_p05": offspeed_x_p05,
                    "pitch_x_p25": offspeed_x_p25,
                    "pitch_x_p50": offspeed_x_p50,
                    "pitch_x_p75": offspeed_x_p75,
                    "pitch_x_p95": offspeed_x_p95,
                    "pitch_z_mean": offspeed_z_mean,
                    "pitch_z_std": offspeed_z_std,
                    "pitch_z_p05": offspeed_z_p05,
                    "pitch_z_p25": offspeed_z_p25,
                    "pitch_z_p50": offspeed_z_p50,
                    "pitch_z_p75": offspeed_z_p75,
                    "pitch_z_p95": offspeed_z_p95,
                },
                "pitch_data_overall": {
                    "pitch_type_probs": pitch_type_probs.to_dict(),
                    "pitch_speed_mean": pitch_speed_mean,
                    "pitch_speed_std": pitch_speed_std,
                    "pitch_speed_p05": pitch_speed_p05,
                    "pitch_speed_p25": pitch_speed_p25,
                    "pitch_speed_p50": pitch_speed_p50,
                    "pitch_speed_p75": pitch_speed_p75,
                    "pitch_speed_p95": pitch_speed_p95,
                    "pitch_x_mean": pitch_x_mean,
                    "pitch_x_std": pitch_x_std,
                    "pitch_x_p05": pitch_x_p05,
                    "pitch_x_p25": pitch_x_p25,
                    "pitch_x_p50": pitch_x_p50,
                    "pitch_x_p75": pitch_x_p75,
                    "pitch_x_p95": pitch_x_p95,
                    "pitch_z_mean": pitch_z_mean,
                    "pitch_z_std": pitch_z_std,
                    "pitch_z_p05": pitch_z_p05,
                    "pitch_z_p25": pitch_z_p25,
                    "pitch_z_p50": pitch_z_p50,
                    "pitch_z_p75": pitch_z_p75,
                    "pitch_z_p95": pitch_z_p95,
                },
            }

            self.logger.info("digest_pitch_data completed successfully")
            return basic, detailed

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _digest_outcome_data(
        self,
        pitches: pd.DataFrame,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Create a final summary of the outcome data.
        """
        self.logger.debug("digest_outcome_data called")

        try:
            outcome_value_counts = pitches["type"].value_counts()
            outcome_probs = outcome_value_counts / outcome_value_counts.sum()

            swing_events = pitches.loc[pitches["bat_speed"].notna()]
            swing_event_value_counts = swing_events["type"].value_counts()
            swing_event_probs = swing_event_value_counts / swing_event_value_counts.sum()
            swing_probability = swing_event_value_counts.sum() / pitches.shape[0]

            swing_speed_mean = swing_events["bat_speed"].mean()
            swing_speed_std = swing_events["bat_speed"].std()
            swing_speed_p05 = swing_events["bat_speed"].quantile(0.05)
            swing_speed_p25 = swing_events["bat_speed"].quantile(0.25)
            swing_speed_p50 = swing_events["bat_speed"].quantile(0.50)
            swing_speed_p75 = swing_events["bat_speed"].quantile(0.75)
            swing_speed_p95 = swing_events["bat_speed"].quantile(0.95)
            swing_length_mean = swing_events["swing_length"].mean()
            swing_length_std = swing_events["swing_length"].std()
            swing_length_p05 = swing_events["swing_length"].quantile(0.05)
            swing_length_p25 = swing_events["swing_length"].quantile(0.25)
            swing_length_p50 = swing_events["swing_length"].quantile(0.50)
            swing_length_p75 = swing_events["swing_length"].quantile(0.75)
            swing_length_p95 = swing_events["swing_length"].quantile(0.95)

            contact_events = pitches.loc[pitches["type"] == "X"]
            contact_event_value_counts = contact_events["events"].value_counts()
            contact_event_probs = contact_event_value_counts / contact_event_value_counts.sum()
            contact_probability = contact_event_value_counts.sum() / pitches.shape[0]

            contact_bb_type_value_counts = contact_events["bb_type"].value_counts()
            contact_bb_type_probs = contact_bb_type_value_counts / contact_bb_type_value_counts.sum()
            contact_hits = contact_events.loc[(contact_events["events"] == "single") | (contact_events["events"] == "double") | (contact_events["events"] == "triple") | (contact_events["events"] == "home_run")]
            contact_ba = contact_hits.shape[0] / contact_events.shape[0]
            contact_bases = contact_events["events"].apply(lambda x: 1 if x == "single" else 2 if x == "double" else 3 if x == "triple" else 4 if x == "home_run" else 0)
            contact_slg = contact_bases.sum() / contact_events.shape[0]
            contact_woba = contact_hits.shape[0] / contact_events.shape[0]
            contact_exit_velocity_mean = contact_events["launch_speed"].mean()
            contact_exit_velocity_std = contact_events["launch_speed"].std()
            contact_exit_velocity_p05 = contact_events["launch_speed"].quantile(0.05)
            contact_exit_velocity_p25 = contact_events["launch_speed"].quantile(0.25)
            contact_exit_velocity_p50 = contact_events["launch_speed"].quantile(0.50)
            contact_exit_velocity_p75 = contact_events["launch_speed"].quantile(0.75)
            contact_exit_velocity_p95 = contact_events["launch_speed"].quantile(0.95)
            contact_launch_angle_mean = contact_events["launch_angle"].mean()
            contact_launch_angle_std = contact_events["launch_angle"].std()
            contact_launch_angle_p05 = contact_events["launch_angle"].quantile(0.05)
            contact_launch_angle_p25 = contact_events["launch_angle"].quantile(0.25)
            contact_launch_angle_p50 = contact_events["launch_angle"].quantile(0.50)
            contact_launch_angle_p75 = contact_events["launch_angle"].quantile(0.75)
            contact_launch_angle_p95 = contact_events["launch_angle"].quantile(0.95)
            contact_xba = contact_events["estimated_ba_using_speedangle"].mean()
            contact_xslg = contact_events["estimated_slg_using_speedangle"].mean()
            contact_xwoba = contact_events["estimated_woba_using_speedangle"].mean()

            new_column_names_outcome = {
                "S": "strike",
                "B": "ball",
                "X": "contact",
            }

            new_column_names_swing = {
                "S": "swinging_strike",
                "X": "contact",
            }

            basic = {
                "outcome_probs": outcome_probs.rename(index=new_column_names_outcome).to_dict(),
                "swing_probability": swing_probability,
                "swing_event_probs": swing_event_probs.rename(index=new_column_names_swing).to_dict(),
                "contact_probability": contact_probability,
                "contact_event_probs": contact_event_probs.to_dict(),
            }

            detailed = {
                "swing_data": {
                    "bat_speed_mean": swing_speed_mean,
                    "bat_speed_std": swing_speed_std,
                    "bat_speed_p05": swing_speed_p05,
                    "bat_speed_p25": swing_speed_p25,
                    "bat_speed_p50": swing_speed_p50,
                    "bat_speed_p75": swing_speed_p75,
                    "bat_speed_p95": swing_speed_p95,
                    "swing_length_mean": swing_length_mean,
                    "swing_length_std": swing_length_std,
                    "swing_length_p05": swing_length_p05,
                    "swing_length_p25": swing_length_p25,
                    "swing_length_p50": swing_length_p50,
                    "swing_length_p75": swing_length_p75,
                    "swing_length_p95": swing_length_p95,
                },
                "contact_data": {
                    "bb_type_probs": contact_bb_type_probs.to_dict(),
                    "BA": contact_ba,
                    "SLG": contact_slg,
                    "wOBA": contact_woba,
                    "exit_velocity_mean": contact_exit_velocity_mean,
                    "exit_velocity_std": contact_exit_velocity_std,
                    "exit_velocity_p05": contact_exit_velocity_p05,
                    "exit_velocity_p25": contact_exit_velocity_p25,
                    "exit_velocity_p50": contact_exit_velocity_p50,
                    "exit_velocity_p75": contact_exit_velocity_p75,
                    "exit_velocity_p95": contact_exit_velocity_p95,
                    "launch_angle_mean": contact_launch_angle_mean,
                    "launch_angle_std": contact_launch_angle_std,
                    "launch_angle_p05": contact_launch_angle_p05,
                    "launch_angle_p25": contact_launch_angle_p25,
                    "launch_angle_p50": contact_launch_angle_p50,
                    "launch_angle_p75": contact_launch_angle_p75,
                    "launch_angle_p95": contact_launch_angle_p95,
                    "xBA": contact_xba,
                    "xSLG": contact_xslg,
                    "xwOBA": contact_xwoba,
                }
            }

            self.logger.info("digest_outcome_data completed successfully")
            return basic, detailed
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_metadata(self) -> dict[str, Any]:
        """
        Get the metadata for the algorithm, including usage information.
        """
        return {
            "algorithm_name": "similarity",
            "instance_name": self.name,
        }

    def get_pitcher_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the pitcher prediction, including usage information.
        """
        self.logger.debug("get_pitcher_prediction_metadata called")

        try:
            start_time = kwargs.get("start_time")
            if start_time is None:
                raise ValueError("start_time is required")
            if not isinstance(start_time, datetime):
                raise ValueError("start_time must be a datetime object")
            end_time = kwargs.get("end_time")
            if end_time is None:
                raise ValueError("end_time is required")
            if not isinstance(end_time, datetime):
                raise ValueError("end_time must be a datetime object")
            n_pitches_total = kwargs.get("n_pitches_total")
            if n_pitches_total is None:
                raise ValueError("n_pitches_total is required")
            if not isinstance(n_pitches_total, int):
                raise ValueError("n_pitches_total must be an integer")
            n_pitches_sampled = kwargs.get("n_pitches_sampled")
            if n_pitches_sampled is None:
                raise ValueError("n_pitches_sampled is required")
            if not isinstance(n_pitches_sampled, int):
                raise ValueError("n_pitches_sampled must be an integer")
            sample_pctg = kwargs.get("sample_pctg")
            if sample_pctg is None:
                raise ValueError("sample_pctg is required")
            if not isinstance(sample_pctg, float):
                raise ValueError("sample_pctg must be a float")

            self.logger.info("get_pitcher_prediction_metadata completed successfully")
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "n_pitches_total": n_pitches_total,
                "n_pitches_sampled": n_pitches_sampled,
                "sample_pctg": sample_pctg,
            }
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_batter_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the batter prediction, including usage information.
        """
        self.logger.debug("get_batter_prediction_metadata called")

        try:
            start_time = kwargs.get("start_time")
            if start_time is None:
                raise ValueError("start_time is required")
            if not isinstance(start_time, datetime):
                raise ValueError("start_time must be a datetime object")
            end_time = kwargs.get("end_time")
            if end_time is None:
                raise ValueError("end_time is required")
            if not isinstance(end_time, datetime):
                raise ValueError("end_time must be a datetime object")
            n_pitches_total = kwargs.get("n_pitches_total")
            if n_pitches_total is None:
                raise ValueError("n_pitches_total is required")
            if not isinstance(n_pitches_total, int):
                raise ValueError("n_pitches_total must be an integer")
            n_pitches_sampled = kwargs.get("n_pitches_sampled")
            if n_pitches_sampled is None:
                raise ValueError("n_pitches_sampled is required")
            if not isinstance(n_pitches_sampled, int):
                raise ValueError("n_pitches_sampled must be an integer")
            sample_pctg = kwargs.get("sample_pctg")
            if sample_pctg is None:
                raise ValueError("sample_pctg is required")
            if not isinstance(sample_pctg, float):
                raise ValueError("sample_pctg must be a float")

            self.logger.info("get_batter_prediction_metadata completed successfully")
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "n_pitches_total": n_pitches_total,
                "n_pitches_sampled": n_pitches_sampled,
                "sample_pctg": sample_pctg,
            }
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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
        """
        Predict batted ball outcome probabilities given exit velocity, launch angle, and optional context.
        """
        self.logger.debug("predict_batted_ball called")

        try:
            start_time = datetime.now()
            self.logger.debug(f"start time: {start_time}")

            # Fetch batted ball data
            batted_balls = await get_all_batted_balls()
            self.logger.debug(f"successfully fetched {batted_balls.shape[0]} batted balls")

            sample_pctg = 0.05
            # Get similar batted balls using continuous similarity scoring
            similar_batted_balls = await self._get_similar_batted_balls(
                batted_balls=batted_balls,
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                spray_angle=spray_angle,
                bb_type=bb_type,
                outs=outs,
                bases_state=bases_state,
                batter_id=batter_id,
                game_date=game_date,
                sample_pctg=sample_pctg,
            )
            self.logger.debug(f"successfully found {similar_batted_balls.shape[0]} similar batted balls")

            # Digest outcome data
            basic_outcome_data, detailed_outcome_data = await self._digest_batted_ball_outcome_data(
                batted_balls=similar_batted_balls,
                outs=outs,
                bases_state=bases_state,
            )
            self.logger.debug("successfully digested batted ball outcome data")

            # Build prediction metadata
            prediction_metadata = self.get_batted_ball_prediction_metadata(
                start_time=start_time,
                end_time=datetime.now(),
                n_batted_balls_total=len(batted_balls),
                n_batted_balls_sampled=len(similar_batted_balls),
                sample_pctg=sample_pctg,
            )

            self.logger.info("predict_batted_ball completed successfully")

            return {
                "algorithm_metadata": self.get_metadata(),
                "basic_outcome_data": basic_outcome_data,
                "detailed_outcome_data": detailed_outcome_data,
                "prediction_metadata": prediction_metadata,
            }

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_similar_batted_balls(
        self,
        batted_balls: pd.DataFrame,
        launch_speed: float,
        launch_angle: float,
        spray_angle: float | None = None,
        bb_type: str | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        batter_id: int | None = None,
        game_date: str | None = None,
        sample_pctg: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get batted balls most similar to the given parameters using continuous similarity scoring.
        """
        self.logger.debug("_get_similar_batted_balls called")

        try:
            # Continuous similarity score for exit velocity (15 mph tolerance)
            batted_balls["score_launch_speed"] = batted_balls["launch_speed"].apply(
                lambda x: max(0, 1 - abs(x - launch_speed) / 15.0)
            )

            # Continuous similarity score for launch angle (20 degree tolerance)
            batted_balls["score_launch_angle"] = batted_balls["launch_angle"].apply(
                lambda x: max(0, 1 - abs(x - launch_angle) / 20.0)
            )

            # Spray angle similarity (if provided)
            if spray_angle is not None and "hc_x" in batted_balls.columns and "hc_y" in batted_balls.columns:
                # Calculate spray angle from hit coordinates (approximate)
                # hc_x: 0 = left field line, 125 = center, 250 = right field line
                # Convert to spray angle: -45 (left) to +45 (right)
                batted_balls["calc_spray_angle"] = ((batted_balls["hc_x"] - 125) / 125) * 45
                batted_balls["score_spray_angle"] = batted_balls["calc_spray_angle"].apply(
                    lambda x: max(0, 1 - abs(x - spray_angle) / 30.0) if pd.notna(x) else 0
                )
            else:
                batted_balls["score_spray_angle"] = 0.0

            # Batted ball type similarity (if provided)
            if bb_type is not None and "bb_type" in batted_balls.columns:
                batted_balls["score_bb_type"] = batted_balls["bb_type"].apply(
                    lambda x: 1.0 if x == bb_type else 0.0
                )
            else:
                batted_balls["score_bb_type"] = 0.0

            # Outs similarity (if provided)
            if outs is not None and "outs_when_up" in batted_balls.columns:
                batted_balls["score_outs"] = batted_balls["outs_when_up"].apply(
                    lambda x: 1.0 if x == outs else 0.0
                )
            else:
                batted_balls["score_outs"] = 0.0

            # Bases state similarity (if provided)
            if bases_state is not None and "on_1b" in batted_balls.columns:
                # Calculate bases state from on_1b, on_2b, on_3b columns
                def calc_bases_state(row):
                    state = 0
                    if pd.notna(row.get("on_1b")):
                        state |= 1
                    if pd.notna(row.get("on_2b")):
                        state |= 2
                    if pd.notna(row.get("on_3b")):
                        state |= 4
                    return state

                batted_balls["calc_bases_state"] = batted_balls.apply(calc_bases_state, axis=1)
                batted_balls["score_bases_state"] = batted_balls["calc_bases_state"].apply(
                    lambda x: 1.0 if x == bases_state else 0.5 if (x > 0 and bases_state > 0) else 0.0
                )
            else:
                batted_balls["score_bases_state"] = 0.0

            # Batter similarity (if provided)
            if batter_id is not None and "batter" in batted_balls.columns:
                batted_balls["score_batter"] = batted_balls["batter"].apply(
                    lambda x: 1.0 if x == batter_id else 0.0
                )
            else:
                batted_balls["score_batter"] = 0.0

            # Date similarity (if provided) - more recent dates get higher scores
            if game_date is not None and "game_date" in batted_balls.columns:
                target_date = datetime.strptime(game_date, "%Y-%m-%d")
                batted_balls["score_date"] = batted_balls["game_date"].apply(
                    lambda x: max(0, 1 - abs((datetime.strptime(str(x)[:10], "%Y-%m-%d") - target_date).days) / 365.0)
                    if pd.notna(x) else 0
                )
            else:
                batted_balls["score_date"] = 0.0

            # Weighted similarity score
            # EV and LA are the most important, optional fields contribute less
            batted_balls["similarity_score"] = (
                batted_balls["score_launch_speed"] * 0.45 +
                batted_balls["score_launch_angle"] * 0.40 +
                batted_balls["score_spray_angle"] * 0.05 +
                batted_balls["score_bases_state"] * 0.05 +
                batted_balls["score_outs"] * 0.03 +
                batted_balls["score_date"] * 0.02
            )

            # Sort by similarity and take top N%
            batted_balls = batted_balls.sort_values(by="similarity_score", ascending=False)
            n_samples = max(100, int(len(batted_balls) * sample_pctg))
            similar = batted_balls.head(n_samples)

            self.logger.info(f"successfully found {similar.shape[0]} similar batted balls")
            return similar

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _digest_batted_ball_outcome_data(
        self,
        batted_balls: pd.DataFrame,
        outs: int | None = None,
        bases_state: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Create a final summary of the batted ball outcome data.
        """
        self.logger.debug("_digest_batted_ball_outcome_data called")

        try:
            # Get outcome probabilities
            outcome_value_counts = batted_balls["events"].value_counts()
            total_events = outcome_value_counts.sum()

            # Map events to outcome categories
            outcome_mapping = {
                "single": "single",
                "double": "double",
                "triple": "triple",
                "home_run": "home_run",
                "field_out": "flyout",
                "grounded_into_double_play": "double_play",
                "double_play": "double_play",
                "force_out": "force_out",
                "fielders_choice": "force_out",
                "fielders_choice_out": "force_out",
                "sac_fly": "sac_fly",
                "sac_fly_double_play": "sac_fly",
                "field_error": "field_error",
                "sac_bunt": "groundout",
                "sac_bunt_double_play": "double_play",
            }

            # Aggregate outcomes into categories
            outcome_probs: dict[str, float] = {}
            for event, count in outcome_value_counts.items():
                category = outcome_mapping.get(str(event), None)
                if category:
                    outcome_probs[category] = outcome_probs.get(category, 0) + count / total_events

            # Infer batted ball type based on launch angle
            la_mean = batted_balls["launch_angle"].mean()
            if la_mean < 10:
                bb_type_inferred = "ground_ball"
            elif la_mean < 25:
                bb_type_inferred = "line_drive"
            elif la_mean < 50:
                bb_type_inferred = "fly_ball"
            else:
                bb_type_inferred = "popup"

            # Classify outs based on batted ball type
            # Redistribute generic "flyout" into more specific categories
            if "flyout" in outcome_probs:
                flyout_prob = outcome_probs.pop("flyout", 0)
                if bb_type_inferred == "ground_ball":
                    outcome_probs["groundout"] = outcome_probs.get("groundout", 0) + flyout_prob
                elif bb_type_inferred == "line_drive":
                    outcome_probs["lineout"] = outcome_probs.get("lineout", 0) + flyout_prob
                elif bb_type_inferred == "popup":
                    outcome_probs["popout"] = outcome_probs.get("popout", 0) + flyout_prob
                else:
                    outcome_probs["flyout"] = flyout_prob

            # Context-aware filtering
            # sac_fly: only if outs < 2 AND runner on 3B
            if outs is not None and bases_state is not None:
                runner_on_3b = (bases_state & 4) != 0
                if not (outs < 2 and runner_on_3b):
                    outcome_probs.pop("sac_fly", None)

                # double_play: only if at least one runner on base
                if bases_state == 0:
                    outcome_probs.pop("double_play", None)

                # force_out: only if force play possible (runner on 1B or bases loaded)
                runner_on_1b = (bases_state & 1) != 0
                if not runner_on_1b:
                    outcome_probs.pop("force_out", None)

            # Calculate hit probability and expected stats
            hits = batted_balls[batted_balls["events"].isin(["single", "double", "triple", "home_run"])]
            hit_probability = len(hits) / len(batted_balls) if len(batted_balls) > 0 else 0

            # Calculate xBA from the data
            xba = batted_balls["estimated_ba_using_speedangle"].mean() if "estimated_ba_using_speedangle" in batted_balls.columns else hit_probability

            basic = {
                "outcome_probs": outcome_probs,
                "hit_probability": hit_probability,
                "xba": xba if pd.notna(xba) else hit_probability,
                "bb_type_inferred": bb_type_inferred,
            }

            # Detailed outcome data
            launch_speed_mean = batted_balls["launch_speed"].mean()
            launch_angle_mean = batted_balls["launch_angle"].mean()

            xslg = batted_balls["estimated_slg_using_speedangle"].mean() if "estimated_slg_using_speedangle" in batted_balls.columns else 0
            xwoba = batted_balls["estimated_woba_using_speedangle"].mean() if "estimated_woba_using_speedangle" in batted_balls.columns else 0

            detailed = {
                "sample_launch_speed_mean": launch_speed_mean if pd.notna(launch_speed_mean) else 0,
                "sample_launch_angle_mean": launch_angle_mean if pd.notna(launch_angle_mean) else 0,
                "expected_stats": {
                    "xBA": xba if pd.notna(xba) else 0,
                    "xSLG": xslg if pd.notna(xslg) else 0,
                    "xwOBA": xwoba if pd.notna(xwoba) else 0,
                }
            }

            self.logger.info("_digest_batted_ball_outcome_data completed successfully")
            return basic, detailed

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_batted_ball_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the batted ball prediction, including usage information.
        """
        self.logger.debug("get_batted_ball_prediction_metadata called")

        try:
            start_time = kwargs.get("start_time")
            if start_time is None:
                raise ValueError("start_time is required")
            if not isinstance(start_time, datetime):
                raise ValueError("start_time must be a datetime object")
            end_time = kwargs.get("end_time")
            if end_time is None:
                raise ValueError("end_time is required")
            if not isinstance(end_time, datetime):
                raise ValueError("end_time must be a datetime object")
            n_batted_balls_total = kwargs.get("n_batted_balls_total")
            if n_batted_balls_total is None:
                raise ValueError("n_batted_balls_total is required")
            if not isinstance(n_batted_balls_total, int):
                raise ValueError("n_batted_balls_total must be an integer")
            n_batted_balls_sampled = kwargs.get("n_batted_balls_sampled")
            if n_batted_balls_sampled is None:
                raise ValueError("n_batted_balls_sampled is required")
            if not isinstance(n_batted_balls_sampled, int):
                raise ValueError("n_batted_balls_sampled must be an integer")
            sample_pctg = kwargs.get("sample_pctg")
            if sample_pctg is None:
                raise ValueError("sample_pctg is required")
            if not isinstance(sample_pctg, float):
                raise ValueError("sample_pctg must be a float")

            similarity_weights = {
                "launch_speed": 0.45,
                "launch_angle": 0.40,
                "spray_angle": 0.05,
                "bases_state": 0.05,
                "outs": 0.03,
                "date": 0.02,
            }

            self.logger.info("get_batted_ball_prediction_metadata completed successfully")
            return {
                "n_batted_balls_sampled": n_batted_balls_sampled,
                "sample_pctg": sample_pctg,
                "similarity_weights": similarity_weights,
            }
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _sample_pitches(
        self,
        pitches: pd.DataFrame,
        n: int = 1,
    ) -> list[api_types.Pitch]:
        """
        Sample n pitches from the given dataframe of similar pitches.
        """
        self.logger.debug("_sample_pitches called")

        try:
            if n > len(pitches):
                raise HTTPException(status_code=400, detail="n is greater than the number of pitches")

            sampled_pitches = pitches.sample(n)
            sampled_pitches_list: list[api_types.Pitch] = []

            def coerce_float(value: Any) -> float | None:
                if value is None or pd.isna(value):
                    return None
                return float(value)

            for pitch in sampled_pitches.itertuples():
                pitch_type = getattr(pitch, "pitch_type", None)
                if pitch_type is None or pd.isna(pitch_type):
                    continue

                speed = coerce_float(getattr(pitch, "release_speed", None))
                spin_rate = coerce_float(getattr(pitch, "release_spin_rate", None))
                spin_axis = coerce_float(getattr(pitch, "spin_axis", None))
                release_pos_x = coerce_float(getattr(pitch, "release_pos_x", None))
                release_pos_z = coerce_float(getattr(pitch, "release_pos_z", None))
                release_extension = coerce_float(getattr(pitch, "release_extension", None))
                vx0 = coerce_float(getattr(pitch, "vx0", None))
                vy0 = coerce_float(getattr(pitch, "vy0", None))
                vz0 = coerce_float(getattr(pitch, "vz0", None))
                ax = coerce_float(getattr(pitch, "ax", None))
                ay = coerce_float(getattr(pitch, "ay", None))
                az = coerce_float(getattr(pitch, "az", None))
                plate_pos_x = coerce_float(getattr(pitch, "plate_x", None))
                plate_pos_z = coerce_float(getattr(pitch, "plate_z", None))

                numeric_fields = [
                    speed,
                    spin_rate,
                    spin_axis,
                    release_pos_x,
                    release_pos_z,
                    release_extension,
                    vx0,
                    vy0,
                    vz0,
                    ax,
                    ay,
                    az,
                    plate_pos_x,
                    plate_pos_z,
                ]
                if any(value is None for value in numeric_fields):
                    continue

                result_value = getattr(pitch, "events", None)
                if result_value is None or pd.isna(result_value):
                    result_value = getattr(pitch, "description", None)
                result = "unknown" if result_value is None or pd.isna(result_value) else str(result_value)

                sampled_pitch = api_types.Pitch(
                    pitch_type=str(pitch_type),
                    speed=speed,
                    spin_rate=spin_rate,
                    spin_axis=spin_axis,
                    release_pos_x=release_pos_x,
                    release_pos_z=release_pos_z,
                    release_extension=release_extension,
                    vx0=vx0,
                    vy0=vy0,
                    vz0=vz0,
                    ax=ax,
                    ay=ay,
                    az=az,
                    plate_pos_x=plate_pos_x,
                    plate_pos_z=plate_pos_z,
                    result=result,
                )
                sampled_pitches_list.append(sampled_pitch)

            if len(sampled_pitches_list) < n:
                self.logger.warning("sampled pitches contain missing fields; returning fewer samples")

            return sampled_pitches_list

        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))
