# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime
import logging
from typing import Any

from fastapi import HTTPException
import pandas as pd

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.fetching import get_pitches_from_pitcher, get_pitches_to_batter, get_player_id_from_name


class SimilarityAlgorithm(PitchPredictAlgorithm):
    """
    The PitchPredict algorithm that uses similarity and nearest-neighbor analysis to predict the next pitch and outcome.
    """

    def __init__(
        self,
        name: str = "similarity",
        sample_pctg: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.sample_pctg = sample_pctg
        self.logger = logging.getLogger("pitchpredict.backend.algs.similarity")

    async def predict_pitcher(
        self,
        pitcher_name: str,
        batter_name: str,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
    ) -> dict[str, Any]:
        """
        Predict the pitcher's next pitch and its outcome.
        """
        self.logger.debug("predict_pitcher called")

        try:
            start_time = datetime.now()
            self.logger.debug(f"start time: {start_time}")

            pitcher_id = await get_player_id_from_name(pitcher_name)
            batter_id = await get_player_id_from_name(batter_name)

            pitches = await get_pitches_from_pitcher(pitcher_id, game_date)
            self.logger.debug(f"successfully fetched {pitches.shape[0]} pitches")

            similar_pitches = await self._get_similar_pitches_for_pitcher(
                pitches=pitches,
                batter_id=batter_id,
                balls=balls,
                strikes=strikes,
                score_bat=score_bat,
                score_fld=score_fld,
                game_date=game_date
            )
            self.logger.debug(f"successfully fetched {similar_pitches.shape[0]} similar pitches")

            basic_pitch_data, detailed_pitch_data = await self._digest_pitch_data(similar_pitches)
            self.logger.debug("successfully digested pitch data")

            basic_outcome_data, detailed_outcome_data = await self._digest_outcome_data(
                pitches=similar_pitches
            )
            self.logger.debug("successfully digested outcome data")

            prediction_metadata = self.get_pitcher_prediction_metadata(
                start_time=start_time,
                end_time=datetime.now(),
                n_pitches_total=len(pitches),
                n_pitches_sampled=len(similar_pitches),
            )

            self.logger.info("predict_pitcher completed successfully")

            return {
                "algorithm_metadata": self.get_metadata(),
                "basic_pitch_data": basic_pitch_data,
                "detailed_pitch_data": detailed_pitch_data,
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

            pitches = await get_pitches_to_batter(batter_id, game_date)
            self.logger.debug(f"successfully fetched {pitches.shape[0]} pitches")

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
        batter_id: int,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
    ) -> pd.DataFrame:
        """
        Get the pitches most similar to the given context for this pitcher.
        """
        self.logger.debug("get_similar_pitches_for_pitcher called")

        try:
            # append "similarity score" column to pitches for each parameter
            # 'score_batter_name': 1 if batter_name is the same as the batter in the pitch, 0 otherwise
            pitches["score_batter_name"] = pitches["batter"].apply(lambda x: 1 if x == batter_id else 0)
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

            # create a single similarity score: a weighted average of the individual similarity scores above
            pitches["similarity_score"] = (
                pitches["score_batter_name"] * 0.35 +
                pitches["score_balls"] * 0.2 +
                pitches["score_strikes"] * 0.2 +
                pitches["score_score_bat"] * 0.1 +
                pitches["score_score_fld"] * 0.1 +
                pitches["score_game_date"] * 0.05
            )

            # sort pitches by similarity score and return the top N pitches
            pitches = pitches.sort_values(by="similarity_score", ascending=False)
            pitches = pitches.head(int(len(pitches) * self.sample_pctg))

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
            pitches = pitches.head(int(len(pitches) * self.sample_pctg))

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
            "sample_pctg": self.sample_pctg,
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

            self.logger.info("get_pitcher_prediction_metadata completed successfully")
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "n_pitches_total": n_pitches_total,
                "n_pitches_sampled": n_pitches_sampled,
                "sample_pctg": self.sample_pctg,
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

            self.logger.info("get_batter_prediction_metadata completed successfully")
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "n_pitches_total": n_pitches_total,
                "n_pitches_sampled": n_pitches_sampled,
                "sample_pctg": self.sample_pctg,
            }
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))