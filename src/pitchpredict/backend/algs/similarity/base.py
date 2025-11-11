# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from collections import defaultdict
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
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)

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
        try:
            pitcher_id = await get_player_id_from_name(pitcher_name)
            batter_id = await get_player_id_from_name(batter_name)

            pitches = await get_pitches_from_pitcher(pitcher_id, game_date)

            similar_pitches = await self._get_similar_pitches(
                pitches=pitches,
                batter_id=batter_id,
                pitcher_id=pitcher_id,
                is_batter=False,
                balls=balls,
                strikes=strikes,
                score_bat=score_bat,
                score_fld=score_fld,
                game_date=game_date
            )

            basic_pitch_data, detailed_pitch_data = await self._digest_pitch_data(similar_pitches)

            print(basic_pitch_data)
            print(detailed_pitch_data)

            raise NotImplementedError("Not implemented")
        except HTTPException as e:
            raise e
        except Exception as e:
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
        try:
            pitcher_id = await get_player_id_from_name(pitcher_name)
            batter_id = await get_player_id_from_name(batter_name)

            pitches = await get_pitches_to_batter(batter_id, game_date)

            similar_pitches = await self._get_similar_pitches(
                pitches=pitches,
                batter_id=batter_id,
                pitcher_id=pitcher_id,
                is_batter=True,
                balls=balls,
                strikes=strikes,
                score_bat=score_bat,
                score_fld=score_fld,
                game_date=game_date
            )

            basic_outcome_data, detailed_outcome_data = await self._digest_outcome_data(similar_pitches, pitch_type, pitch_speed, pitch_x, pitch_z)

            print(basic_outcome_data)
            print(detailed_outcome_data)

            raise NotImplementedError("Not implemented")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_similar_pitches(
        self,
        pitches: pd.DataFrame,
        batter_id: int,
        pitcher_id: int,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
        is_batter: bool = False,
        sample_pctg: float = 0.005,
    ) -> pd.DataFrame:
        """
        Get the pitches most similar to the given context.
        """
        try:
            # append "similarity score" column to pitches for each parameter
            if is_batter:
                # 'score_pitcher_name': 1 if pitcher_name is the same as the pitcher in the pitch, 0 otherwise
                pitches["score_pitcher_name"] = pitches["pitcher"].apply(lambda x: 1 if x == pitcher_id else 0)
            else:
                # 'score_batter_name': 1 if batter_name is the same as the batter in the pitch, 0 otherwise
                pitches["score_batter_name"] = pitches["pitcher"].apply(lambda x: 1 if x == pitcher_id else 0)
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
                (pitches["score_batter_name"] if not is_batter else pitches["score_pitcher_name"]) * 0.35 +
                pitches["score_balls"] * 0.2 +
                pitches["score_strikes"] * 0.2 +
                pitches["score_score_bat"] * 0.1 +
                pitches["score_score_fld"] * 0.1 +
                pitches["score_game_date"] * 0.05
            )

            # sort pitches by similarity score and return the top N pitches
            pitches = pitches.sort_values(by="similarity_score", ascending=False)
            pitches = pitches.head(int(len(pitches) * sample_pctg))

            return pitches

        except HTTPException as e:
            raise e
        except Exception as e:
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

            return basic, detailed
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _digest_outcome_data(
        self,
        pitches: pd.DataFrame,
        pitch_type: str,
        pitch_speed: float,
        pitch_x: float,
        pitch_z: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Create a final summary of the outcome data.
        """
        try:
            outcome_value_counts = pitches["type"].value_counts()
            outcome_probs = outcome_value_counts / outcome_value_counts.sum()
            
            new_column_names = {
                "S": "strike",
                "B": "ball",
                "X": "contact",
            }

            basic = {
                "outcome_probs": outcome_probs.rename(index=new_column_names).to_dict(),
            }

            detailed = {
                "TODO": "TODO",
            }

            return basic, detailed
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))