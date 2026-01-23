# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
import uvicorn

from pitchpredict.api import PitchPredict
from pitchpredict.backend.fetching import (
    get_player_record_from_id,
    get_player_records_from_name,
)
import pitchpredict.utils as utils
import pitchpredict.types.api as api_types


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle server startup and shutdown events.
    """
    await _server_startup(app)

    try:
        yield
    finally:
        await _server_shutdown(app)


async def _server_startup(app: FastAPI):
    """
    Perform necessary startup tasks before the server is ready to handle requests.
    """
    app.state.start_time = datetime.now()
    app.state.api = PitchPredict()


async def _server_shutdown(app: FastAPI):
    """
    Perform necessary shutdown tasks after the server has finished handling requests.
    """
    pass


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {
        "name": "pitchpredict",
        "version": utils.get_version(),
        "status": "running",
        "uptime": datetime.now() - app.state.start_time,
    }


@app.post("/predict/pitcher")
async def predict_pitcher_endpoint(
    request: api_types.PredictPitcherRequest,
) -> api_types.PredictPitcherResponse:
    """
    Predict the pitcher's next pitch and outcome.
    """
    try:
        api = app.state.api
        result = await api.predict_pitcher(
            request=request,
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batter")
async def predict_batter_endpoint(
    request: api_types.PredictBatterRequest,
) -> api_types.PredictBatterResponse:
    """
    Predict the batter's next outcome.
    """
    try:
        api = app.state.api
        result = await api.predict_batter(
            batter_name=request.batter_name,
            pitcher_name=request.pitcher_name,
            balls=request.balls,
            strikes=request.strikes,
            score_bat=request.score_bat,
            score_fld=request.score_fld,
            game_date=request.game_date,
            pitch_type=request.pitch_type,
            pitch_speed=request.pitch_speed,
            pitch_x=request.pitch_x,
            pitch_z=request.pitch_z,
            algorithm=request.algorithm,
        )
        return api_types.PredictBatterResponse(
            basic_outcome_data=result["basic_outcome_data"],
            detailed_outcome_data=result["detailed_outcome_data"],
            prediction_metadata=result["prediction_metadata"],
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batted-ball")
async def predict_batted_ball_endpoint(
    request: api_types.PredictBattedBallRequest,
) -> api_types.PredictBattedBallResponse:
    """
    Predict batted ball outcome probabilities given exit velocity, launch angle, and optional game context.
    """
    try:
        api = app.state.api
        result = await api.predict_batted_ball(
            launch_speed=request.launch_speed,
            launch_angle=request.launch_angle,
            algorithm=request.algorithm,
            spray_angle=request.spray_angle,
            bb_type=request.bb_type,
            outs=request.outs,
            bases_state=request.bases_state,
            batter_id=request.batter_id,
            game_date=request.game_date,
        )
        return api_types.PredictBattedBallResponse(
            basic_outcome_data=result["basic_outcome_data"],
            detailed_outcome_data=result["detailed_outcome_data"],
            prediction_metadata=result["prediction_metadata"],
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players/lookup")
async def lookup_player_endpoint(
    name: str,
    fuzzy: bool = True,
    limit: int = 1,
) -> api_types.PlayerLookupResponse:
    """
    Lookup player IDs and metadata by name.
    """
    try:
        api = app.state.api
        cache = getattr(api, "cache", None)
        results = await get_player_records_from_name(
            player_name=name,
            fuzzy_lookup=fuzzy,
            limit=limit,
            cache=cache,
        )
        return api_types.PlayerLookupResponse(
            query=name,
            fuzzy=fuzzy,
            results=results,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players/{mlbam_id}")
async def get_player_by_id_endpoint(mlbam_id: int) -> api_types.PlayerRecordResponse:
    """
    Lookup player metadata by MLBAM ID.
    """
    try:
        api = app.state.api
        cache = getattr(api, "cache", None)
        record = await get_player_record_from_id(
            mlbam_id=mlbam_id,
            cache=cache,
        )
        return api_types.PlayerRecordResponse(
            mlbam_id=mlbam_id,
            record=record,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8056, reload: bool = False) -> None:
    """
    Run the FastAPI server.
    """
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
