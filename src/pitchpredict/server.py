# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime

from fastapi import FastAPI, HTTPException
import uvicorn

from pitchpredict.api import PitchPredict
import pitchpredict.utils as utils
import pitchpredict.types.server as server_types


async def lifespan(app: FastAPI):
    """
    Handle server startup and shutdown events.
    """
    await _server_startup(app)

    yield

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
        "uptime": datetime.now() - app.state.start_time
    }


@app.post("/predict/pitcher")
async def predict_pitcher_endpoint(request: server_types.PredictPitcherRequest) -> server_types.PredictPitcherResponse:
    """
    Predict the pitcher's next pitch and outcome.
    """
    try:
        api = app.state.api
        result = await api.predict_pitcher(
            pitcher_name=request.pitcher_name,
            batter_name=request.batter_name,
            balls=request.balls,
            strikes=request.strikes,
            score_bat=request.score_bat,
            score_fld=request.score_fld,
            game_date=request.game_date,
            algorithm=request.algorithm,
        )
        return server_types.PredictPitcherResponse(
            basic_pitch_data=result["basic_pitch_data"],
            detailed_pitch_data=result["detailed_pitch_data"],
            basic_outcome_data=result["basic_outcome_data"],
            detailed_outcome_data=result["detailed_outcome_data"],
            prediction_metadata=result["prediction_metadata"],
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batter")
async def predict_batter_endpoint(request: server_types.PredictBatterRequest) -> server_types.PredictBatterResponse:
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
            pitch_y=request.pitch_y,
            algorithm=request.algorithm,
        )
        return server_types.PredictBatterResponse(
            basic_outcome_data=result["basic_outcome_data"],
            detailed_outcome_data=result["detailed_outcome_data"],
            prediction_metadata=result["prediction_metadata"],
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8056,
    reload: bool = False
) -> None:
    """
    Run the FastAPI server.
    """
    uvicorn.run(
        app, 
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()