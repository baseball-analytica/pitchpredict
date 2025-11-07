# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime

from fastapi import FastAPI
import uvicorn

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
async def predict_pitcher(request: server_types.PredictPitcherRequest) -> server_types.PredictPitcherResponse:
    """
    Predict the pitcher's next pitch and outcome.
    """
    raise NotImplementedError("Not implemented")


@app.post("/predict/batter")
async def predict_batter(request: server_types.PredictBatterRequest) -> server_types.PredictBatterResponse:
    """
    Predict the batter's next outcome.
    """
    raise NotImplementedError("Not implemented")


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