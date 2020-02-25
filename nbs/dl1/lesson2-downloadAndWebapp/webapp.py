#
# Example web app using an exported model
#
# setup using:
#   pip install starlette uvicorn aiohttp
#
# serve using:
#   python webapp.py serve
#
# use with:
#   http://localhost:8008/?url=<insert image url here>
#

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

async def homepage(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

# setup the learner

# dir must contain our exported trained model, export.pkl
learner = load_learner('../data/movementSet-v2')


# run the web app

app = Starlette(debug=True, routes=[
    Route('/', homepage),
])

if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)