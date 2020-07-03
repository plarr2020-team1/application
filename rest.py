import io
import cv2
import base64
import numpy as np

from tools import get_res
from base64 import decodestring
from fastapi import FastAPI, File
from starlette.requests import Request
from starlette.responses import StreamingResponse

app = FastAPI()

@app.post("/predict")
def predict(request: Request, 
            file: bytes = File(...)):

    img = cv2.imdecode(np.fromstring(io.BytesIO(file).read(), np.uint8), 1)
    _, res_img = get_res(img)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

    return base64.b64encode(res_img.tobytes())
