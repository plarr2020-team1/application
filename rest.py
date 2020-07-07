import io
import cv2
import base64
import random
import string
import numpy as np

from tools import get_res
from base64 import decodestring
from fastapi import FastAPI, File
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import StreamingResponse

def random_string(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

app = FastAPI()

app.mount("/results", StaticFiles(directory="results"), name="results")

@app.post("/predict")
def predict(request: Request, 
            file: bytes = File(...)):

    img = cv2.imdecode(np.fromstring(io.BytesIO(file).read(), np.uint8), 1)
    _, res_img = get_res(img)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

    img_name = f'results/{random_string()}.png'
    cv2.imwrite(img_name, res_img)

    return {
            "file_name": img_name
    } #base64.b64encode(res_img.tobytes())
