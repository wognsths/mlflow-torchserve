from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
import requests

app = FastAPI()

TORCHSERVE_URL = "http://localhost:8080/predictions/mnist_model"

@app.post("/predict")
async def predict(file: UploadFile):
    img_base64 = base64.b64encode(await file.read()).decode()

    res = requests.post(
        TORCHSERVE_URL,
        json={"body": img_base64},
        headers={"Content-Type": "application/json"},
    )
    if not res.ok:
        try:
            detail = res.json()
        except Exception:
            detail = {"message": res.text}
        raise HTTPException(status_code=res.status_code, detail=detail)
    print(res.json())
    return {"digit": res.json()}

app.mount("/", StaticFiles(directory="app", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/index.html", encoding="utf-8") as f:
        return f.read()