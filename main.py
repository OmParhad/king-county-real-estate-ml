from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = joblib.load("rf_model.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: dict):

    feature_order = model.feature_names_in_

    input_df = pd.DataFrame([data])

    # Add missing columns automatically
    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    # Force correct order
    input_df = input_df[feature_order]

    prediction = model.predict(input_df)
    prediction = int(np.expm1(prediction[0]))

    return {"prediction": float(prediction)}