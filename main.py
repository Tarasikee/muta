import joblib
import pandas as pd

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

model = joblib.load(open("./dumps/model.pkl", "rb"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
        request: Request,
        pregnancies: int = Form(...),
        glucose: int = Form(...),
        bloodpressure: int = Form(...),
        skinthickness: int = Form(...),
        insulin: int = Form(...),
        bmi: float = Form(...),
        dpf: float = Form(...),
        age: int = Form(...),
):
    data = pd.DataFrame(
        [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age",
        ],
    )
    prediction = model.predict(data)
    print(prediction)
    return templates.TemplateResponse(
        "result.html", context={"prediction": prediction, "request": request}
    )
