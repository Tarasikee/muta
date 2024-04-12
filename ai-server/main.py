from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
async def predict(text: str) -> Union[str, dict]:
    return {"text": text, "label": "positive"}
