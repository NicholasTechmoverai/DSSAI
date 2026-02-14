from fastapi import APIRouter, Body
from .predict import Predict as pred

router = APIRouter()

@router.post("/predict-api")
async def prediction_requests(x_features = Body(...)):
    model_results = await pred(X_features=x_features)
    return model_results
