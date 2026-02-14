from fastapi import APIRouter
from .predict import Predict as pred

router = APIRouter()

@router.post("/predict-api")
async def prediction_requests(x_features):
    model_results = pred(X_features=x_features)
    return model_results