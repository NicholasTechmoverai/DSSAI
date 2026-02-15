from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from AI.all.api_predict import predict
router = APIRouter()

FRONTEND_PATH = Path("Frontend/index.html")


@router.get("/", response_class=HTMLResponse)
async def root():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")

    return FRONTEND_PATH.read_text(encoding="utf-8")


@router.post("/predict")
async def prediction_requests(payload: dict = Body(...)):
    symptom1 = payload.get("symptom-1", "").strip()
    symptom2 = payload.get("symptom-2", "").strip()
    symptom3 = payload.get("symptom-3", "").strip()

    if not any([symptom1, symptom2, symptom3]):
        raise HTTPException(
            status_code=400,
            detail="At least one symptom must be provided",
        )

    result = await predict(symptom1, symptom2, symptom3)
    return result
