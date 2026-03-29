from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from inference import ICDRetriever

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ICD Retrieval Service")

retriever = ICDRetriever(
    artifacts_dir=str(BASE_DIR / "artifacts_clean"),
    icd_csv_path=str(BASE_DIR / "artifacts_clean" / "icd_codes_8k.csv"),
    device="cpu",
    txt_backbone="distilbert-base-uncased",
)

class PredictRequest(BaseModel):
    clinical_note: str
    lab_values: Dict[str, float]
    age: float
    sex: str
    top_k: int = 20

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    results = retriever.predict(
        clinical_note=req.clinical_note,
        lab_values=req.lab_values,
        age=req.age,
        sex=req.sex,
        top_k=req.top_k,
    )
    return {"results": results}