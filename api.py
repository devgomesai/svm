from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import joblib
import json
import pandas as pd



app = FastAPI(
    title="SVM Variant Pathogenicity API",
    version="1.0.0",
    description="FastAPI service for SVM-based variant pathogenicity prediction"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"


try:
    model = joblib.load(f"{MODEL_DIR}/svm_model.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/svm_scaler.joblib")

    with open(f"{MODEL_DIR}/svm_features.json", "r") as f:
        FEATURES = json.load(f)["feature_names"]

except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")



def encode_chrom(chrom: str) -> int:
    chrom_str = str(chrom).replace("chr", "").upper()
    mapping = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return mapping.get(chrom_str, int(chrom_str) if chrom_str.isdigit() else -1)


ALLELE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def quick_predict_internal(variant_data: Dict[str, Any]) -> Dict[str, Any]:
    feature_values = {
        "chrom_enc": encode_chrom(variant_data["chrom"]),
        "ref_enc": ALLELE_MAP.get(variant_data["ref"].upper(), -1),
        "alt_enc": ALLELE_MAP.get(variant_data["alt"].upper(), -1),
        "gene_freq": variant_data.get("gene_freq", 0.01),
        "gene_count": variant_data.get("gene_count", 10),
        "sfari_score": variant_data.get("sfari_score", 1.0),
    }

    X = pd.DataFrame([feature_values])[FEATURES]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "Pathogenic" if prediction == 1 else "Non-pathogenic",
    }



class VariantRequest(BaseModel):
    chrom: str = Field(..., example="chr7")
    pos: int = Field(..., example=117199563)
    ref: str = Field(..., example="C")
    alt: str = Field(..., example="T")
    gene: str = Field(..., example="CFTR")
    sfari_score: float = Field(1.0, example=3.0)
    gene_freq: float | None = 0.01
    gene_count: int | None = 10


class PredictionResponse(BaseModel):
    prediction: int
    label: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]



@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_variant(variant: VariantRequest):
    try:
        return quick_predict_internal(variant.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(variants: List[VariantRequest]):
    try:
        results = [
            quick_predict_internal(v.dict()) for v in variants
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
