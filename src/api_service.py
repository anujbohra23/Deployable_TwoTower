import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel


#config
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts_clinical")
ICD_CSV_PATH = os.getenv("ICD_CSV_PATH", "/Users/anujbohra/Desktop/Healthcare/TwoTower/Data/icd_codes_8k.csv")
TXT_BACKBONE = os.getenv("TXT_BACKBONE", "emilyalsentzer/Bio_ClinicalBERT")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

LAB_KEYS = [
    "a1c","glucose","creatinine","egfr","ldl","hdl","triglycerides",
    "wbc","hgb","platelets","crp","troponin","bnp","alt","ast"
]


# Models and scaler
class EHRScaler:
    def __init__(self, means, stds):
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def transform_from_payload(self, age: Optional[float], sex: Optional[str], labs: Dict[str, float]) -> np.ndarray:
        # labs in same order as LAB_KEYS
        vals = []
        for k in LAB_KEYS:
            v = labs.get(k, None)
            if v is None:
                vals.append(np.nan)
            else:
                vals.append(float(v))
        labs_arr = np.array(vals, dtype=np.float32)
        labs_arr = np.where(np.isnan(labs_arr), self.means, labs_arr)
        labs_arr = (labs_arr - self.means) / self.stds

        age_val = float(age) / 100.0 if age is not None else 0.0
        sex_raw = (sex or "M").upper()
        sex_val = 1.0 if sex_raw.startswith("M") else 0.0
        return np.concatenate([labs_arr, [age_val, sex_val]]).astype(np.float32)


class PatientTower(torch.nn.Module):
    def __init__(self, txt_model: str, ehr_dim: int = len(LAB_KEYS) + 2, d: int = 768):
        super().__init__()
        self.txt_encoder = AutoModel.from_pretrained(txt_model)
        h = self.txt_encoder.config.hidden_size
        self.txt_proj = torch.nn.Linear(h, d)
        self.ehr_proj = torch.nn.Sequential(
            torch.nn.Linear(ehr_dim, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, d),
        )
        self.gate = torch.nn.Linear(2 * d, d)

    def forward(self, txt_inputs, ehr_vec):
        out = self.txt_encoder(**txt_inputs)
        # CLS token
        h_cls = out.last_hidden_state[:, 0]
        zt = F.normalize(self.txt_proj(h_cls), dim=-1)
        ze = F.normalize(self.ehr_proj(ehr_vec), dim=-1)
        z = torch.tanh(self.gate(torch.cat([zt, ze], dim=-1)))
        return F.normalize(z, dim=-1)



#fastapi schemas
class PredictRequest(BaseModel):
    note_text: str
    age: Optional[float] = None
    sex: Optional[str] = "M"
    labs: Dict[str, float] = {}
    top_k: int = 10


class PredictedCode(BaseModel):
    code: str
    title: str
    description: str
    score: float


class PredictResponse(BaseModel):
    codes: List[PredictedCode]



app = FastAPI(title="MedRetrieve ICD API")

_tok_txt = None
_scaler: Optional[EHRScaler] = None
_patient_model: Optional[PatientTower] = None
_Z_codes = None
_idx2code = None
_icd_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
def load_artifacts():
    global _tok_txt, _scaler, _patient_model, _Z_codes, _idx2code, _icd_df

    # 1) Load ICD catalog
    _icd_df = pd.read_csv(ICD_CSV_PATH)
    if "text" not in _icd_df.columns:
        # create text field if not present
        for col in ["title", "description", "synonyms"]:
            if col not in _icd_df.columns:
                _icd_df[col] = ""
        _icd_df["text"] = (
            _icd_df["title"].fillna("") + " \n" +
            _icd_df["description"].fillna("") + " \n" +
            _icd_df["synonyms"].fillna("")
        )
    _icd_df = _icd_df.reset_index(drop=True)

    # 2) Load scaler
    import json
    scaler_path = os.path.join(ARTIFACTS_DIR, "ehr_scaler.json")
    with open(scaler_path, "r") as f:
        ss = json.load(f)
    _scaler = EHRScaler(ss["means"], ss["stds"])

    # 3) Load patient model
    _tok_txt = AutoTokenizer.from_pretrained(TXT_BACKBONE, use_fast=True)
    _patient_model = PatientTower(TXT_BACKBONE).to(DEVICE)
    ckpt_pat = os.path.join(ARTIFACTS_DIR, "patient_tower.pt")
    _patient_model.load_state_dict(torch.load(ckpt_pat, map_location=DEVICE))
    _patient_model.eval()

    # 4) Load ICD embeddings (and code index mapping)
    # assuming code_embeds_epoch3.pt exists and includes code2idx/idx2code
    ckpt_codes = os.path.join(ARTIFACTS_DIR, "code_embeds_epoch3.pt")
    codes_obj = torch.load(ckpt_codes, map_location="cpu")
    _Z_codes = codes_obj["embeddings"]  # [N, d], normalized
    _idx2code = codes_obj["idx2code"]

    print("Artifacts loaded. Ready to serve predictions.")



#helpers
def _encode_patient(note_text: str, age: Optional[float], sex: Optional[str], labs: Dict[str, float]) -> torch.Tensor:
    ehr_vec = _scaler.transform_from_payload(age, sex, labs)
    ehr_t = torch.tensor(ehr_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    inputs = _tok_txt(
        note_text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        z = _patient_model(inputs, ehr_t)  # [1, d]
    return z  # [1, d]


def _retrieve_topk(z_patient: torch.Tensor, top_k: int):
    # Dense brute-force similarity; for 8k codes this is fine.
    Z = _Z_codes.to(DEVICE)  # [N, d]
    scores = (z_patient @ Z.T).squeeze(0)  # [N]
    vals, idx = torch.topk(scores, k=min(top_k, Z.size(0)))
    vals = vals.cpu().numpy()
    idx = idx.cpu().numpy()
    return vals, idx



#routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Encode patient
    z = _encode_patient(req.note_text, req.age, req.sex, req.labs)

    # 2) Retrieve top-k ICD codes
    scores, idxs = _retrieve_topk(z, top_k=req.top_k)

    # 3) Build response
    rows = []
    for score, idx in zip(scores, idxs):
        idx = int(idx)
        code = _idx2code.get(idx, "")
        row = _icd_df[_icd_df["code"] == code]
        if len(row) == 0:
            title = ""
            description = ""
        else:
            r0 = row.iloc[0]
            title = str(r0.get("title", ""))
            description = str(r0.get("description", ""))
        rows.append(
            PredictedCode(
                code=code,
                title=title,
                description=description,
                score=float(score),
            )
        )

    return PredictResponse(codes=rows)
