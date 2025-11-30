'''
we will normalize lab values, age and demoqraphics here
'''




from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from .constants import LAB_KEYS


@dataclass
class EHRScaler:
    means: np.ndarray
    stds: np.ndarray


    '''
    Takes the TRAIN subset of patients.Computes mean + std per lab feature.
    These are used to z-score labs.
    '''
    @classmethod
    def fit_from_df(cls, df: pd.DataFrame) -> "EHRScaler":
        lab_cols = [f"lab_{k}" for k in LAB_KEYS]
        X = df[lab_cols].astype(float).values
        means = X.mean(axis=0)
        stds = X.std(axis=0) + 1e-6
        return cls(means=means.astype("float32"), stds=stds.astype("float32"))

    def transform_row(self, row: pd.Series) -> np.ndarray:
        lab_cols = [f"lab_{k}" for k in LAB_KEYS]
        labs = np.array(
            [pd.to_numeric(row.get(c), errors="coerce") for c in lab_cols],
            dtype=np.float32,
        )
        labs = np.where(np.isnan(labs), self.means, labs)
        labs = (labs - self.means) / self.stds

        age = float(pd.to_numeric(row.get("age", 0.0), errors="coerce") or 0.0) / 100.0
        sex_raw = str(row.get("sex", "M")).upper()
        sex = 1.0 if sex_raw.startswith("M") else 0.0

        vec = np.concatenate([labs, np.array([age, sex], dtype=np.float32)])
        return vec.astype(np.float32)

    def to_json(self) -> str:
        payload = {"means": self.means.tolist(), "stds": self.stds.tolist()}
        return json.dumps(payload)

    @classmethod
    def from_json(cls, s: str) -> "EHRScaler":
        payload: Dict[str, Any] = json.loads(s)
        means = np.array(payload["means"], dtype=np.float32)
        stds = np.array(payload["stds"], dtype=np.float32)
        return cls(means=means, stds=stds)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "EHRScaler":
        with open(path, "r") as f:
            s = f.read()
        return cls.from_json(s)
