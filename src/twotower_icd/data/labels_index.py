'''
store which ICD codes are assigned to which encounter, in index space 
(not string space).
'''


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class PatientLabelIndex:
    enc2codes: Dict[int, List[int]]

    @classmethod
    def from_df(cls, labels_df: pd.DataFrame, code2idx: Dict[str, int]) -> "PatientLabelIndex":
        enc2codes: Dict[int, List[int]] = {}
        for enc, g in labels_df.groupby("encounter_id"):
            idxs = []
            for c in g["code"].tolist():
                if c in code2idx:
                    idxs.append(code2idx[c])
            if idxs:
                enc2codes[int(enc)] = sorted(set(idxs))
        return cls(enc2codes=enc2codes)

    def positives(self, enc_id: int) -> List[int]:
        return self.enc2codes.get(int(enc_id), [])
