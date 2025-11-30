'''
this will define the ICD structure. basically, it will unify all ICD 
metadata into a clean object that ICD tower and training loop can use.
'''


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class ICDCatalog:
    items: pd.DataFrame           # columns: ["code", "text"]
    code2idx: Dict[str, int]
    idx2code: Dict[int, str]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ICDCatalog":
        # Prefer leaf codes as items; fallback to all
        if "level" in df.columns:
            leaves = df[df["level"] == "leaf"].copy()
            if leaves.empty:
                leaves = df.copy()
        else:
            leaves = df.copy()

        leaves["title"] = leaves["title"].fillna("").astype(str)
        leaves["description"] = leaves["description"].fillna("").astype(str)
        if "synonyms" in leaves.columns:
            leaves["synonyms"] = leaves["synonyms"].fillna("").astype(str)
        else:
            leaves["synonyms"] = ""

        leaves["text"] = (
            leaves["title"]
            + " \n"
            + leaves["description"]
            + " \n"
            + leaves["synonyms"]
        )
        items = leaves[["code", "text"]].reset_index(drop=True)
        code2idx = {c: i for i, c in enumerate(items["code"].tolist())}
        idx2code = {i: c for c, i in code2idx.items()}
        return cls(items=items, code2idx=code2idx, idx2code=idx2code)

    def code_texts(self) -> List[str]:
        return self.items["text"].tolist()
