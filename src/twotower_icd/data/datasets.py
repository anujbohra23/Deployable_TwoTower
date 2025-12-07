'''
encapsulate one split (train / val / test) and expose rows as PyTorch-ready samples.
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .scalers import EHRScaler
from .labels_index import PatientLabelIndex
from .constants import LAB_KEYS


class PatientDataset(Dataset):
    """
    One row = one encounter.
    Filters to encounters with at least one ICD label.
    """

    def __init__(
        self,
        patients_df: pd.DataFrame,
        splits_df: pd.DataFrame,
        split: str,
        scaler: EHRScaler,
        label_index: PatientLabelIndex,
    ):
        self.scaler = scaler
        self.label_index = label_index

        df = patients_df.merge(splits_df, on="encounter_id", how="left")
        df = df[df["split"] == split].reset_index(drop=True)
        df = df[df["encounter_id"].isin(label_index.enc2codes.keys())].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        enc_id = int(row["encounter_id"])
        ehr_vec = self.scaler.transform_row(row)
        note_text = str(row.get("note_text", ""))[:20000]
        pos_codes = self.label_index.positives(enc_id)
        return {
            "encounter_id": enc_id,
            "ehr": torch.tensor(ehr_vec, dtype=torch.float32),
            "text": note_text,
            "pos_codes": pos_codes,
        }


@dataclass
#convert a list of those dicts (a batch) into batched tensors.
class PatientCollator:
    txt_tokenizer: PreTrainedTokenizerBase
    max_seq_len: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        ehr = torch.stack([b["ehr"] for b in batch], dim=0)
        pos_codes = [b["pos_codes"] for b in batch]
        enc_ids = [b["encounter_id"] for b in batch]

        txt_inputs = self.txt_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return {
            "encounter_id": torch.tensor(enc_ids, dtype=torch.long),
            "ehr": ehr,
            "txt_inputs": txt_inputs,
            "pos_codes": pos_codes,
        }
