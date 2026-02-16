# """
# Inference module for ICD code retrieval system.
# Loads trained models and provides easy-to-use prediction interface.
# """
# from __future__ import annotations
# import os
# from typing import Dict, List, Tuple
# import numpy as np
# import pandas as pd
# import torch
# from transformers import AutoTokenizer

# from twotower_icd.config import TrainingConfig
# from twotower_icd.data.scalers import EHRScaler
# from twotower_icd.data.icd_catalog import ICDCatalog
# from twotower_icd.models.patient_tower import PatientTower
# from twotower_icd.data.constants import LAB_KEYS


# class ICDRetriever:
#     """
#     High-level interface for ICD code retrieval.
#     Loads all necessary artifacts and provides predict() method.
#     """
    
#     def __init__(
#         self,
#         artifacts_dir: str = "./artifacts",
#         icd_csv_path: str = "Data/icd_codes_8k.csv",
#         device: str = "cpu",
#         txt_backbone: str = "emilyalsentzer/Bio_ClinicalBERT",
#     ):
#         """
#         Initialize the retriever with trained models and artifacts.
        
#         Args:
#             artifacts_dir: Directory containing saved models and embeddings
#             icd_csv_path: Path to ICD codes CSV
#             device: Device to run inference on ('cpu', 'cuda', 'mps')
#             txt_backbone: Text encoder backbone model name
#         """
#         self.device = torch.device(device)
#         self.artifacts_dir = artifacts_dir
        
#         # Load ICD catalog
#         print("Loading ICD catalog...")
#         icd_df = pd.read_csv(icd_csv_path)
#         self.icd_catalog = ICDCatalog.from_df(icd_df)
        
#         # Load EHR scaler
#         print("Loading EHR scaler...")
#         scaler_path = os.path.join(artifacts_dir, "ehr_scaler.json")
#         self.scaler = EHRScaler.load(scaler_path)
        
#         # Load tokenizer
#         print("Loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(txt_backbone, use_fast=True)
        
#         # Load patient tower model
#         print("Loading patient tower model...")
#         self.patient_model = PatientTower(
#             txt_model_name=txt_backbone,
#             d=768,
#         ).to(self.device)
        
#         model_path = os.path.join(artifacts_dir, "patient_tower.pt")
#         self.patient_model.load_state_dict(
#             torch.load(model_path, map_location=self.device)
#         )
#         self.patient_model.eval()
        
#         # Load ICD embeddings
#         print("Loading ICD embeddings...")
#         code_files = [f for f in os.listdir(artifacts_dir) if f.startswith("code_embeds_epoch")]
#         if not code_files:
#             raise FileNotFoundError(f"No code embeddings found in {artifacts_dir}")
        
#         latest_embeds = sorted(code_files)[-1]
#         embeds_path = os.path.join(artifacts_dir, latest_embeds)
#         ckpt = torch.load(embeds_path, map_location="cpu")
        
#         self.code_embeddings = ckpt["embeddings"].to(self.device)  # [N_codes, d]
#         self.code2idx = ckpt["code2idx"]
#         self.idx2code = ckpt["idx2code"]
        
#         print(f"✓ Loaded {len(self.idx2code)} ICD codes")
#         print(f"✓ Model ready on {self.device}")
    
#     def predict(
#         self,
#         clinical_note: str,
#         lab_values: Dict[str, float],
#         age: float,
#         sex: str,
#         top_k: int = 20,
#     ) -> List[Dict[str, any]]:
#         """
#         Predict top-K ICD codes for a patient encounter.
        
#         Args:
#             clinical_note: Patient's clinical notes/discharge summary
#             lab_values: Dict of lab values, e.g. {'a1c': 6.5, 'glucose': 120, ...}
#             age: Patient age
#             sex: Patient sex ('M' or 'F')
#             top_k: Number of top codes to return
            
#         Returns:
#             List of dicts with keys: 'code', 'title', 'description', 'score', 'rank'
#         """
#         # 1. Prepare EHR vector
#         ehr_vec = self._prepare_ehr_vector(lab_values, age, sex)
#         ehr_tensor = torch.tensor(ehr_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         # 2. Tokenize clinical note
#         txt_inputs = self.tokenizer(
#             clinical_note[:20000],  # truncate to reasonable length
#             padding=True,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt",
#         )
#         txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
        
#         # 3. Encode patient
#         with torch.no_grad():
#             z_patient = self.patient_model(txt_inputs, ehr_tensor)  # [1, 768]
        
#         # 4. Compute similarity with all ICD codes
#         scores = (z_patient @ self.code_embeddings.T).squeeze(0)  # [N_codes]
        
#         # 5. Get top-K
#         topk_scores, topk_indices = torch.topk(scores, k=min(top_k, len(scores)))
        
#         # 6. Format results
#         results = []
#         for rank, (idx, score) in enumerate(zip(topk_indices.cpu().numpy(), topk_scores.cpu().numpy()), 1):
#             code = self.idx2code[int(idx)]
#             icd_row = self.icd_catalog.items[self.icd_catalog.items['code'] == code].iloc[0]
            
#             # Parse title and description from text field
#             text_parts = icd_row['text'].split('\n')
#             title = text_parts[0] if len(text_parts) > 0 else code
#             description = text_parts[1] if len(text_parts) > 1 else ""
            
#             results.append({
#                 'rank': rank,
#                 'code': code,
#                 'title': title.strip(),
#                 'description': description.strip(),
#                 'score': float(score),
#                 'confidence': self._score_to_confidence(float(score)),
#             })
        
#         return results
    
#     def _prepare_ehr_vector(
#         self,
#         lab_values: Dict[str, float],
#         age: float,
#         sex: str
#     ) -> np.ndarray:
#         """Prepare EHR vector from lab values, age, and sex."""
#         # Create a mock row with lab values
#         row_dict = {}
#         for lab in LAB_KEYS:
#             key = f"lab_{lab}"
#             row_dict[key] = lab_values.get(lab, np.nan)
        
#         row_dict['age'] = age
#         row_dict['sex'] = sex
        
#         row = pd.Series(row_dict)
#         return self.scaler.transform_row(row)
    
#     def _score_to_confidence(self, score: float) -> str:
#         """Convert similarity score to confidence level."""
#         if score >= 0.7:
#             return "High"
#         elif score >= 0.5:
#             return "Medium"
#         else:
#             return "Low"
    
#     def get_lab_keys(self) -> List[str]:
#         """Return list of expected lab keys."""
#         return LAB_KEYS



"""
Inference module for ICD code retrieval system.
Loads trained models and provides easy-to-use prediction interface.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from twotower_icd.config import TrainingConfig
from twotower_icd.data.scalers import EHRScaler
from twotower_icd.data.icd_catalog import ICDCatalog
from twotower_icd.models.patient_tower import PatientTower
from twotower_icd.data.constants import LAB_KEYS


class ICDRetriever:
    """
    High-level interface for ICD code retrieval.
    Loads all necessary artifacts and provides predict() method.
    """
    
    def __init__(
        self,
        artifacts_dir: str = "./artifacts",
        icd_csv_path: str = "Data/icd_codes_8k.csv",
        device: str = "cpu",
        txt_backbone: str = "distilbert-base-uncased",  # Changed to match your trained model
    ):
        """
        Initialize the retriever with trained models and artifacts.
        
        Args:
            artifacts_dir: Directory containing saved models and embeddings
            icd_csv_path: Path to ICD codes CSV
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            txt_backbone: Text encoder backbone model name
        """
        self.device = torch.device(device)
        self.artifacts_dir = artifacts_dir
        
        # Load ICD catalog
        print("Loading ICD catalog...")
        icd_df = pd.read_csv(icd_csv_path)
        self.icd_catalog = ICDCatalog.from_df(icd_df)
        
        # Load EHR scaler
        print("Loading EHR scaler...")
        scaler_path = os.path.join(artifacts_dir, "ehr_scaler.json")
        self.scaler = EHRScaler.load(scaler_path)
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(txt_backbone, use_fast=True)
        
        # Load patient tower model
        print("Loading patient tower model...")
        self.patient_model = PatientTower(
            txt_model_name=txt_backbone,
            d=768,
        ).to(self.device)
        
        model_path = os.path.join(artifacts_dir, "patient_tower.pt")
        self.patient_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.patient_model.eval()
        
        # Load ICD embeddings
        print("Loading ICD embeddings...")
        code_files = [f for f in os.listdir(artifacts_dir) if f.startswith("code_embeds_epoch")]
        if not code_files:
            raise FileNotFoundError(f"No code embeddings found in {artifacts_dir}")
        
        latest_embeds = sorted(code_files)[-1]
        embeds_path = os.path.join(artifacts_dir, latest_embeds)
        ckpt = torch.load(embeds_path, map_location="cpu")
        
        self.code_embeddings = ckpt["embeddings"].to(self.device)  # [N_codes, d]
        self.code2idx = ckpt["code2idx"]
        self.idx2code = ckpt["idx2code"]
        
        print(f"✓ Loaded {len(self.idx2code)} ICD codes")
        print(f"✓ Model ready on {self.device}")
    
    def predict(
        self,
        clinical_note: str,
        lab_values: Dict[str, float],
        age: float,
        sex: str,
        top_k: int = 20,
    ) -> List[Dict[str, any]]:
        """
        Predict top-K ICD codes for a patient encounter.
        
        Args:
            clinical_note: Patient's clinical notes/discharge summary
            lab_values: Dict of lab values, e.g. {'a1c': 6.5, 'glucose': 120, ...}
            age: Patient age
            sex: Patient sex ('M' or 'F')
            top_k: Number of top codes to return
            
        Returns:
            List of dicts with keys: 'code', 'title', 'description', 'score', 'rank'
        """
        # 1. Prepare EHR vector
        ehr_vec = self._prepare_ehr_vector(lab_values, age, sex)
        ehr_tensor = torch.tensor(ehr_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Tokenize clinical note
        txt_inputs = self.tokenizer(
            clinical_note[:20000],  # truncate to reasonable length
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
        
        # 3. Encode patient
        with torch.no_grad():
            z_patient = self.patient_model(txt_inputs, ehr_tensor)  # [1, 768]
        
        # 4. Compute similarity with all ICD codes
        scores = (z_patient @ self.code_embeddings.T).squeeze(0)  # [N_codes]
        
        # 5. Get top-K
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, len(scores)))
        
        # 6. Format results
        results = []
        for rank, (idx, score) in enumerate(zip(topk_indices.cpu().numpy(), topk_scores.cpu().numpy()), 1):
            code = self.idx2code[int(idx)]
            icd_row = self.icd_catalog.items[self.icd_catalog.items['code'] == code].iloc[0]
            
            # Parse title and description from text field
            text_parts = icd_row['text'].split('\n')
            title = text_parts[0] if len(text_parts) > 0 else code
            description = text_parts[1] if len(text_parts) > 1 else ""
            
            results.append({
                'rank': rank,
                'code': code,
                'title': title.strip(),
                'description': description.strip(),
                'score': float(score),
                'confidence': self._score_to_confidence(float(score)),
            })
        
        return results
    
    def _prepare_ehr_vector(
        self,
        lab_values: Dict[str, float],
        age: float,
        sex: str
    ) -> np.ndarray:
        """Prepare EHR vector from lab values, age, and sex."""
        # Create a mock row with lab values
        row_dict = {}
        for lab in LAB_KEYS:
            key = f"lab_{lab}"
            row_dict[key] = lab_values.get(lab, np.nan)
        
        row_dict['age'] = age
        row_dict['sex'] = sex
        
        row = pd.Series(row_dict)
        return self.scaler.transform_row(row)
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert similarity score to confidence level."""
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def get_lab_keys(self) -> List[str]:
        """Return list of expected lab keys."""
        return LAB_KEYS