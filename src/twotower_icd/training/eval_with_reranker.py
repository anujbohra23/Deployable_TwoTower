"""
Evaluation pipeline with re-ranking stage.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from ..models.reranker import LLMReranker, RerankResult


def evaluate_with_reranker(
    patient_tower,
    code_embeddings: torch.Tensor,
    icd_catalog,
    test_loader: DataLoader,
    labels_index,
    reranker: LLMReranker,
    retrieval_k: int = 100,
    rerank_top_n: int = 20,
    device = None
) -> Dict[str, float]:
    """
    Full evaluation pipeline: Retrieval → Re-ranking → Metrics
    
    Args:
        patient_tower: Trained PatientTower model
        code_embeddings: Precomputed ICD embeddings [num_codes, 768]
        icd_catalog: ICDCatalog object
        test_loader: DataLoader for test set
        labels_index: LabelsIndex for ground truth
        reranker: LLMReranker instance
        retrieval_k: Number of candidates to retrieve
        rerank_top_n: Number of final predictions after re-ranking
        device: Device to run on
    
    Returns:
        Dict with metrics for both retrieval and re-ranking stages
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    patient_tower.eval()
    code_embeddings = code_embeddings.to(device)
    
    # Metrics storage
    retrieval_metrics = {k: [] for k in [20, 50, 100]}
    rerank_metrics = {k: [] for k in [5, 10, 20]}
    
    all_rerank_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating with re-ranker"):
            
            # ===== STAGE 1: RETRIEVAL =====
            
            # Encode patients (using actual batch structure)
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            ehr = batch["ehr"].to(device)
            enc_ids = batch["encounter_id"].tolist()
            
            z_patients = patient_tower(txt_inputs, ehr)
            
            # Compute similarity with all ICD codes
            scores = torch.matmul(z_patients, code_embeddings.T)  # [B, num_codes]
            
            # Get top-K for retrieval
            top_k_scores, top_k_indices = torch.topk(scores, retrieval_k, dim=1)
            
            # Compute retrieval metrics
            pos_codes_lists = batch["pos_codes"]
            for k in [20, 50, 100]:
                if k <= retrieval_k:
                    for i, enc_id in enumerate(enc_ids):
                        true_indices = set(pos_codes_lists[i])
                        predicted = set(top_k_indices[i, :k].cpu().tolist())
                        
                        recall = len(predicted & true_indices) / len(true_indices) if true_indices else 0
                        retrieval_metrics[k].append(recall)
            
            # ===== STAGE 2: RE-RANKING =====
            
            # Prepare data for re-ranker
            patient_data = []
            candidate_codes_batch = []
            
            # Get patient data from dataset
            import pandas as pd
            for i, enc_id in enumerate(enc_ids):
                # Find patient row in dataset
                row = test_loader.dataset.df[test_loader.dataset.df["encounter_id"] == enc_id]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                
                # Extract patient info
                patient_info = {
                    "patient_id": str(enc_id),
                    "text": str(row.get("note_text", "")),
                    "demographics": {
                        "age": float(row.get("age", 0)),
                        "sex": str(row.get("sex", "M"))
                    },
                    "labs": {}
                }
                
                # Extract lab values
                for lab in ["a1c", "glucose", "creatinine", "egfr", "ldl", "hdl", 
                           "triglycerides", "wbc", "hgb", "platelets", "crp", 
                           "troponin", "bnp", "alt", "ast"]:
                    val = row.get(f"lab_{lab}")
                    if pd.notna(val):
                        patient_info["labs"][f"lab_{lab}"] = float(val)
                
                patient_data.append(patient_info)
                
                # Get candidate codes with descriptions
                candidate_codes = []
                for j, idx in enumerate(top_k_indices[i]):
                    code_idx = int(idx.item())
                    code = icd_catalog.idx2code[code_idx]
                    icd_row = icd_catalog.items[icd_catalog.items['code'] == code]
                    if len(icd_row) > 0:
                        text_parts = icd_row.iloc[0]['text'].split('\n')
                        description = text_parts[1] if len(text_parts) > 1 else text_parts[0] if text_parts else ""
                        candidate_codes.append((
                            code,
                            description,
                            float(top_k_scores[i, j].item())
                        ))
                candidate_codes_batch.append(candidate_codes)
            
            # Call LLM re-ranker
            rerank_results = reranker.batch_rerank(
                patient_data,
                candidate_codes_batch,
                top_n=rerank_top_n
            )
            
            all_rerank_results.extend(rerank_results)
            
            # Compute re-ranking metrics
            for i, result in enumerate(rerank_results):
                enc_id = int(result.patient_id)
                true_indices = set(labels_index.positives(enc_id))
                
                for k in [5, 10, 20]:
                    if k <= len(result.reranked_codes):
                        predicted_codes = result.reranked_codes[:k]
                        predicted = set([
                            icd_catalog.code2idx[code] 
                            for code in predicted_codes
                            if code in icd_catalog.code2idx
                        ])
                        
                        recall = len(predicted & true_indices) / len(true_indices) if true_indices else 0
                        rerank_metrics[k].append(recall)
    
    # Aggregate metrics
    results = {
        "retrieval": {
            f"recall@{k}": np.mean(v) for k, v in retrieval_metrics.items() if v
        },
        "reranking": {
            f"recall@{k}": np.mean(v) for k, v in rerank_metrics.items() if v
        }
    }
    
    return results, all_rerank_results