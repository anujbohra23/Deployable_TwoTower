#!/usr/bin/env python3
"""
CLI script for evaluating the two-tower model with LLM re-ranking.
Fixed to work with the actual codebase structure.
"""
import argparse
import torch
from pathlib import Path
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from twotower_icd.config import DataPaths, TrainingConfig
from twotower_icd.data.datasets import PatientDataset, PatientCollator
from twotower_icd.data.icd_catalog import ICDCatalog
from twotower_icd.data.labels_index import PatientLabelIndex
from twotower_icd.data.scalers import EHRScaler
from twotower_icd.models.patient_tower import PatientTower
from twotower_icd.reranking import LLMReranker, LLMProvider
from twotower_icd.utils.metrics import evaluate_recall_at_k


def evaluate_with_reranker(
    patient_tower,
    code_embeddings: torch.Tensor,
    icd_catalog: ICDCatalog,
    test_loader: DataLoader,
    label_index: PatientLabelIndex,
    reranker: LLMReranker,
    retrieval_k: int = 100,
    rerank_top_n: int = 20,
    device: torch.device = None,
) -> tuple:
    """
    Full evaluation pipeline: Retrieval → Re-ranking → Metrics
    
    Returns:
        (metrics_dict, detailed_results_list)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    patient_tower.eval()
    code_embeddings = code_embeddings.to(device)
    
    # Metrics storage
    retrieval_recalls = {k: [] for k in [20, 50, 100] if k <= retrieval_k}
    rerank_recalls = {k: [] for k in [5, 10, 20] if k <= rerank_top_n}
    
    detailed_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            enc_ids = batch["encounter_id"].tolist()
            txt_inputs = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
            ehr = batch["ehr"].to(device)
            pos_codes_lists = batch["pos_codes"]
            
            # Stage 1: Retrieval
            z_patients = patient_tower(txt_inputs, ehr)  # [B, 768]
            scores = z_patients @ code_embeddings.T  # [B, num_codes]
            
            # Get top-K for retrieval
            top_k_scores, top_k_indices = torch.topk(scores, retrieval_k, dim=1)
            
            # Compute retrieval metrics
            for i, enc_id in enumerate(enc_ids):
                true_indices = set(pos_codes_lists[i])
                for k in retrieval_recalls.keys():
                    predicted = set(top_k_indices[i, :k].cpu().tolist())
                    recall = len(predicted & true_indices) / len(true_indices) if true_indices else 0.0
                    retrieval_recalls[k].append(recall)
            
            # Stage 2: Re-ranking
            for i, enc_id in enumerate(enc_ids):
                # Get patient info from dataset
                row_idx = None
                for idx, item in enumerate(test_loader.dataset.df.itertuples()):
                    if item.encounter_id == enc_id:
                        row_idx = idx
                        break
                
                if row_idx is None:
                    continue
                
                row = test_loader.dataset.df.iloc[row_idx]
                clinical_note = str(row.get("note_text", ""))
                age = float(row.get("age", 0))
                sex = str(row.get("sex", "M"))
                
                # Extract lab values
                lab_values = {}
                for lab in ["a1c", "glucose", "creatinine", "egfr", "ldl", "hdl", 
                           "triglycerides", "wbc", "hgb", "platelets", "crp", 
                           "troponin", "bnp", "alt", "ast"]:
                    val = row.get(f"lab_{lab}")
                    if pd.notna(val):
                        lab_values[lab] = float(val)
                
                # Prepare candidate codes for reranking
                candidate_codes = []
                for j in range(min(rerank_top_n, retrieval_k)):
                    idx = int(top_k_indices[i, j].item())
                    code = icd_catalog.idx2code[idx]
                    icd_row = icd_catalog.items[icd_catalog.items['code'] == code]
                    if len(icd_row) > 0:
                        text_parts = icd_row.iloc[0]['text'].split('\n')
                        title = text_parts[0] if len(text_parts) > 0 else code
                        description = text_parts[1] if len(text_parts) > 1 else ""
                        candidate_codes.append({
                            'code': code,
                            'title': title.strip(),
                            'description': description.strip(),
                            'score': float(top_k_scores[i, j].item()),
                            'rank': j + 1,
                        })
                
                # Rerank
                try:
                    reranked = reranker.rerank(
                        candidate_codes=candidate_codes,
                        clinical_note=clinical_note,
                        lab_values=lab_values,
                        age=age,
                        sex=sex,
                    )
                    
                    # Compute reranking metrics
                    true_indices = set(pos_codes_lists[i])
                    for k in rerank_recalls.keys():
                        if k <= len(reranked):
                            predicted_codes = [r['code'] for r in reranked[:k]]
                            predicted_indices = set([
                                icd_catalog.code2idx[c] for c in predicted_codes 
                                if c in icd_catalog.code2idx
                            ])
                            recall = len(predicted_indices & true_indices) / len(true_indices) if true_indices else 0.0
                            rerank_recalls[k].append(recall)
                    
                    detailed_results.append({
                        'encounter_id': enc_id,
                        'original_top10': [(c['code'], c['score']) for c in candidate_codes[:10]],
                        'reranked_top10': [(r['code'], r['score']) for r in reranked[:10]],
                    })
                except Exception as e:
                    print(f"Warning: Reranking failed for encounter {enc_id}: {e}")
                    # Use original order
                    for k in rerank_recalls.keys():
                        if k <= len(candidate_codes):
                            predicted_codes = [c['code'] for c in candidate_codes[:k]]
                            predicted_indices = set([
                                icd_catalog.code2idx[c] for c in predicted_codes 
                                if c in icd_catalog.code2idx
                            ])
                            recall = len(predicted_indices & true_indices) / len(true_indices) if true_indices else 0.0
                            rerank_recalls[k].append(recall)
    
    # Aggregate metrics
    metrics = {
        "retrieval": {
            f"recall@{k}": float(np.mean(v)) if v else 0.0
            for k, v in retrieval_recalls.items()
        },
        "reranking": {
            f"recall@{k}": float(np.mean(v)) if v else 0.0
            for k, v in rerank_recalls.items()
        }
    }
    
    return metrics, detailed_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate with LLM re-ranker")
    
    # Data paths
    parser.add_argument("--icd_csv", type=str, required=True)
    parser.add_argument("--patients_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--splits_csv", type=str, required=True)
    
    # Model paths
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing patient_tower.pt and code_embeds.pt")
    
    # LLM config
    parser.add_argument("--llm_backend", type=str, default="openrouter",
                        choices=["openrouter", "huggingface", "openai", "groq", "together"])
    parser.add_argument("--llm_model", type=str, 
                        default="meta-llama/llama-3.2-3b-instruct:free")
    parser.add_argument("--llm_api_key", type=str, default=None,
                        help="API key for LLM service")
    
    # Evaluation params
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--retrieval_k", type=int, default=100)
    parser.add_argument("--rerank_top_n", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--txt_backbone", type=str, default="distilbert-base-uncased")
    
    # Output
    parser.add_argument("--output_json", type=str, default="rerank_results.json")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load data
    print("Loading data...")
    icd_raw = pd.read_csv(args.icd_csv)
    patients = pd.read_csv(args.patients_csv)
    labels = pd.read_csv(args.labels_csv)
    splits = pd.read_csv(args.splits_csv)
    
    # Build ICD catalog and label index
    icd_catalog = ICDCatalog.from_df(icd_raw)
    label_index = PatientLabelIndex.from_df(labels, icd_catalog.code2idx)
    
    # Load scaler
    scaler_path = Path(args.checkpoint_dir) / "ehr_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = EHRScaler.load(str(scaler_path))
    
    # Build dataset
    print(f"Loading {args.split} dataset...")
    dataset = PatientDataset(patients, splits, args.split, scaler, label_index)
    
    tokenizer = AutoTokenizer.from_pretrained(args.txt_backbone, use_fast=True)
    collator = PatientCollator(txt_tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Load patient tower
    print("Loading patient tower...")
    patient_tower_path = Path(args.checkpoint_dir) / "patient_tower.pt"
    if not patient_tower_path.exists():
        raise FileNotFoundError(f"Patient tower not found at {patient_tower_path}")
    
    patient_tower = PatientTower(
        txt_model_name=args.txt_backbone,
        d=768,
    ).to(device)
    patient_tower.load_state_dict(torch.load(patient_tower_path, map_location=device))
    patient_tower.eval()
    
    # Load ICD embeddings
    print("Loading ICD embeddings...")
    code_embed_files = list(Path(args.checkpoint_dir).glob("code_embeds_epoch*.pt"))
    if not code_embed_files:
        raise FileNotFoundError(f"No code embeddings found in {args.checkpoint_dir}")
    code_embeds_path = sorted(code_embed_files)[-1]
    print(f"Using embeddings from: {code_embeds_path}")
    
    ckpt = torch.load(code_embeds_path, map_location="cpu")
    code_embeddings = ckpt["embeddings"]  # [N_codes, d]
    
    # Initialize LLM re-ranker
    print(f"Initializing LLM re-ranker (backend={args.llm_backend}, model={args.llm_model})...")
    
    # Map backend name to provider
    backend_map = {
        "openrouter": "openrouter",
        "huggingface": "huggingface",
        "openai": "openai",
        "groq": "groq",
        "together": "together",
    }
    provider = backend_map.get(args.llm_backend, "huggingface")
    
    # Set API key if provided
    if args.llm_api_key:
        if provider == "openrouter":
            os.environ['OPENROUTER_API_KEY'] = args.llm_api_key
        elif provider == "huggingface":
            os.environ['HUGGINGFACE_API_KEY'] = args.llm_api_key
        elif provider == "openai":
            os.environ['OPENAI_API_KEY'] = args.llm_api_key
        elif provider == "groq":
            os.environ['GROQ_API_KEY'] = args.llm_api_key
        elif provider == "together":
            os.environ['TOGETHER_API_KEY'] = args.llm_api_key
    
    reranker = LLMReranker(
        provider=provider,
        model_name=args.llm_model,
        api_key=args.llm_api_key,
        max_candidates=args.rerank_top_n,
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("STARTING EVALUATION WITH RE-RANKING")
    print("="*60 + "\n")
    
    metrics, detailed_results = evaluate_with_reranker(
        patient_tower=patient_tower,
        code_embeddings=code_embeddings,
        icd_catalog=icd_catalog,
        test_loader=dataloader,
        label_index=label_index,
        reranker=reranker,
        retrieval_k=args.retrieval_k,
        rerank_top_n=args.rerank_top_n,
        device=device,
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\n📊 RETRIEVAL STAGE:")
    for metric, value in metrics["retrieval"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n🤖 RE-RANKING STAGE:")
    for metric, value in metrics["reranking"].items():
        print(f"  {metric}: {value:.4f}")
    
    # Calculate improvement
    if "recall@20" in metrics["retrieval"] and "recall@20" in metrics["reranking"]:
        improvement = (metrics["reranking"]["recall@20"] - metrics["retrieval"]["recall@20"]) * 100
        print(f"\n✨ Improvement at top-20: {improvement:+.2f}%")
    
    # Save results
    output_data = {
        "metrics": metrics,
        "config": {
            "retrieval_k": args.retrieval_k,
            "rerank_top_n": args.rerank_top_n,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "split": args.split,
        },
        "detailed_results": detailed_results[:100],  # Save first 100
    }
    
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
