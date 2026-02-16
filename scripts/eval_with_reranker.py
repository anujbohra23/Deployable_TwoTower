#!/usr/bin/env python3
"""
CLI script for evaluating the two-tower model with LLM re-ranking.
"""
import argparse
import torch
from pathlib import Path
import json

from twotower_icd.data.datasets import PatientDataset, PatientCollator
from twotower_icd.data.icd_catalog import ICDCatalog
from twotower_icd.data.labels_index import PatientLabelIndex
from twotower_icd.data.scalers import EHRScaler
from twotower_icd.models.patient_tower import PatientTower
from twotower_icd.models.reranker import LLMReranker
from twotower_icd.training.eval_with_reranker import evaluate_with_reranker
from torch.utils.data import DataLoader
import pandas as pd


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
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch number for code embeddings (e.g., 3 for code_embeds_epoch3.pt)")
    
    # LLM config
    parser.add_argument("--llm_backend", type=str, default="openrouter",
                        choices=["openrouter", "huggingface", "ollama"])
    parser.add_argument("--llm_model", type=str, 
                        default="meta-llama/llama-3.2-3b-instruct:free")
    parser.add_argument("--llm_api_key", type=str, default=None,
                        help="API key for LLM service (not needed for Ollama)")
    
    # Evaluation params
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--retrieval_k", type=int, default=100)
    parser.add_argument("--rerank_top_n", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--txt_backbone", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    
    # Output
    parser.add_argument("--output_json", type=str, default="rerank_results.json")
    
    args = parser.parse_args()
    
    # Load ICD catalog
    print("Loading ICD catalog...")
    icd_df = pd.read_csv(args.icd_csv)
    icd_catalog = ICDCatalog.from_df(icd_df)
    
    # Load labels index
    print("Loading labels index...")
    labels_df = pd.read_csv(args.labels_csv)
    labels_index = PatientLabelIndex.from_df(labels_df, icd_catalog.code2idx)
    
    # Load EHR scaler
    scaler_path = Path(args.checkpoint_dir) / "ehr_scaler.json"
    ehr_scaler = EHRScaler.load(str(scaler_path))
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    patients_df = pd.read_csv(args.patients_csv)
    splits_df = pd.read_csv(args.splits_csv)
    
    dataset = PatientDataset(
        patients_df=patients_df,
        splits_df=splits_df,
        split=args.split,
        scaler=ehr_scaler,
        label_index=labels_index
    )
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.txt_backbone, use_fast=True)
    collator = PatientCollator(
        txt_tokenizer=tokenizer,
        max_seq_len=args.max_seq_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    # Load patient tower
    print("Loading patient tower...")
    patient_tower_path = Path(args.checkpoint_dir) / "patient_tower.pt"
    
    # Initialize model
    patient_tower = PatientTower(
        txt_model_name=args.txt_backbone,
        d=768
    )
    
    # Load checkpoint to CPU first to avoid device mapping issues
    checkpoint = torch.load(patient_tower_path, map_location="cpu")
    patient_tower.load_state_dict(checkpoint)
    
    # Convert device string to torch.device and move model
    device = torch.device(args.device)
    patient_tower.to(device)
    patient_tower.eval()
    
    # Load ICD embeddings
    print("Loading ICD embeddings...")
    if args.epoch:
        code_embeds_path = Path(args.checkpoint_dir) / f"code_embeds_epoch{args.epoch}.pt"
    else:
        # Find latest epoch
        code_embed_files = list(Path(args.checkpoint_dir).glob("code_embeds_epoch*.pt"))
        if not code_embed_files:
            raise FileNotFoundError("No code embeddings found")
        code_embeds_path = sorted(code_embed_files)[-1]
    
    print(f"Using embeddings from: {code_embeds_path}")
    # Load to CPU first, then move to device
    code_ckpt = torch.load(code_embeds_path, map_location="cpu")
    code_embeddings = code_ckpt["embeddings"]  # Extract embeddings from checkpoint
    
    # Initialize LLM re-ranker
    print(f"Initializing LLM re-ranker (backend={args.llm_backend}, model={args.llm_model})...")
    reranker = LLMReranker(
        backend=args.llm_backend,
        model_name=args.llm_model,
        api_key=args.llm_api_key,
        temperature=0.1
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("STARTING EVALUATION WITH RE-RANKING")
    print("="*60 + "\n")
    
    metrics, rerank_results = evaluate_with_reranker(
        patient_tower=patient_tower,
        code_embeddings=code_embeddings,
        icd_catalog=icd_catalog,
        test_loader=dataloader,
        labels_index=labels_index,
        reranker=reranker,
        retrieval_k=args.retrieval_k,
        rerank_top_n=args.rerank_top_n,
        device=device
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
    
    # Save detailed results
    output_data = {
        "metrics": metrics,
        "config": {
            "retrieval_k": args.retrieval_k,
            "rerank_top_n": args.rerank_top_n,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model
        },
        "detailed_results": [
            {
                "patient_id": r.patient_id,
                "original_top10": r.original_scores[:10],
                "reranked_top10": list(zip(r.reranked_codes[:10], r.reranked_scores[:10]))
            }
            for r in rerank_results[:100]  # Save first 100 for inspection
        ]
    }
    
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {args.output_json}")


if __name__ == "__main__":
    main()