#!/usr/bin/env python
"""
Prepare ICD-10 catalog CSV in the format expected by our two-tower code.

Target columns:
    code, title, description, synonyms, chapter, category, level

You should point this script to a source file that has at least:
    - ICD code
    - short description
    - (optionally) long description / chapter / category

Example usage:

    python scripts/prepare_icd10_catalog.py \
        --src_csv data/raw/icd10_source.csv \
        --code_col code \
        --title_col short_desc \
        --desc_col long_desc \
        --chapter_col chapter \
        --category_col category \
        --out_csv data/icd_codes_full.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True,
                    help="Source ICD-10(-CM) file you downloaded (CSV).")
    ap.add_argument("--code_col", default="code",
                    help="Column name with ICD code.")
    ap.add_argument("--title_col", default="title",
                    help="Column name with short title.")
    ap.add_argument("--desc_col", default=None,
                    help="Column name with long description (optional).")
    ap.add_argument("--chapter_col", default=None,
                    help="Column name with chapter/group (optional).")
    ap.add_argument("--category_col", default=None,
                    help="Column name with category/block (optional).")
    ap.add_argument("--out_csv", required=True,
                    help="Where to write cleaned ICD catalog.")
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)

    # Basic required fields
    out = pd.DataFrame()
    out["code"] = df[args.code_col].astype(str).str.strip()
    out["title"] = df[args.title_col].astype(str).fillna("").str.strip()

    if args.desc_col and args.desc_col in df.columns:
        out["description"] = df[args.desc_col].astype(str).fillna("").str.strip()
    else:
        out["description"] = out["title"]  # fallback

    # Optional fields
    out["synonyms"] = ""  # you can later populate with curated aliases

    if args.chapter_col and args.chapter_col in df.columns:
        out["chapter"] = df[args.chapter_col].astype(str).fillna("").str.strip()
    else:
        out["chapter"] = ""

    if args.category_col and args.category_col in df.columns:
        out["category"] = df[args.category_col].astype(str).fillna("").str.strip()
    else:
        # crude category: first 3 chars of code
        out["category"] = out["code"].str[:3]

    # Mark leaves: for now treat all codes as leaf-level items
    # (this is what you actually retrieve over)
    out["level"] = "leaf"

    # Drop duplicates, keep stable order
    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)
    print(f"Saved cleaned ICD catalog with {len(out):,} codes to {args.out_csv}")


if __name__ == "__main__":
    main()
