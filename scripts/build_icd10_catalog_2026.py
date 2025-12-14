#!/usr/bin/env python
"""
Build ICD-10-CM FY2026 catalog from official CMS/NCHS files.

Inputs (you already downloaded):
  - icd10cm-codes-2026.txt
  - icd10cm-codes-addenda-2026.txt

Outputs:
  - Data/icd10cm_master_2026.csv
  - Data/icd_codes_full_for_model.csv

Notes:
  - This version uses the codes + addenda only (short descriptions).
  - Fields related to chapter/block/hierarchy/notes are left blank
    as placeholders. You can later enrich them by parsing:
      * icd10cm-tabular-2026.pdf
      * icd10cm_index_2026.pdf
"""

import os
import argparse
from pathlib import Path

import pandas as pd


def parse_codes_file(path: Path) -> dict:
    """
    Parse icd10cm-codes-2026.txt into {code: title_short}.

    Lines look like:
      A000    Cholera due to Vibrio cholerae 01, biovar cholerae
      A001    Cholera due to Vibrio cholerae 01, biovar eltor
    """
    code2title = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first whitespace chunk
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            code, title = parts
            code2title[code.strip()] = title.strip()
    return code2title


def apply_addenda(path: Path, code2title: dict) -> dict:
    """
    Parse icd10cm-codes-addenda-2026.txt and update code2title.

    We handle:
      - 'Delete:' lines -> remove code
      - 'Add:'    lines -> add/replace code + title

    The file has lines like:
      Delete:      B880    Other acariasis
      Add:         B8801   Some new description
    """
    if not path.exists():
        print(f"[WARN] Addenda file not found at {path}, skipping.")
        return code2title

    to_delete = set()
    to_add = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Delete:"):
                # Remove 'Delete:' prefix then split
                rest = line[len("Delete:"):].strip()
                parts = rest.split(None, 1)
                if not parts:
                    continue
                code = parts[0].strip()
                to_delete.add(code)

            elif line.startswith("Add:"):
                rest = line[len("Add:"):].strip()
                parts = rest.split(None, 1)
                if len(parts) != 2:
                    continue
                code, title = parts
                to_add[code.strip()] = title.strip()

    # Apply deletes
    for c in to_delete:
        if c in code2title:
            del code2title[c]

    # Apply adds
    code2title.update(to_add)

    print(f"[INFO] Addenda applied. Deleted {len(to_delete)} codes, added/updated {len(to_add)} codes.")
    return code2title


def build_master_df(code2title: dict) -> pd.DataFrame:
    """
    Build icd10cm_master_2026 DataFrame from code->title_short mapping.
    Many fields are left blank as placeholders for future enrichment.
    """
    rows = []
    for code, short in sorted(code2title.items()):
        # Category is first 3 characters (typical ICD-10-CM rule)
        category_code = code[:3]

        # Heuristic: consider "leaf" if length > 3 or contains '.'
        is_leaf = int(len(code) > 3 or "." in code)

        rows.append({
            "code": code,
            "title_short": short,
            "title_long": short,        # placeholder; later from tabular PDF
            "chapter_code": "",         # TODO
            "chapter_name": "",         # TODO
            "block_code": "",           # TODO
            "block_name": "",           # TODO
            "category_code": category_code,
            "category_title": "",       # TODO
            "is_leaf": is_leaf,
            "billable": is_leaf,        # approximation
            "includes": "",
            "excludes1": "",
            "excludes2": "",
            "use_additional_code": "",
            "code_first": "",
            "synonyms": "",             # TODO: from index or manual
        })

    df = pd.DataFrame(rows)
    return df


def build_model_catalog(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build slim model-facing ICD catalog.
    Columns match what the retriever expects.
    """
    def level_from_row(row):
        return "leaf" if row["is_leaf"] else "category"

    df = master_df.copy()
    df["title"] = df["title_short"]
    df["description"] = df["title_long"]
    df["chapter"] = df["chapter_name"]
    df["category"] = df["category_code"]
    df["level"] = df.apply(level_from_row, axis=1)

    # Keep only a subset of columns the model needs
    cols = [
        "code",
        "title",
        "description",
        "synonyms",
        "chapter",
        "category",
        "level",
    ]
    return df[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes_txt", required=True, help="Path to icd10cm-codes-2026.txt")
    ap.add_argument("--addenda_txt", required=False, default=None, help="Path to icd10cm-codes-addenda-2026.txt")
    ap.add_argument("--out_dir", default="Data", help="Directory to write CSVs into")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes_path = Path(args.codes_txt)
    addenda_path = Path(args.addenda_txt) if args.addenda_txt else None

    print(f"[INFO] Reading base codes from: {codes_path}")
    code2title = parse_codes_file(codes_path)

    if addenda_path is not None:
        print(f"[INFO] Applying addenda from: {addenda_path}")
        code2title = apply_addenda(addenda_path, code2title)

    print(f"[INFO] Total codes after addenda: {len(code2title)}")

    master_df = build_master_df(code2title)
    master_csv = out_dir / "icd10cm_master_2026.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"[OK] Wrote master ICD table: {master_csv} (rows={len(master_df)})")

    model_df = build_model_catalog(master_df)
    model_csv = out_dir / "icd_codes_full_for_model.csv"
    model_df.to_csv(model_csv, index=False)
    print(f"[OK] Wrote model ICD catalog: {model_csv} (rows={len(model_df)})")


if __name__ == "__main__":
    main()
