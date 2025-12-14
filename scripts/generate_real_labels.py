#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np

LAB_KEYS = [
    "a1c","glucose","creatinine","egfr","ldl","hdl","triglycerides",
    "wbc","hgb","platelets","crp","troponin","bnp","alt","ast"
]

def safe_num(x):
    try:
        return float(x)
    except:
        return np.nan

def pick_code(pool, rng):
    if len(pool) == 0:
        return None
    return rng.choice(pool)

def build_prefix_pool(icd_df, prefix):
    # Codes are dotless in your official file (e.g., E1165 not E11.65)
    # Keep leaf only
    sub = icd_df[(icd_df["level"] == "leaf") & (icd_df["code"].astype(str).str.startswith(prefix))]
    return sub["code"].astype(str).tolist()

def keyword_hit(text, keywords):
    t = (text or "").lower()
    return any(k in t for k in keywords)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients_csv", required=True)
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--out_labels_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    patients = pd.read_csv(args.patients_csv)
    icd = pd.read_csv(args.icd_csv)

    # Clean NaNs in ICD text fields (safe)
    for c in ["title","description","synonyms","chapter","category","level","code"]:
        if c in icd.columns:
            icd[c] = icd[c].fillna("")

    # Build code pools (broad coverage)
    POOLS = {
        "E11": build_prefix_pool(icd, "E11"),  # Type 2 diabetes
        "E10": build_prefix_pool(icd, "E10"),  # Type 1 diabetes
        "E78": build_prefix_pool(icd, "E78"),  # Hyperlipidemia
        "N18": build_prefix_pool(icd, "N18"),  # CKD
        "D64": build_prefix_pool(icd, "D64"),  # Anemia (other)
        "D50": build_prefix_pool(icd, "D50"),  # Iron deficiency anemia
        "I21": build_prefix_pool(icd, "I21"),  # MI
        "I50": build_prefix_pool(icd, "I50"),  # Heart failure
        "R74": build_prefix_pool(icd, "R74"),  # elevated transaminases
        "K76": build_prefix_pool(icd, "K76"),  # fatty liver / other liver disease
        "J44": build_prefix_pool(icd, "J44"),  # COPD
        "J45": build_prefix_pool(icd, "J45"),  # asthma
        "J18": build_prefix_pool(icd, "J18"),  # pneumonia
        "A41": build_prefix_pool(icd, "A41"),  # sepsis
        "R50": build_prefix_pool(icd, "R50"),  # fever of other origin
    }

    # Helper to fetch lab_ columns
    def lab(row, key):
        return safe_num(row.get(f"lab_{key}", np.nan))

    labels_out = []

    for _, row in patients.iterrows():
        enc = int(row["encounter_id"])
        note = str(row.get("note_text", ""))

        a1c = lab(row, "a1c")
        glu = lab(row, "glucose")
        egfr = lab(row, "egfr")
        cr = lab(row, "creatinine")
        ldl = lab(row, "ldl")
        tg  = lab(row, "triglycerides")
        wbc = lab(row, "wbc")
        hgb = lab(row, "hgb")
        crp = lab(row, "crp")
        trop = lab(row, "troponin")
        bnp = lab(row, "bnp")
        alt = lab(row, "alt")
        ast = lab(row, "ast")

        chosen = set()

        # --- Diabetes signals ---
        # Typical: A1c >= 6.5, fasting glucose >= 126
        if (not np.isnan(a1c) and a1c >= 6.5) or (not np.isnan(glu) and glu >= 126):
            # mostly type 2 in population
            code = pick_code(POOLS["E11"], rng) or pick_code(POOLS["E10"], rng)
            if code: chosen.add(code)

        # --- Hyperlipidemia ---
        if (not np.isnan(ldl) and ldl >= 160) or (not np.isnan(tg) and tg >= 200):
            code = pick_code(POOLS["E78"], rng)
            if code: chosen.add(code)

        # --- CKD ---
        # eGFR < 60 suggests CKD
        if (not np.isnan(egfr) and egfr < 60) or (not np.isnan(cr) and cr >= 1.6):
            code = pick_code(POOLS["N18"], rng)
            if code: chosen.add(code)

        # --- Anemia ---
        # Rough: Hgb < 12 (female) or < 13.5 (male) — but we’ll just use < 11.5 as strong signal
        if (not np.isnan(hgb) and hgb < 11.5):
            # randomize anemia subtype
            code = pick_code(POOLS["D50"], rng) if rng.random() < 0.4 else pick_code(POOLS["D64"], rng)
            if code: chosen.add(code)

        # --- MI / ACS ---
        if (not np.isnan(trop) and trop >= 0.04) or keyword_hit(note, ["chest pain", "stemi", "nstemi", "myocardial infarction"]):
            code = pick_code(POOLS["I21"], rng)
            if code: chosen.add(code)

        # --- Heart failure ---
        if (not np.isnan(bnp) and bnp >= 400) or keyword_hit(note, ["heart failure", "volume overload", "pulmonary edema"]):
            code = pick_code(POOLS["I50"], rng)
            if code: chosen.add(code)

        # --- Liver injury ---
        if (not np.isnan(alt) and alt >= 80) or (not np.isnan(ast) and ast >= 80):
            code = pick_code(POOLS["R74"], rng) if rng.random() < 0.6 else pick_code(POOLS["K76"], rng)
            if code: chosen.add(code)

        # --- Infection / inflammation ---
        if ((not np.isnan(wbc) and wbc >= 12) and (not np.isnan(crp) and crp >= 10)) or keyword_hit(note, ["infection", "sepsis", "fever", "pneumonia"]):
            # mild vs severe
            if keyword_hit(note, ["sepsis"]) or ((not np.isnan(wbc) and wbc >= 18) and (not np.isnan(crp) and crp >= 50)):
                code = pick_code(POOLS["A41"], rng) or pick_code(POOLS["R50"], rng)
            else:
                code = pick_code(POOLS["R50"], rng)
            if code: chosen.add(code)

        # --- Pulmonary (mostly text) ---
        if keyword_hit(note, ["copd", "chronic obstructive"]):
            code = pick_code(POOLS["J44"], rng)
            if code: chosen.add(code)

        if keyword_hit(note, ["asthma", "wheezing"]):
            code = pick_code(POOLS["J45"], rng)
            if code: chosen.add(code)

        if keyword_hit(note, ["pneumonia"]):
            code = pick_code(POOLS["J18"], rng)
            if code: chosen.add(code)

        # Ensure at least 1 label (so training has positives)
        if len(chosen) == 0:
            # fallback: pick a common-ish bucket if exists
            fallback = pick_code(POOLS["E78"], rng) or pick_code(POOLS["R50"], rng) or pick_code(POOLS["K76"], rng)
            if fallback:
                chosen.add(fallback)

        for c in sorted(chosen):
            labels_out.append({"encounter_id": enc, "code": c})

    out_df = pd.DataFrame(labels_out)
    out_df.to_csv(args.out_labels_csv, index=False)
    print("Wrote:", args.out_labels_csv)
    print("Total label rows:", len(out_df))
    print("Unique codes:", out_df["code"].nunique())

if __name__ == "__main__":
    main()
