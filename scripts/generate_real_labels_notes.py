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

def build_prefix_pool(icd_df, prefix):
    sub = icd_df[(icd_df["level"] == "leaf") & (icd_df["code"].astype(str).str.startswith(prefix))]
    return sub["code"].astype(str).tolist()

def pick(pool, rng):
    return None if len(pool) == 0 else rng.choice(pool)

def keyword_hit(text, keywords):
    t = (text or "").lower()
    return any(k in t for k in keywords)

def lab(row, key):
    return safe_num(row.get(f"lab_{key}", np.nan))

def append_note(existing: str, additions: list[str]) -> str:
    base = (existing or "").strip()
    add = " ".join([a.strip() for a in additions if a and a.strip()])
    if not add:
        return base
    # Keep notes from exploding
    if len(base) > 0:
        out = base + "\n\n" + add
    else:
        out = add
    return out[:20000]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients_csv", required=True)
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--out_labels_csv", required=True)
    ap.add_argument("--out_patients_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    patients = pd.read_csv(args.patients_csv)
    icd = pd.read_csv(args.icd_csv)

    for c in ["title","description","synonyms","chapter","category","level","code"]:
        if c in icd.columns:
            icd[c] = icd[c].fillna("")

    # Broad pools (extend anytime)
    POOLS = {
        "E11": build_prefix_pool(icd, "E11"),  # Type 2 diabetes
        "E10": build_prefix_pool(icd, "E10"),  # Type 1 diabetes
        "E78": build_prefix_pool(icd, "E78"),  # Hyperlipidemia
        "N18": build_prefix_pool(icd, "N18"),  # CKD
        "D64": build_prefix_pool(icd, "D64"),  # Anemia
        "D50": build_prefix_pool(icd, "D50"),  # Iron deficiency anemia
        "I21": build_prefix_pool(icd, "I21"),  # MI
        "I50": build_prefix_pool(icd, "I50"),  # Heart failure
        "R74": build_prefix_pool(icd, "R74"),  # elevated transaminases
        "K76": build_prefix_pool(icd, "K76"),  # fatty liver / liver disease
        "J44": build_prefix_pool(icd, "J44"),  # COPD
        "J45": build_prefix_pool(icd, "J45"),  # asthma
        "J18": build_prefix_pool(icd, "J18"),  # pneumonia
        "A41": build_prefix_pool(icd, "A41"),  # sepsis
        "R50": build_prefix_pool(icd, "R50"),  # fever
    }

    labels_out = []
    patients_aug = patients.copy()

    for idx, row in patients.iterrows():
        enc = int(row["encounter_id"])
        note = str(row.get("note_text", ""))

        a1c = lab(row, "a1c"); glu = lab(row, "glucose")
        egfr = lab(row, "egfr"); cr = lab(row, "creatinine")
        ldl = lab(row, "ldl"); tg = lab(row, "triglycerides")
        wbc = lab(row, "wbc"); hgb = lab(row, "hgb"); crp = lab(row, "crp")
        trop = lab(row, "troponin"); bnp = lab(row, "bnp")
        alt = lab(row, "alt"); ast = lab(row, "ast")

        chosen = set()
        additions = []

        # ---- Diabetes ----
        if (not np.isnan(a1c) and a1c >= 6.5) or (not np.isnan(glu) and glu >= 126):
            code = pick(POOLS["E11"], rng) or pick(POOLS["E10"], rng)
            if code: chosen.add(code)
            additions.append(
                f"Metabolic: HbA1c {a1c:.1f}% and glucose {glu:.0f} mg/dL are elevated, consistent with diabetes mellitus."
                if not (np.isnan(a1c) or np.isnan(glu)) else
                "Metabolic: elevated glycemic markers consistent with diabetes mellitus."
            )
            # optionally add symptom keywords sometimes
            if rng.random() < 0.35:
                additions.append("Reports polyuria and polydipsia over the past several weeks.")

        # ---- Hyperlipidemia ----
        if (not np.isnan(ldl) and ldl >= 160) or (not np.isnan(tg) and tg >= 200):
            code = pick(POOLS["E78"], rng)
            if code: chosen.add(code)
            additions.append("Lipids: hyperlipidemia suspected based on elevated LDL/triglycerides.")

        # ---- CKD ----
        if (not np.isnan(egfr) and egfr < 60) or (not np.isnan(cr) and cr >= 1.6):
            code = pick(POOLS["N18"], rng)
            if code: chosen.add(code)
            additions.append("Renal: reduced eGFR / elevated creatinine suggests chronic kidney disease.")

        # ---- Anemia ----
        if (not np.isnan(hgb) and hgb < 11.5):
            code = pick(POOLS["D50"], rng) if rng.random() < 0.4 else pick(POOLS["D64"], rng)
            if code: chosen.add(code)
            additions.append("Hematology: low hemoglobin suggests anemia; evaluate for iron deficiency vs chronic disease.")
            if rng.random() < 0.25:
                additions.append("Symptoms include fatigue and decreased exercise tolerance.")

        # ---- MI / ACS ----
        if (not np.isnan(trop) and trop >= 0.04) or keyword_hit(note, ["chest pain", "stemi", "nstemi", "myocardial infarction"]):
            code = pick(POOLS["I21"], rng)
            if code: chosen.add(code)
            additions.append("Cardiac: chest discomfort with elevated troponin concerning for acute coronary syndrome / MI.")

        # ---- Heart failure ----
        if (not np.isnan(bnp) and bnp >= 400) or keyword_hit(note, ["heart failure", "volume overload", "pulmonary edema"]):
            code = pick(POOLS["I50"], rng)
            if code: chosen.add(code)
            additions.append("Cardiac: dyspnea with elevated BNP suggests heart failure exacerbation.")
            if rng.random() < 0.3:
                additions.append("Exam notable for lower extremity edema and orthopnea.")

        # ---- Liver injury ----
        if (not np.isnan(alt) and alt >= 80) or (not np.isnan(ast) and ast >= 80):
            code = pick(POOLS["R74"], rng) if rng.random() < 0.6 else pick(POOLS["K76"], rng)
            if code: chosen.add(code)
            additions.append("Hepatic: transaminitis noted with elevated ALT/AST; consider hepatic injury or fatty liver disease.")

        # ---- Infection / inflammation ----
        if ((not np.isnan(wbc) and wbc >= 12) and (not np.isnan(crp) and crp >= 10)) or keyword_hit(note, ["infection", "sepsis", "fever", "pneumonia"]):
            severe = keyword_hit(note, ["sepsis"]) or ((not np.isnan(wbc) and wbc >= 18) and (not np.isnan(crp) and crp >= 50))
            code = (pick(POOLS["A41"], rng) or pick(POOLS["R50"], rng)) if severe else pick(POOLS["R50"], rng)
            if code: chosen.add(code)
            additions.append("Inflammatory: leukocytosis and elevated CRP suggest acute infection/inflammation.")
            if rng.random() < 0.35:
                additions.append("Reports fever and chills; infectious workup initiated.")

        # ---- Pulmonary (text-driven) ----
        # We can add these sometimes to diversify notes, but only when already hinted or randomly for realism.
        if keyword_hit(note, ["copd", "chronic obstructive"]) or rng.random() < 0.03:
            code = pick(POOLS["J44"], rng)
            if code: chosen.add(code)
            additions.append("Pulmonary: history of COPD with intermittent shortness of breath and cough.")

        if keyword_hit(note, ["asthma", "wheezing"]) or rng.random() < 0.03:
            code = pick(POOLS["J45"], rng)
            if code: chosen.add(code)
            additions.append("Pulmonary: wheezing episodes consistent with asthma/reactive airway disease.")

        if keyword_hit(note, ["pneumonia"]) or rng.random() < 0.02:
            code = pick(POOLS["J18"], rng)
            if code: chosen.add(code)
            additions.append("Pulmonary: productive cough and suspected pneumonia; consider chest imaging and antibiotics.")

        # Ensure at least 1 label so every encounter is usable
        if len(chosen) == 0:
            fallback = pick(POOLS["E78"], rng) or pick(POOLS["R50"], rng) or pick(POOLS["K76"], rng)
            if fallback:
                chosen.add(fallback)
                additions.append("Assessment: non-specific findings; consider metabolic/inflammatory screening and follow-up.")

        # Write labels
        for c in sorted(chosen):
            labels_out.append({"encounter_id": enc, "code": c})

        # Write augmented note
        patients_aug.at[idx, "note_text"] = append_note(note, additions)

    out_labels = pd.DataFrame(labels_out)
    out_labels.to_csv(args.out_labels_csv, index=False)
    patients_aug.to_csv(args.out_patients_csv, index=False)

    print("Wrote labels:", args.out_labels_csv, "rows:", len(out_labels), "unique codes:", out_labels["code"].nunique())
    print("Wrote patients:", args.out_patients_csv, "rows:", len(patients_aug))

if __name__ == "__main__":
    main()
