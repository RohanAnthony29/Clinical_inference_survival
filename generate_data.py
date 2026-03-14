"""
generate_data.py
Generates a synthetic EHR dataset simulating the Kaggle Heart Failure Prediction Dataset
structure, extended with drug assignment, survival times, and realistic confounding.
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

np.random.seed(42)
N = 2000

# ── Demographics ──────────────────────────────────────────────────────────────
age = np.random.normal(60, 12, N).clip(18, 90)
sex = np.random.binomial(1, 0.55, N)          # 1 = male

# ── Baseline vitals / labs ────────────────────────────────────────────────────
bmi        = np.random.normal(28, 5, N).clip(15, 50)
sys_bp     = np.random.normal(130, 20, N).clip(80, 200)
dia_bp     = np.random.normal(82, 12, N).clip(50, 120)
cholesterol= np.random.normal(210, 40, N).clip(100, 400)
hba1c      = np.random.normal(7.5, 1.5, N).clip(5, 14)

# ── Comorbidities ─────────────────────────────────────────────────────────────
hypertension = (sys_bp > 140).astype(int)
ckd          = np.random.binomial(1, 0.20 + 0.003*age.clip(0, 30), N)
smoker       = np.random.binomial(1, 0.25, N)

# ── Drug assignment (confounded) ──────────────────────────────────────────────
# Older, sicker patients were preferentially given Drug B → classic confounding
logit_treatment = (
    -2.5
    + 0.04 * (age - 60)
    + 0.06 * (bmi - 28)
    + 0.03 * (sys_bp - 130)
    + 0.8  * ckd
    + 0.5  * hypertension
    + 0.3  * smoker
    + np.random.normal(0, 0.3, N)
)
ps_true = expit(logit_treatment)
treatment = np.random.binomial(1, ps_true, N)   # 1 = Drug B

# ── Outcome: cardiovascular event ─────────────────────────────────────────────
# Drug B has a true protective effect (HR ≈ 0.75)
logit_event = (
    -1.8
    + 0.04 * (age - 60)
    + 0.03 * (bmi - 28)
    + 0.025 * (sys_bp - 130)
    + 0.7  * ckd
    + 0.5  * hypertension
    + 0.4  * smoker
    - 0.29 * treatment          # log(0.75) ≈ -0.288  → protective
    + np.random.normal(0, 0.2, N)
)
event_prob = expit(logit_event)
event      = np.random.binomial(1, event_prob, N)

# ── Time-to-event (months, max 60) ────────────────────────────────────────────
# Weibull-ish: events happen sooner for sicker patients
scale = 60 / (event_prob + 0.05)
time_to_event = np.random.exponential(scale, N).clip(1, 60)
# Censor non-events at 60 months
duration = np.where(event == 1, time_to_event, 60)
duration = duration.clip(1, 60)

# ── Introduce realistic missing data ─────────────────────────────────────────
df = pd.DataFrame({
    "patient_id"  : np.arange(1, N+1),
    "age"         : age.round(1),
    "sex"         : sex,
    "bmi"         : bmi.round(1),
    "sys_bp"      : sys_bp.round(1),
    "dia_bp"      : dia_bp.round(1),
    "cholesterol" : cholesterol.round(1),
    "hba1c"       : hba1c.round(2),
    "hypertension": hypertension,
    "ckd"         : ckd,
    "smoker"      : smoker,
    "treatment"   : treatment,        # 0 = Drug A, 1 = Drug B
    "event"       : event,
    "duration"    : duration.round(1),
})

# ~8% missing in bmi, ~5% in cholesterol
miss_bmi  = np.random.choice(N, size=int(0.08*N), replace=False)
miss_chol = np.random.choice(N, size=int(0.05*N), replace=False)
df.loc[miss_bmi,  "bmi"]         = np.nan
df.loc[miss_chol, "cholesterol"] = np.nan

df.to_csv("/home/claude/clinical-causal-inference-survival/data/ehr_synthetic.csv", index=False)
print(f"Dataset saved  →  {N} patients, {event.sum()} events ({event.mean()*100:.1f}%)")
print(df.head())
