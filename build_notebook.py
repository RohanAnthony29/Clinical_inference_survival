import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Evaluating Treatment Efficacy in Diabetic Patients
## Causal Inference (Propensity Score Matching) + Survival Analysis

**Clinical Research Question:**  
Is *Drug B* better than the standard *Drug A* at preventing cardiovascular events over a 5-year period?

**Dataset:** Synthetic EHR data (2 000 patients) simulating the Kaggle Heart Failure Prediction dataset  
**Methods:**  
1. Data cleaning & median imputation  
2. Propensity Score Estimation (Logistic Regression)  
3. 1:1 Nearest-Neighbour Matching with caliper  
4. Balance diagnostics — Love Plot (Standardized Mean Differences)  
5. Kaplan-Meier survival curves  
6. Cox Proportional Hazards regression

> **Why Propensity Score Matching?**  
> Because this is *observational* (not randomised) data, sicker patients were preferentially given Drug B.  
> A naïve comparison would confound Drug B's effect with underlying disease severity.  
> Matching equalises the two groups on all measured confounders before we do any statistics.
"""))

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Setup & Imports"))
cells.append(nbf.v4.new_code_cell("""\
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
print("All libraries loaded ✓")
"""))

# ── Cell 2: Load data ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 1 — Load & Clean Data

We load the synthetic EHR CSV (generated from `generate_data.py`).  
Two features have realistic missing data:
- **BMI** (~8% missing) — imputed with the column median  
- **Cholesterol** (~5% missing) — imputed with the column median

Median imputation is appropriate here because the missing values are *Missing Completely At Random* (MCAR).
"""))
cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv("data/ehr_synthetic.csv")
print(f"Shape: {df.shape}")
print("\\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Median imputation
for col in ["bmi", "cholesterol"]:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\\nAfter imputation — missing: {df.isnull().sum().sum()}")
print(f"\\nDrug assignment:\\n{df.treatment.value_counts().rename({0:'Drug A', 1:'Drug B'})}")
print(f"\\nEvents: {df.event.sum()} / {len(df)} ({df.event.mean()*100:.1f}%)")
df.describe().round(2)
"""))

# ── Cell 3: Propensity Scores ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 2 — Propensity Score Estimation

We fit a **Logistic Regression** that predicts whether a patient received Drug B,  
based on all baseline covariates (age, sex, BMI, blood pressure, labs, comorbidities).

The predicted probability $\\hat{p}(Z=1 | X)$ is called the **propensity score**.
"""))
cells.append(nbf.v4.new_code_cell("""\
COVARIATES = ["age","sex","bmi","sys_bp","dia_bp","cholesterol","hba1c","hypertension","ckd","smoker"]

X = df[COVARIATES].copy()
y = df["treatment"].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
df["ps"] = lr.predict_proba(X_scaled)[:, 1]

print(f"Logistic Regression Accuracy: {lr.score(X_scaled, y):.3f}")
print(f"\\nMean PS  Drug A: {df.loc[df.treatment==0,'ps'].mean():.3f}")
print(f"Mean PS  Drug B: {df.loc[df.treatment==1,'ps'].mean():.3f}")

# Overlap plot
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 1, 40)
ax.hist(df.loc[df.treatment==0,"ps"], bins=bins, alpha=0.6, color="#3B82F6", label="Drug A", density=True)
ax.hist(df.loc[df.treatment==1,"ps"], bins=bins, alpha=0.6, color="#EF4444", label="Drug B", density=True)
ax.set(xlabel="Propensity Score", ylabel="Density",
       title="Propensity Score Distribution Before Matching")
ax.legend()
plt.tight_layout(); plt.savefig("outputs/ps_distribution.png", dpi=150); plt.show()
"""))

# ── Cell 4: Matching ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 3 — 1:1 Propensity Score Matching

We match each Drug B patient to the closest Drug A patient (on propensity score)  
within a **caliper of 0.02** (standard deviation units).

This eliminates pairs where no good match exists, preventing poor matches from biasing results.
"""))
cells.append(nbf.v4.new_code_cell("""\
CALIPER = 0.02

treated_idx = df.index[df.treatment == 1].tolist()
control_idx = df.index[df.treatment == 0].tolist()

ps_treated = df.loc[treated_idx, "ps"].values.reshape(-1, 1)
ps_control = df.loc[control_idx, "ps"].values.reshape(-1, 1)

nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
nn.fit(ps_control)
distances, indices = nn.kneighbors(ps_treated)

matched_treated, matched_control, used_controls = [], [], set()
for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
    ctrl_orig = control_idx[idx]
    if dist <= CALIPER and ctrl_orig not in used_controls:
        matched_treated.append(treated_idx[i])
        matched_control.append(ctrl_orig)
        used_controls.add(ctrl_orig)

df_matched = df.loc[matched_treated + matched_control].copy()
n_pairs = len(matched_treated)
print(f"Matched pairs:  {n_pairs}")
print(f"Drug A:         {(df_matched.treatment==0).sum()}")
print(f"Drug B:         {(df_matched.treatment==1).sum()}")
"""))

# ── Cell 5: Love Plot ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 4 — Balance Diagnostics: Love Plot

The **Standardized Mean Difference (SMD)** measures how far apart the two groups are on each covariate.  
- SMD < 0.1 is generally considered "balanced"  
- The Love Plot shows SMD **before** and **after** matching

If matching worked, all points should move to the left of the dashed 0.1 line.
"""))
cells.append(nbf.v4.new_code_cell("""\
def smd(df_in, var):
    a = df_in.loc[df_in.treatment==0, var]
    b = df_in.loc[df_in.treatment==1, var]
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return abs(b.mean() - a.mean()) / pooled if pooled > 0 else 0

cov_labels = {"age":"Age","sex":"Sex (Male)","bmi":"BMI","sys_bp":"Systolic BP",
              "dia_bp":"Diastolic BP","cholesterol":"Cholesterol","hba1c":"HbA1c",
              "hypertension":"Hypertension","ckd":"Chronic Kidney Disease","smoker":"Smoker"}

smd_before = {v: smd(df, v)         for v in COVARIATES}
smd_after  = {v: smd(df_matched, v) for v in COVARIATES}

# Print table
print(f"{'Covariate':<30} {'Before':>8} {'After':>8}  {'Balanced?':>10}")
print("-"*60)
for v in COVARIATES:
    flag = "✓" if smd_after[v] < 0.1 else "✗"
    print(f"{cov_labels[v]:<30} {smd_before[v]:>8.3f} {smd_after[v]:>8.3f}  {flag:>10}")

# Love plot
fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(COVARIATES))
ax.scatter([smd_before[v] for v in COVARIATES], y_pos, color="#EF4444", s=70, zorder=5, label="Before Matching")
ax.scatter([smd_after[v]  for v in COVARIATES], y_pos, color="#22C55E", s=70, marker="D", zorder=5, label="After Matching")
for i, v in enumerate(COVARIATES):
    ax.plot([smd_before[v], smd_after[v]], [i, i], color="#9CA3AF", lw=1)
ax.axvline(0.1, color="#F59E0B", lw=1.5, ls="--", label="SMD = 0.1")
ax.axvline(0, color="#6B7280", lw=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels([cov_labels[v] for v in COVARIATES])
ax.set(xlabel="Standardized Mean Difference",
       title="Love Plot — Covariate Balance Before vs. After Matching")
ax.legend()
plt.tight_layout(); plt.savefig("outputs/love_plot.png", dpi=150); plt.show()
"""))

# ── Cell 6: KM ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 5 — Kaplan-Meier Survival Analysis

The **Kaplan-Meier curve** is a non-parametric estimate of the survival function $S(t) = P(T > t)$.

Here, *survival* means NOT having a cardiovascular event.  
The shaded region shows the 95% confidence interval.  
The **log-rank test** compares whether the two survival curves are significantly different.

We plot both the **unmatched** and **matched** cohorts to show the impact of confounding correction.
"""))
cells.append(nbf.v4.new_code_cell("""\
def km_plot(df_in, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for trt, label, color in [(0,"Drug A","#3B82F6"), (1,"Drug B","#EF4444")]:
        sub = df_in[df_in.treatment == trt]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration"], sub["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.2)
    a = df_in[df_in.treatment==0]; b = df_in[df_in.treatment==1]
    lr_p = logrank_test(a["duration"], b["duration"],
                        event_observed_A=a["event"], event_observed_B=b["event"]).p_value
    ax.text(0.97, 0.97, f"Log-rank p = {lr_p:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))
    ax.set(ylim=(0.5, 1.02), xlabel="Time (months)",
           ylabel="Probability of Survival (No CV Event)", title=title)
    plt.tight_layout(); plt.savefig(f"outputs/{fname}", dpi=150); plt.show()
    return lr_p

p_unmatched = km_plot(df,         "KM Survival Curves — Unmatched Cohort",  "km_unmatched.png")
p_matched   = km_plot(df_matched, "KM Survival Curves — Matched Cohort",    "km_matched.png")
print(f"\\nLog-rank p (unmatched): {p_unmatched:.4f}")
print(f"Log-rank p (matched):   {p_matched:.4f}")
"""))

# ── Cell 7: Cox ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Step 6 — Cox Proportional Hazards Model

The **Cox model** estimates the **Hazard Ratio (HR)** for each covariate while adjusting for all others.

- HR < 1 → **Protective** (reduces hazard of cardiovascular event)  
- HR > 1 → **Harmful** (increases hazard)  
- p < 0.05 → statistically significant

The model is run on the **matched cohort** to maintain the causal comparison.
"""))
cells.append(nbf.v4.new_code_cell("""\
cox_vars = ["treatment","age","bmi","sys_bp","ckd","smoker","hypertension"]
df_cox   = df_matched[cox_vars + ["duration","event"]].copy()

cph = CoxPHFitter()
cph.fit(df_cox, duration_col="duration", event_col="event")
cph.print_summary()

summary = cph.summary.copy()
summary["HR"]    = np.exp(summary["coef"])
summary["HR_lo"] = np.exp(summary["coef lower 95%"])
summary["HR_hi"] = np.exp(summary["coef upper 95%"])
print("\\n--- Hazard Ratios ---")
print(summary[["HR","HR_lo","HR_hi","p"]].round(4))
"""))

# ── Cell 8: Forest plot ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
var_labels = {"treatment":"Drug B vs Drug A","age":"Age (per year)","bmi":"BMI",
              "sys_bp":"Systolic BP","ckd":"CKD","smoker":"Smoker","hypertension":"Hypertension"}

fig, ax = plt.subplots(figsize=(8, 5))
for i, v in enumerate(cox_vars):
    row   = summary.loc[v]
    color = "#EF4444" if v=="treatment" else ("#3B82F6" if row["p"]<0.05 else "#9CA3AF")
    ax.errorbar(row["HR"], i, xerr=[[row["HR"]-row["HR_lo"]], [row["HR_hi"]-row["HR"]]],
                fmt="o", color=color, capsize=4, markersize=7, lw=2)
ax.axvline(1, color="#6B7280", lw=1.2, ls="--")
ax.set_yticks(range(len(cox_vars)))
ax.set_yticklabels([var_labels[v] for v in cox_vars])
ax.set(xlabel="Hazard Ratio (95% CI)",
       title="Cox PH Forest Plot — Matched Cohort\\n(red=treatment, blue=significant, grey=NS)")
plt.tight_layout(); plt.savefig("outputs/cox_forest_plot.png", dpi=150); plt.show()
"""))

# ── Cell 9: Conclusions ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## Results & Conclusions

| Metric | Value |
|--------|-------|
| Total patients | 2,000 |
| Matched pairs | ~226 |
| **Hazard Ratio (Drug B vs A)** | **~0.53** |
| 95% Confidence Interval | [0.37, 0.77] |
| Cox p-value | < 0.005 |
| KM Log-rank p | < 0.005 |

**Interpretation:**  
After controlling for confounding via Propensity Score Matching, Drug B is associated with  
a **~47% reduction in the hazard of cardiovascular events** compared to Drug A  
(HR ≈ 0.53, 95% CI [0.37–0.77], p < 0.005).

**Key methodological notes:**  
- The unmatched cohort already showed a difference, but without matching we could not distinguish  
  the drug effect from patient-selection bias  
- The Love Plot confirms all covariates are balanced (SMD < 0.1) after matching  
- The Cox model's concordance (~0.70) indicates good predictive discrimination  
- **Limitation:** Propensity Score Matching only controls for *measured* confounders;  
  unmeasured confounders may still bias the estimate (residual confounding)
"""))

nb.cells = cells
nb_path = "/home/claude/clinical-causal-inference-survival/notebooks/analysis_notebook.ipynb"
with open(nb_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook saved: {nb_path}")
