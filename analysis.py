"""
analysis.py
Full pipeline:
  1. Load & clean EHR data (impute missings)
  2. Propensity Score Matching (Logistic Regression + nearest-neighbor matching)
  3. Balance diagnostics — Love Plot (SMD)
  4. Kaplan-Meier survival curves (before & after matching)
  5. Cox Proportional Hazards model
  6. Summary table

All figures saved to  outputs/
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = "/home/claude/clinical-causal-inference-survival"
DATA   = f"{BASE}/data/ehr_synthetic.csv"
OUTDIR = f"{BASE}/outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & CLEAN DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Load & Clean Data")
print("=" * 60)

df = pd.read_csv(DATA)
print(f"Raw shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# Median imputation
for col in ["bmi", "cholesterol"]:
    median = df[col].median()
    n_miss = df[col].isnull().sum()
    df[col].fillna(median, inplace=True)
    print(f"  Imputed {n_miss} missing '{col}' values with median={median:.1f}")

print(f"\nMissing after imputation: {df.isnull().sum().sum()}")
print(f"\nClass balance:\n{df['treatment'].value_counts().rename({0:'Drug A', 1:'Drug B'})}")
print(f"Events: {df['event'].sum()} / {len(df)} ({df['event'].mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PROPENSITY SCORE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Propensity Score Estimation")
print("=" * 60)

COVARIATES = ["age", "sex", "bmi", "sys_bp", "dia_bp",
              "cholesterol", "hba1c", "hypertension", "ckd", "smoker"]

X = df[COVARIATES].copy()
y = df["treatment"].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
df["ps"] = lr.predict_proba(X_scaled)[:, 1]   # P(Drug B)

c_stat = lr.score(X_scaled, y)
print(f"Logistic regression accuracy: {c_stat:.3f}")
print(f"Propensity score  mean(Drug A)={df.loc[df.treatment==0,'ps'].mean():.3f}  "
      f"mean(Drug B)={df.loc[df.treatment==1,'ps'].mean():.3f}")

# ── PS Overlap plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 1, 40)
ax.hist(df.loc[df.treatment==0, "ps"], bins=bins, alpha=0.6,
        color="#3B82F6", label="Drug A", density=True)
ax.hist(df.loc[df.treatment==1, "ps"], bins=bins, alpha=0.6,
        color="#EF4444", label="Drug B", density=True)
ax.set_xlabel("Propensity Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Propensity Score Distribution Before Matching", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/ps_distribution.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  NEAREST-NEIGHBOUR 1:1 PROPENSITY SCORE MATCHING  (caliper = 0.02)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Propensity Score Matching (1:1, caliper=0.02)")
print("=" * 60)

CALIPER = 0.02

treated_idx   = df.index[df.treatment == 1].tolist()
control_idx   = df.index[df.treatment == 0].tolist()

ps_treated = df.loc[treated_idx, "ps"].values.reshape(-1, 1)
ps_control = df.loc[control_idx, "ps"].values.reshape(-1, 1)

nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
nn.fit(ps_control)
distances, indices = nn.kneighbors(ps_treated)

matched_treated = []
matched_control = []
used_controls   = set()

for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
    ctrl_orig_idx = control_idx[idx]
    if dist <= CALIPER and ctrl_orig_idx not in used_controls:
        matched_treated.append(treated_idx[i])
        matched_control.append(ctrl_orig_idx)
        used_controls.add(ctrl_orig_idx)

matched_idx = matched_treated + matched_control
df_matched  = df.loc[matched_idx].copy()

n_pairs = len(matched_treated)
print(f"Matched pairs: {n_pairs}")
print(f"Drug A in matched: {(df_matched.treatment==0).sum()}")
print(f"Drug B in matched: {(df_matched.treatment==1).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BALANCE DIAGNOSTICS — LOVE PLOT (Standardized Mean Differences)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Balance Diagnostics (Love Plot)")
print("=" * 60)

def smd(df_in, var):
    """Standardized Mean Difference between treatment groups."""
    a = df_in.loc[df_in.treatment==0, var]
    b = df_in.loc[df_in.treatment==1, var]
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return abs(b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0

cov_labels = {
    "age": "Age",
    "sex": "Sex (Male)",
    "bmi": "BMI",
    "sys_bp": "Systolic BP",
    "dia_bp": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "hba1c": "HbA1c",
    "hypertension": "Hypertension",
    "ckd": "Chronic Kidney Disease",
    "smoker": "Smoker",
}

smd_before = {v: smd(df, v)         for v in COVARIATES}
smd_after  = {v: smd(df_matched, v) for v in COVARIATES}

for v in COVARIATES:
    print(f"  {cov_labels[v]:<28} Before={smd_before[v]:.3f}  After={smd_after[v]:.3f}")

# Love Plot
fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(COVARIATES))
labels = [cov_labels[v] for v in COVARIATES]

ax.scatter([smd_before[v] for v in COVARIATES], y_pos,
           color="#EF4444", s=70, zorder=5, label="Before Matching")
ax.scatter([smd_after[v]  for v in COVARIATES], y_pos,
           color="#22C55E", s=70, marker="D", zorder=5, label="After Matching")

for i, v in enumerate(COVARIATES):
    ax.plot([smd_before[v], smd_after[v]], [i, i],
            color="#9CA3AF", lw=1, zorder=3)

ax.axvline(0.1, color="#F59E0B", lw=1.5, ls="--", label="SMD = 0.1 threshold")
ax.axvline(0,   color="#6B7280", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Standardized Mean Difference", fontsize=12)
ax.set_title("Love Plot — Covariate Balance\nBefore vs. After Propensity Score Matching",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(-0.02, max(smd_before.values()) * 1.15)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/love_plot.png", dpi=150)
plt.close()
print("  Love plot saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  KAPLAN-MEIER SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Kaplan-Meier Survival Analysis")
print("=" * 60)

def km_plot(df_in, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"Drug A": "#3B82F6", "Drug B": "#EF4444"}

    for trt, label, color in [(0, "Drug A", "#3B82F6"), (1, "Drug B", "#EF4444")]:
        sub = df_in[df_in.treatment == trt]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration"], sub["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color,
                                   linewidth=2.2, at_risk_counts=False)

    # Log-rank test
    a = df_in[df_in.treatment == 0]
    b = df_in[df_in.treatment == 1]
    lr_result = logrank_test(a["duration"], b["duration"],
                             event_observed_A=a["event"],
                             event_observed_B=b["event"])
    p = lr_result.p_value
    ax.text(0.97, 0.97, f"Log-rank p = {p:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#111827",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Probability of Survival\n(No Cardiovascular Event)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}", dpi=150)
    plt.close()
    print(f"  Saved {fname}  (log-rank p={p:.4f})")
    return p

p_unmatched = km_plot(df,         "Kaplan-Meier Survival Curves — Unmatched Cohort",
                      "km_unmatched.png")
p_matched   = km_plot(df_matched, "Kaplan-Meier Survival Curves — Propensity Score Matched Cohort",
                      "km_matched.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COX PROPORTIONAL HAZARDS MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Cox Proportional Hazards Model (matched cohort)")
print("=" * 60)

cox_vars = ["treatment", "age", "bmi", "sys_bp", "ckd", "smoker", "hypertension"]
df_cox   = df_matched[cox_vars + ["duration", "event"]].copy()

cph = CoxPHFitter()
cph.fit(df_cox, duration_col="duration", event_col="event")
cph.print_summary()

summary = cph.summary.copy()
summary["HR"]    = np.exp(summary["coef"])
summary["HR_lo"] = np.exp(summary["coef lower 95%"])
summary["HR_hi"] = np.exp(summary["coef upper 95%"])

# Forest plot
fig, ax = plt.subplots(figsize=(8, 5))
vars_plot = ["treatment", "age", "bmi", "sys_bp", "ckd", "smoker", "hypertension"]
var_labels = {
    "treatment": "Drug B vs Drug A",
    "age": "Age (per year)",
    "bmi": "BMI",
    "sys_bp": "Systolic BP",
    "ckd": "CKD",
    "smoker": "Smoker",
    "hypertension": "Hypertension",
}

y_pos = np.arange(len(vars_plot))
for i, v in enumerate(vars_plot):
    row  = summary.loc[v]
    hr   = row["HR"]
    lo   = row["HR_lo"]
    hi   = row["HR_hi"]
    pval = row["p"]
    color = "#EF4444" if v == "treatment" else ("#3B82F6" if pval < 0.05 else "#9CA3AF")
    ax.errorbar(hr, i, xerr=[[hr-lo], [hi-hr]],
                fmt="o", color=color, capsize=4, markersize=7, lw=2)

ax.axvline(1, color="#6B7280", lw=1.2, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels([var_labels[v] for v in vars_plot], fontsize=10)
ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12)
ax.set_title("Cox Proportional Hazards — Forest Plot\n(Matched Cohort)",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

# Annotate treatment HR
trt_row = summary.loc["treatment"]
ax.annotate(
    f"HR={trt_row['HR']:.2f} [{trt_row['HR_lo']:.2f}–{trt_row['HR_hi']:.2f}]\np={trt_row['p']:.4f}",
    xy=(trt_row["HR"], vars_plot.index("treatment")),
    xytext=(trt_row["HR"]+0.05, vars_plot.index("treatment")+0.5),
    fontsize=9, color="#EF4444",
    arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1),
)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/cox_forest_plot.png", dpi=150)
plt.close()
print("  Cox forest plot saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  COMPOSITE SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Composite Summary Figure")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F8FAFC")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel A: PS distribution ──────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 35)
ax_a.hist(df.loc[df.treatment==0, "ps"], bins=bins, alpha=0.6,
          color="#3B82F6", label="Drug A", density=True)
ax_a.hist(df.loc[df.treatment==1, "ps"], bins=bins, alpha=0.6,
          color="#EF4444", label="Drug B", density=True)
ax_a.set_title("A  Propensity Score Overlap", fontweight="bold", fontsize=11)
ax_a.set_xlabel("Propensity Score"); ax_a.set_ylabel("Density")
ax_a.legend(fontsize=9); ax_a.spines[["top","right"]].set_visible(False)

# ── Panel B: Love plot ────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
y_pos = np.arange(len(COVARIATES))
ax_b.scatter([smd_before[v] for v in COVARIATES], y_pos,
             color="#EF4444", s=50, zorder=5, label="Before")
ax_b.scatter([smd_after[v]  for v in COVARIATES], y_pos,
             color="#22C55E", s=50, marker="D", zorder=5, label="After")
for i, v in enumerate(COVARIATES):
    ax_b.plot([smd_before[v], smd_after[v]], [i, i], color="#D1D5DB", lw=1)
ax_b.axvline(0.1, color="#F59E0B", lw=1.2, ls="--")
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([cov_labels[v] for v in COVARIATES], fontsize=7.5)
ax_b.set_title("B  Love Plot (SMD)", fontweight="bold", fontsize=11)
ax_b.set_xlabel("Standardized Mean Difference")
ax_b.legend(fontsize=8); ax_b.spines[["top","right"]].set_visible(False)

# ── Panel C: KM matched ───────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
for trt, label, color in [(0, "Drug A", "#3B82F6"), (1, "Drug B", "#EF4444")]:
    sub = df_matched[df_matched.treatment == trt]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["duration"], sub["event"], label=label)
    kmf.plot_survival_function(ax=ax_c, ci_show=True, color=color, linewidth=2)
ax_c.set_ylim(0.5, 1.02)
ax_c.text(0.97, 0.97, f"p = {p_matched:.4f}",
          transform=ax_c.transAxes, ha="right", va="top", fontsize=9,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D1D5DB"))
ax_c.set_title("C  KM Curves (Matched)", fontweight="bold", fontsize=11)
ax_c.set_xlabel("Months"); ax_c.set_ylabel("Survival Probability")
ax_c.spines[["top","right"]].set_visible(False)

# ── Panel D: Cox forest plot ──────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, :2])
for i, v in enumerate(vars_plot):
    row   = summary.loc[v]
    color = "#EF4444" if v == "treatment" else ("#3B82F6" if row["p"] < 0.05 else "#9CA3AF")
    ax_d.errorbar(row["HR"], i, xerr=[[row["HR"]-row["HR_lo"]], [row["HR_hi"]-row["HR"]]],
                  fmt="o", color=color, capsize=3, markersize=6, lw=1.5)
ax_d.axvline(1, color="#6B7280", lw=1.2, ls="--")
ax_d.set_yticks(range(len(vars_plot)))
ax_d.set_yticklabels([var_labels[v] for v in vars_plot], fontsize=9)
ax_d.set_xlabel("Hazard Ratio (95% CI)")
ax_d.set_title("D  Cox PH Forest Plot (Matched Cohort)", fontweight="bold", fontsize=11)
ax_d.spines[["top","right"]].set_visible(False)

# ── Panel E: Summary stats ────────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
ax_e.axis("off")
trt_hr = summary.loc["treatment"]
summary_text = (
    f"STUDY SUMMARY\n"
    f"{'─'*30}\n"
    f"Patients enrolled:    {len(df):,}\n"
    f"Cardiovascular events:{df['event'].sum():,}\n"
    f"Matched pairs:        {n_pairs:,}\n\n"
    f"Drug B vs Drug A\n"
    f"{'─'*30}\n"
    f"Hazard Ratio:  {trt_hr['HR']:.3f}\n"
    f"95% CI:        [{trt_hr['HR_lo']:.3f}, {trt_hr['HR_hi']:.3f}]\n"
    f"p-value:       {trt_hr['p']:.4f}\n\n"
    f"{'Significant ✓' if trt_hr['p'] < 0.05 else 'Not significant'}\n"
    f"({'Protective' if trt_hr['HR'] < 1 else 'Harmful'} effect)\n\n"
    f"KM Log-rank p: {p_matched:.4f}"
)
ax_e.text(0.05, 0.95, summary_text, transform=ax_e.transAxes,
          va="top", fontsize=9.5, fontfamily="monospace",
          bbox=dict(boxstyle="round,pad=0.6", fc="#EFF6FF", ec="#3B82F6", lw=1.5))

fig.suptitle("Evaluating Drug B vs Drug A in Diabetic Patients\n"
             "Causal Inference via Propensity Score Matching + Survival Analysis",
             fontsize=14, fontweight="bold", y=1.01)
fig.savefig(f"{OUTDIR}/summary_figure.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Composite summary figure saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT FINAL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
trt_hr = summary.loc["treatment"]
print(f"  Hazard Ratio (Drug B vs A): {trt_hr['HR']:.3f}")
print(f"  95% CI:                     [{trt_hr['HR_lo']:.3f}, {trt_hr['HR_hi']:.3f}]")
print(f"  p-value (Cox):              {trt_hr['p']:.4f}")
print(f"  p-value (Log-rank):         {p_matched:.4f}")
conclusion = "STATISTICALLY SIGNIFICANT" if trt_hr['p'] < 0.05 else "NOT statistically significant"
direction  = "PROTECTIVE" if trt_hr['HR'] < 1 else "HARMFUL"
print(f"\n  Conclusion: Drug B has a {direction} effect ({conclusion})")
print(f"\nAll outputs saved to:  {OUTDIR}/")
print("  • ps_distribution.png")
print("  • love_plot.png")
print("  • km_unmatched.png")
print("  • km_matched.png")
print("  • cox_forest_plot.png")
print("  • summary_figure.png")
