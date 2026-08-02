"""Produit les figures du manuscrit, sans tiret cadratin dans les libelles."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "_figs"
OUT.mkdir(exist_ok=True)

R = json.loads((HERE / "_results_corpora.json").read_text(encoding="utf-8"))
S = json.loads((HERE / "_results_saga.json").read_text(encoding="utf-8"))

ORDER = ["qwen25vl-3b", "qwen25vl-7b", "internvl3-8b", "qwen25-omni-7b"]
NAME = {"qwen25vl-3b": "Qwen2.5-VL-3B", "qwen25vl-7b": "Qwen2.5-VL-7B",
        "internvl3-8b": "InternVL3-8B", "qwen25-omni-7b": "Qwen2.5-Omni-7B"}
CEIL, BLUE, ORANGE, RED, GREEN = 0.634, "#4C72B0", "#DD8452", "#C44E52", "#55A868"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True})

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(ORDER))
A = [R[f"labo/{m}"]["A"] for m in ORDER]
err = np.array([[A[i] - R[f"labo/{m}"]["A_ci"][0] for i, m in enumerate(ORDER)],
                [R[f"labo/{m}"]["A_ci"][1] - A[i] for i, m in enumerate(ORDER)]])
ax.bar(x, A, yerr=err, capsize=4, color=BLUE, width=0.55)
for y, lab, c in [(0.0, "fixed channel", RED), (CEIL, "provenance ceiling", ORANGE),
                  (1.0, "oracle", GREEN)]:
    ax.axhline(y, ls="--", lw=1.1, color=c)
    ax.text(len(ORDER) - 0.4, y + 0.02, lab, fontsize=8, color=c, ha="right")
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ORDER], rotation=15, ha="right")
ax.set_ylabel("Arbitration index A"); ax.set_ylim(-0.55, 1.12)
ax.set_title("Arbitration measured against its reference bounds")
fig.tight_layout(); fig.savefig(OUT / "fig1_arbitration.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
w = 0.36
for ax, key, ttl, ylab in [
        (axes[0], "V0", "Visual channel: laboratory against field",
         "Image only accuracy"),
        (axes[1], "A", "Arbitration: laboratory against field",
         "Arbitration index A")]:
    lab = [R[f"labo/{m}"][key] for m in ORDER]
    fld = [R[f"terrain/{m}"][key] for m in ORDER]
    ax.bar(x - w/2, lab, w, label="laboratory (PlantVillage)", color=BLUE)
    ax.bar(x + w/2, fld, w, label="field (PlantDoc)", color=ORANGE)
    if key == "V0":
        ax.axhline(0.20, ls="--", lw=1.1, color=RED)
        ax.text(3.4, 0.215, "chance", fontsize=8, color=RED, ha="right")
    else:
        ax.axhline(0.0, ls="--", lw=1.1, color=RED)
        ax.text(3.4, 0.012, "fixed channel", fontsize=8, color=RED, ha="right")
        for i, m in enumerate(ORDER):
            for off, corp in ((-w/2, "labo"), (w/2, "terrain")):
                lo, hi = R[f"{corp}/{m}"]["A_ci"]
                v = R[f"{corp}/{m}"][key]
                ax.errorbar(i + off, v, yerr=[[v - lo], [hi - v]],
                            fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=9.5)
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_lab_vs_field.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(figsize=(7.2, 4.0))
imp = [R[f"labo/{m}"]["A"] for m in ORDER]
sag = [S[m]["SAGA - sans conformal (abl.)"]["A"] for m in ORDER]
ax.bar(x - w/2, imp, w, label="implicit controller", color=BLUE)
ax.bar(x + w/2, sag, w, label="externalised arbitration", color=GREEN)
ax.axhline(CEIL, ls="--", lw=1.1, color=ORANGE)
ax.text(3.45, CEIL + 0.02, "provenance ceiling", fontsize=8, color=ORANGE, ha="right")
ax.axhline(0.0, ls="--", lw=1.1, color=RED)
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ORDER], rotation=18, ha="right")
ax.set_ylabel("Arbitration index A"); ax.set_ylim(-0.55, 0.80)
ax.set_title("Moving aggregation outside the controller recovers the ceiling")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig3_saga.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- figure 4
fig, ax = plt.subplots(figsize=(7.2, 4.0))
rc = S["qwen25vl-7b"]["risk_coverage"]
al = sorted(float(a) for a in rc)
cov = [rc[f"{a}"]["coverage"] if f"{a}" in rc else rc[str(a)]["coverage"] for a in al]
rsk = [rc[str(a)]["risk"] for a in al]
ax.plot(cov, rsk, "o-", color=BLUE, lw=1.6, ms=5)
for a, c, r in zip(al, cov, rsk):
    ax.annotate(f"{a:g}", (c, r), textcoords="offset points", xytext=(6, -9), fontsize=7.5)
ax.set_xlabel("Coverage (fraction of decisions committed)")
ax.set_ylabel("Realised error among committed decisions")
ax.set_title("Cost of the abstention guarantee (Qwen2.5-VL-7B, labels give the target risk)")
fig.tight_layout(); fig.savefig(OUT / "fig4_risk_coverage.png", dpi=300); plt.close(fig)

print("figures ecrites :", *(p.name for p in sorted(OUT.glob("*.png"))))
