"""Rebuild every number reported in the article from the raw per-probe records.

No GPU and no model download required: results/raw/*.jsonl.gz contains one record
per probe, with the prediction, the gold letter, which source was chosen, and the
per-letter logits.

    python reproduce.py

Each recomputed value is checked against the value reported in the article, and
any mismatch beyond tolerance is printed as a FAIL. The script exits non-zero if
anything fails, so it can be run in CI.
"""

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
GATES = ("V0_gate", "T0_gate")
CHANCE, N_BOOT, TOL = 0.20, 10_000, 0.002
RNG = np.random.default_rng(20260730)

# Valeurs telles qu'imprimees dans l'article (§6).
REPORTED = {
    "qwen25vl-3b":    {"V0": 0.107, "T0": 0.962, "A": -0.085, "D_adv": +0.192},
    "internvl3-8b":   {"V0": 0.428, "T0": 0.998, "A": -0.117, "D_adv": +0.456},
    "qwen25vl-7b":    {"V0": 0.183, "T0": 0.860, "A": -0.294, "D_adv": +0.661},
    "qwen25-omni-7b": {"V0": 0.191, "T0": 0.958, "A": -0.369, "D_adv": +0.639},
}
CEILING = 0.634

fails = []


def check(label, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    if not ok:
        fails.append(f"{label}: recompute {got:+.3f} vs article {want:+.3f}")
    return "ok  " if ok else "FAIL"


def load():
    rows = []
    for p in sorted(RAW.glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        rows.append(r)
    if not rows:
        sys.exit(f"aucun enregistrement sous {RAW}")
    return rows


def wilson(k, n):
    z = stats.norm.ppf(0.975)
    p, d = k / n, 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def main():
    rows = load()
    models = sorted({r["model"] for r in rows})
    conflict = [r for r in rows if r["cell"] not in GATES]
    print(f"{len(rows)} enregistrements | {len(models)} modeles "
          f"| {len({r['uid'] for r in rows})} items\n")

    print("=" * 74)
    print("1. PORTES  (hasard 0.20, binomial unilateral)")
    print("=" * 74)
    for m in models:
        line = []
        for cell, key in (("V0_gate", "V0"), ("T0_gate", "T0")):
            v = [r["correct"] for r in rows if r["model"] == m and r["cell"] == cell]
            acc = sum(v) / len(v)
            p = stats.binomtest(sum(v), len(v), CHANCE, alternative="greater").pvalue
            line.append(f"{key} {acc:.3f} [{check(f'{m}.{key}', acc, REPORTED[m][key])}] "
                        f"{'ADMIS' if p < 0.05 else 'exclu'}")
        print(f"  {m:<16}" + " | ".join(line))

    print("\n" + "=" * 74)
    print("2. ARBITRAGE  A = acc(S_img) + acc(S_txt) - 1")
    print(f"   reperes : canal fixe 0.000 | Bayes-provenance +{CEILING} | oracle +1.000")
    print("=" * 74)
    for m in models:
        sub = {}
        for s in ("S_img", "S_txt"):
            v = [r["correct"] for r in conflict if r["model"] == m and r["subset"] == s]
            sub[s] = sum(v) / len(v)
        A = sub["S_img"] + sub["S_txt"] - 1

        per = defaultdict(lambda: defaultdict(list))
        for r in conflict:
            if r["model"] == m:
                per[r["uid"]][r["subset"]].append(r["correct"])
        uids = list(per)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            acc = defaultdict(list)
            for i in RNG.choice(len(uids), len(uids)):
                for k, vals in per[uids[i]].items():
                    acc[k].extend(vals)
            boot[b] = np.mean(acc["S_img"]) + np.mean(acc["S_txt"]) - 1
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {m:<16}S_img={sub['S_img']:.3f} S_txt={sub['S_txt']:.3f}  "
              f"A={A:+.3f} [{lo:+.3f},{hi:+.3f}]  {A/CEILING*100:+6.1f}% du plafond  "
              f"[{check(f'{m}.A', A, REPORTED[m]['A'])}]")

    print("\n" + "=" * 74)
    print("3. DEFERENCE ADVERSE  D_adv sur S_img (rapport faux, image juste)")
    print("=" * 74)
    for m in models:
        div = [r for r in conflict if r["model"] == m
               and r["subset"] == "S_img" and r["letter_image"] != r["letter_text"]]
        def frac(q):
            sel = [r for r in div if r["q_rep"] == q]
            return sum(1 for r in sel if r["chose"] == "text") / (len(sel) or 1)
        D = frac("verified") - frac("unverified")
        print(f"  {m:<16}D_adv={D:+.3f}   [{check(f'{m}.D_adv', D, REPORTED[m]['D_adv'])}]")

    print("\n" + "=" * 74)
    if fails:
        print(f"{len(fails)} ECART(S) :")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("Toutes les valeurs recalculees correspondent a l'article "
           f"(tolerance {TOL}).")


if __name__ == "__main__":
    main()
