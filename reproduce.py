"""Rebuild every number reported in the article from the raw per-probe records.

No GPU and no model download required. `results/raw/` and `results/raw_field/`
contain one record per probe: the prediction, the gold letter, which source was
chosen, and the per-letter logits.

    python reproduce.py

Each recomputed value is checked against the value printed in the article.
Any mismatch beyond tolerance is reported as FAIL and the script exits non-zero,
so it can be run in continuous integration.
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
RAW_FIELD = HERE / "results" / "raw_field"
GATES = ("V0_gate", "T0_gate")
CHANCE, N_BOOT, TOL = 0.20, 10_000, 0.002
CEILING = 0.634
RNG = np.random.default_rng(20260730)

# Laboratory corpus, article section 6.
REPORTED = {
    "qwen25vl-3b":    {"V0": 0.107, "T0": 0.962, "A": -0.085, "D_adv": +0.192},
    "internvl3-8b":   {"V0": 0.428, "T0": 0.998, "A": -0.117, "D_adv": +0.456},
    "qwen25vl-7b":    {"V0": 0.183, "T0": 0.860, "A": -0.294, "D_adv": +0.661},
    "qwen25-omni-7b": {"V0": 0.191, "T0": 0.958, "A": -0.369, "D_adv": +0.639},
}
# Field corpus (PlantDoc), article section 6.9.
REPORTED_FIELD = {
    "qwen25vl-3b-field":    {"V0": 0.436, "A": -0.059},
    "qwen25vl-7b-field":    {"V0": 0.471, "A": -0.324},
    "internvl3-8b-field":   {"V0": 0.677, "A": -0.031},
    "qwen25-omni-7b-field": {"V0": 0.358, "A": -0.450},
}

fails = []
RULE = "=" * 74


def check(label, got, want, tol=TOL):
    if abs(got - want) > tol:
        fails.append(f"{label}: recomputed {got:+.3f} vs article {want:+.3f}")
        return "FAIL"
    return "ok  "


def load(root):
    rows = []
    for p in sorted(root.glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        rows.append(r)
    return rows


def subset_acc(rows, model, subset):
    v = [r["correct"] for r in rows
         if r["model"] == model and r["subset"] == subset and r["cell"] not in GATES]
    return sum(v) / len(v) if v else float("nan")


def gate_acc(rows, model, cell):
    v = [r["correct"] for r in rows if r["model"] == model and r["cell"] == cell]
    return sum(v) / len(v), len(v), sum(v)


def boot_ci(rows, model):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["model"] == model and r["cell"] not in GATES:
            per[r["uid"]][r["subset"]].append(r["correct"])
    uids = list(per)
    stat = np.empty(N_BOOT)
    for b in range(N_BOOT):
        acc = defaultdict(list)
        for i in RNG.choice(len(uids), len(uids)):
            for k, vals in per[uids[i]].items():
                acc[k].extend(vals)
        stat[b] = np.mean(acc["S_img"]) + np.mean(acc["S_txt"]) - 1
    return tuple(np.percentile(stat, [2.5, 97.5]))


def main():
    rows = load(RAW)
    if not rows:
        sys.exit(f"no records under {RAW}")
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} records | {len(models)} models "
          f"| {len({r['uid'] for r in rows})} items")

    print("\n" + RULE)
    print("1. ADMISSIBILITY GATES  (chance 0.20, one-sided binomial)")
    print(RULE)
    for m in models:
        parts = []
        for cell, key in (("V0_gate", "V0"), ("T0_gate", "T0")):
            acc, n, k = gate_acc(rows, m, cell)
            p = stats.binomtest(k, n, CHANCE, alternative="greater").pvalue
            parts.append(f"{key} {acc:.3f} [{check(f'{m}.{key}', acc, REPORTED[m][key])}] "
                         f"{'admitted' if p < 0.05 else 'excluded'}")
        print(f"  {m:<16}" + " | ".join(parts))

    print("\n" + RULE)
    print("2. ARBITRATION  A = acc(S_img) + acc(S_txt) - 1")
    print(f"   bounds: fixed channel 0.000 | provenance ceiling +{CEILING} | oracle +1.000")
    print(RULE)
    for m in models:
        si, st = subset_acc(rows, m, "S_img"), subset_acc(rows, m, "S_txt")
        A = si + st - 1
        lo, hi = boot_ci(rows, m)
        print(f"  {m:<16}S_img={si:.3f} S_txt={st:.3f}  A={A:+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"  {A / CEILING * 100:+6.1f}% of ceiling  "
              f"[{check(f'{m}.A', A, REPORTED[m]['A'])}]")

    print("\n" + RULE)
    print("3. ADVERSARIAL DEFERENCE  D_adv on S_img (report wrong, image right)")
    print(RULE)
    for m in models:
        div = [r for r in rows if r["model"] == m and r["cell"] not in GATES
               and r["subset"] == "S_img" and r["letter_image"] != r["letter_text"]]

        def frac(q, sel=div):
            s = [r for r in sel if r["q_rep"] == q]
            return sum(1 for r in s if r["chose"] == "text") / (len(s) or 1)

        D = frac("verified") - frac("unverified")
        print(f"  {m:<16}D_adv={D:+.3f}   [{check(f'{m}.D_adv', D, REPORTED[m]['D_adv'])}]")

    frows = load(RAW_FIELD) if RAW_FIELD.exists() else []
    if frows:
        print("\n" + RULE)
        print("4. FIELD-CONDITION REPLICATION (PlantDoc), article section 6.9")
        print(RULE)
        for m in sorted({r["model"] for r in frows}):
            acc0, _, _ = gate_acc(frows, m, "V0_gate")
            A = subset_acc(frows, m, "S_img") + subset_acc(frows, m, "S_txt") - 1
            w = REPORTED_FIELD.get(m, {})
            c1 = check(f"{m}.V0", acc0, w.get("V0", acc0))
            c2 = check(f"{m}.A", A, w.get("A", A))
            print(f"  {m:<22}V0={acc0:.3f} [{c1}]  A={A:+.3f} [{c2}]")

    print("\n" + RULE)
    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"All recomputed values match the article (tolerance {TOL}).")


if __name__ == "__main__":
    main()
