"""Rebuild every number reported in the article from the raw per-probe records.

No GPU and no model download required. `results/raw/`, `results/raw_field/`,
`results/raw_imageonly/` and `results/raw_arabic/` contain one record per probe:
the prediction, the gold letter, which source was chosen, and the per-letter
logits.

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
RAW_IMGONLY = HERE / "results" / "raw_imageonly"
RAW_ARABIC = HERE / "results" / "raw_arabic"
GATES = ("V0_gate", "T0_gate")
FACTORIAL = ("fresh_verified", "fresh_unverified", "stale_verified", "stale_unverified")
LANGS = ("en", "fr", "ar", "dz")
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
# End-to-end arbitration, article section 6.11.
REPORTED_E2E = {
    "qwen25vl-3b":    {"A_imp": -0.084, "A_or": +0.659, "A_e2e": -0.036, "tool": 0.094},
    "qwen25vl-7b":    {"A_imp": -0.259, "A_or": +0.659, "A_e2e": +0.055, "tool": 0.161},
    "internvl3-8b":   {"A_imp": -0.201, "A_or": +0.659, "A_e2e": +0.232, "tool": 0.344},
    "qwen25-omni-7b": {"A_imp": -0.446, "A_or": +0.659, "A_e2e": +0.057, "tool": 0.134},
}
# Four languages, article section 6.12. Order: en, fr, ar, dz.
REPORTED_LANGS = {
    "qwen25vl-3b":    {"S_img": (0.001, 0.002, 0.003, 0.035),
                       "V0_gate": (0.120, 0.118, 0.161, 0.226)},
    "qwen25vl-7b":    {"S_img": (0.003, 0.007, 0.010, 0.022),
                       "V0_gate": (0.223, 0.152, 0.278, 0.358)},
    "internvl3-8b":   {"S_img": (0.063, 0.064, 0.081, 0.112),
                       "V0_gate": (0.419, 0.466, 0.453, 0.414)},
    "qwen25-omni-7b": {"S_img": (0.004, 0.007, 0.016, 0.025),
                       "V0_gate": (0.227, 0.169, 0.221, 0.213)},
}
# Repository file stem per model key.
STEM = {"qwen25vl-3b": "qwen2.5-vl-3b", "qwen25vl-7b": "qwen2.5-vl-7b",
        "internvl3-8b": "internvl3-8b", "qwen25-omni-7b": "qwen2.5-omni-7b"}
ORDER = ["qwen25vl-3b", "qwen25vl-7b", "internvl3-8b", "qwen25-omni-7b"]

fails = []
RULE = "=" * 74


def check(label, got, want, tol=TOL):
    if abs(got - want) > tol:
        fails.append(f"{label}: recomputed {got:+.3f} vs article {want:+.3f}")
        return "FAIL"
    return "ok  "


def read_records(path):
    """One file of records. Accepts the released .jsonl.gz or a decompressed .jsonl."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [r for r in (json.loads(l) for l in fh if l.strip()) if "error" not in r]


def record_files(root):
    files = [p for p in sorted(root.glob("*.jsonl.gz")) if p.is_file()]
    return files or [p for p in sorted(root.glob("*.jsonl")) if p.is_file()]


def load(root):
    return [r for p in record_files(root) for r in read_records(p)]


def load_one(root, stem):
    for suffix in (".jsonl.gz", ".jsonl"):
        p = root / f"{stem}{suffix}"
        if p.is_file():
            return read_records(p)
    return []


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


# --- end-to-end arbitration, replicating src/end_to_end.py on the released records


def fit_rho(cal):
    a = defaultdict(list)
    for r in cal:
        a[("img", r["q_img"])].append(int(r["img_ok"]))
        a[("rep", r["q_rep"])].append(int(r["rep_ok"]))
    return {k: sum(v) / len(v) for k, v in a.items() if v}


def arbitrate(y_img, y_rep, q_img, q_rep, rho):
    w_i, w_r = rho.get(("img", q_img), .5), rho.get(("rep", q_rep), .5)
    score = defaultdict(float)
    score[y_img] += w_i
    score[y_rep] += w_r
    return max(score, key=score.get)


def index_A(rows, pick):
    acc = defaultdict(list)
    for r in rows:
        acc[r["subset"]].append(int(pick(r) == r["gold"]))
    si = np.mean(acc["S_img"]) if acc["S_img"] else float("nan")
    st = np.mean(acc["S_txt"]) if acc["S_txt"] else float("nan")
    return si + st - 1


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

    if RAW_IMGONLY.exists():
        print("\n" + RULE)
        print("5. END-TO-END ARBITRATION, article section 6.11")
        print("   per-source decisions produced by the controller itself, English only")
        print(RULE)
        print(f"  {'model':<17}{'implicit':>10}{'oracle':>9}{'end to end':>12}"
              f"{'% ceiling':>11}{'img tool':>10}")
        for m in ORDER:
            lab = [r for r in load_one(RAW, STEM[m])
                   if r["cell"] in FACTORIAL and r["lang"] == "en"]
            io = load_one(RAW_IMGONLY, STEM[m])
            if not lab or not io:
                continue
            # The image-only cell carries the IO_ prefix on the originating cell.
            tool = {(r["uid"], r["cell"][3:]): r["pred"] for r in io}
            tool_acc = float(np.mean([r["correct"] for r in io]))

            paired = [{**r, "y_img_tool": tool[(r["uid"], r["cell"])]}
                      for r in lab if (r["uid"], r["cell"]) in tool]
            uids = sorted({r["uid"] for r in paired})
            rng = np.random.default_rng(20261022)
            rng.shuffle(uids)
            cal_ids = set(uids[: len(uids) // 2])
            rho = fit_rho([r for r in paired if r["uid"] in cal_ids])
            test = [r for r in paired if r["uid"] not in cal_ids]

            A_imp = index_A(test, lambda r: r["pred"])
            A_or = index_A(test, lambda r: arbitrate(
                r["letter_image"], r["letter_text"], r["q_img"], r["q_rep"], rho))
            A_e2e = index_A(test, lambda r: arbitrate(
                r["y_img_tool"], r["letter_text"], r["q_img"], r["q_rep"], rho))
            w = REPORTED_E2E[m]
            c = [check(f"{m}.A_imp", A_imp, w["A_imp"]),
                 check(f"{m}.A_oracle", A_or, w["A_or"]),
                 check(f"{m}.A_e2e", A_e2e, w["A_e2e"]),
                 check(f"{m}.img_tool", tool_acc, w["tool"])]
            print(f"  {m:<17}{A_imp:>+10.3f}{A_or:>+9.3f}{A_e2e:>+12.3f}"
                  f"{A_e2e / CEILING * 100:>10.1f}%{tool_acc:>10.3f}")
            print(f"  {'':<17}[{c[0]}] [{c[1]}] [{c[2]}] [{c[3]}]")

    if RAW_ARABIC.exists():
        print("\n" + RULE)
        print("6. FOUR LANGUAGES, article section 6.12")
        print("   items common to all four languages; Arabic and darija await")
        print("   native-speaker validation, see section 6.12")
        print(RULE)
        print(f"  {'model':<17}{'subset':<10}" + "".join(f"{l:>9}" for l in LANGS)
              + f"{'chi2 p':>11}")
        for m in ORDER:
            rows_m = load_one(RAW, STEM[m]) + load_one(RAW_ARABIC, STEM[m])
            by = defaultdict(list)
            for r in rows_m:
                if r["lang"] in LANGS:
                    by[r["lang"]].append(r)
            if any(not by[l] for l in LANGS):
                continue
            common = set.intersection(*[{r["uid"] for r in by[l]} for l in LANGS])
            for label, sel in (
                    ("S_img", lambda r: r["subset"] == "S_img" and r["cell"] not in GATES),
                    ("V0_gate", lambda r: r["cell"] == "V0_gate")):
                counts, accs, cs = [], [], []
                for i, l in enumerate(LANGS):
                    v = [r["correct"] for r in by[l] if r["uid"] in common and sel(r)]
                    accs.append(float(np.mean(v)))
                    counts.append([int(sum(v)), int(len(v) - sum(v))])
                    cs.append(check(f"{m}.{label}.{l}", accs[-1],
                                    REPORTED_LANGS[m][label][i]))
                p = stats.chi2_contingency(np.array(counts).T)[1]
                print(f"  {m:<17}{label:<10}" + "".join(f"{a:>9.3f}" for a in accs)
                      + f"{p:>11.1e}")
                print(f"  {'':<17}{'':<10}" + "".join(f"{x.strip():>9}" for x in cs))

    print("\n" + RULE)
    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"All recomputed values match the article (tolerance {TOL}).")


if __name__ == "__main__":
    main()
