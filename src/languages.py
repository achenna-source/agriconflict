"""Effet de langue sur quatre langues : anglais, francais, arabe standard, darija.

Les sondes arabes et darija sont DERIVEES du corpus publie : memes items, memes
photographies, memes options et meme ordre. Seul le cadre textuel change. Les
quatre langues sont donc directement comparables sur les 1 000 premiers items.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
GATES = ("V0_gate", "T0_gate")
LANGS = ("en", "fr", "ar", "dz")
SRC = {
    "qwen25vl-3b":    ["_q3b/results_qwen25vl-3b.jsonl", "_r_02a-qwen3b"],
    "qwen25vl-7b":    ["_log_qwen7b/results_qwen25vl-7b.jsonl", "_r_02a-qwen7b"],
    "internvl3-8b":   ["_log_internvl/results_internvl3-8b.jsonl", "_r_02a-internvl"],
    "qwen25-omni-7b": ["_log_omni/results_qwen25-omni-7b.jsonl", "_r_02a-omni"],
}


def load(pat):
    base = HERE / pat
    files = ([str(q) for q in base.glob("results_*.jsonl")] if base.is_dir()
             else glob.glob(str(base)))
    rows = []
    for f in files:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    rows.append(r)
    return rows


out = {}
print("Sur les items communs aux quatre langues.\n")
print(f"{'modele':<17}{'cellule':<10}" + "".join(f"{l:>9}" for l in LANGS)
      + f"{'khi2 p':>10}")
for name, pats in SRC.items():
    rows = [r for p in pats for r in load(p)]
    by = defaultdict(list)
    for r in rows:
        if r["lang"] in LANGS:
            by[r["lang"]].append(r)
    common = set.intersection(*[{r["uid"] for r in by[l]} for l in LANGS])
    res = {}
    for label, sel in (("S_img", lambda r: r["subset"] == "S_img" and r["cell"] not in GATES),
                       ("V0_gate", lambda r: r["cell"] == "V0_gate"),
                       ("conflits", lambda r: r["cell"] not in GATES)):
        counts, accs = [], {}
        for l in LANGS:
            v = [r["correct"] for r in by[l] if r["uid"] in common and sel(r)]
            accs[l] = float(np.mean(v)) if v else float("nan")
            counts.append([int(sum(v)), int(len(v) - sum(v))])
        p = stats.chi2_contingency(np.array(counts).T)[1]
        print(f"{name:<17}{label:<10}" + "".join(f"{accs[l]:>9.3f}" for l in LANGS)
              + f"{p:>10.2e}" + ("  <--" if p < 0.05 else ""))
        res[label] = {"acc": accs, "chi2_p": float(p)}
    out[name] = {"n_common_items": len(common), **res}
    print()

(HERE / "_results_langs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("-> _results_langs.json")
