"""Comparaison laboratoire (PlantVillage) contre terrain (PlantDoc).

Meme processus generatif, memes cellules, memes metriques. Seule la source
d'images change. Les intervalles sont des bootstrap groupes par ITEM.
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
GATES = ("V0_gate", "T0_gate")
CHANCE, N_BOOT = 0.20, 10_000
CEILING = 0.634
RNG = np.random.default_rng(20260802)

CORPORA = {
    "labo": {
        "qwen25vl-3b":    "_q3b/results_qwen25vl-3b.jsonl",
        "qwen25vl-7b":    "_log_qwen7b/results_qwen25vl-7b.jsonl",
        "internvl3-8b":   "_log_internvl/results_internvl3-8b.jsonl",
        "qwen25-omni-7b": "_log_omni/results_qwen25-omni-7b.jsonl",
    },
    "terrain": {
        "qwen25vl-3b":    "_f3b/results_*-field.jsonl",
        "qwen25vl-7b":    "_f_qwen7b/results_*-field.jsonl",
        "internvl3-8b":   "_f_internvl/results_*-field.jsonl",
        "qwen25-omni-7b": "_f_omni/results_*-field.jsonl",
    },
}
ORDER = ["qwen25vl-3b", "qwen25vl-7b", "internvl3-8b", "qwen25-omni-7b"]
SHORT = {"qwen25vl-3b": "Qwen2.5-VL-3B", "qwen25vl-7b": "Qwen2.5-VL-7B",
         "internvl3-8b": "InternVL3-8B", "qwen25-omni-7b": "Qwen2.5-Omni-7B"}


def load(pattern: str):
    hits = glob.glob(str(HERE / pattern))
    if not hits:
        return []
    return [r for r in (json.loads(l) for l in open(hits[0], encoding="utf-8") if l.strip())
            if "error" not in r]


def boot_A(rows):
    """A = acc(S_img) + acc(S_txt) - 1, IC bootstrap groupe par item."""
    conf = [r for r in rows if r["cell"] not in GATES]
    per = defaultdict(lambda: defaultdict(list))
    for r in conf:
        per[r["uid"]][r["subset"]].append(r["correct"])
    uids = list(per)
    stat = np.empty(N_BOOT)
    for b in range(N_BOOT):
        acc = defaultdict(list)
        for i in RNG.choice(len(uids), len(uids)):
            for k, v in per[uids[i]].items():
                acc[k].extend(v)
        stat[b] = np.mean(acc["S_img"]) + np.mean(acc["S_txt"]) - 1
    sub = {s: np.mean([r["correct"] for r in conf if r["subset"] == s])
           for s in ("S_img", "S_txt", "S_none")}
    A = sub["S_img"] + sub["S_txt"] - 1
    return A, tuple(np.percentile(stat, [2.5, 97.5])), sub


def metrics(rows):
    out = {}
    v0 = [r for r in rows if r["cell"] == "V0_gate"]
    out["V0"] = np.mean([r["correct"] for r in v0])
    out["V0_p"] = stats.binomtest(int(sum(r["correct"] for r in v0)), len(v0),
                                  CHANCE, alternative="greater").pvalue
    comm = [r for r in v0 if r["chose"] != "abstain"]
    out["commit"] = len(comm) / len(v0)
    out["acc_commit"] = np.mean([r["correct"] for r in comm]) if comm else float("nan")
    t0 = [r for r in rows if r["cell"] == "T0_gate"]
    out["T0"] = np.mean([r["correct"] for r in t0])
    out["A"], out["A_ci"], sub = boot_A(rows)
    out.update(sub)

    conf = [r for r in rows if r["cell"] not in GATES]
    adv = [r for r in conf if r["subset"] == "S_img"
           and r["letter_image"] != r["letter_text"]]
    def ft(q):
        sel = [r for r in adv if r["q_rep"] == q]
        return sum(1 for r in sel if r["chose"] == "text") / (len(sel) or 1)
    out["D_adv"] = ft("verified") - ft("unverified")
    return out


def main():
    data = {c: {m: load(p) for m, p in files.items()} for c, files in CORPORA.items()}
    res = {}
    for corpus in ("labo", "terrain"):
        for m in ORDER:
            rows = data[corpus][m]
            if rows:
                res[(corpus, m)] = metrics(rows)
            else:
                print(f"!! absent : {corpus}/{m}")

    print("=" * 100)
    print("PORTE VISION (V0)  —  hasard 0.20")
    print("=" * 100)
    print(f"  {'modele':<17}{'labo':>10}{'terrain':>10}{'ecart':>10}   admission")
    for m in ORDER:
        a, b = res[("labo", m)], res[("terrain", m)]
        adm = ("labo " + ("OK " if a["V0_p"] < .05 else "non") +
               " | terrain " + ("OK" if b["V0_p"] < .05 else "non"))
        print(f"  {SHORT[m]:<17}{a['V0']:>10.3f}{b['V0']:>10.3f}"
              f"{b['V0']-a['V0']:>+10.3f}   {adm}")

    print("\n" + "=" * 100)
    print("VISION CONDITIONNEE A L'ENGAGEMENT  —  hasard 0.25")
    print("=" * 100)
    print(f"  {'modele':<17}{'engage L':>10}{'acc|eng L':>11}{'engage T':>10}{'acc|eng T':>11}")
    for m in ORDER:
        a, b = res[("labo", m)], res[("terrain", m)]
        print(f"  {SHORT[m]:<17}{a['commit']:>10.3f}{a['acc_commit']:>11.3f}"
              f"{b['commit']:>10.3f}{b['acc_commit']:>11.3f}")

    print("\n" + "=" * 100)
    print("ARBITRAGE  A = acc(S_img) + acc(S_txt) - 1")
    print(f"  canal fixe 0.000 | Bayes-provenance +{CEILING} | oracle +1.000")
    print("=" * 100)
    print(f"  {'modele':<17}{'A labo':>9}{'IC labo':>19}{'A terrain':>11}{'IC terrain':>19}  zero ?")
    for m in ORDER:
        a, b = res[("labo", m)], res[("terrain", m)]
        z = "INCLUS" if b["A_ci"][0] <= 0 <= b["A_ci"][1] else "exclu"
        print(f"  {SHORT[m]:<17}{a['A']:>+9.3f}  [{a['A_ci'][0]:>+6.3f},{a['A_ci'][1]:>+6.3f}]"
              f"{b['A']:>+11.3f}  [{b['A_ci'][0]:>+6.3f},{b['A_ci'][1]:>+6.3f}]  {z}")

    print("\n" + "=" * 100)
    print("S_img (image juste, rapport faux)  et  DEFERENCE ADVERSE D_adv")
    print("=" * 100)
    print(f"  {'modele':<17}{'S_img L':>10}{'S_img T':>10}{'D_adv L':>10}{'D_adv T':>10}")
    for m in ORDER:
        a, b = res[("labo", m)], res[("terrain", m)]
        print(f"  {SHORT[m]:<17}{a['S_img']:>10.3f}{b['S_img']:>10.3f}"
              f"{a['D_adv']:>+10.3f}{b['D_adv']:>+10.3f}")

    out = {f"{c}/{m}": {k: (list(v) if isinstance(v, tuple) else float(v))
                        for k, v in d.items()}
           for (c, m), d in res.items()}
    (HERE / "_results_corpora.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n-> _results_corpora.json")


if __name__ == "__main__":
    main()
