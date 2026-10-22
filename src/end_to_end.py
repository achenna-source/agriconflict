"""SAGA de bout en bout : vraies decisions par source, plus d'oracle.

La reserve methodologique de l'article etait que la couche d'arbitrage recevait
des decisions par source EXACTES. Le corpus image-seule fournit maintenant, pour
chaque cellule factorielle, la decision que le CONTROLEUR LUI-MEME prend sur la
seule photographie de cette cellule. On peut donc comparer, sur les memes items :

  controleur implicite      perception + agregation internes
  SAGA oraculaire           agregation externalisee, sources parfaites
  SAGA bout en bout         agregation externalisee, sources = le controleur
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FACTORIAL = ("fresh_verified", "fresh_unverified", "stale_verified", "stale_unverified")
ALPHA, N_BOOT, CEILING = 0.10, 10_000, 0.634
RNG = np.random.default_rng(20261022)

MODELS = {
    "qwen25vl-3b":    ("_q3b/results_qwen25vl-3b.jsonl", "_r_02i-qwen3b"),
    "qwen25vl-7b":    ("_log_qwen7b/results_qwen25vl-7b.jsonl", "_r_02i-qwen7b"),
    "internvl3-8b":   ("_log_internvl/results_internvl3-8b.jsonl", "_r_02i-internvl"),
    "qwen25-omni-7b": ("_log_omni/results_qwen25-omni-7b.jsonl", "_r_02i-omni"),
}


def load(pat):
    base = HERE / pat
    hits = ([str(q) for q in base.glob("results_*.jsonl")] if base.is_dir()
            else glob.glob(str(base)))
    rows = []
    for f in hits:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    rows.append(r)
    return rows


def fit_rho(cal):
    a = defaultdict(list)
    for r in cal:
        a[("img", r["q_img"])].append(int(r["img_ok"]))
        a[("rep", r["q_rep"])].append(int(r["rep_ok"]))
    return {k: sum(v) / len(v) for k, v in a.items() if v}


def arbitrate(y_img, y_rep, q_img, q_rep, rho):
    w_i, w_r = rho.get(("img", q_img), .5), rho.get(("rep", q_rep), .5)
    score, support = defaultdict(float), defaultdict(list)
    score[y_img] += w_i; support[y_img].append("image")
    score[y_rep] += w_r; support[y_rep].append("report")
    best = max(score, key=score.get)
    return best, score[best] / (sum(score.values()) or 1.0), support[best]


def index_A(rows, pick):
    """A = acc(S_img) + acc(S_txt) - 1 sous la politique pick(r) -> lettre."""
    acc = defaultdict(list)
    for r in rows:
        acc[r["subset"]].append(int(pick(r) == r["gold"]))
    si = np.mean(acc["S_img"]) if acc["S_img"] else float("nan")
    st = np.mean(acc["S_txt"]) if acc["S_txt"] else float("nan")
    return si + st - 1, si, st


def boot(rows, pick):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["uid"]][r["subset"]].append(int(pick(r) == r["gold"]))
    uids = list(per)
    s = np.empty(N_BOOT)
    for b in range(N_BOOT):
        a = defaultdict(list)
        for i in RNG.choice(len(uids), len(uids)):
            for k, v in per[uids[i]].items():
                a[k].extend(v)
        s[b] = np.mean(a["S_img"]) + np.mean(a["S_txt"]) - 1
    return tuple(np.percentile(s, [2.5, 97.5]))


out = {}
print(f"{'modele':<17}{'implicite':>11}{'SAGA oracle':>13}{'SAGA e2e':>11}"
      f"{'IC95 e2e':>19}{'%plaf':>8}{'img tool':>10}")
for name, (lab_pat, io_dir) in MODELS.items():
    lab = [r for r in load(lab_pat) if r["cell"] in FACTORIAL and r["lang"] == "en"]
    io = load(io_dir)
    # decision par source du controleur, indexee par (item, cellule)
    # le champ source_cell n'est pas recopie dans les resultats ;
    # la cellule image-seule porte le prefixe IO_ sur la cellule d'origine
    tool = {(r["uid"], r["cell"][3:]): r["pred"] for r in io}
    tool_acc = np.mean([r["correct"] for r in io])

    rows = []
    for r in lab:
        k = (r["uid"], r["cell"])
        if k not in tool:
            continue
        rows.append({**r, "y_img_tool": tool[k]})
    uids = sorted({r["uid"] for r in rows})
    RNG2 = np.random.default_rng(20261022)
    RNG2.shuffle(uids)
    cal_ids = set(uids[: len(uids) // 2])
    cal = [r for r in rows if r["uid"] in cal_ids]
    test = [r for r in rows if r["uid"] not in cal_ids]
    rho = fit_rho(cal)

    A_imp, _, _ = index_A(test, lambda r: r["pred"])
    A_or, _, _ = index_A(test, lambda r: arbitrate(
        r["letter_image"], r["letter_text"], r["q_img"], r["q_rep"], rho)[0])
    A_e2e, si, st = index_A(test, lambda r: arbitrate(
        r["y_img_tool"], r["letter_text"], r["q_img"], r["q_rep"], rho)[0])
    lo, hi = boot(test, lambda r: arbitrate(
        r["y_img_tool"], r["letter_text"], r["q_img"], r["q_rep"], rho)[0])
    print(f"{name:<17}{A_imp:>+11.3f}{A_or:>+13.3f}{A_e2e:>+11.3f}"
          f"   [{lo:>+6.3f},{hi:>+6.3f}]{A_e2e/CEILING*100:>7.1f}%{tool_acc:>10.3f}")
    out[name] = {"A_implicit": A_imp, "A_oracle": A_or, "A_e2e": A_e2e,
                 "A_e2e_ci": [lo, hi], "pct_ceiling": A_e2e / CEILING,
                 "image_tool_acc": float(tool_acc),
                 "S_img_e2e": float(si), "S_txt_e2e": float(st),
                 "n_test_items": len(uids) - len(cal_ids)}

(HERE / "_results_e2e.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\n-> _results_e2e.json")
