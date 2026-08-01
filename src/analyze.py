"""
AgriConflict v3 — Notebook 3 (CPU) : analyse.

Ce qui change par rapport a la v2. Les sous-ensembles ne sont plus definis par le
plan experimental mais par la CORRECTION REELLE des sources :

  S_both  les deux soutiennent Y     S_img   l'image seule soutient Y
  S_txt   le rapport seul soutient Y  S_none  aucune -> abstention correcte

La provenance ne fait que les predire imparfaitement (0.938 / 0.297 / 0.903 / 0.357),
si bien que rho doit estimer des probabilites au lieu de memoriser une regle.

  A = acc(S_img) + acc(S_txt) - 1

Bornes de reference, calculees sur le corpus lui-meme (notebook 00) :
  canal fixe (image ou rapport)   A = 0.000
  abstention systematique         A = -1.000
  Bayes sur provenance seule      A = +0.634   <- plafond de tout arbitre
  oracle                          A = +1.000
Tout A est rapporte en valeur absolue ET en fraction du plafond atteignable.

Inference : bootstrap groupe par ITEM (uid). Effectif reel = 2 000 items.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).parent / "_an_v3"
OUT.mkdir(exist_ok=True)
ALPHA, N_BOOT = 0.05, 10_000
RNG = np.random.default_rng(20260730)
GATES = ("V0_gate", "T0_gate")


ROOTS = [Path("/kaggle/input"), Path(__file__).parent]     # Kaggle, sinon local


def _find(pattern: str):
    for root in ROOTS:
        if root.exists():
            hits = sorted(root.rglob(pattern))
            if hits:
                return hits
    return []


def load() -> list[dict]:
    rows = []
    for p in _find("results_*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                if "error" not in r and "subset" in r:      # v3 uniquement
                    rows.append(r)
    if not rows:
        raise FileNotFoundError("Aucun results_*.jsonl au format v3.")
    return rows


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(0.975)
    p, d = k / n, 1 + z**2 / n
    c, m = p + z**2 / (2 * n), z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def cluster_boot(per_item: dict, fn) -> tuple[float, float]:
    uids = list(per_item)
    stat = np.empty(N_BOOT)
    for b in range(N_BOOT):
        acc = defaultdict(list)
        for i in RNG.choice(len(uids), len(uids)):
            for key, vals in per_item[uids[i]].items():
                acc[key].extend(vals)
        stat[b] = fn({k: float(np.mean(v)) for k, v in acc.items() if v})
    return tuple(np.percentile(stat, [2.5, 97.5]))


def main() -> None:
    rows = load()
    meta_p = next(iter(_find("probes_meta.json")), None)
    meta = json.loads(meta_p.read_text()) if meta_p else {}
    chance = meta.get("chance", 0.2)
    models = sorted({r["model"] for r in rows})
    langs = sorted({r["lang"] for r in rows})
    res = {"chance": chance, "bounds": {"fixed_channel": 0.0, "bayes_provenance": 0.634,
                                        "oracle": 1.0}, "models": {}}

    conflict = [r for r in rows if r["cell"] not in GATES]

    def acc(m, *, cell=None, subset=None, lang=None) -> np.ndarray:
        src = rows if cell in GATES else conflict
        return np.array([r["correct"] for r in src if r["model"] == m
                         and (cell is None or r["cell"] == cell)
                         and (subset is None or r["subset"] == subset)
                         and (lang is None or r["lang"] == lang)], dtype=float)

    def items(m) -> dict:
        d = defaultdict(lambda: defaultdict(list))
        for r in conflict:
            if r["model"] == m:
                d[r["uid"]][r["subset"]].append(r["correct"])
        return {k: dict(v) for k, v in d.items()}

    # -------------------------------------------------------------- 1. portes
    print("=" * 80)
    print(f"1. PORTES   H0: p <= {chance:.2f}, binomial unilateral, alpha={ALPHA}")
    print("=" * 80)
    gate = {}
    for m in models:
        cells = {}
        for c, name in [("V0_gate", "vision"), ("T0_gate", "texte")]:
            v = acc(m, cell=c)
            k, n = int(v.sum()), v.size
            p = stats.binomtest(k, n, chance, alternative="greater").pvalue
            lo, hi = wilson(k, n)
            cells[name] = (k / n, lo, hi, p)
        gate[m] = all(c[3] < ALPHA for c in cells.values())
        print(f"  {m:<16}" + " | ".join(
            f"{n} {v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}] p={v[3]:.1e}" for n, v in cells.items())
            + f"  -> {'ADMIS' if gate[m] else 'EXCLU'}")
        res["models"][m] = {"gate_passed": bool(gate[m]),
                            "V0": cells["vision"][0], "T0": cells["texte"][0]}
    print(f"\n  >>> REJET AUX PORTES : {sum(1 for m in models if not gate[m])}/{len(models)}")

    # ------------------------------------------- 2. vision conditionnee a l'engagement
    print("\n" + "=" * 80)
    print("2. VISION CONDITIONNEE A L'ENGAGEMENT  [analyse reconduite de la v2]")
    print("   La porte melange competence et volonte de trancher. Conditionnellement")
    print("   a un engagement, le hasard porte sur les 4 classes -> 0.25.")
    print("=" * 80)
    for m in models:
        v0 = [r for r in rows if r["model"] == m and r["cell"] == "V0_gate"]
        comm = [r for r in v0 if r["chose"] != "abstain"]
        if not comm:
            continue
        k = sum(r["correct"] for r in comm)
        p = stats.binomtest(k, len(comm), 0.25, alternative="greater").pvalue
        print(f"  {m:<16}engage {len(comm)/len(v0):.3f}  acc|engage {k/len(comm):.3f}  "
              f"p={p:.1e}")
        res["models"][m].update({"commit_rate": len(comm) / len(v0),
                                 "acc_given_commit": k / len(comm)})

    # ---------------------------------------------------------- 3. arbitrage
    print("\n" + "=" * 80)
    print("3. ARBITRAGE   A = acc(S_img) + acc(S_txt) - 1")
    print("   reperes : canal fixe 0.000 | Bayes-provenance +0.634 | oracle +1.000")
    print("=" * 80)
    print(f"  {'modele':<16}{'S_both':>8}{'S_img':>8}{'S_txt':>8}{'S_none':>8}"
          f"{'A':>8}{'IC 95%':>18}{'% plafond':>11}")
    for m in models:
        s = {k: acc(m, subset=k) for k in ("S_both", "S_img", "S_txt", "S_none")}
        A = s["S_img"].mean() + s["S_txt"].mean() - 1
        lo, hi = cluster_boot(items(m), lambda d: d.get("S_img", 0) + d.get("S_txt", 0) - 1)
        print(f"  {m:<16}" + "".join(f"{s[k].mean():>8.3f}" for k in
              ("S_both", "S_img", "S_txt", "S_none")) +
              f"{A:>8.3f}  [{lo:>6.3f},{hi:>6.3f}]{A/0.634*100:>10.1f}%")
        res["models"][m].update({**{k: s[k].mean() for k in s},
                                 "A": A, "A_ci": [lo, hi], "A_pct_ceiling": A / 0.634})

    # ------------------------------------------------ 4. attribution de source
    print("\n" + "=" * 80)
    print("4. ATTRIBUTION DE SOURCE par sous-ensemble")
    print("=" * 80)
    for m in models:
        print(f"  {m}")
        for sub in ("S_img", "S_txt", "S_none"):
            ch = [r["chose"] for r in conflict if r["model"] == m and r["subset"] == sub]
            n = len(ch) or 1
            d = {k: ch.count(k) / n for k in ("image", "text", "abstain", "other")}
            print(f"    {sub:<8}" + "  ".join(f"{k}={v:.2f}" for k, v in d.items()))
            res["models"][m].setdefault("choice", {})[sub] = d

    # ------------------------------------------- 5. deference a la provenance
    print("\n" + "=" * 80)
    print("5. DEFERENCE A LA PROVENANCE  [non confondue en v3]")
    print("   D = P(suit le rapport | verifie) - P(suit le rapport | non verifie),")
    print("   sur les items ou les deux sources DIVERGENT.")
    print("   D_adv : idem restreint a S_img, ou le rapport est faux et l'image juste.")
    print("=" * 80)
    for m in models:
        div = [r for r in conflict if r["model"] == m
               and r["letter_image"] != r["letter_text"]]
        def frac(sel):
            n = len(sel) or 1
            return sum(1 for r in sel if r["chose"] == "text") / n
        D = frac([r for r in div if r["q_rep"] == "verified"]) - \
            frac([r for r in div if r["q_rep"] == "unverified"])
        adv = [r for r in div if r["subset"] == "S_img"]
        D_adv = frac([r for r in adv if r["q_rep"] == "verified"]) - \
                frac([r for r in adv if r["q_rep"] == "unverified"])
        print(f"  {m:<16}D={D:+.3f}   D_adv={D_adv:+.3f}")
        res["models"][m].update({"D_provenance": D, "D_adversarial": D_adv})

    # ------------------------------------------------------ 6. effet de langue
    if len(langs) > 1:
        print("\n" + "=" * 80)
        print("6. EFFET DE LANGUE sur S_img (image juste, rapport faux)")
        print("=" * 80)
        for m in models:
            per = {l: acc(m, subset="S_img", lang=l) for l in langs}
            if any(v.size == 0 for v in per.values()):
                continue
            tbl = [[int(v.sum()), int(v.size - v.sum())] for v in per.values()]
            p = stats.fisher_exact(tbl)[1] if len(langs) == 2 else float("nan")
            print(f"  {m:<16}" + "  ".join(f"{l}={per[l].mean():.3f}" for l in langs) +
                  f"   p={p:.4f}{'  <-- significatif' if p < ALPHA else ''}")
            res["models"][m]["lang_S_img"] = {l: per[l].mean() for l in langs}
            res["models"][m]["lang_p"] = float(p)

    # ------------------------------------------------------------- sorties
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    xs = np.arange(len(models))
    vals = [res["models"][m]["A"] for m in models]
    err = np.array([[res["models"][m]["A"] - res["models"][m]["A_ci"][0] for m in models],
                    [res["models"][m]["A_ci"][1] - res["models"][m]["A"] for m in models]])
    ax.bar(xs, vals, yerr=err, capsize=4, color="#4C72B0")
    for y, lab, col in [(0.0, "canal fixe", "#C44E52"),
                        (0.634, "plafond Bayes-provenance", "#DD8452"),
                        (1.0, "oracle", "#55A868")]:
        ax.axhline(y, ls="--", lw=1.1, color=col, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("A — arbitrage"); ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=8); ax.set_title("Arbitrage mesure contre ses bornes")
    fig.tight_layout(); fig.savefig(OUT / "fig_arbitration_v3.png", dpi=300)

    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {OUT/'results.json'} et fig_arbitration_v3.png")


if __name__ == "__main__":
    main()
