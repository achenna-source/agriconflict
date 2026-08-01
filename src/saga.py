"""
AgriConflict v3 — Notebook 4 : SAGA sur le meme banc que les modeles audites.

--------------------------------------------------------------------------------
CE QUI EST MESURE, ET CE QUI NE L'EST PAS  (a lire avant tout chiffre)

SAGA se compose de quatre briques. Trois sont evaluees ici, une ne l'est pas :

  evalue    rho, la fiabilite lue sur la provenance d'acquisition
  evalue    l'arbitrage pondere entre decisions par source
  evalue    l'abstention conforme stratifiee (Mondrian)
  evalue    la provenance R, notee contre la source reellement correcte
  NON evalue  la boucle de recherche de preuve : un banc statique n'offre aucun
              outil a appeler. Nous ne la simulons pas et n'en revendiquons rien.

--------------------------------------------------------------------------------
DECISIONS PAR SOURCE : POURQUOI ELLES SONT ORACULAIRES ICI

En v3 chaque cellule peut montrer une photographie DIFFERENTE (l'image perimee
represente l'etat d'avant traitement). La reponse du modele en V0_gate ne porte
donc que sur l'image de cette cellule-la et ne peut pas servir de decision-image
pour les autres. Il aurait fallu une passe image-seule par cellule ; elle n'existe
pas dans ce corpus.

Nous evaluons donc la COUCHE D'ARBITRAGE ISOLEMENT, en lui fournissant les
decisions par source exactes (letter_image, letter_text). Consequences, a enoncer
telles quelles dans l'article :

  * les chiffres SAGA ci-dessous ne sont PAS une performance de bout en bout ;
    une erreur de perception du predicteur par source se propagerait ;
  * ils sont en revanche DIRECTEMENT comparables a la borne Bayes-provenance
    (+0.634), calculee elle aussi sur des sources oraculaires ;
  * la comparaison "controleur implicite vs SAGA" oppose perception+agregation
    d'un cote a agregation seule de l'autre. Elle mesure ce que l'externalisation
    de l'agregation rend possible, pas ce qu'un systeme deploye obtiendrait.

C'est la lecture honnete, et elle reste informative : si l'arbitrage echoue meme
avec des sources parfaites, aucun progres en perception ne le sauvera.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ALPHA = 0.10
SEED = 20260730
GATES = ("V0_gate", "T0_gate")

FILES = {
    "qwen25vl-3b":    "_q3b/results_qwen25vl-3b.jsonl",
    "qwen25vl-7b":    "_log_qwen7b/results_qwen25vl-7b.jsonl",
    "internvl3-8b":   "_log_internvl/results_internvl3-8b.jsonl",
    "qwen25-omni-7b": "_log_omni/results_qwen25-omni-7b.jsonl",
}


def load(model: str) -> list[dict]:
    p = HERE / FILES[model]
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or "subset" not in r or r["cell"] in GATES:
            continue
        out.append(r)
    return out


def fit_rho(cal: list[dict]) -> dict[tuple[str, str], float]:
    """rho : provenance -> fiabilite. Ajustee sur la calibration seule ; a
    l'inference elle ne consulte QUE la provenance, jamais une etiquette."""
    a = defaultdict(list)
    for r in cal:
        a[("img", r["q_img"])].append(int(r["img_ok"]))
        a[("rep", r["q_rep"])].append(int(r["rep_ok"]))
    return {k: sum(v) / len(v) for k, v in a.items() if v}


def arbitrate(r: dict, rho: dict, uniform: bool):
    w_i = 1.0 if uniform else rho.get(("img", r["q_img"]), 0.5)
    w_r = 1.0 if uniform else rho.get(("rep", r["q_rep"]), 0.5)
    score, support = defaultdict(float), defaultdict(list)
    score[r["letter_image"]] += w_i; support[r["letter_image"]].append("image")
    score[r["letter_text"]] += w_r; support[r["letter_text"]].append("report")
    best = max(score, key=score.get)
    return best, score[best] / (sum(score.values()) or 1.0), support[best]


def thresholds_at(cal: list[dict], rho: dict, uniform: bool,
                  alpha: float) -> dict[str, float]:
    """Seuils Mondrian pour un risque cible donne."""
    th = {}
    for cell in {r["cell"] for r in cal}:
        sub = [(arbitrate(r, rho, uniform)[1],
                int(arbitrate(r, rho, uniform)[0] == r["gold"]))
               for r in cal if r["cell"] == cell]
        best = float("inf")
        for t in sorted({c for c, _ in sub}, reverse=True):
            sel = [ok for c, ok in sub if c >= t]
            if sel and (len(sel) - sum(sel)) / len(sel) <= alpha:
                best = t
        th[cell] = best
    return th


def thresholds(cal: list[dict], rho: dict, uniform: bool) -> dict[str, float]:
    """Mondrian : un seuil par cellule de provenance, pour borner l'erreur
    DANS CHAQUE strate et non en moyenne. Si aucun seuil n'atteint ALPHA,
    la strate s'abstient entierement — c'est le comportement correct."""
    th = {}
    for cell in {r["cell"] for r in cal}:
        sub = [(arbitrate(r, rho, uniform)[1], int(arbitrate(r, rho, uniform)[0] == r["gold"]))
               for r in cal if r["cell"] == cell]
        best = float("inf")
        for t in sorted({c for c, _ in sub}, reverse=True):
            sel = [ok for c, ok in sub if c >= t]
            if sel and (len(sel) - sum(sel)) / len(sel) <= ALPHA:
                best = t
        th[cell] = best
    return th


def evaluate(test: list[dict], policy) -> dict:
    per = defaultdict(lambda: {"n": 0, "ok": 0, "abst": 0, "pn": 0, "pok": 0})
    for r in test:
        dec, R = policy(r)
        s = per[r["subset"]]
        s["n"] += 1
        s["ok"] += int(dec == r["gold"])
        if dec == r["letter_abstain"]:
            s["abst"] += 1
        elif r["subset"] in ("S_img", "S_txt"):
            s["pn"] += 1
            s["pok"] += int(("image" if r["subset"] == "S_img" else "report") in R)
    out = {k: {"acc": v["ok"] / v["n"], "abst": v["abst"] / v["n"],
               "prov": v["pok"] / v["pn"] if v["pn"] else float("nan")}
           for k, v in per.items()}
    out["A"] = out.get("S_img", {}).get("acc", 0) + out.get("S_txt", {}).get("acc", 0) - 1
    return out


def main() -> None:
    rng = random.Random(SEED)
    print(f"SAGA v3 — alpha={ALPHA} | bornes : canal fixe 0.000, "
          f"Bayes-provenance +0.634, oracle +1.000\n")
    summary = {}

    for model in FILES:
        data = load(model)
        if not data:
            print(f"{model}: resultats v3 absents, ignore\n"); continue

        uids = sorted({r["uid"] for r in data})
        rng.shuffle(uids)
        cal_ids = set(uids[: len(uids) // 2])
        cal = [r for r in data if r["uid"] in cal_ids]
        test = [r for r in data if r["uid"] not in cal_ids]

        rho = fit_rho(cal)
        th, th_u = thresholds(cal, rho, False), thresholds(cal, rho, True)

        def mk(uniform, conformal):
            t = th_u if uniform else th
            def pol(r):
                dec, conf, R = arbitrate(r, rho, uniform)
                if conformal and conf < t.get(r["cell"], float("inf")):
                    return r["letter_abstain"], R
                return dec, R
            return pol

        variants = {
            "controleur implicite":        lambda r: (r["pred"], []),
            "SAGA - rho uniforme (abl.)":  mk(True, True),
            "SAGA - sans conformal (abl.)": mk(False, False),
            "SAGA (complet)":              mk(False, True),
        }

        print("=" * 92)
        print(f"{model}   rho ajustee sur {len(cal_ids)} items, testee sur "
              f"{len(uids)-len(cal_ids)}")
        print("  rho :", {f"{k[0]}/{k[1]}": round(v, 3) for k, v in sorted(rho.items())})
        print(f"  {'methode':<30}{'S_img':>8}{'S_txt':>8}{'S_none':>8}"
              f"{'A':>8}{'%plaf':>8}{'prov':>8}")
        summary[model] = {"rho": {f"{k[0]}/{k[1]}": v for k, v in rho.items()}}
        for name, pol in variants.items():
            e = evaluate(test, pol)
            prov = e.get("S_img", {}).get("prov", float("nan"))
            print(f"  {name:<30}{e.get('S_img',{}).get('acc',0):>8.3f}"
                  f"{e.get('S_txt',{}).get('acc',0):>8.3f}"
                  f"{e.get('S_none',{}).get('abst',0):>8.3f}"
                  f"{e['A']:>8.3f}{e['A']/0.634*100:>7.1f}%{prov:>8.3f}")
            summary[model][name] = e

        # ------------------------------------------------ courbe risque-couverture
        # Un alpha unique est trompeur : si le risque cible est inatteignable a
        # partir de la seule provenance, la strate s'abstient entierement et SAGA
        # parait s'effondrer. Le compromis doit etre montre en entier.
        print(f"\n  Risque-couverture (alpha balaye)")
        print(f"    {'alpha':>7}{'couverture':>12}{'erreur reelle':>15}{'A':>9}{'%plaf':>8}")
        rc = {}
        for a in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.00):
            th_a = thresholds_at(cal, rho, False, a)
            def pol_a(r, t=th_a):
                dec, conf, R = arbitrate(r, rho, False)
                return ((r["letter_abstain"], R)
                        if conf < t.get(r["cell"], float("inf")) else (dec, R))
            comm = [r for r in test if pol_a(r)[0] != r["letter_abstain"]]
            cov = len(comm) / len(test)
            err = (sum(1 for r in comm if pol_a(r)[0] != r["gold"]) / len(comm)
                   if comm else float("nan"))
            e = evaluate(test, pol_a)
            print(f"    {a:>7.2f}{cov:>12.3f}{err:>15.3f}{e['A']:>9.3f}"
                  f"{e['A']/0.634*100:>7.1f}%")
            rc[a] = {"coverage": cov, "risk": err, "A": e["A"]}
        summary[model]["risk_coverage"] = rc
        print()

    (HERE / "_results_saga.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-> _results_saga.json")
    print("\nRAPPEL : sources oraculaires. Ces chiffres isolent la couche d'arbitrage ;")
    print("ils ne sont pas une performance de bout en bout (cf. en-tete du fichier).")


if __name__ == "__main__":
    main()
