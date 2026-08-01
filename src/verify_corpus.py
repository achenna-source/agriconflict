"""Verification du corpus v3. Cible les deux proprietes qui ont motive la v3."""
import json
from collections import Counter, defaultdict
from pathlib import Path

meta = json.loads(next(Path("/kaggle/input").rglob("probes_meta.json")).read_text())
jl = next(Path("/kaggle/input").rglob("probes.jsonl"))
rows = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l]
root = jl.parent

print("=== META ==="); print(json.dumps(meta, indent=2)[:900])
print(f"\n=== VOLUME ===\nlignes {len(rows)} | items {len({r['uid'] for r in rows})}")
print("cellules :", dict(Counter(r["cell"] for r in rows)))
print("langues  :", dict(Counter(r["lang"] for r in rows)))
print("gold     :", dict(Counter(r["gold_letter"] for r in rows)))
imgs = {r["image"] for r in rows}
print(f"images {len(imgs)} | manquantes", sum(1 for i in imgs if not (root / i).exists()))

# --- 1. rho n'est plus degeneree : fiabilite empirique par provenance
print("\n=== 1. FIABILITE EMPIRIQUE PAR PROVENANCE (doit etre dans ]0,1[) ===")
acc = defaultdict(list)
for r in rows:
    if r["q_img"] != "none":
        acc[("image", r["q_img"])].append(int(r["img_ok"]))
    if r["q_rep"]:
        acc[("rapport", r["q_rep"])].append(int(r["rep_ok"]))
for k, v in sorted(acc.items()):
    m = sum(v) / len(v)
    flag = "  << DEGENERE" if m in (0.0, 1.0) else ""
    print(f"  {k[0]:<8} {k[1]:<11} {m:.3f}  (n={len(v)}){flag}")

# --- 2. aucune cellule ne determine la verite
print("\n=== 2. AUCUNE CELLULE NE DETERMINE LA VERITE ===")
for cell in meta["cells"]:
    sub = [r for r in rows if r["cell"] == cell]
    c = Counter(r["subset"] for r in sub)
    n = len(sub) or 1
    print(f"  {cell:<18} " + "  ".join(f"{k}={c[k]/n:.2f}"
          for k in ("S_both", "S_img", "S_txt", "S_none")))

# --- 3. bornes : politiques de reference simulees sur les sous-ensembles
print("\n=== 3. BORNES  A = acc(S_img) + acc(S_txt) - 1 ===")
def sim(pol):
    a = defaultdict(list)
    for r in rows:
        if r["subset"] in ("S_img", "S_txt", "S_none") and r["cell"] not in ("V0_gate", "T0_gate"):
            p = pol(r)
            a[r["subset"]].append(int(p == r["gold_letter"]))
    return {k: sum(v) / len(v) for k, v in a.items() if v}

pi = meta["p_img_right"]; pr = meta["p_rep_right"]
for name, pol in [
    ("suit toujours l'image", lambda r: r["letter_image"]),
    ("suit toujours le rapport", lambda r: r["letter_text"]),
    ("s'abstient toujours", lambda r: r["letter_abstain"]),
    ("Bayes sur provenance seule",
     lambda r: r["letter_image"] if pi[r["q_img"]] >= pr[r["q_rep"]] else r["letter_text"]),
    ("oracle (connait la source juste)", lambda r: r["gold_letter"]),
]:
    a = sim(pol)
    A = a.get("S_img", 0) + a.get("S_txt", 0) - 1
    print(f"  {name:<34} S_img={a.get('S_img',0):.3f} S_txt={a.get('S_txt',0):.3f} "
          f"S_none={a.get('S_none',0):.3f} -> A={A:+.3f}")
print("  attendu : canal fixe -> A=0 ; oracle -> A=+1 ; Bayes-provenance -> entre les deux")

print("\n=== EXEMPLE stale_verified, sous-ensemble S_txt ===")
ex = next(r for r in rows if r["cell"] == "stale_verified"
          and r["subset"] == "S_txt" and r["lang"] == "en")
print(ex["prompt"])
print(f"-> gold={ex['gold_letter']} | image={ex['letter_image']} rapport={ex['letter_text']} "
      f"abst={ex['letter_abstain']}")
