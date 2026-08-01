"""
AgriConflict v3 — Notebook 1 (CPU) : construction du jeu de sondes.

--------------------------------------------------------------------------------
POURQUOI v3 : casser la circularite de rho

En v2, chaque combinaison de provenance determinait la verite. Un rapport non
verifie etait TOUJOURS faux, si bien que la fonction de fiabilite apprenait
rho(non verifie) = 0.000 exactement : elle memorisait le plan experimental au lieu
d'estimer une fiabilite. L'arbitrage de SAGA devenait alors trivial.

v3 rend la provenance INFORMATIVE MAIS NON DETERMINANTE. On declare un modele
generatif explicite, et la verite est echantillonnee independamment de la
provenance :

    P(image juste | fraiche)        = 0.90
    P(image juste | perimee)        = 0.30
    P(rapport juste | verifie)      = 0.85
    P(rapport juste | non verifie)  = 0.35

Aucune de ces valeurs n'est 0 ni 1 : rho doit estimer des probabilites reelles,
et aucune cellule ne se reduit a une regle. Le plafond de tout arbitre
conditionne a la provenance devient le taux de Bayes implique par ces quatre
nombres, calcule et rapporte comme borne superieure.

--------------------------------------------------------------------------------
MODELE GENERATIF

  1. Y        = etat COURANT de la parcelle (la verite a predire)
  2. Y_img    = classe reellement montree par la photographie
                 fraiche : Y avec p=0.90, sinon une autre classe
                 perimee : Y avec p=0.30, sinon l'etat d'avant traitement
  3. Y_rep    = classe annoncee par le rapport de terrain
                 verifie      : Y avec p=0.85
                 non verifie  : Y avec p=0.35
  4. VERITE   = Y si au moins une source le soutient (conflit RESOLUBLE)
                ABSTENTION sinon (conflit IRREDUCTIBLE, §3.3)

Les conflits irreductibles ne sont donc plus une cellule construite : ils
EMERGENT du processus, a une frequence qui decoule des quatre parametres.

--------------------------------------------------------------------------------
CELLULES

  V0_gate            image fraiche juste, aucun rapport      -> Y     (porte vision)
  T0_gate            aucune image, rapport verifie juste     -> Y     (porte texte)
  fresh_verified     image fraiche  x rapport verifie        -> stochastique
  fresh_unverified   image fraiche  x rapport non verifie    -> stochastique
  stale_verified     image perimee  x rapport verifie        -> stochastique
  stale_unverified   image perimee  x rapport non verifie    -> stochastique

Chaque sonde enregistre son SOUS-ENSEMBLE, defini par la correction reelle des
sources et non par la provenance :

  S_both  les deux soutiennent Y      S_img  l'image seule soutient Y
  S_txt   le rapport seul soutient Y  S_none aucune  -> abstention correcte

L'indice d'arbitrage se lit sur S_img et S_txt :
    A = acc(S_img) + acc(S_txt) - 1
canal fixe -> 0, arbitre parfait -> +1. Mais les sous-ensembles sont maintenant
definis par la verite observee, non par le plan : la provenance ne fait que les
predire imparfaitement.
"""

import hashlib
import json
import os
import random
from pathlib import Path

from PIL import Image

# --------------------------------------------------------------------------
SEED = 20260730
N_ITEMS = 2000
N_CLASS_OPTIONS = 4          # + abstention = 5 -> hasard = 20 %
LANGS = ["en", "fr"]
OUT = Path("/kaggle/working")

# Fiabilites generatives declarees. Strictement dans ]0,1[ par construction.
P_IMG_RIGHT = {"fresh": 0.90, "stale": 0.30}
P_REP_RIGHT = {"verified": 0.85, "unverified": 0.35}

CELLS = ["V0_gate", "T0_gate",
         "fresh_verified", "fresh_unverified",
         "stale_verified", "stale_unverified"]

rng = random.Random(SEED)


def find_plantvillage_root() -> Path:
    if os.environ.get("PV_ROOT"):
        return Path(os.environ["PV_ROOT"])
    for cand in Path("/kaggle/input").rglob("color"):
        if cand.is_dir() and len(list(cand.iterdir())) > 20:
            return cand
    raise FileNotFoundError("PlantVillage introuvable.")


def prettify(c: str) -> str:
    crop, _, dis = c.partition("___")
    crop, dis = crop.replace("_", " ").strip(), dis.replace("_", " ").replace("  ", " ").strip()
    return f"{crop}, healthy" if not dis or dis.lower() == "healthy" else f"{crop}, {dis}"


T = {
    "en": {
        "img_fresh": "Photograph of the parcel, taken today.",
        "img_stale": ("Photograph of the parcel, taken three weeks ago, "
                      "before the most recent treatment."),
        "img_none": "No photograph of the parcel is available.",
        "rep_verified": ("Field report, laboratory-confirmed, sampled today, validated by "
                         "the regional plant-health service: the parcel shows {claim}."),
        "rep_unverified": ("Field report, unverified, submitted by the grower twelve days "
                           "ago, not confirmed by an agronomist: the parcel shows {claim}."),
        "question": ("Based on the evidence above, what is the CURRENT state of this "
                     "parcel? Answer with a single letter."),
        "abstain": "Cannot be determined from the available evidence",
    },
    "fr": {
        "img_fresh": "Photographie de la parcelle, prise aujourd'hui.",
        "img_stale": ("Photographie de la parcelle, prise il y a trois semaines, "
                      "avant le dernier traitement."),
        "img_none": "Aucune photographie de la parcelle n'est disponible.",
        "rep_verified": ("Rapport de terrain, confirme par analyse en laboratoire, preleve "
                         "aujourd'hui, valide par le service regional de protection des "
                         "vegetaux : la parcelle presente {claim}."),
        "rep_unverified": ("Rapport de terrain, non verifie, transmis par l'exploitant il y "
                           "a douze jours, sans confirmation d'un agronome : la parcelle "
                           "presente {claim}."),
        "question": ("D'apres les elements ci-dessus, quel est l'etat ACTUEL de cette "
                     "parcelle ? Repondez par une seule lettre."),
        "abstain": "Ne peut pas etre determine a partir des elements disponibles",
    },
}

SPEC = {                       # cellule -> (provenance image, provenance rapport)
    "V0_gate":          ("fresh", None),
    "T0_gate":          ("none",  "verified"),
    "fresh_verified":   ("fresh", "verified"),
    "fresh_unverified": ("fresh", "unverified"),
    "stale_verified":   ("stale", "verified"),
    "stale_unverified": ("stale", "unverified"),
}


def main() -> None:
    root = find_plantvillage_root()
    by_class = {d.name: sorted(p.name for p in d.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                for d in root.iterdir() if d.is_dir()}
    by_class = {c: f for c, f in by_class.items() if len(f) >= 5}
    classes = sorted(by_class)
    print(f"PlantVillage : {len(classes)} classes")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (448, 448), (128, 128, 128)).save(OUT / "images/_blank.jpg", quality=92)

    records, n = [], 0
    fh = (OUT / "probes.jsonl").open("w", encoding="utf-8")
    subset_count = {k: 0 for k in ("S_both", "S_img", "S_txt", "S_none")}

    for i in range(N_ITEMS):
        Y = rng.choice(classes)                      # etat courant = verite

        # Le POOL d'options est fixe AVANT les tirages, et toute classe erronee
        # est tiree dans ce pool. Garantit que la classe montree par l'image et
        # celle annoncee par le rapport figurent TOUJOURS parmi les options :
        # sans cela, "suivre l'image" n'est pas exprimable et l'attribution de
        # source est indefinie (defaut detecte a la verification de la v3).
        opts = [Y] + rng.sample([c for c in classes if c != Y], N_CLASS_OPTIONS - 1)
        wrong = [c for c in opts if c != Y]

        # Tirages independants par cellule : la provenance ne determine rien.
        draw = {}
        for cell, (qi, qr) in SPEC.items():
            if cell == "V0_gate":
                y_img, y_rep = Y, None               # porte : image juste
            elif cell == "T0_gate":
                y_img, y_rep = None, Y               # porte : rapport juste
            else:
                y_img = Y if rng.random() < P_IMG_RIGHT[qi] else rng.choice(wrong)
                y_rep = Y if rng.random() < P_REP_RIGHT[qr] else rng.choice(wrong)
            draw[cell] = (y_img, y_rep)

        rng.shuffle(opts)

        abst_pos = rng.randrange(N_CLASS_OPTIONS + 1)

        def letter(cls: str | None) -> str | None:
            if cls is None or cls not in opts:
                return None
            p = opts.index(cls)
            return "ABCDE"[p + 1 if p >= abst_pos else p]

        uid = hashlib.sha1(f"{Y}/{i}".encode()).hexdigest()[:12]
        rel = f"images/{uid}.jpg"
        # La photographie montre Y_img de la cellule stale_* si different ; on
        # stocke une image par classe citee pour rester coherent.
        img_files = {}
        for cls in {c for c in (y for y, _ in draw.values()) if c}:
            fn = rng.choice(by_class[cls])
            rp = f"images/{uid}_{abs(hash(cls)) % 10**6}.jpg"
            with Image.open(root / cls / fn) as im:
                im = im.convert("RGB"); im.thumbnail((448, 448), Image.LANCZOS)
                im.save(OUT / rp, quality=92)
            img_files[cls] = rp

        labels = [prettify(c) for c in opts]
        for lang in LANGS:
            full = labels[:abst_pos] + [T[lang]["abstain"]] + labels[abst_pos:]
            block = "\n".join(f"{'ABCDE'[j]}. {o}" for j, o in enumerate(full))
            L_abst = "ABCDE"[abst_pos]

            for cell in CELLS:
                qi, qr = SPEC[cell]
                y_img, y_rep = draw[cell]

                lines = [T[lang][f"img_{qi}"]]
                if qr:
                    lines.append(T[lang][f"rep_{qr}"].format(claim=prettify(y_rep)))
                prompt = "\n".join(lines) + "\n\n" + T[lang]["question"] + "\n" + block

                img_ok = y_img == Y
                rep_ok = y_rep == Y
                subset = ("S_both" if img_ok and rep_ok else
                          "S_img" if img_ok else
                          "S_txt" if rep_ok else "S_none")
                gold = letter(Y) if (img_ok or rep_ok) else L_abst
                if gold is None:
                    continue
                subset_count[subset] += 1

                fh.write(json.dumps({
                    "probe_id": f"{uid}_{lang}_{cell}", "uid": uid,
                    "lang": lang, "cell": cell, "subset": subset,
                    "image": img_files[y_img] if y_img else "images/_blank.jpg",
                    "prompt": prompt, "gold_letter": gold,
                    "letter_image": letter(y_img), "letter_text": letter(y_rep),
                    "letter_abstain": L_abst, "letter_truth": letter(Y),
                    "q_img": qi, "q_rep": qr,
                    "img_ok": img_ok, "rep_ok": rep_ok,
                    "n_options": N_CLASS_OPTIONS + 1, "true_class": Y,
                }, ensure_ascii=False) + "\n")
                n += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N_ITEMS}")
    fh.close()

    (OUT / "probes_meta.json").write_text(json.dumps({
        "version": 3, "seed": SEED, "n_items": N_ITEMS,
        "n_options": N_CLASS_OPTIONS + 1, "chance": 1 / (N_CLASS_OPTIONS + 1),
        "langs": LANGS, "cells": CELLS, "n_records": n, "n_classes": len(classes),
        "p_img_right": P_IMG_RIGHT, "p_rep_right": P_REP_RIGHT,
        "subset_counts": subset_count,
        "note": ("v3: provenance informative but NOT determinative. Reliabilities "
                 "strictly in (0,1) so rho must estimate probabilities rather than "
                 "memorise the design. Irreducible conflicts emerge from the process."),
        "arbitration_index": "A = acc(S_img) + acc(S_txt) - 1",
        "source": str(root),
    }, indent=2), encoding="utf-8")

    print(f"\n{n} sondes | hasard {1/(N_CLASS_OPTIONS+1):.1%}")
    print("sous-ensembles :", subset_count)


if __name__ == "__main__":
    main()
