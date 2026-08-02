"""
AgriConflict — replication en conditions de TERRAIN (PlantDoc).

Meme processus generatif, memes fiabilites declarees, memes cellules et memes
metriques que le corpus principal. Seule la source d'images change :
PlantDoc est photographie au champ, avec encombrement naturel, occlusion et
eclairage variable, la ou PlantVillage montre une feuille isolee sur fond uni.

L'argument du §5.1 de l'article — PlantVillage est le MEILLEUR cas pour le canal
visuel, donc un echec y est d'autant plus probant — reste une hypothese tant
qu'il n'est pas teste. C'est ce que fait ce corpus.

DEUX DIFFERENCES, declarees :

 1. 1 000 items au lieu de 2 000. PlantDoc ne contient que ~2 700 images
    d'entrainement sur 27 classes exploitables ; au-dela, les memes photographies
    se repeteraient trop souvent entre items. L'effectif effectif reste 1 000
    items, suffisant pour les intervalles groupes par item.

 2. Libelles de classe issus d'une table explicite (ci-dessous). La nomenclature
    PlantDoc est irreguliere (Apple_Scab_Leaf, grape_leaf_black_rot,
    Bell_pepper_leaf_spot) et une heuristique produirait des libelles faux,
    du type "Bell, pepper leaf spot".
"""

import hashlib
import json
import os
import random
from pathlib import Path

from PIL import Image

SEED = 20260801
N_ITEMS = 1000
N_CLASS_OPTIONS = 4
LANGS = ["en", "fr"]
OUT = Path("/kaggle/working")

P_IMG_RIGHT = {"fresh": 0.90, "stale": 0.30}
P_REP_RIGHT = {"verified": 0.85, "unverified": 0.35}

CELLS = ["V0_gate", "T0_gate", "fresh_verified", "fresh_unverified",
         "stale_verified", "stale_unverified"]
SPEC = {
    "V0_gate":          ("fresh", None),
    "T0_gate":          ("none",  "verified"),
    "fresh_verified":   ("fresh", "verified"),
    "fresh_unverified": ("fresh", "unverified"),
    "stale_verified":   ("stale", "verified"),
    "stale_unverified": ("stale", "unverified"),
}

# Table explicite : repertoire PlantDoc -> libelle affiche.
LABELS = {
    "Apple_Scab_Leaf":                     "Apple, apple scab",
    "Apple_leaf":                          "Apple, healthy",
    "Apple_rust_leaf":                     "Apple, cedar apple rust",
    "Bell_pepper_leaf":                    "Bell pepper, healthy",
    "Bell_pepper_leaf_spot":               "Bell pepper, bacterial spot",
    "Blueberry_leaf":                      "Blueberry, healthy",
    "Cherry_leaf":                         "Cherry, healthy",
    "Corn_Gray_leaf_spot":                 "Corn, gray leaf spot",
    "Corn_leaf_blight":                    "Corn, northern leaf blight",
    "Corn_rust_leaf":                      "Corn, common rust",
    "Peach_leaf":                          "Peach, healthy",
    "Potato_leaf_early_blight":            "Potato, early blight",
    "Potato_leaf_late_blight":             "Potato, late blight",
    "Raspberry_leaf":                      "Raspberry, healthy",
    "Soyabean_leaf":                       "Soybean, healthy",
    "Squash_Powdery_mildew_leaf":          "Squash, powdery mildew",
    "Strawberry_leaf":                     "Strawberry, healthy",
    "Tomato_Early_blight_leaf":            "Tomato, early blight",
    "Tomato_Septoria_leaf_spot":           "Tomato, Septoria leaf spot",
    "Tomato_leaf":                         "Tomato, healthy",
    "Tomato_leaf_bacterial_spot":          "Tomato, bacterial spot",
    "Tomato_leaf_late_blight":             "Tomato, late blight",
    "Tomato_leaf_mosaic_virus":            "Tomato, mosaic virus",
    "Tomato_leaf_yellow_virus":            "Tomato, yellow leaf curl virus",
    "Tomato_mold_leaf":                    "Tomato, leaf mold",
    "grape_leaf":                          "Grape, healthy",
    "grape_leaf_black_rot":                "Grape, black rot",
}

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

rng = random.Random(SEED)


def find_root() -> Path:
    if os.environ.get("PD_ROOT"):
        return Path(os.environ["PD_ROOT"])
    for cand in Path("/kaggle/input").rglob("train"):
        if cand.is_dir() and sum(1 for d in cand.iterdir() if d.is_dir()) > 20:
            return cand
    raise FileNotFoundError("PlantDoc train/ introuvable.")


def main() -> None:
    root = find_root()
    IMG = {".jpg", ".jpeg", ".png"}
    by_class = {}
    for d in root.iterdir():
        if not d.is_dir() or d.name not in LABELS:
            continue
        files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMG)
        if len(files) >= 5:
            by_class[d.name] = files
    classes = sorted(by_class)
    print(f"PlantDoc : {len(classes)} classes, "
          f"{sum(len(v) for v in by_class.values())} images sous {root}")
    if len(classes) < N_CLASS_OPTIONS:
        raise RuntimeError("pas assez de classes")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (448, 448), (128, 128, 128)).save(
        OUT / "images/_blank.jpg", quality=92)

    n = 0
    counts = {k: 0 for k in ("S_both", "S_img", "S_txt", "S_none")}
    fh = (OUT / "probes.jsonl").open("w", encoding="utf-8")

    for i in range(N_ITEMS):
        Y = rng.choice(classes)
        opts = [Y] + rng.sample([c for c in classes if c != Y], N_CLASS_OPTIONS - 1)
        wrong = [c for c in opts if c != Y]

        draw = {}
        for cell, (qi, qr) in SPEC.items():
            if cell == "V0_gate":
                y_img, y_rep = Y, None
            elif cell == "T0_gate":
                y_img, y_rep = None, Y
            else:
                y_img = Y if rng.random() < P_IMG_RIGHT[qi] else rng.choice(wrong)
                y_rep = Y if rng.random() < P_REP_RIGHT[qr] else rng.choice(wrong)
            draw[cell] = (y_img, y_rep)

        rng.shuffle(opts)
        abst_pos = rng.randrange(N_CLASS_OPTIONS + 1)

        def letter(cls):
            if cls is None or cls not in opts:
                return None
            p = opts.index(cls)
            return "ABCDE"[p + 1 if p >= abst_pos else p]

        uid = hashlib.sha1(f"pd/{Y}/{i}".encode()).hexdigest()[:12]
        img_files = {}
        for cls in {c for c, _ in draw.values() if c}:
            fn = rng.choice(by_class[cls])
            rp = f"images/{uid}_{abs(hash(cls)) % 10**6}.jpg"
            with Image.open(root / cls / fn) as im:
                im = im.convert("RGB")
                im.thumbnail((448, 448), Image.LANCZOS)
                im.save(OUT / rp, quality=92)
            img_files[cls] = rp

        labels = [LABELS[c] for c in opts]
        for lang in LANGS:
            full = labels[:abst_pos] + [T[lang]["abstain"]] + labels[abst_pos:]
            block = "\n".join(f"{'ABCDE'[j]}. {o}" for j, o in enumerate(full))
            L_abst = "ABCDE"[abst_pos]

            for cell in CELLS:
                qi, qr = SPEC[cell]
                y_img, y_rep = draw[cell]
                lines = [T[lang][f"img_{qi}"]]
                if qr:
                    lines.append(T[lang][f"rep_{qr}"].format(claim=LABELS[y_rep]))
                prompt = "\n".join(lines) + "\n\n" + T[lang]["question"] + "\n" + block

                img_ok, rep_ok = y_img == Y, y_rep == Y
                subset = ("S_both" if img_ok and rep_ok else "S_img" if img_ok
                          else "S_txt" if rep_ok else "S_none")
                gold = letter(Y) if (img_ok or rep_ok) else L_abst
                if gold is None:
                    continue
                counts[subset] += 1

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
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_ITEMS}")
    fh.close()

    (OUT / "probes_meta.json").write_text(json.dumps({
        "version": 3, "corpus": "plantdoc-field", "seed": SEED, "n_items": N_ITEMS,
        "n_options": N_CLASS_OPTIONS + 1, "chance": 1 / (N_CLASS_OPTIONS + 1),
        "langs": LANGS, "cells": CELLS, "n_records": n, "n_classes": len(classes),
        "p_img_right": P_IMG_RIGHT, "p_rep_right": P_REP_RIGHT,
        "subset_counts": counts,
        "note": ("field-condition replication of the main corpus; identical generative "
                 "process and metrics, PlantDoc imagery instead of PlantVillage"),
        "source": str(root),
    }, indent=2), encoding="utf-8")

    print(f"\n{n} sondes | {N_ITEMS} items | hasard {1/(N_CLASS_OPTIONS+1):.1%}")
    print("sous-ensembles :", counts)


if __name__ == "__main__":
    main()
