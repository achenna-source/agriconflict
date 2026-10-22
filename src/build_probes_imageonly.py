"""
AgriConflict, corpus "image seule" pour l'evaluation SAGA de bout en bout.

POURQUOI. Dans l'article, la couche d'arbitrage de SAGA recoit des decisions par
source EXACTES (oraculaires). C'est la reserve methodologique la plus previsible :
les chiffres isolent l'arbitrage mais ne mesurent pas un systeme deploye, puisque
l'erreur de perception du predicteur par source se propagerait.

Ce corpus la leve. Pour chaque cellule factorielle, il presente la MEME photographie
que la cellule correspondante, avec sa meme ligne de provenance, mais SANS le rapport
de terrain. La reponse du modele devient alors une vraie decision par source, produite
par le controleur lui-meme, et SAGA peut etre evalue de bout en bout.

CONSTRUCTION PAR DERIVATION, PAS PAR REGENERATION. Le generateur d'origine choisit le
fichier image en parcourant un ensemble, dont l'ordre d'iteration depend du hachage des
chaines. Regenerer le corpus risquerait donc d'assigner d'autres photographies. On lit
a la place le corpus publie et on en retire la ligne de rapport : les images, les
options, leur ordre et la position de l'abstention sont identiques par construction.

ANGLAIS SEULEMENT. Le prompt image-seule ne contient aucun rapport, donc la langue n'y
joue que sur la question et les libelles d'options. On s'en tient a l'anglais pour
diviser par deux le cout GPU, et l'evaluation de bout en bout de SAGA porte sur la
moitie anglaise. C'est declare dans l'article.
"""

import json
import shutil
from pathlib import Path

OUT = Path("/kaggle/working")
FACTORIAL = ("fresh_verified", "fresh_unverified", "stale_verified", "stale_unverified")
REPORT_PREFIX = ("Field report", "Rapport de terrain")


def main() -> None:
    src = next(Path("/kaggle/input").rglob("probes.jsonl"))
    root = src.parent
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l]
    print(f"corpus source : {src}  ({len(rows)} sondes)")

    shutil.copytree(root / "images", OUT / "images")
    n_img = len(list((OUT / "images").iterdir()))
    print(f"images copiees : {n_img}")

    n, skipped = 0, 0
    with (OUT / "probes.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            if r["cell"] not in FACTORIAL or r["lang"] != "en":
                continue
            lines = r["prompt"].split("\n")
            kept = [l for l in lines if not l.startswith(REPORT_PREFIX)]
            if len(kept) == len(lines):
                skipped += 1
                continue
            fh.write(json.dumps({
                "probe_id": r["probe_id"].replace(f"_{r['cell']}", f"_IO_{r['cell']}"),
                "uid": r["uid"], "lang": "en", "cell": f"IO_{r['cell']}",
                "source_cell": r["cell"],
                "image": r["image"],
                "prompt": "\n".join(kept),
                # verite du canal image : la classe que la photographie montre
                "gold_letter": r["letter_image"],
                "letter_image": r["letter_image"], "letter_text": r["letter_text"],
                "letter_abstain": r["letter_abstain"],
                "subset": r["subset"], "q_img": r["q_img"], "q_rep": r["q_rep"],
                "img_ok": r["img_ok"], "rep_ok": r["rep_ok"],
                "n_options": r["n_options"], "true_class": r["true_class"],
            }, ensure_ascii=False) + "\n")
            n += 1

    assert skipped == 0, f"{skipped} sondes sans ligne de rapport reconnue"
    (OUT / "probes_meta.json").write_text(json.dumps({
        "version": 3, "corpus": "image-only-derived", "n_records": n,
        "langs": ["en"], "cells": [f"IO_{c}" for c in FACTORIAL],
        "chance": 1 / 5, "derived_from": str(src),
        "note": ("image-only variant of the four factorial cells, derived from the "
                 "published corpus by removing the field-report line; images, options, "
                 "option order and abstention position identical by construction"),
    }, indent=2), encoding="utf-8")

    print(f"\n{n} sondes image-seule ecrites")
    ex = [json.loads(l) for l in (OUT / "probes.jsonl").read_text(encoding="utf-8").splitlines()[:1]][0]
    print("\n--- exemple ---")
    print(ex["prompt"])
    print(f"-> gold (classe montree) = {ex['gold_letter']}")


if __name__ == "__main__":
    main()
