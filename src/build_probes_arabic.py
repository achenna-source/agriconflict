# -*- coding: utf-8 -*-
"""
AgriConflict, bras arabe et darija algerienne.

DERIVE du corpus publie, comme le corpus image-seule : on relit les sondes et on
remplace uniquement le CADRE textuel (lignes de provenance, question, libelle
d'abstention). Les photographies, les options de classe, leur ordre et la position
de l'abstention restent identiques par construction, donc les resultats se
comparent directement a l'anglais et au francais.

DEUX CHOIX DECLARES, a enoncer tels quels dans l'article.

1. Les libelles de CLASSE restent en anglais. Traduire trente-huit couples
   culture-maladie en arabe agronomique exige un glossaire valide que nous n'avons
   pas ; une traduction approximative corromprait la tache plutot que la traduire.
   Ce choix est aussi defendable sur le terrain : dans la vulgarisation agricole
   maghrebine les noms de maladies circulent couramment en francais ou en latin
   au milieu d'un texte arabe.

2. La darija est ecrite en caracteres arabes. Elle s'ecrit aussi en caracteres
   latins, et le registre retenu ici est celui de la vulgarisation orale transcrite.

VALIDATION OBLIGATOIRE AVANT PUBLICATION. Les gabarits ci-dessous n'ont pas ete
rediges par un locuteur natif. Tant qu'un co-auteur arabophone ne les a pas valides
et que cette validation n'est pas decrite dans l'article, ce bras ne doit pas etre
rapporte : un relecteur arabophone le demontera, et il aura raison.
"""

import json
from pathlib import Path

OUT = Path("/kaggle/working")
ITEM_LIMIT = 1000          # moitie du corpus, pour tenir le budget GPU
LANGS = ("ar", "dz")

FRAME = {
    "ar": {
        "img_fresh": "صورة فوتوغرافية للقطعة، مأخوذة اليوم.",
        "img_stale": "صورة فوتوغرافية للقطعة، مأخوذة منذ ثلاثة أسابيع، قبل آخر معالجة.",
        "img_none": "لا تتوفر أي صورة فوتوغرافية للقطعة.",
        "rep_verified": ("تقرير ميداني، مؤكد بتحليل مخبري، أخذت العينة اليوم، ومصادق عليه "
                         "من المصلحة الجهوية لحماية النباتات: القطعة تظهر {claim}."),
        "rep_unverified": ("تقرير ميداني، غير مؤكد، أرسله المستثمر منذ اثني عشر يوما، دون "
                           "تأكيد من مهندس زراعي: القطعة تظهر {claim}."),
        "question": ("بناء على المعطيات أعلاه، ما هي الحالة الحالية لهذه القطعة؟ "
                     "أجب بحرف واحد."),
        "abstain": "لا يمكن تحديدها من المعطيات المتوفرة",
    },
    "dz": {
        "img_fresh": "تصويرة للقطعة، مدية اليوم.",
        "img_stale": "تصويرة للقطعة، مدية هاذي تلاثة سيمانات، قبل آخر معالجة.",
        "img_none": "ما كاينة حتى تصويرة للقطعة.",
        "rep_verified": ("تقرير من الميدان، مأكد بتحليل في المخبر، العينة مدية اليوم، "
                         "ومصادق عليه من مصلحة حماية النباتات: القطعة فيها {claim}."),
        "rep_unverified": ("تقرير من الميدان، ماشي مأكد، بعثو الفلاح هاذي طناش يوم، بلا "
                           "ما يشوفو مهندس زراعي: القطعة فيها {claim}."),
        "question": ("على حساب المعلومات اللي فوق، واش هي الحالة تاع هاذ القطعة ضرك؟ "
                     "جاوب بحرف واحد."),
        "abstain": "ما يمكنش نعرفوها من المعلومات اللي كاينة",
    },
}

SPEC = {                       # cellule -> (provenance image, provenance rapport)
    "V0_gate": ("fresh", None), "T0_gate": ("none", "verified"),
    "fresh_verified": ("fresh", "verified"), "fresh_unverified": ("fresh", "unverified"),
    "stale_verified": ("stale", "verified"), "stale_unverified": ("stale", "unverified"),
}


def main() -> None:
    src = next(Path("/kaggle/input").rglob("probes.jsonl"))
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l]
    en = [r for r in rows if r["lang"] == "en"]
    uids = sorted({r["uid"] for r in en})[:ITEM_LIMIT]
    keep = set(uids)
    print(f"source {src} | {len(rows)} sondes | items retenus {len(keep)}")

    import shutil
    shutil.copytree(src.parent / "images", OUT / "images")
    print(f"images copiees : {len(list((OUT / 'images').iterdir()))}")

    n = 0
    with (OUT / "probes.jsonl").open("w", encoding="utf-8") as fh:
        for r in en:
            if r["uid"] not in keep:
                continue
            # recupere le bloc d'options tel quel, en traduisant la seule ligne
            # d'abstention ; les libelles de classe restent inchanges
            opt_lines = [l for l in r["prompt"].split("\n")
                         if len(l) > 2 and l[0] in "ABCDE" and l[1] == "."]
            assert len(opt_lines) == 5, f"{len(opt_lines)} options pour {r['probe_id']}"
            qi, qr = SPEC[r["cell"]]
            for lang in LANGS:
                F = FRAME[lang]
                block = []
                for l in opt_lines:
                    block.append(f"{l[0]}. {F['abstain']}"
                                 if l[0] == r["letter_abstain"] else l)
                head = [F[f"img_{qi}"]]
                if qr:
                    # la classe annoncee par le rapport est TOUJOURS letter_text,
                    # que le rapport ait raison ou tort ; on relit son libelle
                    # depuis le bloc d'options a cette lettre
                    lab = next(l[3:] for l in opt_lines if l[0] == r["letter_text"])
                    head.append(F[f"rep_{qr}"].format(claim=lab))
                prompt = "\n".join(head) + "\n\n" + F["question"] + "\n" + "\n".join(block)
                fh.write(json.dumps({**r,
                    "probe_id": r["probe_id"].replace("_en_", f"_{lang}_"),
                    "lang": lang, "prompt": prompt,
                }, ensure_ascii=False) + "\n")
                n += 1

    (OUT / "probes_meta.json").write_text(json.dumps({
        "version": 3, "corpus": "arabic-darija-derived", "n_records": n,
        "langs": list(LANGS), "n_items": len(keep), "chance": 1 / 5,
        "derived_from": str(src),
        "class_labels": "kept in English, see module docstring",
        "validation_required": "templates not written by a native speaker; must be "
                               "validated by an Arabic-speaking co-author and that "
                               "validation described in the paper before reporting",
    }, indent=2), encoding="utf-8")

    print(f"\n{n} sondes ecrites ({len(keep)} items x 6 cellules x 2 langues)")
    for lang in LANGS:
        ex = next(json.loads(l) for l in
                  (OUT / "probes.jsonl").read_text(encoding="utf-8").splitlines()
                  if json.loads(l)["lang"] == lang
                  and json.loads(l)["cell"] == "stale_verified")
        print(f"\n--- exemple {lang} / stale_verified ---")
        print(ex["prompt"])
        print(f"-> gold {ex['gold_letter']}")


if __name__ == "__main__":
    main()
