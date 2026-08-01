"""
AgriConflict — Notebook 2 (GPU T4) : evaluation d'un modele.

UN NOTEBOOK PAR MODELE. Reglez MODEL_ID ci-dessous, puis Save Version.

Methode : choix multiple force, lecture des logits du PREMIER TOKEN.
Un seul passage avant par sonde — pas de generation autoregressive.
=> deterministe, aucun echec de parsing, 10-30x plus rapide.

Reprise sur interruption : relancer le notebook reprend ou il s'est arrete
(indispensable avec le plafond de 12 h de Kaggle).

Prerequis notebook : Accelerator = GPU T4 x2, Internet = ON.
    !pip install -q -U transformers accelerate bitsandbytes qwen-vl-utils
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# --------------------------------------------------------------------------
# Configuration — A MODIFIER pour chaque modele
# --------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SHORT_NAME = "qwen25vl-3b"
LOAD_4BIT = False        # True pour tout modele >= 7B sur T4 16 Go
MODEL_ALTS: list[str] = []   # identifiants de repli, essayes dans l'ordre
# Options passees au processeur. Sert a neutraliser le decoupage dynamique en
# tuiles d'InternVL : sans cela une sonde produit des milliers de tokens visuels
# et le run depasse la limite de session Kaggle (constate : annule apres 12 h).
# Nos images font 448 px au maximum, donc une vue unique correspond exactement a
# ce que voient les autres modeles -- la comparaison en devient PLUS homogene.
PROC_KWARGS: dict = {}

# Grille suggeree (un notebook chacun) :
#   Qwen/Qwen2.5-VL-3B-Instruct    qwen25vl-3b    4bit=False   generaliste
#   Qwen/Qwen2.5-VL-7B-Instruct    qwen25vl-7b    4bit=True    generaliste
#   OpenGVLab/InternVL3-8B         internvl3-8b   4bit=True    generaliste
#   <AgriGPT-VL sur HuggingFace>   agrigpt-vl     4bit=True    AGRICOLE  <- indispensable
#   <OLLM natif>                   ollm           4bit=True    omni      <- test Beyond Text-Dominance

PROBES_DIR = Path("/kaggle/input/agriconflict-probes")   # sortie du notebook 1
OUT = Path("/kaggle/working") / f"results_{SHORT_NAME}.jsonl"
LETTERS = "ABCDEFGH"
LOG_EVERY = 200


def resolve_probes_dir() -> Path:
    if (PROBES_DIR / "probes.jsonl").exists():
        return PROBES_DIR
    for cand in Path("/kaggle/input").rglob("probes.jsonl"):
        return cand.parent
    raise FileNotFoundError(
        "probes.jsonl introuvable. Ajoutez la sortie du notebook 1 "
        "comme Dataset d'entree."
    )


def letter_token_ids(tokenizer, n_options: int) -> dict[str, list[int]]:
    """Ids du premier token pour chaque lettre, en variantes avec/sans espace."""
    table = {}
    for letter in LETTERS[:n_options]:
        ids = set()
        for variant in (letter, f" {letter}", f"\n{letter}"):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if enc:
                ids.add(enc[0])
        if not ids:
            raise RuntimeError(f"Impossible de tokeniser la lettre {letter}")
        table[letter] = sorted(ids)
    return table


def _ensure_bitsandbytes() -> None:
    """L'image Kaggle n'embarque pas bitsandbytes : sans lui, tout chargement
    4-bit leve ImportError et les 28 000 sondes echouent en quelques minutes."""
    if not LOAD_4BIT:
        return
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("bitsandbytes absent -> installation")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "bitsandbytes>=0.46.1"], check=False)
        import bitsandbytes  # noqa: F401


def _build(model_id: str, kwargs: dict):
    """Charge le modele en essayant l'AutoClass, puis les classes specifiques.

    Qwen2.5-Omni n'est pas couvert par AutoModelForImageTextToText (architecture
    Thinker-Talker) : on cible le Thinker, seul composant qui produit des logits
    textuels exploitables pour un choix force.
    """
    try:
        return AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, **kwargs)
    except (ValueError, KeyError) as exc:
        print(f"  AutoModelForImageTextToText refuse ({type(exc).__name__}), repli")

    from transformers import AutoConfig
    cfg_name = type(AutoConfig.from_pretrained(
        model_id, trust_remote_code=True)).__name__
    print(f"  config = {cfg_name}")

    if "Omni" in cfg_name:
        try:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration as C
            return C.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        except Exception as exc:                          # noqa: BLE001
            print(f"  Thinker direct impossible ({exc}), chargement complet")
            from transformers import Qwen2_5OmniForConditionalGeneration as C
            return C.from_pretrained(model_id, trust_remote_code=True, **kwargs).thinker

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, **kwargs)


def load_model():
    _ensure_bitsandbytes()
    kwargs = {
        "torch_dtype": torch.float16,        # T4 = Turing : PAS de bf16
        "attn_implementation": "sdpa",       # flash-attn indisponible sur T4
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if LOAD_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    last = None
    for mid in [MODEL_ID, *MODEL_ALTS]:
        try:
            print(f"tentative : {mid}")
            processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
            model = _build(mid, kwargs)
            model.eval()
            print(f"charge : {mid}  ({type(model).__name__})")
            return model, processor, mid
        except Exception as exc:                          # noqa: BLE001
            print(f"  ECHEC {mid} : {type(exc).__name__}: {str(exc)[:180]}")
            last = exc
    raise RuntimeError(f"Aucun identifiant utilisable parmi "
                       f"{[MODEL_ID, *MODEL_ALTS]}") from last


@torch.inference_mode()
def score_probe(model, processor, image: Image.Image, prompt: str,
                tok_ids: dict[str, list[int]]) -> tuple[str, dict[str, float]]:
    """Renvoie la lettre predite et les logits par lettre."""
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           **PROC_KWARGS)
    except TypeError:            # processeur qui n'accepte pas ces options
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    logits = model(**inputs).logits[0, -1, :].float()

    scores = {letter: max(logits[i].item() for i in ids)
              for letter, ids in tok_ids.items()}
    return max(scores, key=scores.get), scores


def main() -> None:
    probes_dir = resolve_probes_dir()
    probes = [json.loads(l) for l in
              (probes_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines() if l]
    print(f"{len(probes)} sondes chargees depuis {probes_dir}")

    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line:
                done.add(json.loads(line)["probe_id"])
        print(f"Reprise : {len(done)} sondes deja traitees")

    todo = [p for p in probes if p["probe_id"] not in done]
    if not todo:
        print("Rien a faire — evaluation deja complete.")
        return
    print(f"A traiter : {len(todo)}")

    model, processor, used_id = load_model()
    tokenizer = getattr(processor, "tokenizer", processor)
    tok_ids = letter_token_ids(tokenizer, todo[0]["n_options"])
    print(f"Modele charge. Ids des lettres : "
          f"{ {k: v[0] for k, v in tok_ids.items()} }")

    t0 = time.time()
    n_ok = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo):
            img_path = probes_dir / p["image"]
            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
                    pred, scores = score_probe(
                        model, processor, image, p["prompt"], tok_ids
                    )
                correct = int(pred == p["gold_letter"])
                n_ok += correct
                # attribution de source : distingue le suivisme de la confusion
                if pred == p["letter_image"]:
                    chose = "image"
                elif pred == p["letter_text"]:
                    chose = "text"
                elif pred == p["letter_abstain"]:
                    chose = "abstain"
                else:
                    chose = "other"
                fh.write(json.dumps({
                    "probe_id": p["probe_id"], "uid": p["uid"],
                    "model": SHORT_NAME, "model_id": used_id,
                    "lang": p["lang"], "cell": p["cell"],
                    "pred": pred, "gold": p["gold_letter"], "correct": correct,
                    "chose": chose,
                    # champs v3 : la stratification se fait sur la correction REELLE
                    # des sources, pas sur la provenance.
                    "subset": p["subset"], "q_img": p["q_img"], "q_rep": p["q_rep"],
                    "img_ok": p["img_ok"], "rep_ok": p["rep_ok"],
                    "letter_image": p["letter_image"], "letter_text": p["letter_text"],
                    "letter_abstain": p["letter_abstain"],
                    "logits": {k: round(v, 4) for k, v in scores.items()},
                }, ensure_ascii=False) + "\n")
            except Exception as exc:                      # noqa: BLE001
                fh.write(json.dumps({
                    "probe_id": p["probe_id"], "model": SHORT_NAME,
                    "cell": p["cell"], "lang": p["lang"], "error": repr(exc),
                }, ensure_ascii=False) + "\n")

            if (i + 1) % LOG_EVERY == 0:
                fh.flush()
                el = time.time() - t0
                rate = el / (i + 1)
                print(f"  {i+1}/{len(todo)} | {rate:.3f} s/sonde | "
                      f"acc courante {n_ok/(i+1):.3f} | "
                      f"reste ~{rate*(len(todo)-i-1)/60:.0f} min")

    print(f"\nTermine en {(time.time()-t0)/60:.1f} min -> {OUT}")
    print("Save Version, puis ajouter cette sortie comme Dataset du notebook 3.")


if __name__ == "__main__":
    main()
