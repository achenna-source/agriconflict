"""
AgriConflict — pousse et lance les notebooks sur Kaggle via l'API.

PREREQUIS (une seule fois, 2 min) :
  1. https://www.kaggle.com/settings  ->  section API  ->  "Create New Token"
  2. Placer le kaggle.json telecharge dans  C:\\Users\\AJ\\.kaggle\\kaggle.json
  3. pip install kaggle

USAGE :
  python push_and_run.py build          # notebook 1 (CPU) : construit les sondes
  python push_and_run.py eval           # notebook 2 (GPU) : tous les modeles
  python push_and_run.py eval qwen3b     # un seul modele
  python push_and_run.py analyze        # notebook 3 (CPU) : tableaux + figures
  python push_and_run.py status         # etat de tous les kernels pousses
  python push_and_run.py pull           # recupere les sorties en local

Les notebooks sont crees en PRIVE. Rien n'est publie.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
BUILD = HERE / "_push_build"
PLANTVILLAGE = "abdallahalidev/plantvillage-dataset"

# Grille de modeles : slug -> (MODEL_ID HuggingFace, SHORT_NAME, 4bit)
MODELS = {
    # VLM classiques -- litterature : penchent vers le TEXTE
    "qwen3b":   ("Qwen/Qwen2.5-VL-3B-Instruct",  "qwen25vl-3b",  False, []),
    "qwen7b":   ("Qwen/Qwen2.5-VL-7B-Instruct",  "qwen25vl-7b",  True, []),
    "internvl": ("OpenGVLab/InternVL3-8B-hf",    "internvl3-8b", True,
                 ["OpenGVLab/InternVL3_5-8B-HF", "OpenGVLab/InternVL3-8B"]),
    # OLLM natif -- litterature : penche vers la VISION. Test de H3.
    "omni":     ("Qwen/Qwen2.5-Omni-7B",         "qwen25-omni-7b", True,
                 ["Qwen/Qwen2.5-Omni-3B"]),
    # AgriGPT-VL / Agri-R1 / AgroNVILA : aucun poids public au 2026-07-29.
    # Voir l'audit de disponibilite date, section 6.5 du manuscrit.
}


# Options de processeur par modele (voir PROC_KWARGS dans 02_run_eval.py).
PROC = {"internvl": {"crop_to_patches": False}}


def kaggle_username() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        sys.exit(
            "kaggle.json introuvable.\n"
            "  1. https://www.kaggle.com/settings -> API -> Create New Token\n"
            f"  2. deposer le fichier dans {cfg}\n"
        )
    return json.loads(cfg.read_text(encoding="utf-8"))["username"]


def run(cmd: list[str]) -> str:
    print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print("   " + out.strip().replace("\n", "\n   "))
    return out


def push(slug: str, code_file: Path, *, gpu: bool,
         dataset_sources: list[str], kernel_sources: list[str],
         env: dict[str, str] | None = None) -> str:
    """Prepare un dossier et pousse un kernel Kaggle."""
    user = kaggle_username()
    kid = f"{user}/{slug}"
    d = BUILD / slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

    src = code_file.read_text(encoding="utf-8")
    if env:
        # Injecte la configuration en tete du script, sans toucher au fichier source.
        header = "".join(f"{k} = {v!r}\n" for k, v in env.items())
        for k in env:
            # Neutralise l'affectation d'origine. Doit reconnaitre les affectations
            # ANNOTEES (MODEL_ALTS: list[str] = []) : sinon l'original s'execute apres
            # l'en-tete injecte et ecrase silencieusement la configuration.
            pat = re.compile(rf"^{re.escape(k)}\s*(:[^=]+)?=")
            src = "\n".join(
                (f"# [overridden] {ln}" if pat.match(ln) else ln)
                for ln in src.splitlines()
            )
        src = f"# --- configuration injectee par push_and_run.py ---\n{header}\n{src}"
    (d / code_file.name).write_text(src, encoding="utf-8")

    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": kid,
        "title": slug.replace("-", " "),
        "code_file": code_file.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,          # rien n'est publie
        "enable_gpu": gpu,
        # IMPORTANT : champ machine_shape, valeurs acceptees NvidiaTeslaT4 / NvidiaTeslaP100
        # Sans "GpuT4x2", Kaggle attribue machine_shape="Gpu" = Tesla P100 (sm_60), que les
        # builds PyTorch recents ne compilent plus (sm_70 minimum) :
        # "CUDA error: no kernel image is available for execution on the device".
        "machine_shape": "NvidiaTeslaT4" if gpu else "",
        "enable_internet": True,     # requis : telechargement HuggingFace
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
        "competition_sources": [],
    }, indent=2), encoding="utf-8")

    print(f"\n>>> push {kid}  (gpu={gpu})")
    run(["kaggle", "kernels", "push", "-p", str(d)])
    return kid


def wait(kid: str, poll: int = 60, timeout: int = 4 * 3600) -> str:
    """Attend la fin d'un kernel. Kaggle coupe a 12 h ; le script est reprenable."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = run(["kaggle", "kernels", "status", kid])
        # Ne se fier QU'a l'etat explicite du kernel : une coupure reseau pendant
        # l'appel status contient "Error" et faisait abandonner un run encore vivant.
        if "KernelWorkerStatus.COMPLETE" in out:
            return "complete"
        if "KernelWorkerStatus.ERROR" in out:
            return "error"
        if "KernelWorkerStatus.CANCEL" in out:
            return "cancelled"
        print(f"   ... {(time.time()-t0)/60:.0f} min ecoulees")
        time.sleep(poll)
    return "timeout"


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    user = kaggle_username()
    print(f"Compte Kaggle : {user}")

    if cmd == "build":
        kid = push("agriconflict-01-build", HERE / "01_build_probes.py",
                   gpu=False, dataset_sources=[PLANTVILLAGE], kernel_sources=[])
        print(f"\nEtat : {wait(kid)}")
        print("Ensuite :  python push_and_run.py eval")

    elif cmd == "eval":
        only = sys.argv[2] if len(sys.argv) > 2 else None
        probes = f"{user}/agriconflict-01-build"
        for slug, (mid, short, q4, alts) in MODELS.items():
            if only and slug != only:
                continue
            kid = push(f"agriconflict-02-{slug}", HERE / "02_run_eval.py",
                       gpu=True, dataset_sources=[], kernel_sources=[probes],
                       env={"MODEL_ID": mid, "SHORT_NAME": short,
                            "LOAD_4BIT": q4, "MODEL_ALTS": alts,
                            "PROC_KWARGS": PROC.get(slug, {})})
            print(f"\n{slug} -> {wait(kid)}")

    elif cmd == "analyze":
        srcs = [f"{user}/agriconflict-01-build"] + \
               [f"{user}/agriconflict-02-{s}" for s in MODELS]
        kid = push("agriconflict-03-analyze", HERE / "03_analyze.py",
                   gpu=False, dataset_sources=[], kernel_sources=srcs)
        print(f"\nEtat : {wait(kid)}")
        print("Ensuite :  python push_and_run.py pull")

    elif cmd == "pull":
        out = HERE / "_results"
        out.mkdir(exist_ok=True)
        for slug in ["agriconflict-01-build", "agriconflict-03-analyze"] + \
                    [f"agriconflict-02-{s}" for s in MODELS]:
            run(["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(out)])
        print(f"\nSorties recuperees dans {out}")

    elif cmd == "status":
        for slug in ["agriconflict-01-build", "agriconflict-03-analyze"] + \
                    [f"agriconflict-02-{s}" for s in MODELS]:
            run(["kaggle", "kernels", "status", f"{user}/{slug}"])

    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
