# AgriConflict

A probe corpus for measuring how agricultural vision–language models arbitrate between sources that disagree.

Companion code and results for *When the Sensors Disagree: Agricultural Vision–Language Models Read the Image but Act on the Report* (Athmani, Chenna & Boubiche, LASTIC Laboratory, Université Batna 2).

---

## What this measures

Agricultural decision systems combine photographs with written field reports. These sources routinely disagree, and existing agricultural benchmarks score models only on clean, mutually consistent inputs. AgriConflict measures what a model does when they conflict.

Each probe pairs a photograph with a field report. The **declared provenance** of each source — a fresh or three-week-old photograph, a laboratory-confirmed or unverified grower report — is informative about reliability but deliberately **not determinative** of the truth:

| Source | Provenance | P(supports the truth) |
|---|---|---|
| Photograph | fresh (taken today) | 0.90 |
| Photograph | stale (three weeks old, pre-treatment) | 0.30 |
| Field report | laboratory-confirmed, sampled today | 0.85 |
| Field report | unverified, grower-submitted, twelve days old | 0.35 |

No value is 0 or 1, so no configuration determines the answer and a reliability function must estimate probabilities rather than memorise the design. An explicit "cannot be determined" option makes abstention measurable.

Probes are labelled by which source is *actually* correct — `S_both`, `S_img`, `S_txt`, `S_none` — and the arbitration index is

```
A = acc(S_img) + acc(S_txt) − 1
```

Because the reliable source differs between the two subsets while both channels stay legible in each, **any fixed-channel policy scores exactly 0**, and only a policy that switches source scores positively.

### Reference bounds

Simulated on the corpus itself, so any measured value is interpretable:

| Policy | A |
|---|---|
| Always follow the photograph | 0.000 |
| Always follow the report | 0.000 |
| Always abstain | −1.000 |
| **Bayes-optimal using provenance alone** | **+0.634** |
| Oracle | +1.000 |

`+0.634` is the ceiling for any provenance-conditioned arbiter.

---

## Headline results

Four models, 24,000 probes each, zero inference failures.

| Model | Vision gate | S_img | S_txt | **A** | 95% CI |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 0.107 ✗ | 0.002 | 0.913 | −0.085 | [−0.097, −0.074] |
| InternVL3-8B | **0.428 ✓** | 0.061 | 0.822 | −0.117 | [−0.134, −0.100] |
| Qwen2.5-VL-7B | 0.183 ✗ | 0.005 | 0.701 | −0.294 | [−0.312, −0.276] |
| Qwen2.5-Omni-7B | 0.191 ✗ | 0.003 | 0.628 | −0.369 | [−0.388, −0.350] |

Chance on the vision gate is 0.200. All four models pass the *text* gate at 0.86–0.998.

Three findings:

1. **The failure is not perceptual.** Conditional on committing to an answer at all, every model identifies the parcel state from the photograph at 0.60–0.88. Three of four simply decline to act on an image without a corroborating report.
2. **Arbitration is negative for all four**, including the one model whose visual channel is demonstrably functional — so arbitration failure is not explained by perceptual failure.
3. **Provenance gates the text channel, never the image.** A "laboratory-confirmed" marker on a *false* report increases wrong-report following by 19–66 points. The image is chosen at most 6% of the time under conflict.

---

## Repository layout

```
src/
  build_probes.py     generate the corpus from PlantVillage (deterministic, seeded)
  run_eval.py         evaluate one model; forced choice, first-token logits
  analyze.py          gates, arbitration index, source attribution, deference, language
  saga.py             reliability-weighted arbitration + conformal abstention
  verify_corpus.py    corpus integrity and identification checks
  kaggle_runner.py    push and run the pipeline on Kaggle via the API
notebooks/
  colab_internvl.ipynb
docs/
  preregistration.md  analysis plan and interpretation rules, fixed before the runs
results/
  raw/*.jsonl.gz      per-probe records: prediction, gold, chosen source, per-letter logits
  analysis/           results.json, results_saga.json, fig_arbitration.png
  probes_meta.json    corpus parameters and realised reliabilities
reproduce.py          rebuild every reported number from the raw records
```

---

## Reproducing the reported numbers

No GPU required — the raw per-probe records are included.

```bash
pip install -r requirements.txt
python reproduce.py
```

This recomputes the gates, the commitment-conditional accuracies, the arbitration index with item-clustered bootstrap intervals, source attribution, provenance deference and the language test, and checks them against the values reported in the article.

## Regenerating the corpus

The corpus is deterministic given the seed, so it is not redistributed — PlantVillage carries its own licence.

```bash
# obtain PlantVillage, then:
PV_ROOT=/path/to/plantvillage/color python src/build_probes.py
```

Seed `20260730`, 2,000 items, 6 cells, 2 languages → 24,000 probes. `results/probes_meta.json` records the realised reliabilities for comparison.

## Evaluating a new model

```bash
python src/run_eval.py   # set MODEL_ID, SHORT_NAME, LOAD_4BIT at the top
```

Evaluation is a single forward pass per probe comparing the logits of the option letters, so it is deterministic and free of parsing failure. Records are written incrementally and a re-run resumes from the last completed probe.

**Note on InternVL3-8B.** Its default policy splits each image into up to twelve 448-pixel tiles, producing thousands of visual tokens per probe. Our images are at most 448 pixels, so a single view carries the same information; `crop_to_patches=False` is set for this model, which makes the comparison across models more homogeneous rather than less.

---

## Scope of the released evaluation

Stated plainly, because the article does the same:

- **The evidence-seeking loop of SAGA is not evaluated.** A static corpus offers no tool to call, and it is not simulated.
- **`saga.py` supplies oracle per-source decisions.** It measures the arbitration layer in isolation. These are not end-to-end figures; per-source perception error would propagate. They are directly comparable to the +0.634 ceiling, which is computed the same way.
- **PlantVillage is laboratory imagery** — single leaves, uniform backgrounds. This is the easiest possible case for the visual channel, which strengthens a negative result but leaves field generalisation untested.
- **No dedicated agricultural VLM is included.** None had publicly retrievable weights when the audit was run; the dated record is in the article.
- **Two languages**, English and French.

---

## Licence

Code and evaluation outputs: MIT (see `LICENSE`). PlantVillage imagery is not redistributed and carries its own terms.
