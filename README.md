# AgriConflict

A probe corpus for measuring how agricultural vision–language models arbitrate between sources that disagree.

Companion code and results for *When the Sensors Disagree: Agricultural Vision–Language Models Read the Image but Act on the Report* (Athmani, Chenna & Boubiche, LASTIC Laboratory, Université Batna 2).

---

## What this measures

Agricultural decision systems combine photographs with written field reports. These sources routinely disagree, and existing agricultural benchmarks score models only on clean, mutually consistent inputs. AgriConflict measures what a model does when they conflict.

Each probe pairs a photograph with a field report. The **declared provenance** of each source (a fresh or three-week-old photograph, a laboratory-confirmed or unverified grower report) is informative about reliability but deliberately **not determinative** of the truth:

| Source | Provenance | P(supports the truth) |
|---|---|---|
| Photograph | fresh (taken today) | 0.90 |
| Photograph | stale (three weeks old, pre-treatment) | 0.30 |
| Field report | laboratory-confirmed, sampled today | 0.85 |
| Field report | unverified, grower-submitted, twelve days old | 0.35 |

No value is 0 or 1, so no configuration determines the answer and a reliability function must estimate probabilities rather than memorise the design. An explicit "cannot be determined" option makes abstention measurable.

Probes are labelled by which source is *actually* correct (`S_both`, `S_img`, `S_txt`, `S_none`), and the arbitration index is

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
2. **Arbitration is negative for all four**, including the one model whose visual channel is demonstrably functional, so arbitration failure is not explained by perceptual failure.
3. **Provenance gates the text channel, never the image.** A "laboratory-confirmed" marker on a *false* report increases wrong-report following by 19–66 points. The image is chosen at most 6% of the time under conflict.

### Field-condition replication

The same protocol was replicated on PlantDoc, whose images are photographed in the field. 12,000 probes per model, four models.

| Model | Vision gate lab → field | A lab → field | 95% CI field |
|---|---|---|---|
| Qwen2.5-VL-3B | 0.107 ✗ → **0.436 ✓** | −0.085 → −0.059 | [−0.087, −0.031] |
| Qwen2.5-VL-7B | 0.183 ✗ → **0.471 ✓** | −0.294 → −0.324 | [−0.359, −0.290] |
| InternVL3-8B | 0.428 ✓ → **0.677 ✓** | −0.117 → −0.031 | [−0.065, +0.003] |
| Qwen2.5-Omni-7B | 0.191 ✗ → **0.358 ✓** | −0.369 → −0.450 | [−0.481, −0.418] |

All four models pass the vision gate on field imagery; three failed it on laboratory imagery. This contradicts the expectation that laboratory conditions are the easier case, and it is not an artefact of option difficulty: the probability that four randomly drawn options belong to four distinct crops is 0.539 for PlantVillage and 0.530 for PlantDoc.

**Arbitration stays negative in seven of eight measurements and is never positive.** The one interval containing zero belongs to InternVL3-8B on field imagery, which means it becomes indistinguishable from a fixed-channel policy, not that it arbitrates. Two models do *worse* on field imagery despite seeing better. The central finding is not an artefact of laboratory conditions.

### End-to-end arbitration

The figures above measure the controller as deployed. A separate question is whether
relocating aggregation outside the frozen controller helps. Supplying an explicit
reliability-weighted arbitration layer with *exact* per-source decisions reaches the
provenance ceiling, but that isolates the layer rather than measuring a pipeline.

We therefore ran an image-only variant of each factorial cell: the same photograph with
the same provenance line and no field report, so the decision the controller returns on
the image alone becomes a genuine per-source decision produced by the controller itself.
Eight thousand additional probes per model, English only.

| Model | Implicit controller | Oracle sources | **End to end** | 95% CI | % of ceiling | Image tool |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | -0.084 | +0.659 | **-0.036** | [-0.062, -0.010] | -5.7% | 0.094 |
| Qwen2.5-VL-7B | -0.259 | +0.659 | **+0.055** | [+0.022, +0.087] | 8.6% | 0.161 |
| InternVL3-8B | -0.201 | +0.659 | **+0.232** | [+0.193, +0.269] | 36.5% | 0.344 |
| Qwen2.5-Omni-7B | -0.446 | +0.659 | **+0.057** | [+0.025, +0.089] | 9.0% | 0.134 |

The oracle column is identical across models by construction: with exact sources the
decision depends only on the fitted reliability function, so it is a property of the
corpus and not of any model.

**Three of four models cross into positive arbitration, and these are the only positive
values anywhere in the study.** The information needed to arbitrate is present in the
prompt and is lost in aggregation. Recovery is nonetheless partial, between 8.6 and 36.5
per cent of the ceiling, and the shortfall tracks the last column: the image tool returns
the class the photograph shows on 0.094 to 0.344 of probes against a chance rate of 0.200.
The model with the best image tool recovers the most; the model whose tool sits below
chance stays negative. Arbitration machinery cannot repair a channel the controller
refuses to read.

### Four languages

The Arabic and Algerian-dialect (darija) arms are derived from the published corpus: same
items, same photographs, same options and same option order, with only the textual frame
replaced. Twelve thousand probes per model over the first 1,000 items. Accuracies are
computed on items common to all four languages.

| Model | Subset | English | French | Arabic | Darija | chi-square *p* |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | S_img | 0.001 | 0.002 | 0.003 | **0.035** | 9e-17 |
| | V0 gate | 0.120 | 0.118 | 0.161 | **0.226** | 1e-12 |
| Qwen2.5-VL-7B | S_img | 0.003 | 0.007 | 0.010 | **0.022** | 5e-04 |
| | V0 gate | 0.223 | 0.152 | 0.278 | **0.358** | 6e-26 |
| InternVL3-8B | S_img | 0.063 | 0.064 | 0.081 | **0.112** | 1e-04 |
| | V0 gate | 0.419 | 0.466 | 0.453 | 0.414 | 0.048 |
| Qwen2.5-Omni-7B | S_img | 0.004 | 0.007 | 0.016 | **0.025** | 2e-04 |
| | V0 gate | 0.227 | 0.169 | 0.221 | 0.213 | 0.006 |

On S_img, the subset where the photograph is right and the report wrong, every model
resists the false report more as the language of the report moves away from English, and
darija exceeds English by a factor of between seven and thirty-five. The earlier
two-language round found nothing because English and French sit close together on this
axis. The parsimonious reading is that a report in a language the model processes less
fluently is less persuasive. The effect is small in absolute terms, so the asymmetric
evidentiary standard weakens in lower-resource languages without reversing.

> **Validation status.** The Arabic and darija templates were not written by a native
> speaker. Class labels were deliberately left in English, because translating
> crop-disease pairs into agronomic Arabic without a validated glossary would corrupt the
> task rather than translate it. Darija is rendered in Arabic script, though it is also
> commonly written in Latin script. These records are released for inspection; the
> magnitude of the language effect should be treated as provisional until an
> Arabic-speaking agronomist has validated the register.

---

## Repository layout

```
src/
  build_probes.py           generate the laboratory corpus from PlantVillage (deterministic, seeded)
  build_probes_field.py     generate the field corpus from PlantDoc, identical process
  build_probes_imageonly.py derive the image-only variant by stripping the report line
  build_probes_arabic.py    derive the Arabic and darija variants, frame text only
  run_eval.py               evaluate one model; forced choice, first-token logits
  analyze.py                gates, arbitration index, source attribution, deference, language
  saga.py                   reliability-weighted arbitration + conformal abstention
  end_to_end.py             arbitration on the controller's own per-source decisions
  languages.py              the four-language comparison on common items
  compare_corpora.py        laboratory against field, item-clustered intervals
  figures.py                manuscript figures
  verify_corpus.py          corpus integrity and identification checks
  kaggle_runner.py          push and run the pipeline on Kaggle via the API
notebooks/
  colab_internvl.ipynb
docs/
  preregistration.md  analysis plan and interpretation rules, fixed before the runs
results/
  raw/*.jsonl.gz            laboratory corpus, per-probe records: prediction, gold,
                            chosen source, per-letter logits
  raw_field/*.jsonl.gz      field corpus, same schema
  raw_imageonly/*.jsonl.gz  image-only variant, same schema
  raw_arabic/*.jsonl.gz     Arabic and darija variants, same schema
  analysis/                 results.json, results_saga.json, results_corpora.json,
                            results_e2e.json, results_langs.json
  figures/                  the four manuscript figures
  probes_meta.json          corpus parameters and realised reliabilities
reproduce.py          rebuild every reported number from the raw records
```

---

## Reproducing the reported numbers

No GPU required: the raw per-probe records are included.

```bash
pip install -r requirements.txt
python reproduce.py
```

This recomputes the gates, the commitment-conditional accuracies, the arbitration index with item-clustered bootstrap intervals, source attribution, provenance deference, the field replication, the end-to-end arbitration figures and the four-language comparison, and checks each against the value reported in the article. A mismatch beyond 0.002 exits non-zero.

## Regenerating the corpus

The corpus is deterministic given the seed, so it is not redistributed, because PlantVillage carries its own licence.

```bash
# obtain PlantVillage, then:
PV_ROOT=/path/to/plantvillage/color python src/build_probes.py

# obtain PlantDoc, then:
PD_ROOT=/path/to/plantdoc/train python src/build_probes_field.py
```

The image-only and Arabic variants are *derived* from the generated corpus rather than
regenerated, because image-file selection iterates a set whose order depends on string
hashing and is therefore not stable across interpreter runs:

```bash
python src/build_probes_imageonly.py   # strips the report line, keeps everything else
python src/build_probes_arabic.py      # replaces the frame text only
```

Laboratory: seed `20260730`, 2,000 items, 6 cells, 2 languages, 24,000 probes.
Field: seed `20260801`, 1,000 items, 12,000 probes.
Image-only: 4 cells over 2,000 items, English, 8,000 probes.
Arabic and darija: 6 cells over the first 1,000 items, 2 languages, 12,000 probes.
`results/probes_meta.json` records the realised reliabilities for comparison.

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
- **`saga.py` supplies oracle per-source decisions.** It measures the arbitration layer in isolation, and is directly comparable to the +0.634 ceiling, which is computed the same way. `end_to_end.py` is the version that does not rely on an oracle, and it is the one to read for a deployment estimate.
- **End-to-end arbitration was measured in English only**, on the laboratory corpus, to halve its cost.
- **Laboratory imagery is not the easier case.** We expected it to be, and the field replication refuted that: all four models read PlantDoc photographs better than PlantVillage ones, because a detached leaf on a uniform background removes the context that identifies the crop. Both corpora are nonetheless curated disease-classification collections, so neither establishes behaviour on imagery captured by a grower's own handset.
- **No dedicated agricultural VLM is included.** None had publicly retrievable weights on either of two checks almost three months apart; the dated record is in the article.
- **SAGA is evaluated on the laboratory corpus only.**
- **The Arabic and darija arms await native-speaker validation**, as stated above.
- **Provenance is declared in text**, not derived from sensor metadata. This isolates the arbitration question at the cost of realism about how provenance reaches a deployed system.

---

## Licence

Code and evaluation outputs: MIT (see `LICENSE`). PlantVillage imagery is not redistributed and carries its own terms.
