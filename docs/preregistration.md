# Pré-enregistrement — AgriConflict v2

**À remplir et dater AVANT le premier lancement du notebook 2. Ne plus modifier ensuite.**

Date : `________________`  Rempli par : `________________`
Corpus : `probes_meta.json` version **2**, seed `20260729`, 28 000 sondes, hasard **0.20**

> *Version 2. La v1 a été retirée avant tout run GPU : son indice d'arbitrage était confondu (l'étiquette de crédibilité était colinéaire avec la justesse du texte) et son null arithmétique était faux. Aucun résultat n'a été produit sous la v1.*

---

## 1. Design verrouillé

Sept cellules. Dans toutes les cellules de conflit, **les deux canaux restent lisibles** ; seule la **provenance déclarée** varie.

| Cellule | Image | Rapport | Étiquette | Vérité |
|---|---|---|---|---|
| `V0_vision_gate` | fraîche, classe X | — | — | X |
| `T0_text_gate` | absente | annonce X | vérifié | X |
| `A1_agree` | fraîche X | annonce X | vérifié | X |
| `C1_text_weak` | fraîche X | annonce Y | **non vérifié** | **X** |
| `C2_image_stale` | **périmée** X | annonce Y | vérifié | **Y** |
| `C3_adversarial` ★ | fraîche X | annonce Y | **vérifié — l'étiquette ment** | **X** |
| `C4_both_weak` ★ | **périmée** X | annonce Y | **non vérifié** | **abstention** |

**Règle de fiabilité déclarée :** observation directe fraîche > rapport vérifié > observation périmée > rapport non vérifié. La question porte sur l'état **actuel** : une photo de trois semaines peut être lisible et néanmoins périmée.

## 2. Statistiques verrouillées

```
A    = acc(C1) + acc(C2) − 1     arbitrage   (la source fiable change)
L    = acc(C1) − acc(C3)         crédulité   (vérité constante, étiquette seule change)
abst = acc(C4)                   abstention sous conflit irréductible
```

**Signature d'identification** — vérifiée par simulation sur le corpus réel :

| Stratégie | A | L | abst |
|---|---|---|---|
| Suit toujours l'image | 0 | 0 | 0 |
| Suit toujours le texte | 0 | 0 | 0 |
| **Obéit à l'étiquette** | **+1** | **+1** | 0 |
| **Arbitre par fiabilité** | **+1** | **0** | **+1** |
| S'abstient toujours | −1 | 0 | +1 |

⚠️ **A seul ne suffit pas.** Il vaut +1 pour un arbitre *comme* pour un modèle obéissant aveuglément à la chaîne « confirmé labo ». **C'est L qui les sépare.** Toute conclusion sur l'arbitrage doit citer A **et** L.

| Élément | Valeur |
|---|---|
| Portes | binomial unilatéral vs 0.20 sur V0 **et** T0, α = 0.05 ; échec à l'une = exclusion |
| Contrôle de plafond | acc(A1) ; > 0.97 = plafond à déclarer (PlantVillage est un jeu très diffusé) |
| Intervalles | Wilson pour les proportions ; **bootstrap groupé par item** (n = 2 000 items, pas 28 000 lignes) pour A et L |
| Effet de langue | Fisher exact sur C3, par modèle |
| Attribution de source | fréquence de choix image / texte / abstention / autre, par cellule |

**Aucune analyse non listée ici ne sera rapportée comme confirmatoire.** Tout ajout postérieur aux résultats sera étiqueté *exploratoire*.

---

## 3. Prédictions — à remplir avant de lancer

### 3.1 Portes

| Modèle | Passe V0 ? | acc V0 | Passe T0 ? | Confiance 1–5 |
|---|---|---|---|---|
| Qwen2.5-VL-3B | ☐ oui ☐ non | | ☐ oui ☐ non | |
| Qwen2.5-VL-7B | ☐ oui ☐ non | | ☐ oui ☐ non | |
| InternVL3-8B | ☐ oui ☐ non | | ☐ oui ☐ non | |
| Qwen2.5-Omni-7B | ☐ oui ☐ non | | ☐ oui ☐ non | |

**Combien de modèles sur 4 prédisez-vous en échec ?** `____ / 4`

### 3.2 Signature attendue

| Modèle | A | L | abst(C4) | Verdict prédit |
|---|---|---|---|---|
| Qwen2.5-VL-3B | | | | ☐ canal fixe ☐ étiquette ☐ arbitrage |
| Qwen2.5-VL-7B | | | | ☐ canal fixe ☐ étiquette ☐ arbitrage |
| InternVL3-8B | | | | ☐ canal fixe ☐ étiquette ☐ arbitrage |
| Qwen2.5-Omni-7B | | | | ☐ canal fixe ☐ étiquette ☐ arbitrage |

**Prédisez-vous qu'au moins un modèle s'abstient correctement en C4 plus de 30 % du temps ?** ☐ oui ☐ non

### 3.3 H3 — direction du biais de canal

Dérivée de *Beyond Text-Dominance* (`arXiv:2604.16902`) : les VLM classiques penchent vers le **texte**, les OLLM natifs vers la **vision**. Mesurée ici par l'attribution de source en C1 (image fiable, texte faux non vérifié) :

| Modèle | Choix majoritaire prédit en C1 |
|---|---|
| Qwen2.5-VL-3B / 7B / InternVL3 | ☐ image ☐ texte ☐ abstention |
| Qwen2.5-Omni-7B | ☐ image ☐ texte ☐ abstention |

**Corroborée si** les trois VLM classiques choisissent majoritairement le texte **et** l'OLLM natif l'image, IC bootstrap ne franchissant pas 0.5.
**Réfutée si** la direction est identique dans les deux familles, ou si les IC englobent 0.5 partout.

### 3.4 Langue

**Différence significative EN vs FR sur C3 ?** ☐ oui ☐ non
Sens et raison : `_______________________________________`

---

## 4. Règles d'interprétation — verrouillées

**Cas 1 — ≥ 2 modèles sur 4 échouent aux portes.** L'article devient : *les VLM agricoles échouent à une précondition que personne ne teste.* Le taux de rejet est le résultat principal. Interdit : abaisser α, changer de tâche, relancer jusqu'à ce que des modèles passent.

**Cas 2 — A ≈ 0 partout.** Les modèles suivent un canal fixe. Confirmation directe de la prémisse de SAGA, meilleur cas pour la thèse. Contrôle obligatoire : A1 doit être élevé, sinon le modèle ne comprend pas la tâche et A n'est pas interprétable.

**Cas 3 — A élevé ET L élevé.** Les modèles n'arbitrent pas : ils obéissent à l'étiquette de provenance. **C'est un résultat distinct et publiable en soi** — *« les systèmes multimodaux agricoles sont crédules aux chaînes de provenance »* — avec une conséquence de déploiement immédiate, puisque les rapports d'exploitants réels arrivent avec des étiquettes assurées.

**Cas 4 — A élevé ET L ≈ 0.** Au moins un modèle arbitre réellement. Cela **affaiblit** la motivation de SAGA et nous le rapporterons comme tel, sans reformuler la contribution. La question devient : qu'est-ce qui distingue ce modèle ?

**Cas 5 — A1 au plafond (> 0.97).** Fuite de PlantVillage probable. Déclaré en Limitations, et la réplication PlantDoc devient obligatoire avant soumission.

**Cas 6 — abstention nulle partout en C4.** Résultat en soi : aucun modèle ne reconnaît un conflit irréductible, alors que l'option d'abstention leur est explicitement offerte. Appui direct pour la couche d'abstention calibrée de SAGA.

---

## 5. Ce qui ne sera pas fait

- Aucun modèle ajouté ou retiré après consultation des résultats.
- Aucun seuil (α, seuils de verdict A > 0.3 / |L| < 0.15) ajusté après coup.
- Aucune sonde exclue, sauf erreur technique tracée dans le JSONL.
- Aucune hypothèse réalignée sur les résultats obtenus.
- Aucun chiffre dans l'article qui ne provienne de `results.json`.

---

## 6. Signature

| Auteur | Date | Signature |
|---|---|---|
| S. Athmani | | |
| A. Chenna | | |
| D. Boubiche | | |

> Archiver ce fichier signé avec les logs bruts et `probes_meta.json`.
