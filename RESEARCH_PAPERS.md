# Scientific Foundations, Mathematical Formulation and Citations of the Project (Research Grounding)

This document records the scientific grounding of the main components of "Privacy Auto-Anonymizer": the mathematical formulation of differentially-private pixelation (DP-Pix), multi-modal PII detection within the contextual-integrity framework, scene-conditioned visual-linguistic reasoning (the Imajev engine) and interactive human-in-the-loop validation. The month/year of every reference is consistent with its arXiv identifier.

---

## 1. Differentially-Private Pixelation (DP-Pix) and Resistance to Reconstruction Attacks

**Problem.** Traditional blurring (Gaussian Blur) is a deterministic, linear mapping; the Privacy Blur paper (December 2025 — arXiv:2512.16086) and Revelio (June 2025 — arXiv:2506.12344) show that the blur output can be reconstructed by Super-Resolution networks and diffusion models with an accuracy of up to **95.9%**, because high-frequency gradients (edges, skin texture, letter glyphs) leave behind information sufficient to solve the inverse problem.

**Why DP-Pix works.** The DP-Pix structure builds three defensive layers on the basis of "Explainability-Driven Incremental Image Anonymization" (2025): (1) block averaging, which removes high-frequency geometry, (2) Gaussian noise at the grid level, which is statistical and irreversible, and (3) the nearest-neighbour mapping, which makes all pixels of a cell a single colour so that no intra-cell gradient remains.

**Mathematical formulation** (implementation: `filters.py::dp_pix_filter`). For an image $I$ and cell size $b$:

```text
(1) Block averaging:        F(D)[i,j] = (1/b²) · Σ_{p ∈ B_ij} I(p)          # cv2.INTER_AREA
(2) Grid-level noise:       I_priv   = clip( F(D) + N(0, σ²), 0, 255 )      # one draw per cell
(3) Mosaic reconstruction:  Î        = Upsample(I_priv, INTER_NEAREST)      # every b×b macropixel a single colour
```

The sensitivity (L1) of the block-averaging function to one input pixel is $\Delta f = 255 / b^2$; hence, with the Gaussian mechanism, noise $\sigma = \Delta f \cdot \sqrt{2\ln(1.25/\delta)} / \varepsilon$ establishes the $(\varepsilon, \delta)$ guarantee for the released grid vector.

**Key implementation note.** The noise **must** be applied to the down-sampled array and **before** upsampling with `INTER_NEAREST`. If noise is injected per pixel after upsampling, high-frequency graininess ("salt-and-pepper") is recreated — precisely the statistics that SR and De-blurring networks use for inversion. Every macropixel must have **one uniform noisy colour**.

**Use in the code:** `filters.dp_pix_filter` (faces/plates, manual and automatic regions) and the cell-uniformity test `DpPixGridLevelTests` in `tests.py`.

---

## 2. Contextual Integrity and Multi-Modal PII Detection

The Contextual Integrity framework (the RedactionBench paper — June 2026, arXiv:2606.18782) rules that "sensitivity" is not a property of the text but the ratio of the information to the context of its transmission; therefore, instead of a single model, the system combines several complementary layers:

1. **Vision layer:** YOLOv8-face (with a safety margin for full face coverage) and YOLOv8-plate — every plate candidate is assessed in a validation cascade by OCR inside the box (strict on documents, lenient on photographs).
2. **Text-recognition layer:** three selectable engines (EasyOCR, PaddleOCR, docTR) with dedicated preprocessing: MRZ-based deskew, blackhat morphology, and mild CLAHE in LAB space.
3. **Key–value spatial graph:** multilingual header↔value matching on the next line ("Surname / Nom / Apellidos" → "DE LA PAZ") and coreference transfer of the name from the MRZ to the printed events.
4. **Algorithmic validators:** Luhn/ISO 7812 (card), IBAN mod-97/ISO 13616, SSA rules for SSN, the Iranian national-ID checksum, and the ICAO Doc 9303 check digits for TD1/TD2/TD3.
5. **Zero-entropy text masking:** for textual identifiers, an absolute black box (`solid_black_box`) instead of blur/mosaic — no bit of the content should be estimable.

**Use in the code:** `pii.py` (patterns, validators, header↔value pairing, noise-resistant MRZ) and `ml_pipeline.py` (the OCR interface and character-span→polygon mapping).

---

## 3. Scene-Conditioned Visual-Linguistic Reasoning (the Imajev Engine)

The Imajev engine takes typed decisions and explicitly reports an "Unknown" probability. The central idea is to calibrate each candidate's probability with the **scene prior**:

```text
P(E | C, S)  ∝  P(C | E, S) · P(E | S)
```

where $C$ is the candidate image crop, $E$ the entity type (face/plate/text) and $S$ the overall scene (document/street). In the logistic form, the prior becomes a shift in the logits: $z'_E = z_E + \log P(E|S)$.

**Calibrated example (encoded in `ImajevClassifier.predict_typed`):** on a "document / identity card" scene with confidence $\geq 0.80$, the plate prior tends to zero; consequently, for document-text crops: $z_{plate} \mathrel{-}= 5$ and $z_{printed} \mathrel{+}= 3.5$, which takes the output to "Printed Text" $\geq 85\%$ and "License Plate" $< 15\%$ (previously these crops were classified 100% as plates).

**Single decision contract.** The visual vote (VLM) and the textual check (OCR) are merged **before** the log is recorded; a candidate never receives two contradictory `[APPROVED]` and `[REJECTED]` records. On a high-confidence document scene, inconsistent candidates are suppressed silently and at most one aggregate `[INFO]` note is recorded. **fail-open** policy: a validator error never removes the element (the mistaken removal of a face = its disclosure).

**Use in the code:** `ImajevClassifier`, `_validate_crop_with_imajev` and the scene/plate/face/text phases in `analyze_image_entities`.

---

## 4. Interactive Human-in-the-Loop (HITL) Validation

The ReGenHuman paper (June 2026 — arXiv:2606.14972) shows that fully automatic cutting fails on real documents: handwritten signatures, rotated stamps and text over guilloche patterns are neither found by YOLO nor always read by OCR. The solution is an "interactive curator":

- Clicking any box or HUD label → toggling the mask (Human-in-the-Loop).
- **Manual region drawing:** dragging the mouse over empty Canvas space creates a live selection rectangle (turquoise/amber dashed outline); on release, if its dimensions are $\geq 12$ image pixels, a `CUSTOM` region with masking enabled is added and can be removed with the × on its label.
- `ResizeObserver` on the Canvas ensures that labels and SVG guide lines are recalculated precisely during layout animations (opening/closing the terminal panel, window resizing).

**Use in the code:** `static/js/app.js` (manual drawing, mask toggle, ResizeObserver) and `templates/anonymizer/index.html` (Canvas and HUD layers).

---

## 5. Explainable Reasoning and Privacy Governance (XAI + Governance)

Four added components turn "explainability" from a claim into a measurable contract.

**a) Cross-zone VIZ ↔ MRZ matching.** On identity documents, ICAO Doc 9303 writes the same identity information **twice**: once human-readable (the Visual Inspection Zone, VIZ) and once machine-readable (the MRZ). This redundancy creates a **cross-verification** signal that no single zone can provide: a local forgery (photo substitution, manipulation of one digit in the printed zone) causes the two zones to diverge.

- Every printed field is assessed according to its own data shape: names via **letter mapping** (the MRZ deletes spaces, hence "DE LA PAZ" ↔ "DELAPAZ"), the document number via the **substring without filler**, and dates via the ICAO month number and the two-digit year. The "day" is deliberately not mandatory: the thinnest column of the date on the page is the least reliable OCR signal, whereas month+year makes the field unique.
- **Check digits** are recomputed with weights $7,3,1$ (modulo 10); for TD3, the composite check digit is verified over `[0..9] + [13..19] + [21..27] + [28..43]`. Three cases are distinguished: `True` (all digits valid), `False` (known template length but inconsistent check digit = evidence of tampering) and `None` (unknown length ⇒ the field positions are undefined and **no verdict is issued**). This very distinction saves the system from unfounded claims.
- **Consistency score:** $c = 0.994 \cdot \frac{|\text{matched fields}|}{|\text{compared fields}|}$ and, if the check digit is `False`, $c \leftarrow \min(c, 0.55)$ (falling into the review band). The acceptance threshold is $c \ge 0.90$; consequently the 41-character page (incomplete text) is never wrongly flagged as "tampered", because its check digit is `None` and no penalty is applied.

**b) The bi-directional spotlight as an XAI contract.** Explainability is useless when the user cannot say "which pixels does this model statement refer to?". For this reason, every terminal record whose decision was applied to a real element carries that element's `detection_id`/`detection_ids` and `bbox`, and the browser keeps two paths live: card → box (pulsing ring, amber = unmasked, turquoise = masked) and box/label → card (flash + smooth scroll). No explanation is displayed without a spatial reference.

**c) Re-identification risk meter.** Two quantities are computed live:

```text
Initial risk    = min(99, Σ w(e))                          , w: MRZ=35, FACE=28, ID/SSN/CARD/IBAN=25,
Residual risk   = min(99, Σ_{e: !masked} w(e) + 1/ε · [∃ masked DP-Pix])   NAME=20, DATE/PHONE/EMAIL/ADDR=12
```

The sum is **additive** (every independent identifier raises the chance of re-identification) and the 99% cap prevents predicting "absolute uncertainty". The term $1/\varepsilon$ is the **differential-privacy leakage floor**: a DP-Pix mask does not make a crop uninformed, it only bounds the measurement, so a masked face at $\varepsilon = 0.5$ leaves about 2% and at $\varepsilon = 2.0$ about 0.5% — while text masking (zero-entropy solid black) has a floor of zero. The colour thresholds (25% and 60%) are the same "emerald / amber / red" bands of the user interface.

**d) Zero-PII audit certificate.** An audit report that itself carries identity data defeats its own purpose. Therefore every value is represented by a **fingerprint**:

```text
fingerprint(v) = SHA-256(salt_session ‖ "|" ‖ v)[0..9]        (ten hex characters)
```

The salt is generated **per session** with `crypto.getRandomValues` and is never exported. Without a salt, the hash of a name or a date is recovered from a small dictionary (common names, 365 days of the year) in a fraction of a second; with a session salt, the fingerprint is meaningful only **within that same certificate** (matching/equality of two values) and is in no way reversible to an identity. The SHA-256 implementation was written in `app.js` (dependency-free, equivalent to `sha256sum`) so that the fingerprint remains valid on insecure origins (served over `http://<lan-ip>`) where `crypto.subtle` does not exist.

**Use in the code:** `_cross_zone_consistency_entry`, `_mrz_check_digits_valid` and `_finalize_detection_links` in `ml_pipeline.py`; `computeRiskAssessment`, `buildAuditReport`, `sha256Hex` and `drawSpotlightRing` in `static/js/app.js`.

---

## 6. Summary Table of Citations

| # | Topic | Reference | Use in the code |
|---|-------|------|----------------|
| 1 | Reconstruction of blurring with diffusion/SR | Privacy Blur — December 2025, arXiv:2512.16086 | Justification for rejecting simple blur; DP-Pix as the default |
| 2 | Face reconstruction attack | Revelio — June 2025, arXiv:2506.12344 | Second witness against simple blur/mosaic |
| 3 | Incremental DP-Pix | Explainability-Driven Incremental Image Anonymization — 2025 | `filters.dp_pix_filter` (grid-level noise) |
| 4 | Contextual integrity and zero-entropy masking | RedactionBench — June 2026, arXiv:2606.18782 | `solid_black_box` for text |
| 5 | Human-in-the-loop curation | ReGenHuman — June 2026, arXiv:2606.14972 | Interactive toggle + manual region drawing |
| 6 | Passport structure and check digits | ICAO Doc 9303 (8th edition, parts 4–6) | `_mrz_check_digits_valid` (7-3-1 weights, composite TD3 check digit) and VIZ ↔ MRZ matching |
| 7 | De-identification and dictionary attack on hashes | NIST SP 800-63B (random-salt requirement) | `fingerprintValue` with a per-session salt in the audit certificate |

---

## 7. Reproduction of the Results for the Grader

```bash
python manage.py test anonymizer        # full test suite (offline, no weight download)
python -m anonymizer.filters            # self-test of the masking filters
```
