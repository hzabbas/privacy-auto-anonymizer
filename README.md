# Privacy Auto-Anonymizer

**An interactive system for the automatic redaction of sensitive data in images and documents** — face, license-plate and identity-text detection, with masking resistant to reconstruction attacks and fully client-side output.

Undergraduate capstone project in Computer Engineering — technology stack: Python, Django, YOLOv8, EasyOCR/PaddleOCR/docTR, HTML5 Canvas, Tailwind CSS.

---

## 1. Overview

The user uploads an image (a scene photograph, an identity document, a letter, a form). Using object-detection and text-recognition engines, the server extracts the sensitive regions and returns only their **coordinates** as JSON. All box rendering, user interaction and **final masking take place on the browser Canvas**; consequently the redacted image is never produced or stored on the server.

## 2. Features

**Multi-layer detection**
- Faces: YOLOv8-face with a safety margin so that the facial region is covered completely.
- License plates: YOLOv8-plate; **every** plate candidate is validated by text recognition inside the box — strictly on document scans (the box text must genuinely resemble a plate: short, with at least one digit and no lowercase letters) and leniently on photographs (an unreadable box is retained so that a real plate is not missed).
- Text: three selectable engines — EasyOCR (default), PaddleOCR and docTR — with dedicated preprocessing (MRZ-based deskew, blackhat morphology, CLAHE).
- Document layout analysis (three modes): "Regex & Fuzzy" with line reconstruction, "DocTR Smart Layout" with the LW-DETR model (DocLayNet classes; only text regions are OCR'd and images/graphics are ignored) and "LayoutParser" with Detectron2.
  - Note: `layoutparser` cannot be installed on Windows (it requires `detectron2`, and Detectron2 is not published on PyPI and has no Windows wheel); for this reason the "DocTR Smart Layout" path was implemented, which requires only `python-doctr` (already installed) and also works on Windows.

**Recognition of identity information (PII) on OCR text**
- Checksum validation: credit cards (Luhn / ISO 7812), IBAN (mod-97), SSN (SSA assignment rules), Iranian national ID (checksum 11).
- Passport/identity-card MRZ (ICAO Doc 9303) and attribution of the extracted name to other parts of the page.
- Phone (Google libphonenumber + NANP/Iranian patterns), email, address, postal code, IP, URL, date of birth, password/API key (with entropy assessment).
- Person names: contextual rules (Dear/Mr./Name:/Sincerely) + optional BERT NER model (dslim/bert-base-NER).
- Normalization of Persian/Arabic digits and correction of common OCR errors within numeric strings (`l→1`, `O→0`).

**Imajev reasoner (optional)**
- Scene routing: separating documents/forms from street scenes in order to skip pointless OCR.
- Validation of low-confidence boxes with a typed decision and an explicit uncertainty (Unknown) probability.
- **fail-open** policy: if the validator fails, the element is not removed; the mistaken removal of a real face would amount to disclosing it.
- If a local Imajev server is running (`IMAJEV_ENDPOINT`), inference is delegated to it; otherwise the internal heuristic engine answers.

**Explainable reasoning (XAI) and privacy governance**
- **Cross-zone VIZ ↔ MRZ consistency verification:** on identity documents, the printed fields of the "Visual Inspection Zone" are matched against the "Machine Readable Zone" (ICAO Doc 9303); every field (person name, document number, date) is assessed separately, TD2/TD3 **check digits** are recomputed with the 7-3-1 weights, and the result is recorded in the Imajev VLM Terminal as a `CROSS-CHECK` event with a two-class distribution (`VIZ ↔ MRZ Consistent` / `Tampered / Mismatch`). A mismatch or an invalid check digit = flagging for manual review.
- **Bi-directional interactive spotlight:** every terminal record that relates to a real element carries its identifier and box (`detection_id`/`detection_ids` + `bbox`). Holding the pointer over a card illuminates the corresponding box on the Canvas with a pulsing turquoise ring (masked) or amber ring (unmasked); the reverse path also works: holding the pointer over a box or HUD label makes the corresponding terminal card flash and, if necessary, scroll smoothly into view. Clicking a card toggles the mask of those same elements directly from the terminal.
- **Live re-identification risk meter:** the dashboard above the terminal ("Re-Identification Risk Meter") shows two numbers live: **Initial Risk** (the sum of the weights of the identified identifiers: `MRZ` 35, face 28, `ID_NUMBER`/`SSN`/card/IBAN 25, person name 20, date/phone/email/address 12 — capped at 99%) and **Residual Risk** (computed only from **unmasked** elements plus the DP-Pix differential-privacy leakage floor of $1/\varepsilon$). The status label reads *Low Risk / Safe*, *Moderate Risk*, *Critical Exposure* or *No Data*, and the coloured bar transitions between emerald/amber/red at the 25% and 60% thresholds, being recomputed on every click, on manual drawing or on "Auto-Redact All"/"Deselect All".
- **Adaptive privacy policy (3 modes):** "Balanced (eps = 2.0) - Standard Thresholds" ($\varepsilon = 2.0$, `block=12`, `sigma=18`), "Strict Zero-Trust (eps = 0.5) - Auto-Mask All & High DP Noise" ($\varepsilon = 0.5$: automatic masking of all identifiers, larger macropixels and more noise, confidence threshold lowered by 0.10) and "Fast Audit - Highlight High-Risk Identifiers" (highlighting high-risk identifiers for manual review).
- **Zero-PII audit certificate:** the "Export Audit JSON" button produces a `privacy-audit-report.json` file containing the ISO 8601 timestamp, the policy mode and the DP-Pix parameters, the image dimensions, the risk assessment, the masking summary, the list of detection records (geometry/score/mask status), **salted SHA-256 fingerprints (ten hex characters) instead of raw text** and the complete Imajev reasoning trace. The certificate is constructed and downloaded entirely in the browser.

**User interface**
- Interactive masking: clicking the box or the label of any element (ReGenHuman human-in-the-loop).
- **Manual region drawing:** dragging the mouse over empty image space creates a live selection rectangle (turquoise/amber dashed outline); custom regions (such as a handwritten signature or a stamp) are added as `CUSTOM` elements with masking enabled and are removed with the × on the label; these regions are also preserved across re-analysis.
- Face/plate filter: **DP-Pix** (block averaging + grid-level Gaussian noise before upsampling); text/manual regions: **zero-entropy solid black block**.
- Labels and HUD guide lines are recalculated precisely via `ResizeObserver` during layout animations (opening/closing the terminal panel, window resizing).
- Before/after comparison by holding the "Hold to Compare" button or the `Space` key; "Auto-Redact All"; group "Select All"/"Deselect All".
- PNG export at the original resolution, without guide lines, coloured boxes or labels.
- English LTR interface (this release replaces the earlier fully RTL Persian interface); Tailwind and the Vazirmatn/Inter/JetBrains Mono fonts are served **locally** (no internet connection required).

## 3. Architecture and Data Flow

```
┌────────────────────┐   multipart    ┌────────────────────────────────────────────┐
│      Browser       │ ──────────────▶│  Django: /api/analyze/                     │
│      (Canvas)      │                │  ┌──────────────────────────────────────┐  │
│                    │                │  │ ml_pipeline.py                       │  │
│ 1. Upload          │                │  │  YOLO face ──┐                       │  │
│ 5. Render boxes    │                │  │  YOLO plate ─┤ validation cascade    │  │
│ 6. Mask            │                │  │  OCR text ───┘                       │  │
│ 7. Export PNG      │                │  │  pii.py  → PII spans                 │  │
│                    │ ◀──────────────│  │  Imajev  → typed decision            │  │
└────────────────────┘  JSON geometry │  └──────────────────────────────────────┘  │
                                      └────────────────────────────────────────────┘
```

1. The image is read **in memory only** (`FILE_UPLOAD_MAX_MEMORY_SIZE`) and no file is written to disk.
2. The detection pipeline returns the box and the polygon of every element in the original image coordinates.
3. The browser draws the boxes as an interactive HUD (SVG guide line + HTML label).
4. The final masking is performed on the Canvas itself and the output is downloaded directly.

```
anonymizer/
├── ml_pipeline.py   # detection pipeline (YOLO + OCR + Imajev + span→polygon mapping)
├── pii.py           # PII recognizers with checksum and contextual validation
├── filters.py       # server-side filters: DP-Pix, solid black block, heavy blur
├── views.py         # main page + JSON API
└── tests.py         # 155 unit/integration tests (no model loading)
config/              # Django configuration (settings/urls/wsgi)
templates/anonymizer/index.html
static/{css,js,fonts,vendor}/
weights/             # model weights (not in git; downloaded on first run)
```

## 4. Installation and Setup

Prerequisite: Python 3.10+ (development was carried out with 3.12).

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
python manage.py migrate          # only for the default Django database
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.

**First run:** the model weights are downloaded automatically (anonymization/plate: about 6 MB each, the EasyOCR models, and about 400 MB if NER is enabled). For offline simulation, place the files in `weights/` in advance.

**Optional engines:** if `paddleocr`/`docTR`/`LayoutParser` are not installed, the corresponding options are shown disabled ("not installed") in the user interface and the pipeline falls back to EasyOCR/Regex. The DocTR layout model downloads about 60 MB on first use and is cached in `~/.cache/doctr` (Linux/macOS: a similar path). (Configuring `layoutparser` requires `detectron2`, which has no Windows wheel; on Windows use "DocTR Smart Layout".)

## 5. Usage

1. Upload the image with the "Upload Image" button or by dragging and dropping it onto the work area.
2. Select the OCR engine and the layout analysis method; changing either selection triggers **automatic re-analysis**.
3. If needed, enable the "Imajev Smart Reasoner" toggle so that the terminal panel displays decisions live.
4. The suggested boxes appear on the image; clicking any box or its label toggles its mask on or off.
5. With "Auto-Redact All" or "Select All" all elements can be masked at once.
6. For review, hold the "Hold to Compare" button (or the `Space` key).
7. "Export Redacted PNG" downloads a PNG at the original resolution with only the active masks.

**Working with the explainable-reasoning layer (the "Imajev Smart Reasoner" toggle):**

8. The Imajev VLM Terminal (Bayesian Visual & Contextual Verifier) opens on the right; set the "Privacy Policy" selector to `Balanced`, `Strict Zero-Trust` or `Fast Audit`. In the strict mode all identifiers are masked immediately and the DP-Pix noise is intensified; in the audit mode, high-risk identifiers (risk ≥ 25%) are highlighted with an amber ring for manual review.
9. The "Re-Identification Risk Meter" dashboard at the top of the terminal shows the initial risk and the residual risk live; every click on a box, manual region drawing, click on a terminal card or "Auto-Redact All"/"Deselect All" immediately updates the bar and the status label.
10. Terminal cards linked to an element become interactive with the pointer: holding the pointer → a spotlight on the corresponding box; the reverse path from the box/HUD label → the corresponding terminal card flashes; and a click → toggling the mask of those same elements.
11. On identity documents, the `[CROSS-CHECK]` card reports the result of matching the printed zone against the machine-readable strip (together with the two-class distribution and the semantic consistency).
12. "Export Audit JSON" downloads the `privacy-audit-report.json` file: geometry, scores, risk assessment and the complete reasoning trace — with hash fingerprints instead of any raw PII value.

## 6. API

`POST /api/analyze/` — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | input image (JPG/PNG/WEBP); maximum 30 MB |
| `conf_threshold` | float | detector confidence threshold (default 0.40, clamped to the range 0.05–0.95) |
| `ocr_engine` | str | `easyocr` \| `paddleocr` \| `doctr` |
| `layout_engine` | str | `regex` \| `doctr_layout` \| `layoutparser` |
| `use_imajev` | bool | `true` \| `false` |

Response (example):

```json
{
  "status": "success",
  "image_dimensions": {"width": 1200, "height": 800},
  "detections": [
    {"id": 1, "type": "face",  "bbox": [120, 90, 240, 210], "polygon": [[120,90],[240,90],[240,210],[120,210]], "score": 0.91},
    {"id": 2, "type": "text",  "entity": "CREDIT_CARD", "label": "Credit Card", "text": "4111 1111 1111 1111", "bbox": [...], "polygon": [...], "score": 0.94}
  ],
  "imajev_logs": [
    "{\"entity\": \"face\", \"action\": \"approved\", \"confidence\": 0.99, \"detection_id\": 1, \"detection_ids\": [1], \"bbox\": [120, 90, 240, 210], \"message\": \"...\"}",
    "{\"entity\": \"mrz\", \"stage\": \"CROSS-CHECK\", \"status\": \"APPROVED\", \"consistency\": 0.994, \"matched_fields\": [\"Person Name\", \"Passport / ID Number\"], \"distribution\": {\"VIZ ↔ MRZ Consistent\": 0.99, \"Tampered / Mismatch\": 0.01}, \"detection_ids\": [1, 2], \"message\": \"...\"}"
  ]
}
```

Note: `imajev_logs` is a list of JSON records (as strings). Any record whose decision was applied to a real element carries that element's `detection_id` / `detection_ids` and `bbox`, and these very fields establish the bi-directional terminal ↔ Canvas link (spotlight); global records (scene, summary, error) do not carry these fields.

Example invocation with curl (the CSRF token is read from the cookie):

```bash
curl -c cookies.txt http://127.0.0.1:8000/ -o /dev/null
TOKEN=$(grep csrftoken cookies.txt | awk '{print $7}')
curl -b cookies.txt -H "X-CSRFToken: $TOKEN" -F "image=@doc.jpg" http://127.0.0.1:8000/api/analyze/
```

## 7. Privacy and Security

- **In-memory processing:** the image is decoded with `np.frombuffer`/`cv2.imdecode` and no copy is written to disk; the `FILE_UPLOAD_MAX_MEMORY_SIZE` ceiling in the settings has been raised for this purpose (the Django default is 2.5 MB, and larger files are written temporarily to disk).
- **Client-side output:** masking is applied in the browser; the server does not even produce the redacted image.
- **CSRF protection** on the API, upload-size limit (HTTP 413), image-dimension limit (60 megapixels) and whitelist validation of the parameters.
- **No sensitive-data logging:** error messages return only the error type and the extracted text is never printed to the server console.
- **Zero-PII audit certificate:** in the audit report, every recovered value is represented only by a **salted SHA-256 fingerprint** (ten hex characters). The salt is generated **randomly per session**, is never exported or stored, and for this reason the hash of a name or a date cannot be reversed by a dictionary/rainbow-table attack. The geometry (boxes), the scores, the mask status and the Imajev reasoning trace remain in the certificate; the raw PII text does not. (The only exception: the terminal messages themselves, which for operator transparency show up to 24 sample characters — the very content the user has already seen on screen — and this is explicitly documented in the `privacy_guarantees` section of the certificate.)
- The `SECRET_KEY`, `DEBUG` and `ALLOWED_HOSTS` keys are read from environment variables (the defaults are for development only).

**Why is simple blur not used?** The Privacy Blur (arXiv:2512.16086) and Revelio (arXiv:2506.12344) papers show that Gaussian blur is invertible by diffusion models with an accuracy of up to 95.9%; for this reason the default for faces/plates is **DP-Pix** and for text a **zero-entropy solid black block**.

## 8. Scientific Grounding

| Paper | Application in the implementation |
|---|---|
| Explainability-Driven Incremental Image Anonymization (2025) | The DP-Pix formula: block averaging → grid-level Gaussian noise → `INTER_NEAREST` upsampling (every macropixel a single colour and resistant to reconstruction) in `filters.py` and on the client-side mask path |
| Privacy Blur (2025) and Revelio (2025) | Justification for not using simple blur as the default |
| RedactionBench (2026) | Requirement of a zero-entropy solid black block for sensitive text and tokens |
| ReGenHuman (2026) | Design of the interactive Human-in-the-Loop box review |
| ICAO Doc 9303 (8th edition) | TD1/TD2/TD3 structure, check digits with 7-3-1 weights and the basis of the VIZ ↔ MRZ consistency check (`_cross_zone_consistency_entry`) |

Full details in `RESEARCH_PAPERS.md`.

## 9. Tests

```bash
python manage.py test anonymizer -v 2        # 155 tests: PII, filters, pipeline, API, Imajev, DP-Pix, VIZ↔MRZ, XAI
python -m anonymizer.filters                 # self-test of the redaction filters
python -m anonymizer.ml_pipeline             # end-to-end self-test (requires models)
```

The tests deliberately load no model so that they run quickly and offline (the pipeline is mocked in the tests). Added coverage:

- `IcaoCheckDigitTests` — recomputation of the check digits on the sample page of ICAO Doc 9303 (the Appendix to Section 4) and detection of a single-digit change in the date of birth.
- `CrossZoneVerificationTests` — emission of the `CROSS-CHECK` event, simultaneous linking of the printed fields and the MRZ strip, and flagging for "review" when the printed zone disagrees with the MRZ.
- `ImajevDetectionLinkTests` — correctness of `detection_id`/`detection_ids`/`bbox` on the records, absence of a link for rejected candidates, and no leakage of internal references into the response.
- `XaiGovernanceUiTests` — presence of the governance controls (policy selector, risk meter, audit-certificate button) in the rendered page and bi-directional isolation of the meter rows.

## 10. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OCR_LANGS` | `en` | EasyOCR languages, e.g. `en,fa` |
| `ENABLE_NER` | `1` | Enable/disable the person-name NER model (`0` = off) |
| `PRELOAD_MODELS` | `1` | Preload the models on `runserver` (`0` = off) |
| `IMAJEV_ENDPOINT` | `http://127.0.0.1:8765/v1/systemone` | address of the optional Imajev server |
| `IMAJEV_TIMEOUT` | `2.0` | deadline for an inference request to the Imajev server (seconds) |
| `IMAJEV_REMOTE_COOLDOWN` | `120` | after a connection error, this many seconds are skipped before contacting the Imajev server again |
| `DOCTR_LAYOUT_ARCH` | `lw_detr_s` | docTR layout-model architecture (e.g. `lw_detr_m` for higher accuracy and greater latency) |
| `DJANGO_SECRET_KEY` / `DJANGO_DEBUG` / `DJANGO_ALLOWED_HOSTS` | development | deployment settings |
| `ANONYMIZER_LOG_LEVEL` | `INFO` | log level of the `anonymizer` component |

## 11. Limitations and Future Work

**Practical comparison of the OCR engines (on the project's test document, 8 lines of English/numeric text):**

| Engine | Lines read | Measured note |
|---|---|---|
| EasyOCR (default) | 8 of 8 | reads the phone parentheses, the card digit groups and the IP correctly |
| docTR (crnn_vgg16_bn) | 8 of 8 | scrambles formatted numbers: `(159) 357-8426` → `1 9) ( 357-8426` and `4111 1111 1111 1111` → `41 11 I - 11 11 1 1111`; accurate for plain English text (name, email, address) |

Conclusion: for documents containing a phone/card/IP, the combination **"DocTR Smart Layout" + the EasyOCR engine** is the best configuration (DocTR supplies the layout and EasyOCR reads the text). In the hybrid mode, the text detections are OCR'd on each block individually and the quality (for instance the completeness of an email and the confidence score) improves.

- Persian text recognition in documents depends on the OCR engines (EasyOCR with `OCR_LANGS=en,fa` or PaddleOCR); the default docTR recognizer is Latin-only.
- The address/date patterns are primarily designed for English documents; adding Persian patterns is the next step.
- The face/plate detection models are generic; for higher accuracy they can be fine-tuned on the target domain. The plate detector previously fired on ordinary document text ("Utility Bill", "Ref: 88294", "Billing Currency: USD ($)") with a confidence of 0.6 to 0.8; now every plate candidate is validated by OCR inside the box and on documents these spurious boxes are removed (the rule "only a real plate may cover the text inside it" is also preserved).
- Document reference numbers (Ref / Invoice / Order / Tracking / Serial) are no longer flagged as identity data; person-linked identifiers (Account / Client / Patient / Passport / Policy, etc.) remain sensitive as before.
- For real deployment: `DEBUG=0`, running under WSGI, `collectstatic` and serving the static files.

## 12. Licence and Acknowledgements

The code of this project was written for academic submission. The libraries used (Ultralytics YOLOv8, EasyOCR, PaddleOCR, docTR, transformers, phonenumbers) are subject to their own licences; the model weights are not included in the repository and are downloaded on first run.
