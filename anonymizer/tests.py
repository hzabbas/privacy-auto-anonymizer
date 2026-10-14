"""
Test suite for the Privacy Auto-Anonymizer.

Coverage:
  * pii.py      - checksum validators, pattern recognizers, multilingual
                  header/value pairing, MRZ parsing, context recognizers and
                  span merging.
  * filters.py  - the three redaction filters plus coordinate clamping.
  * ml_pipeline - OCR line reconstruction, span -> polygon mapping, plate text
                  validation, image decoding and the fail-open validation policy.
  * views.py    - the UI route, the JSON API contract (400/413/405/500) and the
                  engine fallbacks.

The suite is deliberately model-free: no YOLO / EasyOCR / NER weights are loaded
or downloaded, so `python manage.py test` stays fast and offline-friendly.
"""

import json
from io import BytesIO
from unittest import mock

import numpy as np
from django.test import SimpleTestCase
from django.urls import reverse

from . import filters, pii
from .ml_pipeline import (
    _dedupe_ocr_boxes,
    _dedupe_text_detections,
    _doctr_pages_to_ocr_results,
    _enhance_contrast,
    _group_lines,
    _layout_blocks_from_prediction,
    _layout_blocks_with_policy,
    _layout_region_blocks,
    _load_image,
    _OCRBox,
    _ocr_pii_detections,
    _passport_number_zone_spans,
    _plate_text_is_valid,
    _plate_text_looks_like_plate,
    _suppressed_by_plate,
    _validate_crop_with_imajev,
    analyze_image_entities,
    FACE_OPTIONS,
    get_engine_status,
    ImajevClassifier,
    ModelManager,
    PLATE_OPTIONS,
    SCENE_OPTIONS,
)

NL = chr(10)  # keeps multi-line sample documents readable without escapes


# --------------------------------------------------------------------------- #
# pii.py                                                                      #
# --------------------------------------------------------------------------- #

class ValidatorTests(SimpleTestCase):
    def test_luhn_accepts_known_cards(self):
        self.assertTrue(pii.luhn_valid("4111111111111111"))     # Visa test number
        self.assertTrue(pii.luhn_valid("5555555555554444"))     # Mastercard test number
        self.assertTrue(pii.luhn_valid("4111 1111 1111 1111"))  # spacing is ignored

    def test_luhn_rejects_bad_checksum_and_short_input(self):
        self.assertFalse(pii.luhn_valid("4111111111111112"))
        self.assertFalse(pii.luhn_valid("1234"))

    def test_iban_valid(self):
        self.assertTrue(pii.iban_valid("GB82 WEST 1234 5698 7654 32"))
        self.assertFalse(pii.iban_valid("GB82 WEST 1234 5698 7654 33"))
        self.assertFalse(pii.iban_valid("not an iban"))

    def test_ssn_rules(self):
        self.assertTrue(pii.ssn_valid("457", "21", "9021"))
        self.assertFalse(pii.ssn_valid("000", "12", "3456"))   # area 000
        self.assertFalse(pii.ssn_valid("666", "12", "3456"))   # area 666
        self.assertFalse(pii.ssn_valid("900", "12", "3456"))   # area starting with 9
        self.assertFalse(pii.ssn_valid("123", "00", "3456"))   # group 00
        self.assertFalse(pii.ssn_valid("123", "45", "0000"))   # serial 0000
        self.assertFalse(pii.ssn_valid("123", "45", "6789"))   # known invalid sample

    def test_iranian_national_id_checksum(self):
        self.assertTrue(pii.iranian_national_id_valid("0499370899"))
        self.assertFalse(pii.iranian_national_id_valid("0499370898"))   # wrong check digit
        self.assertFalse(pii.iranian_national_id_valid("1111111111"))   # all digits equal
        self.assertFalse(pii.iranian_national_id_valid("12345"))

    def test_shannon_entropy(self):
        self.assertEqual(pii.shannon_entropy(""), 0.0)
        self.assertEqual(pii.shannon_entropy("aaaa"), 0.0)
        self.assertGreater(pii.shannon_entropy("a8Fk2zQ1"), 2.0)


class PatternRecognizerTests(SimpleTestCase):
    def _spans(self, text, **kwargs):
        return pii.find_pii(text, **kwargs)

    def _entities(self, text, **kwargs):
        return {s.entity for s in self._spans(text, **kwargs)}

    def test_email_and_phone_and_card(self):
        text = "Contact anthony@caldwell.com or (159) 357-8426. Card 4111 1111 1111 1111"
        entities = self._entities(text)
        self.assertIn("EMAIL", entities)
        self.assertIn("PHONE", entities)
        self.assertIn("CREDIT_CARD", entities)

    def test_spans_point_at_the_original_text(self):
        text = "Card: 4111 1111 1111 1111 end"
        span = next(s for s in self._spans(text) if s.entity == "CREDIT_CARD")
        self.assertEqual(text[span.start:span.end], "4111 1111 1111 1111")

    def test_persian_digits_are_normalized_without_breaking_offsets(self):
        # 10-digit Iranian national ID written with Persian digits (valid checksum).
        text = "کد ملی: ۰۴۹۹۳۷۰۸۹۹"
        spans = self._spans(text)
        id_spans = [s for s in spans if s.entity == "ID_NUMBER"]
        self.assertTrue(id_spans, "Persian-digit national ID was not detected")
        self.assertEqual(text[id_spans[0].start:id_spans[0].end], "۰۴۹۹۳۷۰۸۹۹")

    def test_ocr_digit_confusion_inside_numbers(self):
        # A letter that visually reads as a digit, sitting between digits, must be
        # repaired before the checksum runs ("411l" -> "4111").
        self.assertTrue(pii.luhn_valid(pii._numeric_view("411l 1111 1111 1111")))
        # A plain word is never rewritten.
        self.assertEqual(pii._numeric_view("total amount"), "total amount")

    def test_label_value_secret_is_redacted_but_prose_is_not(self):
        self.assertIn("SECRET", self._entities("password: hunter2xyz"))
        self.assertNotIn("SECRET", self._entities("We maintain confidential records."))

    def test_document_reference_numbers_are_not_pii(self):
        # "Ref: 88294" identifies a document, not a person.
        for text in ("Ref: 88294", "Invoice No: 55-1024", "Order #: 99881", "Tracking: 1Z999AA10123456784"):
            self.assertNotIn("ID_NUMBER", self._entities(text), text)
        # Real account / identity identifiers stay sensitive.
        for text in ("Client Account ID: NILTEP-9812", "Passport No: 1234567890", "Patient ID: 4471902"):
            self.assertIn("ID_NUMBER", self._entities(text), text)

    def test_mrz_name_is_propagated_to_the_page(self):
        # ICAO Doc 9303 sample passport MRZ, followed by prose that mentions the
        # same surname. The MRZ parser must recognise the name and coreference it
        # onto the plain-text occurrence further down the page.
        text = NL.join([
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
            "Passenger ERIKSSON boarded the flight.",
            "",
        ])
        spans = self._spans(text)
        person_texts = [text[s.start:s.end] for s in spans if s.entity == "PERSON"]
        self.assertIn("ERIKSSON", person_texts)

    def test_labelled_name_context(self):
        self.assertIn("PERSON", self._entities("Patient: Jane Doe"))
        # A bare department word is not a person.
        self.assertNotIn("PERSON", self._entities("Department: Human Resources"))

    def test_ip_and_url(self):
        entities = self._entities("Host 192.168.10.24 see https://example.com/privacy")
        self.assertIn("IP_ADDRESS", entities)
        self.assertIn("URL", entities)

    def test_merge_prefers_higher_priority_entity(self):
        text = "4111 1111 1111 1111"
        spans = [s for s in self._spans(text) if s.entity in ("ID_NUMBER", "CREDIT_CARD")]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].entity, "CREDIT_CARD")

    def test_merge_trims_surrounding_punctuation(self):
        text = "email: jane@acme.com, please reply"
        span = next(s for s in self._spans(text) if s.entity == "EMAIL")
        self.assertEqual(text[span.start:span.end], "jane@acme.com")

    def test_empty_text(self):
        self.assertEqual(pii.find_pii(""), [])
        self.assertEqual(pii.find_pii("   "), [])

    def test_entity_whitelist(self):
        text = "anthony@caldwell.com 4111 1111 1111 1111"
        entities = self._entities(text, entities={"EMAIL"})
        self.assertEqual(entities, {"EMAIL"})

    def test_entity_label_fallback(self):
        self.assertEqual(pii.entity_label("CREDIT_CARD"), "Credit card")
        self.assertEqual(pii.entity_label("SOMETHING_NEW"), "Something New")


# --------------------------------------------------------------------------- #
# filters.py                                                                  #
# --------------------------------------------------------------------------- #

class FilterTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)

    def test_solid_black_box_zeroes_the_region(self):
        out = filters.apply_redaction(self.image, [20, 20, 80, 60], filter_type="solid_black", copy=True)
        self.assertTrue(np.all(out[20:60, 20:80] == 0))
        # Outside the box nothing changed.
        self.assertTrue(np.array_equal(out[0:10, 0:10], self.image[0:10, 0:10]))

    def test_dp_pix_changes_the_region_and_keeps_shape(self):
        np.random.seed(0)
        crop = self.image[10:50, 10:60]
        out = filters.dp_pix_filter(crop, pixel_size=8, noise_scale=10.0)
        self.assertEqual(out.shape, crop.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, crop))

    def test_dp_pix_without_noise_is_pure_pixelation(self):
        out = filters.dp_pix_filter(self.image[10:50, 10:60], pixel_size=16, noise_scale=0.0)
        # Each 16x16 cell must now be a single flat colour.
        self.assertEqual(out[0, 0].tolist(), out[8, 8].tolist())
        self.assertEqual(len(np.unique(out[0:16, 0:16].reshape(-1, 3), axis=0)), 1)

    def test_blur_changes_the_region(self):
        out = filters.apply_redaction(self.image, [10, 10, 90, 70], filter_type="blur", ksize=21, copy=True)
        self.assertFalse(np.array_equal(out[10:70, 10:90], self.image[10:70, 10:90]))

    def test_blur_handles_even_oversized_kernel(self):
        # Even kernel and a kernel larger than the crop must not crash.
        out = filters.heavy_gaussian_blur(self.image[:5, :5], ksize=100)
        self.assertEqual(out.shape, (5, 5, 3))

    def test_out_of_bounds_coordinates_are_clamped(self):
        out = filters.apply_redaction(self.image, [-50, -50, 10_000, 10_000], filter_type="solid_black", copy=True)
        self.assertTrue(np.all(out == 0))

    def test_inverted_box_is_a_no_op(self):
        out = filters.apply_redaction(self.image, [80, 60, 20, 20], filter_type="solid_black", copy=True)
        self.assertTrue(np.array_equal(out, self.image))

    def test_unknown_filter_raises(self):
        with self.assertRaises(ValueError):
            filters.apply_redaction(self.image, [0, 0, 10, 10], filter_type="magic")

    def test_grayscale_and_empty_inputs(self):
        gray = np.zeros((40, 40), dtype=np.uint8)
        out = filters.apply_redaction(gray, [0, 0, 20, 20], filter_type="solid_black", copy=True)
        self.assertTrue(np.all(out[0:20, 0:20] == 0))
        self.assertEqual(filters.dp_pix_filter(np.zeros((0, 0, 3), dtype=np.uint8)).size, 0)


# --------------------------------------------------------------------------- #
# ml_pipeline helpers                                                         #
# --------------------------------------------------------------------------- #

def _png_bytes(width=64, height=48, color=(200, 30, 30)):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class ImageLoadingTests(SimpleTestCase):
    def test_decodes_bytes_to_bgr(self):
        image = _load_image(_png_bytes(30, 20))
        self.assertEqual(image.shape, (20, 30, 3))

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            _load_image(b"this is not an image")

    def test_converts_grayscale_source_to_three_channels(self):
        from PIL import Image

        buf = BytesIO()
        Image.new("L", (10, 10), 128).save(buf, format="PNG")
        self.assertEqual(_load_image(buf.getvalue()).shape, (10, 10, 3))


class PlateTextTests(SimpleTestCase):
    def test_accepts_plate_like_strings(self):
        # An unreadable crop must be kept (fail open), not discarded.
        for text in ("B 11 HOY", "21B34522", "IR-24 991B77", ""):
            self.assertTrue(_plate_text_is_valid(text), text)

    def test_rejects_printed_words(self):
        for text in ("Anthony Caldwell", "confidential records maintained", "12345678901234567890"):
            self.assertFalse(_plate_text_is_valid(text), text)


class StrictPlateTextTests(SimpleTestCase):
    """Document-scan policy: the crop must actually read like a plate."""

    def test_accepts_real_plate_formats(self):
        for text in ("B 11 HOY", "21B34522", "IR-24 991B77", "12 ب 345", "۷۸۹ الف ۱۲۳"):
            self.assertTrue(_plate_text_looks_like_plate(text), text)

    def test_rejects_document_text_false_positives(self):
        # Measured on a utility bill: the plate model fired on all of these.
        for text in ("Utility Bill", "Ref: 88294", "Validity: Sep 15, 2026",
                     "Billing Currency: USD ($)", "State Utility Tax (6%)",
                     "Residential Utility Service - Standard Monthly", ""):
            self.assertFalse(_plate_text_looks_like_plate(text), text)

    def test_rejects_long_runs_and_pure_letters(self):
        self.assertFalse(_plate_text_looks_like_plate("ABCDEFGH"))       # no digit
        self.assertFalse(_plate_text_looks_like_plate("1234567890123"))  # 13 chars


class PreprocessTests(SimpleTestCase):
    """preprocess_for_ocr must survive degenerate inputs (it feeds every OCR pass)."""

    def test_uniform_image_does_not_crash(self):
        from .ml_pipeline import preprocess_for_ocr

        flat = np.full((150, 240, 3), 70, dtype=np.uint8)   # no gradients at all
        gray, mrz_crop, y_offset, m_inv = preprocess_for_ocr(flat)
        self.assertEqual(gray.shape[:2], (150, 240))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(mrz_crop.dtype, np.uint8)
        self.assertIsNone(m_inv)
        self.assertGreaterEqual(y_offset, 0)

    def test_white_and_black_pages(self):
        from .ml_pipeline import preprocess_for_ocr

        for value in (0, 255):
            page = np.full((120, 200, 3), value, dtype=np.uint8)
            gray, mrz_crop, _, _ = preprocess_for_ocr(page)
            self.assertEqual(gray.shape[:2], (120, 200))


class OcrLineGroupingTests(SimpleTestCase):
    def test_groups_words_into_lines(self):
        boxes = [
            _OCRBox([[0, 0], [40, 0], [40, 10], [0, 10]], "Name:", 0.9),
            _OCRBox([[50, 0], [110, 0], [110, 10], [50, 10]], "Jane", 0.9),
            _OCRBox([[0, 20], [60, 20], [60, 30], [0, 30]], "Surname:", 0.9),
        ]
        lines = _group_lines(boxes)
        self.assertEqual(len(lines), 2)
        self.assertEqual([b.text for b in lines[0]], ["Name:", "Jane"])

    def test_large_horizontal_gap_starts_a_new_segment(self):
        boxes = [
            _OCRBox([[0, 0], [40, 0], [40, 10], [0, 10]], "left", 0.9),
            _OCRBox([[600, 0], [660, 0], [660, 10], [600, 10]], "right", 0.9),
        ]
        self.assertEqual(len(_group_lines(boxes)), 2)


class OcrDetectionMappingTests(SimpleTestCase):
    """Feeds synthetic OCR results through the span -> polygon/bbox mapper."""

    def _run(self, text_boxes, width=400, height=200):
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            return _ocr_pii_detections(text_boxes, width, height)

    def test_maps_pii_span_to_polygon_within_bounds(self):
        boxes = [
            ([[10, 10], [200, 10], [200, 40], [10, 40]], "Card 4111 1111 1111 1111", 0.95),
        ]
        dets = self._run(boxes)
        self.assertTrue(dets)
        det = next(d for d in dets if d["entity"] == "CREDIT_CARD")
        self.assertEqual(det["type"], "text")
        self.assertEqual(det["label"], "Credit card")
        self.assertIn("polygon", det)
        self.assertEqual(len(det["polygon"]), 4)
        for x, y in det["polygon"]:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 400)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 200)
        x1, y1, x2, y2 = det["bbox"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertIn("4111 1111 1111 1111", det["text"])

    def test_low_confidence_noise_is_ignored(self):
        boxes = [([[0, 0], [50, 0], [50, 10], [0, 10]], "junk", 0.02)]
        self.assertEqual(self._run(boxes), [])

    def test_no_boxes(self):
        self.assertEqual(self._run([]), [])


class ImajevValidationPolicyTests(SimpleTestCase):
    def test_failure_keeps_the_detection_fail_open(self):
        crop = np.full((40, 40, 3), 120, dtype=np.uint8)
        with mock.patch(
            "anonymizer.ml_pipeline.ImajevClassifier.predict_typed",
            side_effect=RuntimeError("engine exploded"),
        ):
            result = _validate_crop_with_imajev(crop, "face", initial_score=0.55)
        self.assertEqual(result["action"], "error")          # not "rejected"
        self.assertEqual(result["detector_score"], 0.55)
        self.assertIn("message", result)

    def test_empty_crop_is_an_error_not_a_rejection(self):
        result = _validate_crop_with_imajev(np.zeros((0, 0, 3), dtype=np.uint8), "text")
        self.assertEqual(result["action"], "error")

    def test_normal_path_returns_the_full_contract(self):
        # A skin-tone patch stands in for a face crop; the built-in classifier
        # answers offline (the remote endpoint simply refuses the connection).
        from .ml_pipeline import ImajevClassifier

        with mock.patch.object(ImajevClassifier, "_query_remote_server", return_value=None):
            crop = np.full((60, 60, 3), (150, 170, 210), dtype=np.uint8)
            result = _validate_crop_with_imajev(crop, "face", initial_score=0.5)
        self.assertIn(result["action"], ("approved", "rejected"))
        self.assertIn("confidence", result)
        self.assertIn("unknown_prob", result)


class EngineStatusTests(SimpleTestCase):
    def test_reports_expected_keys(self):
        status = get_engine_status()
        self.assertEqual(
            set(status),
            {"easyocr", "paddleocr", "doctr", "layoutparser", "doctr_layout"},
        )
        self.assertTrue(all(isinstance(v, bool) for v in status.values()))

    def test_doctr_layout_availability_follows_doctr(self):
        status = get_engine_status()
        self.assertEqual(status["doctr_layout"], status["doctr"])


class LayoutBlockTests(SimpleTestCase):
    """The docTR layout stage is model-free-testable through its converter."""

    def _prediction(self):
        import numpy as np

        return {
            "class_names": ["Section-header", "Text", "Picture", "Text"],
            "boxes": np.array([
                [0.05, 0.09, 0.70, 0.15],     # title bar
                [0.05, 0.26, 0.42, 0.30],     # text line
                [0.50, 0.50, 0.90, 0.60],     # a picture -> must be dropped
                [0.05, 0.26, 0.06, 0.30],     # degenerate (1px wide) -> dropped
            ], dtype="float32"),
            "scores": [0.9, 0.8, 0.7, 0.6],
        }

    def test_converts_normalised_boxes_to_pixels_and_drops_non_text(self):
        blocks = _layout_blocks_from_prediction(self._prediction(), width=1000, height=500)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], [50, 45, 700, 75])
        self.assertEqual(blocks[1], [50, 130, 420, 150])

    def test_handles_empty_or_missing_boxes(self):
        self.assertEqual(_layout_blocks_from_prediction({}, 100, 100), [])
        self.assertEqual(_layout_blocks_from_prediction({"class_names": [], "boxes": []}, 100, 100), [])

    def test_out_of_range_boxes_are_clamped(self):
        import numpy as np

        pred = {"class_names": ["Text"], "boxes": np.array([[-0.5, -0.2, 1.4, 1.3]], dtype="float32")}
        self.assertEqual(_layout_blocks_from_prediction(pred, 200, 100), [[0, 0, 200, 100]])


class PlateSuppressionTests(SimpleTestCase):
    """A marginal plate box must not swallow the PII labels inside it."""

    TEXT = {"type": "text", "bbox": [117, 554, 319, 588], "entity": "IP_ADDRESS", "score": 0.79}

    def test_confident_plate_absorbs_its_own_characters(self):
        plate = {"type": "plate", "bbox": [55, 548, 335, 593], "score": 0.85}
        self.assertTrue(_suppressed_by_plate(self.TEXT, [plate]))

    def test_marginal_plate_keeps_the_text_label(self):
        # The measured false positive: plate score 0.51 over an IP address.
        plate = {"type": "plate", "bbox": [55, 548, 335, 593], "score": 0.51}
        self.assertFalse(_suppressed_by_plate(self.TEXT, [plate]))

    def test_unrelated_plate_does_not_suppress(self):
        plate = {"type": "plate", "bbox": [900, 40, 1050, 90], "score": 0.9}
        self.assertFalse(_suppressed_by_plate(self.TEXT, [plate]))

    def test_empty_plate_list(self):
        self.assertFalse(_suppressed_by_plate(self.TEXT, []))


class DedupeTests(SimpleTestCase):
    def test_duplicate_ocr_readings_are_collapsed_to_the_best_one(self):
        dets = [
            {"type": "text", "bbox": [10, 10, 200, 40], "text": "National ID: 0499370899", "score": 0.85},
            {"type": "text", "bbox": [12, 11, 202, 41], "text": "National ID: 0499370899", "score": 0.90},
            {"type": "text", "bbox": [10, 60, 200, 90], "text": "E-mail: jane@acme.com", "score": 0.80},
            {"type": "face", "bbox": [300, 10, 400, 110], "score": 0.90},
        ]
        out = _dedupe_text_detections(dets)
        self.assertEqual(len(out), 3)
        kept = next(d for d in out if d.get("text") == "National ID: 0499370899")
        self.assertEqual(kept["score"], 0.90)

    def test_different_texts_in_overlapping_boxes_are_both_kept(self):
        dets = [
            {"type": "text", "bbox": [0, 0, 100, 30], "text": "Phone: (159) 357-8426", "score": 0.8},
            {"type": "text", "bbox": [5, 5, 105, 35], "text": "Date of birth: 12/03/1988", "score": 0.8},
        ]
        self.assertEqual(len(_dedupe_text_detections(dets)), 2)


# --------------------------------------------------------------------------- #
# pii.py - US-passport data page (cross-line pairing, MRZ, dates)             #
# --------------------------------------------------------------------------- #

class PassportFieldPairingTests(SimpleTestCase):
    """
    Regression suite for the US-passport data-page layout: multilingual header
    rows, values on the row below, alphanumeric dates and both MRZ lines.
    """

    SAMPLE = NL.join([
        "Surname / Nom / Apellidos",
        "DE LA PAZ",
        "Given Names / Prénoms / Nombres",
        "MICHELLE",
        "Passport No. / No. du Passeport / No. de Pasaporte",
        "910239248",
        "Date of Birth / Date de naissance",
        "07 Aug 1999",
        "Date of Issue / Date de délivrance",
        "06 Feb 2008",
        "Date of Expiration / Date d'expiration",
        "05 Feb 2018",
        "P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<",
        "9102392482USA6401171F1812051900781200<129676",
    ])
    MRZ_1 = "P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<"
    MRZ_2 = "9102392482USA6401171F1812051900781200<129676"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.spans = pii.find_pii(cls.SAMPLE)

    def _texts(self, entity):
        return [self.SAMPLE[s.start:s.end] for s in self.spans if s.entity == entity]

    def test_header_rows_are_never_flagged(self):
        # "Surname / Nom / Apellidos" and friends are *labels*: no span may sit
        # on them - least of all a PERSON on "Nom / Apellidos".
        for header in ("Surname / Nom / Apellidos",
                       "Given Names / Prénoms / Nombres",
                       "Passport No. / No. du Passeport / No. de Pasaporte"):
            start = self.SAMPLE.index(header)
            end = start + len(header)
            overlapping = [s.entity for s in self.spans if s.start < end and s.end > start]
            self.assertEqual(overlapping, [], f"unexpected spans on header {header!r}")

    def test_all_caps_names_from_the_rows_below_their_labels(self):
        persons = self._texts("PERSON")
        self.assertIn("DE LA PAZ", persons)     # multi-part surname with particles
        self.assertIn("MICHELLE", persons)      # single given name, all caps

    def test_passport_number_on_the_row_below_its_label(self):
        self.assertIn("910239248", self._texts("ID_NUMBER"))

    def test_alphanumeric_dates(self):
        found = {(s.entity, self.SAMPLE[s.start:s.end]) for s in self.spans}
        self.assertIn(("DATE_OF_BIRTH", "07 Aug 1999"), found)
        self.assertIn(("DATE", "06 Feb 2008"), found)
        self.assertIn(("DATE", "05 Feb 2018"), found)

    def test_both_mrz_lines_are_redacted_in_full(self):
        mrz = {self.SAMPLE[s.start:s.end].strip() for s in self.spans if s.entity == "MRZ"}
        self.assertIn(self.MRZ_1, mrz)
        self.assertIn(self.MRZ_2, mrz)

    def test_header_only_text_produces_no_spans(self):
        header_only = NL.join([
            "Surname / Nom / Apellidos",
            "Given Names / Prénoms / Nombres",
            "Passport No. / No. du Passeport / No. de Pasaporte",
            "SEE PAGE",
        ])
        self.assertEqual(pii.find_pii(header_only), [])

    def test_decomposed_accents_still_classify_the_header(self):
        # OCR sometimes returns "Pre" + combining acute instead of "Pré".
        header = "Given Names / Pre" + chr(0x301) + "noms / Nombres"
        text = NL.join([header, "MICHELLE"])
        self.assertIn("PERSON", {s.entity for s in pii.find_pii(text)})

    def test_persian_field_labels(self):
        text = NL.join(["نام خانوادگی", "رضایی", "شماره پاسپورت", "D23145890"])
        found = {(s.entity, text[s.start:s.end]) for s in pii.find_pii(text)}
        self.assertIn(("PERSON", "رضایی"), found)
        self.assertIn(("ID_NUMBER", "D23145890"), found)

    def test_entity_labels_for_new_entities(self):
        self.assertEqual(pii.entity_label("MRZ"), "MRZ")
        self.assertEqual(pii.entity_label("DATE"), "Date")


class MrzRobustnessTests(SimpleTestCase):
    """The MRZ detector must survive real-world OCR noise (ICAO Doc 9303)."""

    def test_whitespace_between_characters_is_tolerated(self):
        spaced = "P < D E L A P A Z < < M I C H E L L E " + "< " * 20
        spans = [s for s in pii.find_pii(spaced) if s.entity == "MRZ"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spaced[spans[0].start:spans[0].end].strip(), spaced.strip())

    def test_misread_filler_runs_are_repaired(self):
        for line in (
            "P<DELAPAZKKMICHELLE" + "K" * 26,      # K read for "<"
            "P<DELAPAZ(LMICHELLE" + "L" * 26,      # "(" and L read for "<"
            "P<DELAPAZ(((MICHELLE" + "(" * 22,     # "(" read for "<"
        ):
            self.assertTrue(
                [s for s in pii.find_pii(line) if s.entity == "MRZ"], line
            )

    def test_single_letters_inside_names_are_not_repaired(self):
        self.assertEqual(pii._repair_mrz_fillers("UTOERIKSSON"), "UTOERIKSSON")
        self.assertEqual(pii._repair_mrz_fillers("JACK<<"), "JACK<<")
        self.assertEqual(pii._repair_mrz_fillers("(KK)"), "<<<<")

    def test_missing_country_code_still_links_to_the_printed_surname(self):
        text = NL.join([
            "Surname / Nom / Apellidos",
            "DE LA PAZ",
            "P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<",
        ])
        people = [text[s.start:s.end] for s in pii.find_pii(text) if s.entity == "PERSON"]
        self.assertIn("DE LA PAZ", people)

    def test_td1_three_row_mrz(self):
        text = NL.join([
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
        ])
        mrz = [s for s in pii.find_pii(text) if s.entity == "MRZ"]
        self.assertEqual(len(mrz), 3)

    def test_td1_name_row_is_propagated(self):
        text = NL.join([
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
            "Passenger ERIKSSON boarded the flight.",
        ])
        people = [text[s.start:s.end] for s in pii.find_pii(text) if s.entity == "PERSON"]
        self.assertIn("ERIKSSON", people)

    def test_mrz_span_covers_the_whole_line(self):
        line = "P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<"
        text = NL.join(["HEADER LINE", line])
        span = next(s for s in pii.find_pii(text) if s.entity == "MRZ")
        self.assertEqual(text[span.start:span.end], line)


class DateRecognizerTests(SimpleTestCase):
    """Standalone DATE recognition (passport / ID-card date formats)."""

    def _entities(self, text):
        return [(s.entity, text[s.start:s.end]) for s in pii.find_pii(text)]

    def test_day_month_year_forms(self):
        for text in ("Issued 07 Aug 1999", "Issued 5 Feb 2018",
                     "Issued 7 August 1999", "Issued 1 Sept 2020"):
            self.assertIn("DATE", [e for e, _ in self._entities(text)], text)

    def test_month_day_year_with_comma(self):
        self.assertIn("DATE", [e for e, _ in self._entities("Expires Aug 07, 1999")])

    def test_numeric_dates(self):
        self.assertIn("DATE", [e for e, _ in self._entities("Valid until 2024-03-15")])
        self.assertIn("DATE", [e for e, _ in self._entities("Valid until 15/03/2024")])

    def test_jalali_dates(self):
        entities = self._entities("تاریخ تولد 1402/03/15")
        self.assertTrue({"DATE", "DATE_OF_BIRTH"} & {e for e, _ in entities})

    def test_dob_label_upgrades_the_entity(self):
        entities = self._entities("Date of Birth: 07 Aug 1999")
        self.assertIn(("DATE_OF_BIRTH", "07 Aug 1999"), entities)

    def test_document_reference_numbers_stay_dates_free(self):
        self.assertNotIn("DATE", [e for e, _ in self._entities("Ref: 88294")])


# --------------------------------------------------------------------------- #
# ml_pipeline - layout policy, docTR reconstruction, passport zone, CLAHE     #
# --------------------------------------------------------------------------- #

class LayoutPictureBypassTests(SimpleTestCase):
    """ID cards / passports must never lose text to a Picture misclassification."""

    def test_picture_dominant_page_bypasses_the_exclusion(self):
        import numpy as np

        prediction = {
            "class_names": ["Picture", "Text", "Text"],
            "boxes": np.array([[0.05, 0.05, 0.95, 0.60],    # 54% card body
                               [0.05, 0.70, 0.40, 0.74],
                               [0.05, 0.80, 0.40, 0.84]], dtype="float32"),
        }
        blocks = _layout_blocks_with_policy([prediction], 1000, 1000)
        self.assertEqual(len(blocks), 3)                # the Picture region is kept
        self.assertEqual(blocks[0], [50, 50, 950, 600])

    def test_few_text_blocks_bypass_the_exclusion(self):
        import numpy as np

        prediction = {
            "class_names": ["Text", "Picture"],
            "boxes": np.array([[0.05, 0.05, 0.40, 0.09],
                               [0.50, 0.30, 0.70, 0.40]], dtype="float32"),
        }
        self.assertEqual(len(_layout_blocks_with_policy([prediction], 1000, 1000)), 2)

    def test_normal_page_still_drops_non_text_regions(self):
        import numpy as np

        prediction = {
            "class_names": ["Text", "Text", "Text", "Text", "Picture"],
            "boxes": np.array([[0.05, 0.05, 0.40, 0.09],
                               [0.05, 0.15, 0.40, 0.19],
                               [0.05, 0.25, 0.40, 0.29],
                               [0.05, 0.35, 0.40, 0.39],
                               [0.60, 0.60, 0.90, 0.90]], dtype="float32"),  # 10%
        }
        self.assertEqual(len(_layout_blocks_with_policy([prediction], 1000, 1000)), 4)

    def test_picture_area_ratio_is_reported(self):
        import numpy as np

        prediction = {"class_names": ["Picture"], "boxes": np.array([[0.0, 0.0, 0.5, 0.5]], dtype="float32")}
        blocks, ratio = _layout_region_blocks(prediction, 200, 200)
        self.assertEqual(blocks, [])
        self.assertAlmostEqual(ratio, 0.25, places=3)


class _StubWord:
    def __init__(self, value, geometry, confidence=0.9):
        self.value, self.geometry, self.confidence = value, geometry, confidence


class _StubLine:
    def __init__(self, words):
        self.words = words


class _StubBlock:
    def __init__(self, lines):
        self.lines = lines


class _StubPage:
    def __init__(self, blocks):
        self.blocks = blocks


class DoctrReconstructionTests(SimpleTestCase):
    """docTR word boxes must be rebuilt into lines before PII runs."""

    def _page(self):
        return _StubPage([
            _StubBlock([_StubLine([
                _StubWord("07", ((0.10, 0.10), (0.14, 0.13))),
                _StubWord("Aug", ((0.15, 0.10), (0.21, 0.13))),
                _StubWord("1999", ((0.22, 0.10), (0.30, 0.13))),
            ])]),
            _StubBlock([_StubLine([
                _StubWord("P<DELAPAZ<<MICHELLE" + "<" * 15,
                          ((0.05, 0.80), (0.90, 0.88))),
            ])]),
        ])

    def test_words_are_converted_with_their_own_geometry(self):
        results = _doctr_pages_to_ocr_results([self._page()], 750, 1000)
        self.assertEqual(
            [t for _, t, _ in results],
            ["07", "Aug", "1999", "P<DELAPAZ<<MICHELLE" + "<" * 15],
        )
        polygon = results[0][0]
        self.assertAlmostEqual(polygon[0][0], 75.0, places=3)
        self.assertAlmostEqual(polygon[0][1], 100.0, places=3)
        self.assertAlmostEqual(polygon[2][0], 105.0, places=3)
        self.assertAlmostEqual(polygon[2][1], 130.0, places=3)

    def test_isolated_words_are_recombined_into_one_line(self):
        results = _doctr_pages_to_ocr_results([self._page()], 750, 1000)
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(results, 750, 1000)

        date = next(d for d in detections if d["entity"] == "DATE")
        self.assertEqual(date["text"], "07 Aug 1999")     # never "07", "Aug", "1999"
        self.assertIn("MRZ", {d["entity"] for d in detections})


class PassportNumberZoneTests(SimpleTestCase):
    """Upper-right passport corner fallback for the document number."""

    def _lines(self):
        header = [_OCRBox([[10, 10], [300, 10], [300, 40], [10, 40]],
                          "Passport No. / No. du Passeport", 0.9)]
        number = [_OCRBox([[320, 10], [420, 10], [420, 40], [320, 40]],
                          "910239248", 0.9)]
        return [header, number]

    @staticmethod
    def _bookkeep(lines):
        pos = 0
        for line in lines:
            for index, box in enumerate(line):
                if index:
                    pos += 1
                box.start = pos
                pos += len(box.text)
                box.end = pos
            pos += 1

    def test_zone_number_detected_on_a_passport_page(self):
        lines = self._lines()
        self._bookkeep(lines)
        spans = _passport_number_zone_spans(lines, [], 500, 200)
        self.assertEqual([s.entity for s in spans], ["ID_NUMBER"])

    def test_zone_fallback_needs_passport_evidence(self):
        lines = self._lines()
        lines[0] = [_OCRBox([[10, 10], [200, 10], [200, 40], [10, 40]], "Invoice", 0.9)]
        self._bookkeep(lines)
        self.assertEqual(_passport_number_zone_spans(lines, [], 500, 200), [])

    def test_lower_left_numbers_are_not_in_the_passport_zone(self):
        lines = self._lines()
        lines[1] = [_OCRBox([[10, 180], [110, 180], [110, 210], [10, 210]],
                            "910239248", 0.9)]
        self._bookkeep(lines)
        self.assertEqual(_passport_number_zone_spans(lines, [], 500, 220), [])

    def test_zone_spans_skip_regions_already_covered(self):
        lines = self._lines()
        self._bookkeep(lines)
        covered = [pii.PIISpan(lines[1][0].start, lines[1][0].end, "ID_NUMBER", 0.9)]
        self.assertEqual(_passport_number_zone_spans(lines, covered, 500, 200), [])

    def test_zone_maps_to_a_detection(self):
        lines = self._lines()
        self._bookkeep(lines)
        results = [(box.pts, box.text, box.conf) for line in lines for box in line]
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(results, 500, 200)
        number = next(d for d in detections if d["entity"] == "ID_NUMBER")
        self.assertGreater(number["bbox"][0], 250)


class ContrastEnhancementTests(SimpleTestCase):
    """preprocess_for_ocr enhances contrast gently - it never binarises."""

    def test_low_contrast_print_gains_local_contrast(self):
        image = np.full((60, 80, 3), 100, dtype=np.uint8)
        image[20:40, 20:60] = 104
        enhanced = _enhance_contrast(image)
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(float(np.std(enhanced)), float(np.std(image)))

    def test_uniform_frame_stays_calm(self):
        flat = np.full((40, 50, 3), 90, dtype=np.uint8)
        self.assertEqual(int(np.std(_enhance_contrast(flat))), 0)

    def test_grayscale_input_is_passed_through(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        self.assertIs(_enhance_contrast(gray), gray)


class OcrBoxDedupeTests(SimpleTestCase):
    """Full-frame + MRZ-band double reads must collapse, not merge into one line."""

    def test_identical_readings_of_the_same_box_collapse(self):
        boxes = [
            _OCRBox([[10, 10], [200, 10], [200, 40], [10, 40]], "07 Aug 1999", 0.7),
            _OCRBox([[11, 11], [201, 11], [201, 41], [11, 41]], "07 Aug 1999", 0.95),
        ]
        kept = _dedupe_ocr_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].conf, 0.95)

    def test_different_readings_on_the_same_spot_are_both_kept(self):
        boxes = [
            _OCRBox([[10, 10], [200, 10], [200, 40], [10, 40]], "910239248", 0.9),
            _OCRBox([[11, 11], [201, 11], [201, 41], [11, 41]], "910239249", 0.9),
        ]
        self.assertEqual(len(_dedupe_ocr_boxes(boxes)), 2)

    def test_near_identical_mrz_readings_still_collapse(self):
        # Measured on a real run: the two reads of a 44-char MRZ line differed
        # by one inserted digit and used to be joined into a doubled line.
        boxes = [
            _OCRBox([[10, 10], [500, 10], [500, 40], [10, 40]],
                    "9102392482USA640117171F1812051900781200<129676", 0.8),
            _OCRBox([[11, 11], [501, 11], [501, 41], [11, 41]],
                    "9102392482USA6401171F1812051900781200<129676", 0.95),
        ]
        kept = _dedupe_ocr_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].conf, 0.95)

    def test_truncated_reading_defers_to_the_more_complete_one(self):
        # A band read cut the MRZ filler tail off; the fuller frame read must
        # win even though the truncation scored a higher confidence.
        short_text = "P<DELAPAZ< <MICHELLE<<<<<"
        long_text = "P<DELAPAZ< <MICHELLE" + "<" * 25
        boxes = [
            _OCRBox([[10, 10], [300, 10], [300, 40], [10, 40]], short_text, 0.95),
            _OCRBox([[11, 11], [460, 11], [460, 41], [11, 41]], long_text, 0.7),
        ]
        kept = _dedupe_ocr_boxes(boxes)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].text, long_text)

    def test_doubled_words_no_longer_break_date_detection(self):
        results = []
        for text, x in (("07", 70), ("Aug", 120), ("1999", 170)):
            results.append(([[x, 100], [x + 40, 100], [x + 40, 130], [x, 130]], text, 0.8))
            results.append(([[x + 1, 101], [x + 41, 101], [x + 41, 131], [x + 1, 131]], text, 0.9))
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(results, 500, 200)
        date = next(d for d in detections if d["entity"] == "DATE")
        self.assertEqual(date["text"], "07 Aug 1999")


# --------------------------------------------------------------------------- #
# views.py                                                                    #
# --------------------------------------------------------------------------- #

class IndexViewTests(SimpleTestCase):
    def test_index_renders_the_studio(self):
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "editor-canvas")
        self.assertContains(response, "imageUploadInput")
        self.assertIn("engine_status", response.context)

    def test_index_sets_the_csrf_cookie(self):
        response = self.client.get(reverse("index"))
        self.assertIn("csrftoken", response.cookies)


class AnalyzeApiTests(SimpleTestCase):
    url_name = "analyze_image_api"

    def test_get_is_not_allowed(self):
        self.assertEqual(self.client.get(reverse(self.url_name)).status_code, 405)

    def test_missing_file_is_rejected(self):
        response = self.client.post(reverse(self.url_name), data={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_empty_file_is_rejected(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("empty.png", b"", content_type="image/png")
        response = self.client.post(reverse(self.url_name), data={"image": upload})
        self.assertEqual(response.status_code, 400)

    def test_undecodable_file_is_rejected_with_400(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("x.png", b"not-an-image-at-all", content_type="image/png")
        response = self.client.post(reverse(self.url_name), data={"image": upload})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unable to decode", response.json()["message"])

    def test_oversized_upload_is_rejected_with_413(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("big.png", b"x" * 2048, content_type="image/png")
        with mock.patch("anonymizer.views.MAX_UPLOAD_SIZE", 1024):
            response = self.client.post(reverse(self.url_name), data={"image": upload})
        self.assertEqual(response.status_code, 413)

    def test_successful_analysis_shape_and_engine_whitelist(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        fake = {
            "status": "success",
            "image_dimensions": {"width": 10, "height": 10},
            "detections": [],
            "imajev_logs": [],
        }
        upload = SimpleUploadedFile("ok.png", _png_bytes(10, 10), content_type="image/png")

        with mock.patch("anonymizer.views.analyze_image_entities", return_value=fake) as pipeline:
            response = self.client.post(reverse(self.url_name), data={
                "image": upload,
                "conf_threshold": "0.5",
                "ocr_engine": "definitely-not-an-engine",
                "layout_engine": "nope",
                "use_imajev": "true",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fake)
        _, kwargs = pipeline.call_args
        self.assertEqual(kwargs["ocr_engine"], "easyocr")      # unknown engine -> fallback
        self.assertEqual(kwargs["layout_engine"], "regex")     # unknown engine -> fallback
        self.assertTrue(kwargs["use_imajev"])
        self.assertAlmostEqual(kwargs["conf_threshold"], 0.5)

    def test_known_layout_engine_is_forwarded(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        fake = {
            "status": "success",
            "image_dimensions": {"width": 10, "height": 10},
            "detections": [],
            "imajev_logs": [],
        }
        upload = SimpleUploadedFile("ok.png", _png_bytes(10, 10), content_type="image/png")
        with mock.patch("anonymizer.views.analyze_image_entities", return_value=fake) as pipeline:
            self.client.post(reverse(self.url_name), data={
                "image": upload,
                "layout_engine": "doctr_layout",
                "ocr_engine": "paddleocr",
            })
        _, kwargs = pipeline.call_args
        self.assertEqual(kwargs["layout_engine"], "doctr_layout")
        self.assertEqual(kwargs["ocr_engine"], "paddleocr")

    def test_conf_threshold_is_clamped(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        fake = {
            "status": "success",
            "image_dimensions": {"width": 10, "height": 10},
            "detections": [],
            "imajev_logs": [],
        }
        upload = SimpleUploadedFile("ok.png", _png_bytes(10, 10), content_type="image/png")
        with mock.patch("anonymizer.views.analyze_image_entities", return_value=fake) as pipeline:
            self.client.post(reverse(self.url_name), data={"image": upload, "conf_threshold": "42"})
        _, kwargs = pipeline.call_args
        self.assertEqual(kwargs["conf_threshold"], 0.95)

    def test_pipeline_value_error_becomes_400(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("ok.png", _png_bytes(10, 10), content_type="image/png")
        with mock.patch(
            "anonymizer.views.analyze_image_entities",
            side_effect=ValueError("Image is too large"),
        ):
            response = self.client.post(reverse(self.url_name), data={"image": upload})
        self.assertEqual(response.status_code, 400)

    def test_unexpected_pipeline_error_becomes_500_json(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("ok.png", _png_bytes(10, 10), content_type="image/png")
        with mock.patch(
            "anonymizer.views.analyze_image_entities",
            side_effect=RuntimeError("boom"),
        ):
            response = self.client.post(reverse(self.url_name), data={"image": upload})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["status"], "error")


class ImageSizeGuardTests(SimpleTestCase):
    def test_oversized_pixel_count_is_rejected(self):
        from .ml_pipeline import analyze_image_entities

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        with mock.patch("anonymizer.ml_pipeline.MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(ValueError):
                analyze_image_entities(image)


# --------------------------------------------------------------------------- #
# Bilingual passport dates (pii.py)                                           #
# --------------------------------------------------------------------------- #

class BilingualPassportDateTests(SimpleTestCase):
    """Canadian style "DD MON /MON YY" dates must read as DATE (raised to
    DATE_OF_BIRTH when they sit under a birth label)."""

    SAMPLES = ("01 JAN /JAN 85", "14 JAN /JAN 13", "14 JAN /JAN 23", "05 FEV /FEB 24")

    # OCR routinely reads the thin slash as I / l / 1 / | ("14 JAN IJAN 23").
    MISREAD_SAMPLES = ("14 JAN IJAN 23", "14 JAN lJAN 13", "14 JAN |JAN 23")

    def test_each_bilingual_shape_is_detected(self):
        for sample in self.SAMPLES:
            spans = {(s.entity, sample[s.start:s.end]) for s in pii.find_pii(sample)}
            self.assertIn(("DATE", sample), spans, sample)

    def test_slash_misread_as_i_l_pipe_is_still_a_date(self):
        for sample in self.MISREAD_SAMPLES:
            spans = {(s.entity, sample[s.start:s.end]) for s in pii.find_pii(sample)}
            self.assertIn(("DATE", sample), spans, sample)

    def test_bilingual_value_under_a_birth_label_is_a_dob(self):
        text = NL.join(["Date of Birth / Date de naissance", "14 JAN /JAN 13"])
        spans = {(s.entity, text[s.start:s.end]) for s in pii.find_pii(text)}
        self.assertIn(("DATE_OF_BIRTH", "14 JAN /JAN 13"), spans)

    def test_bilingual_value_under_an_issue_header_is_a_date(self):
        text = NL.join(["Date de délivrance / Date of issue", "01 JAN /JAN 85"])
        spans = {(s.entity, text[s.start:s.end]) for s in pii.find_pii(text)}
        self.assertIn(("DATE", "01 JAN /JAN 85"), spans)

    def test_fecha_de_nacimiento_header_pairs_its_value(self):
        text = NL.join(["Fecha de nacimiento", "07 Aug 1999"])
        spans = {(s.entity, text[s.start:s.end]) for s in pii.find_pii(text)}
        self.assertIn(("DATE_OF_BIRTH", "07 Aug 1999"), spans)

    def test_non_month_words_never_match_the_bilingual_shape(self):
        entities = {s.entity for s in pii.find_pii("10 USD /EUR 20")}
        self.assertNotIn("DATE", entities)


# --------------------------------------------------------------------------- #
# Imajev scene prior (ml_pipeline.py)                                         #
# --------------------------------------------------------------------------- #

class ImajevScenePriorTests(SimpleTestCase):
    """Document-scene calibration: plate-shaped *text* must read as printed
    text (>= 85%), never as a licence plate (< 15%)."""

    @staticmethod
    def _text_crop():
        # White crop with letter-like dark blobs: the shape of a "plate
        # candidate" that in reality is a document text row (passport number).
        crop = np.full((40, 160, 3), 245, dtype=np.uint8)
        for index in range(6):
            x = 12 + index * 24
            crop[10:30, x:x + 12] = 35
        return crop

    def _verdict(self, state):
        with mock.patch.object(ImajevClassifier, "_query_remote_server", return_value=None):
            return ImajevClassifier().predict_typed(
                self._text_crop(), options=list(PLATE_OPTIONS), state=state,
            )

    def test_document_scene_reads_text_not_a_plate(self):
        scores = self._verdict({"scene": "document"})["scores"]
        self.assertGreaterEqual(scores["Printed Text"], 0.85)
        self.assertLessEqual(scores["License Plate"], 0.15)

    def test_without_the_prior_the_shape_alone_still_reads_plate_like(self):
        scores = self._verdict(None)["scores"]
        self.assertGreater(scores["License Plate"], 0.5)


# --------------------------------------------------------------------------- #
# Imajev log consistency on identity documents (ml_pipeline.py)               #
# --------------------------------------------------------------------------- #

class _StubTensorValue:
    """Mimics the Ultralytics tensor access pattern (box.conf[0].item())."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return self._value


class _StubBox:
    def __init__(self, xyxy, conf):
        self.conf = [_StubTensorValue(conf)]
        self.xyxy = [_StubTensorValue(xyxy)]


class _StubYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubYolo:
    """Stands in for a YOLO model: returns fixed boxes, ignores kwargs."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, **kwargs):
        return [_StubYoloResult(self._boxes)]


class _StubOcrReader:
    """Answers full-frame / MRZ-band / plate-crop reads by shape."""

    def __init__(self, frame_results, frame_shape,
                 plate_ok="B 11 HOY", plate_bad="Anthony Caldwell"):
        self.frame_results = frame_results
        self.frame_w, self.frame_h = frame_shape
        self.plate_ok = plate_ok
        self.plate_bad = plate_bad
        self.frame_reads = 0

    def readtext(self, image, **kwargs):
        height, width = image.shape[:2]
        if width >= self.frame_w * 0.9 and height >= self.frame_h * 0.5:
            self.frame_reads += 1
            return list(self.frame_results) if self.frame_reads == 1 else []
        if width >= self.frame_w * 0.9:
            return []                                    # bottom MRZ-band read
        text = self.plate_ok if width >= 220 else self.plate_bad
        return [([[0, 0], [width, 0], [width, height], [0, height]], text, 0.9)]


def _run_imajev_pipeline(frame, face_boxes, plate_boxes, ocr_reader):
    """analyze_image_entities() with stubbed detectors and the offline Imajev."""
    face_model = None if face_boxes is None else _StubYolo(face_boxes)
    plate_model = None if plate_boxes is None else _StubYolo(plate_boxes)
    face_patch = mock.patch.object(ModelManager, "face_model", new_callable=mock.PropertyMock, return_value=face_model)
    plate_patch = mock.patch.object(ModelManager, "plate_model", new_callable=mock.PropertyMock, return_value=plate_model)
    ocr_patch = mock.patch.object(ModelManager, "ocr_reader", new_callable=mock.PropertyMock, return_value=ocr_reader)
    with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False), \
            mock.patch.object(ImajevClassifier, "_query_remote_server", return_value=None), \
            face_patch, plate_patch, ocr_patch:
        return analyze_image_entities(frame, use_imajev=True)


def _log_entries(result):
    return [json.loads(entry) for entry in result["imajev_logs"]]


def _passport_frame(width=900, height=560):
    """A synthetic scanned passport page: white paper, text rows, photo, MRZ."""
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    for row_y in (60, 96, 132, 168, 204, 240, 276, 312):
        for index in range(10):
            x = 30 + index * 34
            frame[row_y:row_y + 24, x:x + 18] = 40
    photo = np.full((140, 120, 3), (120, 140, 180), dtype=np.uint8)
    photo[:, 12::24] = (170, 190, 225)                   # skin-tone stripes
    frame[60:200, 700:820] = photo
    frame[420:452, 20:620] = 50                          # MRZ rows
    frame[470:502, 20:680] = 50
    return frame


def _street_frame(width=640, height=420):
    """A synthetic photograph: colourful noise plus two plate-like panels."""
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, size=(height, width, 3)).astype(np.uint8)
    for x1, x2 in ((60, 300), (340, 540)):
        frame[300:350, x1:x2] = 55
        for index in range((x2 - x1) // 34):
            x = x1 + 16 + index * 30
            frame[312:338, x:x + 14] = 235
    return frame


class ImajevLogConsistencyTests(SimpleTestCase):
    """A passport must never produce the contradictory five-log PLATE storm:

    * no candidate receives both an [APPROVED] and a [REJECTED] entry;
    * plate false positives on a document scene are suppressed (one note only);
    * the terminal carries reasoning for the entities that were really found.
    """

    def _document_inputs(self):
        frame = _passport_frame()
        frame_results = [
            ([[30, 60], [420, 60], [420, 90], [30, 90]], "Surname / Nom / Apellidos", 0.95),
            ([[30, 96], [260, 96], [260, 126], [30, 126]], "MARTIN", 0.95),
            ([[30, 132], [420, 132], [420, 162], [30, 162]], "Given Names / Prénoms / Nombres", 0.9),
            ([[30, 168], [260, 168], [260, 198], [30, 198]], "SARAH", 0.95),
            ([[30, 204], [430, 204], [430, 234], [30, 234]], "Date of Birth / Date de naissance", 0.9),
            ([[30, 240], [220, 240], [220, 270], [30, 270]], "14 JAN /JAN 13", 0.9),
            ([[30, 276], [420, 276], [420, 306], [30, 306]], "Passport No. / No. du Passeport", 0.9),
            ([[30, 312], [200, 312], [200, 342], [30, 342]], "ZE000509", 0.9),
            ([[30, 420], [620, 420], [620, 452], [30, 452]],
             "P<CANMARTIN<<SARAH" + "<" * 25, 0.9),
            ([[30, 470], [680, 470], [680, 502], [30, 502]],
             "ZE0005098CAN9401131F2501184" + "<" * 12 + "04", 0.9),
        ]
        return frame, _StubOcrReader(frame_results, (frame.shape[1], frame.shape[0]))

    def test_document_scene_suppresses_plate_false_positives(self):
        frame, reader = self._document_inputs()
        face_boxes = [_StubBox([700, 60, 820, 200], 0.92)]
        plate_boxes = [
            _StubBox([30, 60, 190, 100], 0.85),     # document text -> suppressed
            _StubBox([30, 420, 220, 460], 0.90),    # MRZ band -> suppressed
            _StubBox([500, 240, 640, 280], 0.60),   # below the raised document floor
        ]
        result = _run_imajev_pipeline(frame, face_boxes, plate_boxes, reader)
        logs = _log_entries(result)

        plate_entries = [entry for entry in logs if entry.get("entity") == "plate"]
        # Exactly one aggregate note: never a [APPROVED] + [REJECTED] pair.
        self.assertEqual([entry["action"] for entry in plate_entries], ["info"])
        self.assertIn("2", plate_entries[0]["message"])
        self.assertEqual([d for d in result["detections"] if d["type"] == "plate"], [])

        face_entries = [entry for entry in logs if entry.get("entity") == "face"]
        self.assertEqual([entry["action"] for entry in face_entries], ["approved"])
        self.assertTrue(any(d["type"] == "face" for d in result["detections"]))

        scene = next(entry for entry in logs if entry.get("entity") == "scene")
        self.assertEqual(scene["action"], "approved")
        self.assertEqual(set(scene["scores"]), set(SCENE_OPTIONS))

    def test_document_reports_reasoning_for_the_detected_identity_fields(self):
        frame, reader = self._document_inputs()
        result = _run_imajev_pipeline(frame, None, [], reader)
        logs = _log_entries(result)

        summaries = [entry for entry in logs if entry.get("entity") == "text"]
        previews = {entry.get("text_preview") for entry in summaries}
        self.assertIn("PERSON", previews)
        self.assertIn("MRZ", previews)
        self.assertIn("ID_NUMBER", previews)
        for entry in summaries:
            self.assertEqual(entry["action"], "approved")
            self.assertTrue(entry["message"])
            self.assertTrue(entry["scores"])

        entities = {d.get("entity") for d in result["detections"] if d["type"] == "text"}
        self.assertIn("PERSON", entities)
        self.assertIn("MRZ", entities)


class ImajevPlateDecisionTests(SimpleTestCase):
    """One plate candidate -> one combined decision -> at most one log."""

    def test_visual_approval_plus_ocr_rejection_is_a_single_rejected_log(self):
        frame = _street_frame()
        reader = _StubOcrReader([], (frame.shape[1], frame.shape[0]))
        plate_boxes = [
            _StubBox([60, 300, 300, 350], 0.90),     # reads "B 11 HOY" -> approved
            _StubBox([340, 300, 540, 350], 0.90),    # reads prose     -> rejected
        ]
        result = _run_imajev_pipeline(frame, None, plate_boxes, reader)
        logs = _log_entries(result)

        plate_entries = [entry for entry in logs if entry.get("entity") == "plate"]
        # Two candidates, two entries: the second one must NOT also carry the
        # [APPROVED] entry the old two-pass pipeline appended before its OCR
        # check rejected it (that alone produced three entries for two boxes).
        self.assertEqual(sorted(entry["action"] for entry in plate_entries),
                         ["approved", "rejected"])
        approved = next(entry for entry in plate_entries if entry["action"] == "approved")
        self.assertIn("confidence", approved)            # visual verdict carries scores
        rejected = next(entry for entry in plate_entries if entry["action"] == "rejected")
        self.assertIn("message", rejected)               # OCR verdict carries its reason

        self.assertEqual(len([d for d in result["detections"] if d["type"] == "plate"]), 1)

        scene = next(entry for entry in logs if entry.get("entity") == "scene")
        self.assertEqual(scene["action"], "rejected")   # street scene -> OCR skipped


# --------------------------------------------------------------------------- #
# Split-box bilingual dates (ml_pipeline.py)                                  #
# --------------------------------------------------------------------------- #

class SplitDateStitchingTests(SimpleTestCase):
    """Canadian passports print "14 JAN /JAN    13": OCR hands back two boxes."""

    @staticmethod
    def _results(year_text):
        return [
            ([[30, 300], [210, 300], [210, 330], [30, 330]], "14 JAN /JAN", 0.82),
            ([[330, 302], [370, 302], [370, 328], [330, 328]], year_text, 0.71),
        ]

    def test_wide_gap_split_date_is_one_date_detection(self):
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(self._results("13"), 900, 560)
        dates = [d for d in detections if d["entity"] == "DATE"]
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0]["text"], "14 JAN /JAN 13")
        # The union redaction must cover both halves (gap and watermark included).
        self.assertLessEqual(dates[0]["bbox"][0], 31)
        self.assertGreaterEqual(dates[0]["bbox"][2], 369)

    def test_second_bilingual_date_variant(self):
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(self._results("23"), 900, 560)
        dates = [d for d in detections if d["entity"] == "DATE"]
        self.assertEqual([d["text"] for d in dates], ["14 JAN /JAN 23"])

    def test_ocr_year_misreads_are_repaired_when_stitching(self):
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(self._results("I3"), 900, 560)
        dates = [d for d in detections if d["entity"] == "DATE"]
        self.assertEqual([d["text"] for d in dates], ["14 JAN /JAN 13"])

    def test_complete_single_box_year_misread_is_repaired(self):
        results = [([[30, 300], [240, 300], [240, 330], [30, 330]], "14 JAN /JAN I3", 0.8)]
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(results, 900, 560)
        dates = [d for d in detections if d["entity"] == "DATE"]
        self.assertEqual([d["text"] for d in dates], ["14 JAN /JAN 13"])

    def test_value_row_under_a_date_header_is_collected_across_split_lines(self):
        # The year sits far enough right that line grouping splits the row into
        # separate lines; only the header-aware row collection can join them.
        results = [
            ([[30, 260], [300, 260], [300, 290], [30, 290]], "Date of birth / Date de naissance", 0.9),
            ([[30, 300], [210, 300], [210, 330], [30, 330]], "14 JAN /JAN", 0.82),
            ([[700, 302], [740, 302], [740, 328], [700, 328]], "13", 0.71),
        ]
        with mock.patch("anonymizer.ml_pipeline.ENABLE_NER", False):
            detections = _ocr_pii_detections(results, 900, 560)
        dob = [d for d in detections if d["entity"] == "DATE_OF_BIRTH"]
        self.assertEqual([d["text"] for d in dob], ["14 JAN /JAN 13"])
        self.assertGreaterEqual(dob[0]["bbox"][2], 739)


# --------------------------------------------------------------------------- #
# DP-Pix grid-level noise (filters.py)                                        #
# --------------------------------------------------------------------------- #

class DpPixGridLevelTests(SimpleTestCase):
    """Noise is drawn per grid cell BEFORE the nearest-neighbour upsample, so
    every b x b macro-pixel ends up one uniform noisy colour."""

    @staticmethod
    def _image():
        rng = np.random.default_rng(9)
        return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    def test_macro_pixels_are_uniform_with_noise_enabled(self):
        np.random.seed(5)
        out = filters.dp_pix_filter(self._image(), pixel_size=16, noise_scale=10.0)
        cells = []
        for by in range(4):
            for bx in range(4):
                cell = out[by * 16:(by + 1) * 16, bx * 16:(bx + 1) * 16].reshape(-1, 3)
                self.assertEqual(len(np.unique(cell, axis=0)), 1)
                cells.append(tuple(out[by * 16, bx * 16]))
        # The grid cells themselves must differ: one noise draw per cell.
        self.assertGreater(len(set(cells)), 1)

    def test_noise_actually_changes_the_grid(self):
        np.random.seed(5)
        noisy = filters.dp_pix_filter(self._image(), pixel_size=16, noise_scale=10.0)
        clean = filters.dp_pix_filter(self._image(), pixel_size=16, noise_scale=0.0)
        self.assertFalse(np.array_equal(noisy, clean))


# --------------------------------------------------------------------------- #
# Submission hygiene                                                          #
# --------------------------------------------------------------------------- #

class SubmissionHygieneTests(SimpleTestCase):
    """The packaged project must not ship internal notes or a dev database."""

    def test_internal_notes_and_dev_database_are_deleted_from_disk(self):
        from pathlib import Path

        from django.conf import settings

        base = Path(settings.BASE_DIR)
        self.assertFalse((base / "PROJECT_MEMORY.md").exists())
        self.assertFalse((base / "db.sqlite3").exists())
        self.assertFalse((base / "db.sqlite3-journal").exists())
