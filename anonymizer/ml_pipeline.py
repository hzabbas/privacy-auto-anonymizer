import os
import sys
import threading
import urllib.request
import difflib
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
import numpy as np
import cv2
import re
import json
import math
import logging
import time
import warnings

from . import pii

logger = logging.getLogger(__name__)

# Silence only the known-noisy numeric-library warnings (OpenMP duplicate libs,
# future-deprecation spam from torch/paddle). Everything else stays visible.
for _noisy in ("torch", "paddle", "ultralytics", "easyocr", "doctr"):
    warnings.filterwarnings("ignore", module=rf"^{_noisy}")

# Tunable thresholds -------------------------------------------------------- #
# Plate detector floor. The detector is unreliable on documents (it fires on
# printed words), so every surviving candidate is verified against its OCR text
# in the plate stage below - confidence alone is not evidence of a plate.
PLATE_MIN_CONFIDENCE = 0.45
# A plate at/above this score counts as "confident" and may absorb the text
# detections that overlap it (see _suppressed_by_plate).
PLATE_CASCADE_BELOW = 0.60
# YOLOv8-face boxes are tight; expand each side by this ratio before masking.
FACE_EXPAND_RATIO = 0.10
# OCR words below this recognition confidence are treated as noise.
OCR_MIN_CONFIDENCE = 0.20
# Refuse absurdly large uploads before decoding them (DoS / memory guard).
MAX_IMAGE_PIXELS = 60_000_000
# Above this fraction of near-white pixels the input is treated as a scanned
# document / screenshot, where plate candidates must be verified strictly.
DOCUMENT_WHITE_RATIO = 0.35
# OCR languages. The project targets English documents; English-only recognition
# (english_g2) is markedly more accurate on Latin text than the Arabic-script model
# that EasyOCR selects when 'fa' is enabled. Override with e.g. OCR_LANGS="en,fa".
OCR_LANGS = [s.strip() for s in os.environ.get("OCR_LANGS", "en").split(",") if s.strip()]
# Transformer NER (PERSON / LOCATION) is optional; disable with ENABLE_NER=0.
ENABLE_NER = os.environ.get("ENABLE_NER", "1") != "0"
# Timeout for the optional remote Imajev inference server.
IMAJEV_TIMEOUT = float(os.environ.get("IMAJEV_TIMEOUT", "2.0"))
# After a failed remote call, skip the endpoint for this many seconds.
IMAJEV_REMOTE_COOLDOWN = int(os.environ.get("IMAJEV_REMOTE_COOLDOWN", "120"))
# Page-layout model used by the 'doctr_layout' engine (DocLayNet classes).
DOCTR_LAYOUT_ARCH = os.environ.get("DOCTR_LAYOUT_ARCH", "lw_detr_s")

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated model weights and reliable download URLs
FACE_MODEL_NAME = "yolov8n-face.pt"
FACE_MODEL_PATH = WEIGHTS_DIR / FACE_MODEL_NAME
FACE_MODEL_URL = "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"

PLATE_MODEL_NAME = "yolov8n-plate.pt"
PLATE_MODEL_PATH = WEIGHTS_DIR / PLATE_MODEL_NAME
PLATE_MODEL_URL = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"


def _download_file(url: str, destination: Path) -> bool:
    """
    Safely downloads model weights: streams to a temporary ``.part`` file and only
    renames it into place once the transfer completed, so an interrupted download
    can never leave a truncated ``.pt`` behind.
    """
    if destination.exists() and destination.stat().st_size > 0:
        return True
    part_path = destination.with_suffix(destination.suffix + ".part")
    try:
        logger.info("Downloading model weights to %s ...", destination)
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response, open(part_path, "wb") as out_file:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out_file.write(chunk)
        if part_path.stat().st_size == 0:
            raise IOError("downloaded file is empty")
        part_path.replace(destination)
        logger.info("Successfully downloaded %s", destination.name)
        return True
    except Exception as e:
        logger.warning("Could not download %s: %s", destination.name, e)
        part_path.unlink(missing_ok=True)
        return False

class ImajevClassifier:
    """
    Jev-style typed-decision model engine for Imajev VLM.
    Outputs calibrated probabilities over specified options
    and explicitly handles the unknown state according to the Jev contract.
    """
    def __init__(self, endpoint: str = None):
        self.endpoint = endpoint or os.environ.get("IMAJEV_ENDPOINT", "http://127.0.0.1:8765/v1/systemone")
        self._remote_error_logged = False
        self._remote_disabled_until = 0.0

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        scaled = logits / max(temperature, 1e-4)
        exps = np.exp(scaled - np.max(scaled))
        return exps / np.sum(exps)

    def _query_remote_server(self, image: np.ndarray, options: List[str], state: dict = None) -> Union[dict, None]:
        """
        Optional remote inference: forwards the crop to a local Imajev VLM server
        when one is running. Any network / protocol problem falls back to the
        built-in heuristic classifier below.

        A failed attempt also opens a short circuit breaker: without it, every
        crop would wait for the connect timeout again and one unreachable server
        would slow the whole analysis down by seconds.
        """
        if not self.endpoint:
            return None
        if time.monotonic() < self._remote_disabled_until:
            return None
        try:
            _, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_bytes = encoded.tobytes()
            payload = {
                "state": state or {},
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": "Select the best matching category.",
                        "criteria": {opt: None for opt in options}
                    }
                }
            }
            boundary = "----ImajevBoundary7MA4YWxkTrZu0gW"
            crlf = chr(13) + chr(10)  # multipart bodies require CRLF separators
            body = bytearray()
            body += (
                f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="request"{crlf}{crlf}'
            ).encode("utf-8")
            body += json.dumps(payload).encode("utf-8")
            body += crlf.encode("utf-8")
            body += (
                f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="image"; filename="crop.jpg"{crlf}'
                f"Content-Type: image/jpeg{crlf}{crlf}"
            ).encode("utf-8")
            body += image_bytes
            body += f"{crlf}--{boundary}--{crlf}".encode("utf-8")

            req = urllib.request.Request(
                self.endpoint,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req, timeout=IMAJEV_TIMEOUT) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    ans = data.get("answers", {}).get("decision", {})
                    if "probabilities" in ans:
                        return {
                            "type": "choice",
                            "choice": ans.get("choice", options[0]),
                            "scores": ans["probabilities"],
                            "probabilities": ans["probabilities"],
                            "unknown_probability": float(ans.get("unknown_probability", 0.0)),
                            "abstained": ans.get("abstained", False)
                        }
        except Exception as exc:
            # Open the breaker for a while so the next crops answer instantly.
            self._remote_disabled_until = time.monotonic() + IMAJEV_REMOTE_COOLDOWN
            if not self._remote_error_logged:
                logger.info(
                    "Imajev remote endpoint unavailable (%s); using the built-in classifier "
                    "and skipping remote calls for %ds.", exc, IMAJEV_REMOTE_COOLDOWN,
                )
                self._remote_error_logged = True
        return None

    def predict_typed(self, image: np.ndarray, options: List[str], state: dict = None) -> dict:
        if image is None or image.size == 0 or not options:
            return {
                "type": "choice",
                "choice": options[0] if options else "unknown",
                "scores": {opt: 0.0 for opt in options},
                "probabilities": {opt: 0.0 for opt in options},
                "unknown_probability": 1.0,
                "abstained": True
            }

        remote_res = self._query_remote_server(image, options, state)
        if remote_res is not None:
            return remote_res

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 0 else 0.0
        mean_lum = float(np.mean(gray)) if gray.size > 0 else 0.0
        contrast = float(np.std(gray)) if gray.size > 0 else 0.0

        blur_penalty = 2.0 if lap_var < 15.0 else (0.9 if lap_var < 45.0 else 0.05)
        exposure_penalty = 1.4 if (mean_lum < 20 or mean_lum > 240) else 0.0
        contrast_penalty = 1.2 if contrast < 12.0 else 0.0
        z_unknown = -1.2 + blur_penalty + exposure_penalty + contrast_penalty

        doc_key = next((opt for opt in options if "document" in opt.lower() or "id card" in opt.lower() or "form" in opt.lower()), None)
        street_key = next((opt for opt in options if "street" in opt.lower() or "landscape" in opt.lower() or "crowd" in opt.lower()), None)

        if doc_key and street_key and len(options) == 2:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else None
            mean_sat = float(np.mean(hsv[:, :, 1])) if hsv is not None else 0.0
            hue_std = float(np.std(hsv[:, :, 0])) if hsv is not None else 0.0
            white_ratio = float(np.mean(gray > 185))

            sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_ratio = float(np.mean(np.abs(sob_x)) / (np.mean(np.abs(sob_y)) + 1e-4))

            z_doc = (white_ratio * 4.2) + (min(grad_ratio, 3.0) * 0.8) - (mean_sat / 255.0 * 4.0) + (1.0 if contrast > 40 else 0.0)
            z_street = (mean_sat / 255.0 * 5.0) + (hue_std / 180.0 * 2.5) - (white_ratio * 3.5) + (1.2 if contrast > 30 else 0.0)

            logits = np.array([z_doc if opt == doc_key else z_street for opt in options] + [z_unknown], dtype=np.float64)
            probs = self._softmax(logits, temperature=0.9)

            opt_probs = {opt: float(probs[i]) for i, opt in enumerate(options)}
            unknown_p = float(probs[-1])
            best_opt = max(opt_probs, key=opt_probs.get)

            return {
                "type": "choice",
                "choice": best_opt,
                "scores": opt_probs,
                "probabilities": opt_probs,
                "unknown_probability": unknown_p,
                "abstained": unknown_p > 0.4
            }

        plate_key = next((opt for opt in options if "license plate" in opt.lower()), None)
        if plate_key:
            ar = float(w) / max(float(h), 1.0)
            ar_score = 3.5 if 2.0 <= ar <= 5.5 else (1.5 if 1.5 <= ar <= 6.5 else -2.5)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            char_count = sum(1 for stat in stats[1:] if 0.25 * h <= stat[cv2.CC_STAT_HEIGHT] <= 0.9 * h and 0.03 * w <= stat[cv2.CC_STAT_WIDTH] <= 0.3 * w)
            char_score = 2.5 if 3 <= char_count <= 10 else (-0.5 if char_count > 15 else 0.5)

            sob_v = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))))
            sob_h = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))))
            barcode_tendency = (sob_v / (sob_h + 1e-4)) if sob_v > 20 else 0.5

            z_plate = ar_score + char_score + (1.0 if contrast > 35 else -1.0)
            z_printed = (2.0 if ar < 1.8 or char_count > 10 else -1.0) + (1.5 if contrast > 20 else 0.0)
            z_barcode = (3.0 if barcode_tendency > 2.8 else -1.5)

            logits_list = []
            for opt in options:
                if opt == plate_key:
                    logits_list.append(z_plate)
                elif "barcode" in opt.lower() or "graphic" in opt.lower():
                    logits_list.append(z_barcode)
                else:
                    logits_list.append(z_printed)
            logits_list.append(z_unknown)

            probs = self._softmax(np.array(logits_list, dtype=np.float64), temperature=0.85)
            opt_probs = {opt: float(probs[i]) for i, opt in enumerate(options)}
            unknown_p = float(probs[-1])
            best_opt = max(opt_probs, key=opt_probs.get)

            return {
                "type": "choice",
                "choice": best_opt,
                "scores": opt_probs,
                "probabilities": opt_probs,
                "unknown_probability": unknown_p,
                "abstained": unknown_p > 0.4
            }

        face_key = next((opt for opt in options if "human face" in opt.lower()), None)
        if face_key:
            ar = float(w) / max(float(h), 1.0)
            ar_face = 2.0 if 0.7 <= ar <= 1.4 else (0.5 if 0.5 <= ar <= 1.8 else -2.0)

            skin_ratio = 0.0
            if image.ndim == 3:
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                cr = ycrcb[:, :, 1]
                cb = ycrcb[:, :, 2]
                skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                skin_ratio = float(np.mean(skin_mask))

            skin_score = 3.0 if 0.25 <= skin_ratio <= 0.85 else (-1.0 if skin_ratio < 0.08 else 0.8)

            h_third, w_third = max(1, h // 3), max(1, w // 3)
            center = gray[h_third:2*h_third, w_third:2*w_third]
            center_var = float(np.std(center)) if center.size > 0 else 0.0
            sym_score = 1.5 if center_var > 20 else -0.5

            z_face = ar_face + skin_score + sym_score + (1.0 if lap_var > 40 else -0.5)
            z_illustration = (2.5 if skin_ratio < 0.15 and contrast > 30 else -1.0)
            z_noise = (3.0 if ar_face < -1.0 or lap_var < 15 or skin_ratio < 0.05 else -1.5)

            logits_list = []
            for opt in options:
                if opt == face_key:
                    logits_list.append(z_face)
                elif "illustration" in opt.lower() or "statue" in opt.lower():
                    logits_list.append(z_illustration)
                else:
                    logits_list.append(z_noise)
            logits_list.append(z_unknown)

            probs = self._softmax(np.array(logits_list, dtype=np.float64), temperature=0.85)
            opt_probs = {opt: float(probs[i]) for i, opt in enumerate(options)}
            unknown_p = float(probs[-1])
            best_opt = max(opt_probs, key=opt_probs.get)

            return {
                "type": "choice",
                "choice": best_opt,
                "scores": opt_probs,
                "probabilities": opt_probs,
                "unknown_probability": unknown_p,
                "abstained": unknown_p > 0.4
            }

        text_key = next((opt for opt in options if "identification text" in opt.lower() or "valid" in opt.lower()), None)
        if text_key:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            proj_y = np.sum(binary, axis=1)
            line_density = float(np.sum(proj_y > 0)) / max(float(h), 1.0)
            text_score = 2.5 if 0.25 <= line_density <= 0.85 and contrast > 30 else 0.0

            z_id_text = text_score + (1.5 if lap_var > 30 else -1.0)
            z_bg_text = (1.5 if contrast < 25 else -0.5)
            z_garbled = (2.0 if lap_var < 15 or line_density > 0.9 else -1.0)

            logits_list = []
            for opt in options:
                if opt == text_key:
                    logits_list.append(z_id_text)
                elif "random" in opt.lower() or "background" in opt.lower():
                    logits_list.append(z_bg_text)
                else:
                    logits_list.append(z_garbled)
            logits_list.append(z_unknown)

            probs = self._softmax(np.array(logits_list, dtype=np.float64), temperature=0.85)
            opt_probs = {opt: float(probs[i]) for i, opt in enumerate(options)}
            unknown_p = float(probs[-1])
            best_opt = max(opt_probs, key=opt_probs.get)

            return {
                "type": "choice",
                "choice": best_opt,
                "scores": opt_probs,
                "probabilities": opt_probs,
                "unknown_probability": unknown_p,
                "abstained": unknown_p > 0.4
            }

        num = len(options)
        p_known = (1.0 - 0.05) / num
        return {
            "type": "choice",
            "choice": options[0],
            "scores": {opt: p_known for opt in options},
            "probabilities": {opt: p_known for opt in options},
            "unknown_probability": 0.05,
            "abstained": False
        }

    def __call__(self, image: np.ndarray, options: List[str] = None, **kwargs) -> dict:
        return self.predict_typed(image, options=options or kwargs.get('options', []))

    def predict(self, image: np.ndarray, options: List[str] = None, **kwargs) -> dict:
        return self.predict_typed(image, options=options or kwargs.get('options', []))


class ModelManager:
    """
    Singleton manager to lazily load and cache YOLO & EasyOCR models in memory.
    Ensures zero disk reload overhead on consecutive inference calls.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._face_model = None
            cls._instance._plate_model = None
            cls._instance._ocr_reader = None
            cls._instance._ner = None
            cls._instance._paddle_ocr_reader = None
            cls._instance._doctr_reader = None
            cls._instance._layout_parser = None
            cls._instance._doctr_layout_model = None
            cls._instance._imajev_model = None
            # Failure flags: a broken engine must not be re-initialized (and
            # re-logged) on every single request.
            cls._instance._face_failed = False
            cls._instance._plate_failed = False
            cls._instance._ocr_failed = False
            cls._instance._paddle_failed = False
            cls._instance._doctr_failed = False
            cls._instance._layout_failed = False
            cls._instance._doctr_layout_failed = False
        return cls._instance

    @staticmethod
    def _load_yolo(path: Path, url: str, name: str):
        """
        Loads a dedicated YOLO model. Returns None on failure instead of silently
        falling back to COCO ``yolov8n.pt``: a generic COCO detector would report
        every person / car / object as a "face" or "plate".
        """
        from ultralytics import YOLO
        if not path.exists():
            _download_file(url, path)
        if not path.exists():
            logger.warning("%s weights unavailable; %s detection disabled.", name, name)
            return None
        try:
            return YOLO(str(path))
        except Exception as e:
            logger.warning("Error loading %s model: %s", name, e)
            return None

    @property
    def face_model(self):
        if self._face_model is None and not self._face_failed:
            with self._lock:
                if self._face_model is None and not self._face_failed:
                    self._face_model = self._load_yolo(FACE_MODEL_PATH, FACE_MODEL_URL, "face")
                    self._face_failed = self._face_model is None
        return self._face_model

    @property
    def plate_model(self):
        if self._plate_model is None and not self._plate_failed:
            with self._lock:
                if self._plate_model is None and not self._plate_failed:
                    self._plate_model = self._load_yolo(PLATE_MODEL_PATH, PLATE_MODEL_URL, "plate")
                    self._plate_failed = self._plate_model is None
        return self._plate_model

    @property
    def ocr_reader(self):
        """Lazy EasyOCR reader (the always-available base OCR engine)."""
        if self._ocr_reader is None and not self._ocr_failed:
            with self._lock:
                if self._ocr_reader is None and not self._ocr_failed:
                    import easyocr
                    try:
                        import torch
                        use_gpu = torch.cuda.is_available()
                    except Exception:
                        use_gpu = False
                    easyocr_storage = WEIGHTS_DIR / "easyocr"
                    easyocr_storage.mkdir(parents=True, exist_ok=True)
                    try:
                        # verbose=False avoids Windows cp1252 charmap encoding errors
                        # with progress-bar characters.
                        self._ocr_reader = easyocr.Reader(
                            OCR_LANGS,
                            gpu=use_gpu,
                            model_storage_directory=str(easyocr_storage),
                            verbose=False
                        )
                        logger.info("Loaded EasyOCR (%s).", ", ".join(OCR_LANGS))
                    except Exception as e:
                        logger.error("EasyOCR could not be initialized (%s); text detection disabled.", e)
                        self._ocr_failed = True
        return self._ocr_reader

    @property
    def ocr_available(self) -> bool:
        return self.ocr_reader is not None

    @property
    def ner(self):
        """Optional BERT NER recognizer; returns None when disabled or unavailable."""
        if not ENABLE_NER:
            return None
        if self._ner is None:
            self._ner = pii.NERRecognizer(cache_dir=str(WEIGHTS_DIR / "hf"))
        return self._ner if self._ner.available else None


    @property
    def imajev_model(self):
        """Built-in typed-decision reasoning engine (optionally backed by a local Imajev server)."""
        if self._imajev_model is None:
            with self._lock:
                if self._imajev_model is None:
                    try:
                        self._imajev_model = ImajevClassifier()
                        logger.info("Loaded Imajev typed-decision engine.")
                    except Exception as e:
                        logger.warning("Failed to initialize Imajev (%s).", e)
                        self._imajev_model = None
        return self._imajev_model

    @property
    def paddle_ocr_reader(self):
        """Optional PaddleOCR engine; None (and no repeated retries) when unavailable."""
        if self._paddle_ocr_reader is None and not self._paddle_failed:
            with self._lock:
                if self._paddle_ocr_reader is None and not self._paddle_failed:
                    try:
                        import inspect
                        from paddleocr import PaddleOCR

                        paddle_kwargs = {
                            'use_textline_orientation': False,
                            'lang': 'en',
                            'enable_mkldnn': False,
                        }
                        # Only pass the document-preprocessing switches that this
                        # PaddleOCR version actually supports.
                        sig = inspect.signature(PaddleOCR.__init__)
                        if 'use_doc_orientation_classify' in sig.parameters:
                            paddle_kwargs['use_doc_orientation_classify'] = False
                        if 'use_doc_unwarping' in sig.parameters:
                            paddle_kwargs['use_doc_unwarping'] = False

                        self._paddle_ocr_reader = PaddleOCR(**paddle_kwargs)
                        logger.info("Loaded PaddleOCR.")
                    except Exception as e:
                        logger.warning("PaddleOCR unavailable (%s); the UI will fall back to EasyOCR.", e)
                        self._paddle_ocr_reader = None
                        self._paddle_failed = True
        return self._paddle_ocr_reader

    @property
    def doctr_reader(self):
        """Optional DocTR engine; None (and no repeated retries) when unavailable."""
        if self._doctr_reader is None and not self._doctr_failed:
            with self._lock:
                if self._doctr_reader is None and not self._doctr_failed:
                    try:
                        from doctr.models import ocr_predictor
                        self._doctr_reader = ocr_predictor(
                            det_arch='db_resnet50',
                            reco_arch='crnn_vgg16_bn',
                            pretrained=True,
                            assume_straight_pages=True,
                        )
                        logger.info("Loaded DocTR (Latin-only recognizer: Persian text may not be recognized).")
                    except Exception as e:
                        logger.warning("DocTR unavailable (%s); the UI will fall back to EasyOCR.", e)
                        self._doctr_reader = None
                        self._doctr_failed = True
        return self._doctr_reader

    @property
    def layout_parser(self):
        """Optional LayoutParser (Detectron2) block detector; not installable on every platform."""
        if self._layout_parser is None and not self._layout_failed:
            with self._lock:
                if self._layout_parser is None and not self._layout_failed:
                    try:
                        import layoutparser as lp
                        self._layout_parser = lp.Detectron2LayoutModel(
                            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                        )
                        logger.info("Loaded LayoutParser.")
                    except Exception as e:
                        logger.warning("LayoutParser unavailable (%s); falling back to the regex layout.", e)
                        self._layout_parser = None
                        self._layout_failed = True
        return self._layout_parser

    @property
    def doctr_layout_model(self):
        """
        Page-layout model (LW-DETR, DocLayNet classes) shipped with docTR.

        Unlike LayoutParser this needs no Detectron2, so it also runs on Windows;
        the whole engine is optional and degrades to the regex layout.
        """
        if self._doctr_layout_model is None and not self._doctr_layout_failed:
            with self._lock:
                if self._doctr_layout_model is None and not self._doctr_layout_failed:
                    try:
                        from doctr.models import layout_predictor
                        self._doctr_layout_model = layout_predictor(
                            arch=DOCTR_LAYOUT_ARCH, pretrained=True
                        )
                        logger.info("Loaded the docTR layout model (%s).", DOCTR_LAYOUT_ARCH)
                    except Exception as e:
                        logger.warning("docTR layout model unavailable (%s); falling back to the regex layout.", e)
                        self._doctr_layout_model = None
                        self._doctr_layout_failed = True
        return self._doctr_layout_model

# Global singleton instance
_model_manager = ModelManager()


def get_engine_status() -> Dict[str, bool]:
    """
    Reports which OCR / layout engines can actually be imported in the current
    environment, so the UI can disable (instead of silently falling back from)
    the engines that are not installed.
    """
    import importlib.util

    def _installed(*modules: str) -> bool:
        for module in modules:
            try:
                if importlib.util.find_spec(module) is None:
                    return False
            except (ImportError, ValueError):
                return False
        return True

    return {
        "easyocr": _installed("easyocr", "torch"),
        "paddleocr": _installed("paddleocr", "paddle"),
        "doctr": _installed("doctr", "torch"),
        "layoutparser": _installed("layoutparser", "detectron2"),
        "doctr_layout": _installed("doctr", "torch"),
    }


def _load_image(image_input: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
    """
    Decodes image input from file path, bytes, or returns existing ndarray.
    Supports Windows non-ASCII / Persian file paths and correctly applies EXIF rotation.
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    import io
    from PIL import Image, ImageOps

    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image path does not exist: {path_str}")
        with open(path_str, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_input, (bytes, bytearray)):
        image_bytes = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def _clamp(val: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(val, max_val))


def _rect_polygon(x1: int, y1: int, x2: int, y2: int) -> List[List[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _clamp_polygon(polygon, width: int, height: int) -> List[List[int]]:
    """
    Rounds every vertex to whole pixels and clamps it into the image frame.

    Shared by every polygon path: freshly mapped OCR polygons and the ones
    rotated back through the deskew matrix (whose vertices can land slightly
    outside the frame).
    """
    return [
        [_clamp(int(round(x)), 0, width), _clamp(int(round(y)), 0, height)]
        for x, y in polygon
    ]


def _polygon_bbox(polygon) -> List[int]:
    """Axis-aligned bounding box [x1, y1, x2, y2] of a polygon's vertices."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _plate_text_is_valid(text: str) -> bool:
    """
    Lenient licence-plate test used on photographs.

    Real licence plates (US/EU/UK/Iran/...) contain alphanumeric characters,
    often with digits or uppercase letters (e.g. 'B 11 HOY', '21B34522').
    An unreadable crop is *kept* here: on a photograph the plate is often blurry
    or distant, and dropping it would leave it visible.
    """
    if not text.strip():
        return True
    alnum = re.sub(r"[^A-Za-z0-9\u0660-\u0669\u06F0-\u06F9\u0600-\u06FF]", "", text)
    if len(alnum) < 2 or len(alnum) > 14:
        return False
    # If it has digits, it's valid
    if any(c.isdigit() for c in alnum):
        return True
    # If it has uppercase characters and OCR digit confusions ('I', 'L', 'O')
    if alnum.isupper() and any(c in "ILO10" for c in alnum):
        return True
    # If all uppercase and short (typical plate format)
    if alnum.isupper() and 3 <= len(alnum) <= 8:
        return True
    return False


_PLATE_DIGITS = "0123456789\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9"


def _plate_text_looks_like_plate(text: str) -> bool:
    """
    Strict plate test used on document scans.

    The plate detector is not trustworthy on documents - it fires at 0.6-0.8 on
    ordinary text ("Utility Bill", "Ref: 88294", "Billing Currency: USD ($)") - so
    there a candidate must actually *read* like a plate: a short run of uppercase
    letters / digits / Persian characters containing at least one digit
    ("B 11 HOY", "21B34522", "12 ب 345").
    """
    alnum = re.sub(r"[^A-Za-z0-9\u0600-\u06FF]", "", (text or "").strip())
    if not alnum or not (2 <= len(alnum) <= 12):
        return False
    if not any(ch in _PLATE_DIGITS for ch in alnum):
        return False
    # Prose carries lowercase letters ("Ref88294", "UtilityBill"); plate lettering
    # is uppercase or Persian script.
    lowercase = sum(1 for ch in alnum if ch.islower())
    return lowercase <= 1


def _read_plate_text(ocr_reader, crop: np.ndarray) -> str:
    """OCR of a plate candidate; small crops are upscaled so the recognizer can read them."""
    verify_crop = crop
    if verify_crop.shape[0] < 32:
        scale = 32.0 / verify_crop.shape[0]
        verify_crop = cv2.resize(verify_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = ocr_reader.readtext(verify_crop, detail=1, paragraph=False)
    return "".join(str(item[1]) for item in results)


# --------------------------------------------------------------------------- #
# OCR line reconstruction and span -> polygon mapping
# --------------------------------------------------------------------------- #

class _OCRBox:
    __slots__ = ("pts", "text", "conf", "cx", "cy", "h", "x1", "x2", "y1", "y2", "start", "end")

    def __init__(self, bbox, text: str, conf: float):
        raw_pts = [[float(p[0]), float(p[1])] for p in bbox]
        self.pts = raw_pts
        self.text = text
        self.conf = conf
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self.x1, self.x2, self.y1, self.y2 = min(xs), max(xs), min(ys), max(ys)
        self.cx, self.cy = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
        self.h = max(1.0, math.hypot(self.pts[3][0] - self.pts[0][0], self.pts[3][1] - self.pts[0][1]))
        self.start = self.end = 0


def _group_lines(boxes: List[_OCRBox]) -> List[List[_OCRBox]]:
    """
    Groups word boxes into reading-order lines. Two boxes share a line when their
    vertical centres are within half a line height; a large horizontal gap
    (multi-column layouts, forms) starts a new logical line so that patterns
    never bridge unrelated columns.
    """
    lines: List[List[_OCRBox]] = []
    for b in sorted(boxes, key=lambda b: (b.cy, b.x1)):
        placed = False
        for line in lines:
            ref_cy = sum(x.cy for x in line) / len(line)
            ref_h = sum(x.h for x in line) / len(line)
            if abs(b.cy - ref_cy) <= 0.5 * min(ref_h, b.h) + 1:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    result: List[List[_OCRBox]] = []
    for line in lines:
        line.sort(key=lambda b: b.x1)
        seg = [line[0]]
        for b in line[1:]:
            gap = b.x1 - seg[-1].x2
            if gap > 2.5 * max(b.h, seg[-1].h):
                result.append(seg)
                seg = [b]
            else:
                seg.append(b)
        result.append(seg)
    result.sort(key=lambda seg: (sum(b.cy for b in seg) / len(seg), seg[0].x1))
    return result


def _sub_polygon(box: _OCRBox, a: int, b: int) -> List[List[float]]:
    """
    Returns the oriented sub-quadrilateral of ``box`` covering characters [a, b).
    Characters are assumed to be evenly spread along the box (good enough for
    word-level EasyOCR boxes). Half a character of padding is added on both sides
    so that proportional-font glyphs are never left partially visible.
    """
    n = max(1, len(box.text))
    pad = 0.05
    fa = max(0.0, (a - pad) / n)
    fb = min(1.0, (b + pad) / n)
    p0, p1, p2, p3 = box.pts

    def lerp(p, q, t):
        return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]

    return [lerp(p0, p1, fa), lerp(p0, p1, fb), lerp(p3, p2, fb), lerp(p3, p2, fa)]


def _expand_polygon(poly: List[List[float]], ratio_h: float, ratio_w: float) -> List[List[float]]:
    """Expands an oriented quad outward along its own axes (privacy-safe margin)."""
    p0, p1, p2, p3 = poly
    ux, uy = p1[0] - p0[0], p1[1] - p0[1]           # width axis
    vx, vy = p3[0] - p0[0], p3[1] - p0[1]           # height axis
    dw = (ratio_w * ux, ratio_w * uy)
    dh = (ratio_h * vx, ratio_h * vy)
    return [
        [p0[0] - dw[0] - dh[0], p0[1] - dw[1] - dh[1]],
        [p1[0] + dw[0] - dh[0], p1[1] + dw[1] - dh[1]],
        [p2[0] + dw[0] + dh[0], p2[1] + dw[1] + dh[1]],
        [p3[0] - dw[0] + dh[0], p3[1] - dw[1] + dh[1]],
    ]


# Evidence markers for a passport / ID data page (used by the corner-zone
# fallback below): multilingual passport words plus the raw MRZ prefix.
_PASSPORT_HINT_RE = re.compile(
    r"(?i)passport|passeport|pasaporte|پاسپورت|گذرنامه|machine[- ]readable|P<"
)


def _passport_number_zone_spans(lines, spans, width: int, height: int) -> List[pii.PIISpan]:
    """
    Document-number fallback for the upper-right passport zone.

    A passport data page prints its number twice: under the multilingual label
    (handled by the cross-line pairing in pii.py) and again in the upper-right
    corner. When the page shows passport evidence (an MRZ line or a passport
    header was read), a standalone 8-10 digit line in the upper-right zone is
    accepted as the document number even though no label preceded it.
    """
    passport_like = any(span.entity == "MRZ" for span in spans) or any(
        _PASSPORT_HINT_RE.search(box.text) for line in lines for box in line
    )
    if not passport_like:
        return []

    extra: List[pii.PIISpan] = []
    for line in lines:
        text = " ".join(box.text for box in line).strip()
        digits = re.sub(r"\D", "", text)
        if not (8 <= len(digits) <= 10) or len(text) > len(digits) + 3:
            continue
        center_x = sum(box.cx for box in line) / len(line)
        center_y = sum(box.cy for box in line) / len(line)
        if center_x <= 0.5 * width or center_y >= 0.55 * height:
            continue
        start, end = line[0].start, line[-1].end
        if any(span.start <= start and end <= span.end for span in spans):
            continue                       # already covered by another span
        extra.append(pii.PIISpan(start, end, "ID_NUMBER", 0.85, "passport_zone"))
    return extra


def _fingerprints_are_similar(a: str, b: str) -> bool:
    """
    True when two OCR readings of the same spot are the same text up to OCR
    noise: one contains the other, or the readings agree at >= 90% similarity.
    An extra or flapped character inside a 40-character MRZ line must not
    defeat the collapse of a double read.
    """
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    if abs(len(a) - len(b)) > max(2, int(0.12 * max(len(a), len(b)))):
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


def _dedupe_ocr_boxes(boxes: List[_OCRBox]) -> List[_OCRBox]:
    """
    Collapses near-identical readings of the same word/line.

    The OCR stage deliberately reads the frame twice (whole image + bottom MRZ
    band), so a word inside that band comes back twice with (nearly) the same
    box. Without this collapse the two readings are joined into a single
    logical line ("07 Aug 1999 07 Aug 1999"), which breaks the date and MRZ
    patterns and can mint false identifiers out of the doubled digits.

    When one reading is a truncated version of the other (a filler tail cut
    off mid-MRZ), the *more complete* text wins - a short reading would drop a
    full MRZ line below the 28-character structural minimum.
    """
    kept: List[_OCRBox] = []
    for box in sorted(boxes, key=lambda b: -b.conf):
        fingerprint = _text_fingerprint(box.text)
        duplicate_index = -1
        for index, existing in enumerate(kept):
            if not fingerprint or not _fingerprints_are_similar(
                    fingerprint, _text_fingerprint(existing.text)):
                continue
            if _iou([existing.x1, existing.y1, existing.x2, existing.y2],
                    [box.x1, box.y1, box.x2, box.y2]) > 0.45:
                duplicate_index = index
                break

        if duplicate_index == -1:
            kept.append(box)
            continue

        existing = kept[duplicate_index]
        # Fingerprints strip fillers and spaces, so truncation detection must
        # compare the whitespace-free *full* text (fillers intact): a band read
        # cut mid-filler is a prefix of the fuller frame read.
        compact = re.sub(r"\s+", "", box.text).lower()
        existing_compact = re.sub(r"\s+", "", existing.text).lower()
        if len(compact) > len(existing_compact) and existing_compact in compact:
            kept[duplicate_index] = box          # fuller reading supersedes the truncation
        elif len(compact) == len(existing_compact) and box.conf > existing.conf:
            kept[duplicate_index] = box
        # otherwise the already-kept reading stands
    return kept


def _ocr_pii_detections(ocr_results, width: int, height: int, M_inv=None) -> List[Dict[str, Any]]:
    boxes = []
    for bbox, text, conf in ocr_results:
        t = str(text).strip()
        # MRZ lines and long codes are often assigned very low confidence (e.g. 0.05) by EasyOCR
        # because they look like gibberish. Allow long alphanumeric strings to bypass the threshold.
        is_long_code = len(t) >= 15 and sum(c.isalnum() or c == '<' for c in t) >= 15
        if t and (float(conf) >= OCR_MIN_CONFIDENCE or is_long_code):
            boxes.append(_OCRBox(bbox, t, float(conf)))
    if not boxes:
        return []

    boxes = _dedupe_ocr_boxes(boxes)
    lines = _group_lines(boxes)
    parts: List[str] = []
    pos = 0
    for li, line in enumerate(lines):
        if li:
            parts.append("\n")
            pos += 1
        for bi, b in enumerate(line):
            if bi:
                parts.append(" ")
                pos += 1
            b.start = pos
            parts.append(b.text)
            pos += len(b.text)
            b.end = pos
    full_text = "".join(parts)

    spans = pii.find_pii(full_text, ner=_model_manager.ner)
    # Spatial fallback for the second printed copy of a passport number (the
    # upper-right corner of the data page, away from its label).
    spans.extend(_passport_number_zone_spans(lines, spans, width, height))

    detections: List[Dict[str, Any]] = []
    for span in spans:
        for line in lines:
            hit = [b for b in line if b.start < span.end and b.end > span.start]
            if not hit:
                continue
            first, last = hit[0], hit[-1]
            left = _sub_polygon(first, max(0, span.start - first.start), first.end - first.start)
            right = _sub_polygon(last, 0, min(span.end, last.end) - last.start)
            poly = _expand_polygon([left[0], right[1], right[2], left[3]], ratio_h=0.02, ratio_w=0.005)
            polygon = _clamp_polygon(poly, width, height)

            if M_inv is not None:
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                # Deskewing rotates the coordinates back; clamp them to the frame again.
                polygon = _clamp_polygon(cv2.transform(pts, M_inv).reshape(-1, 2), width, height)

            x1, y1, x2, y2 = _polygon_bbox(polygon)
            if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                continue
            ocr_conf = sum(b.conf for b in hit) / len(hit)
            detections.append({
                "type": "text",
                "entity": span.entity,
                "label": pii.entity_label(span.entity),
                "text": full_text[max(span.start, first.start):min(span.end, last.end)],
                "bbox": [x1, y1, x2, y2],
                "polygon": polygon,
                # Combined evidence: recognizer confidence tempered by OCR confidence.
                "score": round(span.score * (0.75 + 0.25 * ocr_conf), 2),
            })
    return detections


def _enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gentle CLAHE contrast enhancement in LAB colour space.

    Passports / ID cards are printed on textured, guilloche-patterned stock and
    photographed under uneven light. Stretching only the L channel's *local*
    contrast lifts the print without binarising the image - a hard THRESH_OTSU
    threshold would destroy the anti-aliased glyph edges that EasyOCR and DocTR
    recognise most reliably, so it is deliberately avoided.
    """
    if img_bgr is None or img_bgr.size == 0 or img_bgr.ndim != 3:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def preprocess_for_ocr(img_bgr: np.ndarray):
    """
    Prepares the frame for OCR: gentle LAB-CLAHE contrast enhancement plus an
    MRZ-based deskew. The bottom band is returned as a separate black-hat crop
    (robust against guilloche patterns); the OCR frame itself is never
    binarised.

    Returns (deskewed_gray, blackhat_bottom, y_offset, M_inv).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Try deskewing by finding the MRZ
    ratio = 600.0 / w
    resized = cv2.resize(gray, (600, int(h * ratio)))
    
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)
    
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    if maxVal > minVal:
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    else:
        # A perfectly flat image (no gradients at all) must still yield a uint8
        # frame: THRESH_OTSU below rejects float input.
        gradX = np.zeros_like(gradX, dtype=np.uint8)
    
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    angle = 0.0
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh) if bh > 0 else 0
        if ar > 3.0 and bw > 300:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box) / ratio
            rect_center, rect_size, rect_angle = cv2.minAreaRect(box.astype(np.float32))
            
            angle = rect_angle
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            break
            
    M_inv = None
    if abs(angle) > 0.5:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Gentle LAB-space CLAHE: lifts the print contrast on textured passport
    # stock without destructive binarisation (see _enhance_contrast).
    gray = cv2.cvtColor(_enhance_contrast(img_bgr), cv2.COLOR_BGR2GRAY)
    
    # Extract bottom 35% for MRZ BlackHat processing (robust against Guilloche patterns)
    y_offset = int(h * 0.65)
    bottom_crop = gray[y_offset:h, 0:w]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat_mrz = cv2.morphologyEx(bottom_crop, cv2.MORPH_BLACKHAT, kernel)
    blackhat_inv = 255 - blackhat_mrz
    
    return gray, blackhat_inv, y_offset, M_inv


def _run_easyocr_pipeline(image: np.ndarray, width: int, height: int, is_crop: bool = False) -> List[Dict[str, Any]]:
    ocr_reader = _model_manager.ocr_reader
    longest = max(width, height)
    mag_ratio = 1.5 if longest < 1280 else 1.0
    
    if is_crop:
        # Layout-block crops bypass preprocess_for_ocr; the same gentle CLAHE
        # enhancement is applied here so recall stays uniform across engines.
        if image.ndim == 3:
            image = _enhance_contrast(image)
        ocr_results = ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            canvas_size=2560,
            mag_ratio=mag_ratio,
            text_threshold=0.5,
            low_text=0.35,
            link_threshold=0.4,
            width_ths=0.5,
            add_margin=0.05,
        )
        return _ocr_pii_detections(ocr_results, width, height)
        
    ocr_image, mrz_crop, y_offset, M_inv = preprocess_for_ocr(image)
    
    ocr_results = ocr_reader.readtext(
        ocr_image,
        detail=1,
        paragraph=False,
        canvas_size=2560,
        mag_ratio=mag_ratio,
        text_threshold=0.5,
        low_text=0.35,
        link_threshold=0.4,
        width_ths=0.5,
        add_margin=0.05,
    )
    
    mrz_results = ocr_reader.readtext(
        mrz_crop,
        detail=1,
        paragraph=False,
        canvas_size=2560,
        mag_ratio=mag_ratio,
        text_threshold=0.5,
        low_text=0.35,
        link_threshold=0.4,
        width_ths=0.7,
        add_margin=0.05,
    )
    
    for bbox, text, conf in mrz_results:
        shifted_bbox = [[p[0], p[1] + y_offset] for p in bbox]
        ocr_results.append((shifted_bbox, text, conf))

    return _ocr_pii_detections(ocr_results, width, height, M_inv)


def _run_paddleocr_pipeline(image: np.ndarray, width: int, height: int, is_crop: bool = False) -> List[Dict[str, Any]]:
    paddle = _model_manager.paddle_ocr_reader
    if paddle is None:
        logger.warning("PaddleOCR unavailable; falling back to EasyOCR.")
        return _run_easyocr_pipeline(image, width, height, is_crop)
    
    def _parse_paddle_results(results):
        parsed = []
        if not results or not results[0]:
            return parsed
        if isinstance(results[0], dict):
            for res_dict in results:
                if 'rec_texts' in res_dict and 'rec_polys' in res_dict and 'rec_scores' in res_dict:
                    for text, box, score in zip(res_dict['rec_texts'], res_dict['rec_polys'], res_dict['rec_scores']):
                        text_str = str(text).strip()
                        if text_str:
                            box_list = box.tolist() if hasattr(box, 'tolist') else box
                            parsed.append((box_list, text_str, float(score)))
        elif isinstance(results[0], list):
            for line in results[0]:
                if len(line) == 2:
                    bbox, (text, conf) = line
                    text_str = str(text).strip()
                    if text_str:
                        bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                        parsed.append((bbox_list, text_str, float(conf)))
        return parsed

    def _ensure_3channel(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    results = paddle.ocr(_ensure_3channel(image))
    ocr_results = _parse_paddle_results(results)
    return _ocr_pii_detections(ocr_results, width, height)


def _doctr_pages_to_ocr_results(pages, img_w: int, img_h: int):
    """
    Converts docTR result pages into (polygon, text, confidence) tuples.

    docTR nests one object per *word* (blocks -> lines -> words) with
    normalised geometry. Each word is emitted with its own polygon - exactly
    like EasyOCR's boxes - so ``_ocr_pii_detections`` rebuilds the
    reading-order line ("07 Aug 1999" instead of "07", "Aug", "1999") and maps
    a character span back to the precise word polygons it covers.
    """
    results: List[Tuple[List[List[float]], str, float]] = []
    for page in pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    value = str(word.value or "")
                    if not value.strip():
                        continue
                    # docTR geometry is normalised to the page (0..1) and the
                    # page *is* the image handed to the model, so the pixel
                    # position is the fraction times the image size.
                    (x1, y1), (x2, y2) = word.geometry
                    polygon = [
                        [x1 * img_w, y1 * img_h],
                        [x2 * img_w, y1 * img_h],
                        [x2 * img_w, y2 * img_h],
                        [x1 * img_w, y2 * img_h],
                    ]
                    results.append((polygon, value, float(word.confidence)))
    return results


def _run_doctr_pipeline(image: np.ndarray, width: int, height: int, is_crop: bool = False) -> List[Dict[str, Any]]:
    # Same cached singleton as the layout stage (_model_manager.doctr_reader):
    # both engines share docTR's model zoo, so nothing is re-instantiated and
    # no redundant preprocessing pass runs here.
    doctr = _model_manager.doctr_reader
    if doctr is None:
        logger.warning("DocTR unavailable; falling back to EasyOCR.")
        return _run_easyocr_pipeline(image, width, height, is_crop)

    from doctr.io import DocumentFile

    def _run_doctr_on_img(img_array):
        _, encoded_image = cv2.imencode('.png', img_array)
        doc = DocumentFile.from_images([encoded_image.tobytes()])
        result = doctr(doc)
        img_h, img_w = img_array.shape[:2]
        return _doctr_pages_to_ocr_results(result.pages, img_w, img_h)

    if is_crop:
        # Layout-block crops bypass preprocess_for_ocr; apply the same gentle
        # CLAHE enhancement so recall stays uniform across engines.
        if image.ndim == 3:
            image = _enhance_contrast(image)
        ocr_results = _run_doctr_on_img(image)
        return _ocr_pii_detections(ocr_results, width, height)

    ocr_image, mrz_crop, y_offset, M_inv = preprocess_for_ocr(image)

    ocr_results = _run_doctr_on_img(ocr_image)

    if mrz_crop is not None and mrz_crop.size > 0:
        mrz_results = _run_doctr_on_img(mrz_crop)
        for bbox, text, conf in mrz_results:
            shifted_bbox = [[p[0], p[1] + y_offset] for p in bbox]
            ocr_results.append((shifted_bbox, text, conf))

    return _ocr_pii_detections(ocr_results, width, height, M_inv)


# DocLayNet-style classes returned by docTR's layout model that contain no text:
# OCR is not worth running inside those regions, everything else is kept (page
# headers/footers routinely carry names and document numbers).
_NON_TEXT_LAYOUT_CLASSES = {"picture", "figure", "image", "graphic"}

# The LW-DETR layout model is trained on modern office documents and regularly
# labels an entire ID card / passport / photographed form as "Picture".
# Trusting that exclusion there would delete every field printed on the card,
# so it is bypassed when the picture regions cover a suspicious share of the
# page or when the model found almost no text blocks at all - both typical for
# camera-scanned identity documents.
PICTURE_AREA_BYPASS_RATIO = 0.35
MIN_LAYOUT_TEXT_BLOCKS = 3


def _layout_region_blocks(page_prediction: Dict[str, Any],
                          width: int, height: int,
                          min_size: int = 10,
                          include_non_text: bool = False) -> Tuple[List[List[int]], float]:
    """
    Converts one page of an LW-DETR layout prediction into pixel rectangles.

    The predictor yields ``{"class_names": [...], "boxes": ndarray(N,4), "scores": [...]}``
    with boxes normalised to [x1, y1, x2, y2] in 0..1. Returns
    ``(blocks, picture_area_ratio)``: ``blocks`` holds the clamped regions
    (non-text classes only when ``include_non_text`` is set) and
    ``picture_area_ratio`` the share of the page covered by Picture / Figure
    regions, which drives the bypass policy below. Degenerate boxes are ignored.
    """
    class_names = page_prediction.get("class_names") or []
    boxes = page_prediction.get("boxes")
    if boxes is None:
        return [], 0.0

    blocks: List[List[int]] = []
    picture_area = 0.0
    page_area = float(max(1, width) * max(1, height))
    for name, box in zip(class_names, boxes):
        is_non_text = str(name).strip().lower() in _NON_TEXT_LAYOUT_CLASSES
        x1, y1, x2, y2 = (float(v) for v in np.asarray(box).flatten()[:4])
        bx1 = _clamp(int(round(x1 * width)), 0, width)
        by1 = _clamp(int(round(y1 * height)), 0, height)
        bx2 = _clamp(int(round(x2 * width)), 0, width)
        by2 = _clamp(int(round(y2 * height)), 0, height)
        if bx2 - bx1 <= min_size or by2 - by1 <= min_size:
            continue
        if is_non_text:
            picture_area += (bx2 - bx1) * (by2 - by1) / page_area
            if not include_non_text:
                continue
        blocks.append([bx1, by1, bx2, by2])
    return blocks, min(1.0, picture_area)


def _layout_blocks_from_prediction(page_prediction: Dict[str, Any],
                                   width: int, height: int,
                                   min_size: int = 10) -> List[List[int]]:
    """Text blocks of one layout page (raw converter; policy in _layout_blocks_with_policy)."""
    return _layout_region_blocks(page_prediction, width, height, min_size)[0]


def _layout_blocks_with_policy(predictions: List[Dict[str, Any]],
                               width: int, height: int) -> List[List[int]]:
    """
    Applies the Picture/Figure exclusion policy to a whole page's predictions.

    Non-text regions are dropped *only* when the layout looks like a normal
    multi-block document. A page whose picture share exceeds
    PICTURE_AREA_BYPASS_RATIO, or which yields fewer than
    MIN_LAYOUT_TEXT_BLOCKS text blocks, is treated as an ID card / passport /
    photographed form: every region is then kept so OCR still reads the whole
    card.
    """
    text_blocks: List[List[int]] = []
    picture_area = 0.0
    for prediction in predictions:
        blocks, area = _layout_region_blocks(prediction, width, height)
        text_blocks.extend(blocks)
        picture_area += area

    if picture_area > PICTURE_AREA_BYPASS_RATIO or len(text_blocks) < MIN_LAYOUT_TEXT_BLOCKS:
        all_blocks: List[List[int]] = []
        for prediction in predictions:
            all_blocks.extend(
                _layout_region_blocks(prediction, width, height, include_non_text=True)[0]
            )
        if all_blocks:
            logger.info(
                "docTR layout: bypassing the Picture/Figure exclusion "
                "(picture area %.0f%%, %d text block(s)) - reading every region "
                "so ID-card / passport text is not dropped.",
                picture_area * 100.0, len(text_blocks),
            )
        return all_blocks
    return text_blocks


def _doctr_layout_blocks(image: np.ndarray) -> List[List[int]]:
    """Detects the text-bearing page blocks of ``image`` with docTR's layout model."""
    model = _model_manager.doctr_layout_model
    if model is None:
        return []

    try:
        from doctr.io import DocumentFile

        _, encoded = cv2.imencode('.png', image)
        height, width = image.shape[:2]
        pages = model(DocumentFile.from_images([encoded.tobytes()]))
    except Exception as exc:
        logger.warning("docTR layout detection failed: %s", exc)
        return []

    predictions = [page for page in pages if isinstance(page, dict)]
    return _layout_blocks_with_policy(predictions, width, height)


def _suppressed_by_plate(text_det: Dict[str, Any],
                         plate_dets: List[Dict[str, Any]],
                         min_plate_score: float = PLATE_CASCADE_BELOW) -> bool:
    """
    True when a text detection is really the plate's own characters.

    Only a *confident* plate may absorb the text detections that overlap it. The
    plate model also fires on formatted numbers (a dotted IP address, a card
    number), and those marginal boxes must not swallow the PII labels sitting
    inside them: the region would still be masked by the plate, but the entity
    type would be reported as "licence plate" instead of "IP address".
    """
    return any(
        float(plate.get("score", 0.0)) >= min_plate_score
        and _iou(text_det.get("bbox", [0, 0, 0, 0]), plate.get("bbox", [0, 0, 0, 0])) > 0.5
        for plate in plate_dets
    )


def _text_fingerprint(value: str) -> str:
    """Lower-cased alphanumeric fingerprint used to compare two OCR readings."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _dedupe_text_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drops text detections that describe the same characters twice.

    The OCR stage reads the document twice (whole frame + bottom MRZ band), so a
    line inside that band comes back twice with slightly different confidences
    and boxes. Two detections count as duplicates when their boxes overlap
    strongly *and* their recognised text agrees; the higher-scoring one is kept.
    """
    kept: List[Dict[str, Any]] = []
    for det in detections:
        if det.get("type") != "text":
            kept.append(det)
            continue

        fingerprint = _text_fingerprint(det.get("text", ""))
        match_index = -1
        for index, existing in enumerate(kept):
            if existing.get("type") != "text":
                continue
            if _iou(det.get("bbox", [0, 0, 0, 0]), existing.get("bbox", [0, 0, 0, 0])) < 0.55:
                continue
            other = _text_fingerprint(existing.get("text", ""))
            if fingerprint and other and (fingerprint in other or other in fingerprint):
                match_index = index
                break

        if match_index == -1:
            kept.append(det)
        elif float(det.get("score", 0.0)) > float(kept[match_index].get("score", 0.0)):
            kept[match_index] = det
    return kept


def _validate_crop_with_imajev(crop_img: np.ndarray, entity_type: str, initial_score: float = None) -> dict:
    """
    Second-stage validation of a low-confidence crop by the Imajev typed-decision engine.

    Failure policy: the only outcome that drops a detection is an explicit model
    *rejection*. If the validator itself fails (missing engine, decoding issue, ...)
    the crop is kept, because silently discarding a real face / plate / line of text
    would leave sensitive content unmasked — the worst possible failure for a
    privacy tool.
    """
    detector_score = round(float(initial_score), 3) if initial_score is not None else None
    if crop_img is None or crop_img.size == 0:
        return {
            "action": "error",
            "confidence": 0.0,
            "unknown_prob": 1.0,
            "reason": "empty crop",
            "detector_score": detector_score,
        }

    entity_fa = {"plate": "پلاک", "face": "چهره"}.get(entity_type, "متن")
    try:
        vlm = _model_manager.imajev_model
        if not vlm:
            raise RuntimeError("Imajev engine is not available")

        if entity_type == "plate":
            options = ["valid license plate", "printed text", "graphic/barcode"]
            target = "valid license plate"
        elif entity_type == "face":
            options = ["human face", "illustration/statue", "background noise"]
            target = "human face"
        else:
            options = ["valid identification text", "random background text", "garbled ocr noise"]
            target = "valid identification text"

        if hasattr(vlm, 'predict_typed'):
            resp = vlm.predict_typed(crop_img, options=options)
        elif callable(vlm):
            resp = vlm(crop_img, options=options)
        elif hasattr(vlm, 'predict'):
            resp = vlm.predict(crop_img, options=options)
        else:
            raise RuntimeError("VLM does not support typed decision API")

        if not isinstance(resp, dict):
            raise ValueError(f"Invalid model response format: {type(resp).__name__}")

        scores = resp.get("scores") or resp.get("probabilities")
        if not scores and all(opt in resp for opt in options):
            scores = {opt: resp[opt] for opt in options}
        if not scores:
            raise ValueError("No option probabilities returned by model")

        unknown_prob = float(resp.get("unknown_probability", resp.get("unknown_prob", 0.0)))
        target_score = float(scores.get(target, 0.0))

        if target_score < 0.5 or unknown_prob > 0.4:
            return {
                "action": "rejected",
                "confidence": round(target_score, 3),
                "unknown_prob": round(unknown_prob, 3),
                "reason": "model decision",
                "detector_score": detector_score,
                "message": (
                    f"❌ Imajev: کادر {entity_fa} نامعتبر ارزیابی شد "
                    f"(اطمینان {target_score * 100:.1f}٪ / عدم‌قطعیت {unknown_prob * 100:.1f}٪) و حذف گردید."
                ),
                "scores": {k: round(v, 3) for k, v in scores.items()}
            }

        return {
            "action": "approved",
            "confidence": round(target_score, 3),
            "unknown_prob": round(unknown_prob, 3),
            "reason": "model decision",
            "detector_score": detector_score,
            "message": (
                f"✅ Imajev: کادر {entity_fa} تأیید شد "
                f"(اطمینان {target_score * 100:.1f}٪ / عدم‌قطعیت {unknown_prob * 100:.1f}٪)."
            ),
            "scores": {k: round(v, 3) for k, v in scores.items()}
        }
    except Exception as e:
        logger.warning("Imajev validation failed for %s: %s", entity_type, e)
        return {
            "action": "error",
            "confidence": 0.0,
            "unknown_prob": 1.0,
            "reason": f"validator error: {e}",
            "detector_score": detector_score,
            "message": (
                f"⚠️ Imajev: اعتبارسنجی کادر {entity_fa} ممکن نشد ({e})؛ "
                "عنصر بدون فیلتر ثانویه نگه داشته شد."
            ),
        }


def analyze_image_entities(
    image_input: Union[str, Path, np.ndarray, bytes], 
    conf_threshold: float = 0.40,
    ocr_engine: str = 'easyocr',
    layout_engine: str = 'regex',
    use_imajev: bool = False
) -> Dict[str, Any]:
    """
    Analyzes an input image and returns every sensitive entity with its geometry.

    Pipeline: YOLO face detection -> YOLO plate detection (with an OCR/Imajev
    confidence cascade) -> OCR text recognition (EasyOCR / PaddleOCR / DocTR)
    -> PII recognition over the reconstructed text lines -> optional Imajev
    typed-decision routing and crop validation.

    Args:
        image_input: Raw bytes, path, or an already decoded BGR ndarray.
        conf_threshold: Minimum detector confidence (also the face/plate floor).
        ocr_engine: 'easyocr' | 'paddleocr' | 'doctr' (unknown values fall back
                    to EasyOCR; a missing engine degrades to EasyOCR as well).
        layout_engine: 'regex' (reconstruct lines ourselves) | 'doctr_layout'
                       (docTR LW-DETR page layout, works on Windows) |
                       'layoutparser' (Detectron2 blocks; needs detectron2).
        use_imajev: Enables the Imajev reasoning pass: scene routing (skip OCR on
                    outdoor photos) plus second-stage validation of low-confidence
                    crops. Validation failures never drop a detection.

    Returns:
        {
            "status": "success",
            "image_dimensions": {"width": int, "height": int},
            "detections": [{"id", "type", "bbox", "polygon", "score", ...}, ...],
            "imajev_logs": [json-stringified decision records, ...]
        }
    """
    image = _load_image(image_input)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({width}x{height} px). "
            f"The maximum supported resolution is {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
        )

    detections: List[Dict[str, Any]] = []
    imajev_logs: List[str] = []

    skip_ocr = False
    if use_imajev:
        imajev_logs.append(json.dumps({
            "entity": "system",
            "action": "info",
            "message": "🚀 سامانه استدلال تایپ‌شده Imajev Jev-Engine آماده تحلیل است."
        }))
        try:
            vlm = _model_manager.imajev_model
            if vlm:
                scene_options = ["Document or ID Card or Form", "Street scene or landscape or crowd"]
                if hasattr(vlm, 'predict_typed'):
                    resp = vlm.predict_typed(image, options=scene_options)
                elif callable(vlm):
                    resp = vlm(image, options=scene_options)
                elif hasattr(vlm, 'predict'):
                    resp = vlm.predict(image, options=scene_options)
                else:
                    raise RuntimeError("Imajev model missing typed inference capability")

                scores = resp.get("scores") or resp.get("probabilities") or {}
                unknown_prob = float(resp.get("unknown_probability", resp.get("unknown_prob", 0.0)))
                doc_prob = float(scores.get("Document or ID Card or Form", 0.0))
                street_prob = float(scores.get("Street scene or landscape or crowd", 0.0))

                if street_prob > doc_prob:
                    skip_ocr = True
                    log_msg = f"⚡ Imajev: تصویر به عنوان فضای باز / خیابان طبقه‌بندی شد ({street_prob*100:.1f}٪)؛ پردازش OCR اسناد لغو گردید."
                    imajev_logs.append(json.dumps({
                        "entity": "scene",
                        "action": "rejected",
                        "confidence": round(street_prob, 3),
                        "unknown_prob": round(unknown_prob, 3),
                        "message": log_msg,
                        "reason": "model decision",
                        "scores": {k: round(v, 3) for k, v in scores.items()}
                    }))
                    logger.info("%s", log_msg)
                else:
                    skip_ocr = False
                    log_msg = f"🔍 Imajev: تصویر به عنوان سند یا کارت شناسایی طبقه‌بندی شد ({doc_prob*100:.1f}٪)؛ پردازش متن و OCR فعال است."
                    imajev_logs.append(json.dumps({
                        "entity": "scene",
                        "action": "approved",
                        "confidence": round(doc_prob, 3),
                        "unknown_prob": round(unknown_prob, 3),
                        "message": log_msg,
                        "reason": "model decision",
                        "scores": {k: round(v, 3) for k, v in scores.items()}
                    }))
                    logger.info("%s", log_msg)
            else:
                raise RuntimeError("Imajev model instance unavailable")
        except Exception as e:
            err_msg = f"⚠️ Imajev: خطا در استدلال تایپ‌شده طبقه‌بندی صحنه ({e})"
            imajev_logs.append(json.dumps({
                "entity": "scene",
                "action": "error",
                "confidence": 0.0,
                "unknown_prob": 1.0,
                "message": err_msg,
                "reason": f"exception: {str(e)}"
            }))
            logger.warning("%s", err_msg)


    # Licence plates are verified strictly on document scans: the plate model is
    # unreliable there (it fires at 0.6-0.8 on ordinary text), whereas on a
    # photograph an unreadable plate crop is normal and must be kept.
    _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _white_ratio = float(np.mean(_gray > 200))
    document_like = _white_ratio >= DOCUMENT_WHITE_RATIO and not skip_ocr

    # 1. Faces -------------------------------------------------------------- #
    try:
        face_model = _model_manager.face_model
        if face_model is not None:
            longest_dim = max(width, height)
            yolo_imgsz = 1280 if longest_dim >= 960 else 640
            for r in face_model.predict(source=image, conf=conf_threshold, imgsz=yolo_imgsz, verbose=False):
                for box in r.boxes:
                    score = float(box.conf[0].item())
                    if score < conf_threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    w, h = x2 - x1, y2 - y1
                    x1 -= w * FACE_EXPAND_RATIO
                    x2 += w * FACE_EXPAND_RATIO
                    y1 -= h * FACE_EXPAND_RATIO * 1.5
                    y2 += h * FACE_EXPAND_RATIO
                    bx = [_clamp(int(x1), 0, width), _clamp(int(y1), 0, height),
                          _clamp(int(math.ceil(x2)), 0, width), _clamp(int(math.ceil(y2)), 0, height)]
                    if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
                        continue
                        
                    if use_imajev and score < 0.60:
                        crop = image[bx[1]:bx[3], bx[0]:bx[2]]
                        if crop.size > 0:
                            result = _validate_crop_with_imajev(crop, "face", initial_score=score)
                            result["entity"] = "face"
                            if result["action"] == "approved":
                                result["id"] = len(detections) + 1
                            imajev_logs.append(json.dumps(result))
                            if result["action"] == "rejected":
                                continue

                    detections.append({
                        "type": "face",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        logger.warning("Face detection skipped: %s", e)

    # 2. Licence plates ----------------------------------------------------- #
    try:
        plate_model = _model_manager.plate_model
        if plate_model is not None:
            plate_conf = max(conf_threshold, PLATE_MIN_CONFIDENCE)
            longest_dim = max(width, height)
            yolo_imgsz = 1280 if longest_dim >= 600 else 640
            for r in plate_model.predict(source=image, conf=plate_conf, imgsz=yolo_imgsz, iou=0.45, verbose=False):
                for box in r.boxes:
                    score = float(box.conf[0].item())
                    if score < plate_conf:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bx = [_clamp(x1, 0, width), _clamp(y1, 0, height), _clamp(x2, 0, width), _clamp(y2, 0, height)]
                    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                    if bw < 8 or bh < 4:
                        continue
                    aspect = bw / float(bh)
                    if not (0.8 <= aspect <= 8.0):
                        continue

                    # Every plate candidate is verified, whatever its confidence: the
                    # detector also fires at 0.6-0.8 on ordinary document text, so a
                    # high score alone is not evidence of a plate.
                    crop = image[bx[1]:bx[3], bx[0]:bx[2]]
                    if crop.size == 0:
                        continue
                    if use_imajev:
                        result = _validate_crop_with_imajev(crop, "plate", initial_score=score)
                        result["entity"] = "plate"
                        imajev_logs.append(json.dumps(result))
                        if result["action"] == "rejected":
                            continue

                    # OCR confirmation: on a document the crop must actually read like
                    # a plate; on a photograph the lenient test applies and an
                    # unreadable crop is kept. When no OCR engine could be
                    # initialized the candidate is kept (fail open).
                    ocr_reader = _model_manager.ocr_reader
                    if ocr_reader is not None:
                        plate_text = _read_plate_text(ocr_reader, crop)
                        verified = (_plate_text_looks_like_plate(plate_text) if document_like
                                    else _plate_text_is_valid(plate_text))
                        if not verified:
                            if use_imajev:
                                imajev_logs.append(json.dumps({
                                    "entity": "plate",
                                    "action": "rejected",
                                    "reason": "invalid OCR pattern",
                                    "message": "❌ Imajev/OCR: محتوای کادر با الگوی پلاک همخوانی نداشت و حذف شد",
                                }))
                            continue

                    detections.append({
                        "type": "plate",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        logger.warning("Plate detection skipped: %s", e)

    # 3. Dynamic OCR & Layout strategy -------------------------------------- #
    try:
        lp_blocks = []
        if not skip_ocr and layout_engine == 'layoutparser':
            lp_model = _model_manager.layout_parser
            if lp_model is None:
                logger.info("LayoutParser not available, falling back to regex layout.")
                layout_engine = 'regex'
            else:
                layout = lp_model.detect(image)
                # Filter for Text blocks
                for block in layout:
                    if block.type in ["Text", "Title", "List"]:
                        x1, y1, x2, y2 = block.coordinates
                        lp_blocks.append([int(x1), int(y1), int(x2), int(y2)])
        elif not skip_ocr and layout_engine == 'doctr_layout':
            lp_blocks = _doctr_layout_blocks(image)
            if lp_blocks:
                logger.info("docTR layout: %d text region(s) detected.", len(lp_blocks))
            else:
                logger.info("docTR layout found no text regions, falling back to regex layout.")
                layout_engine = 'regex'

        text_dets = []
        if skip_ocr:
            pass
        elif layout_engine in ('layoutparser', 'doctr_layout') and lp_blocks:
            # Run OCR ONLY on detected layout blocks
            for block in lp_blocks:
                bx1, by1, bx2, by2 = block
                # Ensure valid crop
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(width, bx2), min(height, by2)
                if bx2 - bx1 > 10 and by2 - by1 > 10:
                    crop = image[by1:by2, bx1:bx2]
                    crop_h, crop_w = crop.shape[:2]
                    
                    if ocr_engine == 'paddleocr':
                        block_dets = _run_paddleocr_pipeline(crop, crop_w, crop_h, is_crop=True)
                    elif ocr_engine == 'doctr':
                        block_dets = _run_doctr_pipeline(crop, crop_w, crop_h, is_crop=True)
                    else:
                        block_dets = _run_easyocr_pipeline(crop, crop_w, crop_h, is_crop=True)
                        
                    # Shift coordinates back to original image
                    for det in block_dets:
                        d_bx = det["bbox"]
                        det["bbox"] = [d_bx[0] + bx1, d_bx[1] + by1, d_bx[2] + bx1, d_bx[3] + by1]
                        if "polygon" in det:
                            det["polygon"] = [[p[0] + bx1, p[1] + by1] for p in det["polygon"]]
                        text_dets.append(det)
        else:
            if ocr_engine == 'paddleocr':
                text_dets = _run_paddleocr_pipeline(image, width, height)
            elif ocr_engine == 'doctr':
                text_dets = _run_doctr_pipeline(image, width, height)
            else: # easyocr
                text_dets = _run_easyocr_pipeline(image, width, height)

        # The EasyOCR stage reads the document twice (full frame + bottom MRZ band),
        # so ordinary lines inside that band arrive twice; keep one of each pair.
        text_dets = _dedupe_text_detections(text_dets)

        plates = [d for d in detections if d["type"] == "plate"]
        for d in text_dets:
            if _suppressed_by_plate(d, plates):
                continue
            
            score = d.get("score", 1.0)
            if use_imajev and score < 0.5:
                bx = d["bbox"]
                crop = image[max(0, int(bx[1])):min(height, int(bx[3])), max(0, int(bx[0])):min(width, int(bx[2]))]
                if crop.size > 0:
                    result = _validate_crop_with_imajev(crop, "text", initial_score=score)
                    result["entity"] = "text"
                    result["text_preview"] = d.get('text', '')[:12]
                    if result["action"] == "approved":
                        result["id"] = len(detections) + 1
                    imajev_logs.append(json.dumps(result))
                    if result["action"] == "rejected":
                        continue
                    
            detections.append(d)
    except ValueError:
        raise
    except Exception as e:
        logger.warning("Text OCR stage skipped: %s", e)
        # Any other engine failure is tolerated: faces and plates still get returned
        # even when the OCR stage could not run.

    if use_imajev and len(imajev_logs) == 1:
        imajev_logs.append(json.dumps({"entity": "system", "action": "info", "message": "✨ Imajev: همه عناصر شناسایی‌شده دارای ضریب اطمینان بالا بودند و نیازی به فیلتر ثانویه نبود."}))

    for i, d in enumerate(detections, start=1):
        d["id"] = i

    return {
        "status": "success",
        "image_dimensions": {
            "width": int(width),
            "height": int(height)
        },
        "detections": detections,
        "imajev_logs": imajev_logs
    }


if __name__ == "__main__":
    import json

    print("--- Running ML Pipeline Self-Test ---")

    test_img = np.full((520, 900, 3), 255, dtype=np.uint8)
    lines = [
        "Anthony Caldwell",
        "(159) 357-8426",
        "anthony@caldwell.com",
        "Card: 4111 1111 1111 1111",
        "We maintain confidential records.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(test_img, line, (40, 70 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

    print("Executing analyze_image_entities on test frame...")
    results = analyze_image_entities(test_img, conf_threshold=0.30)

    print("\n--- Inference Output JSON ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nTotal entities detected: {len(results['detections'])}")
    print("--- ML Pipeline Self-Test Completed Successfully ---")

