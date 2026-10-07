import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
import cv2
import re
import math
import warnings
warnings.filterwarnings("ignore")

from . import pii

# Tunable thresholds -------------------------------------------------------- #
# Plate detector is noisy on documents (it fires on printed words), so it uses a
# stricter floor and OCR verification below PLATE_CASCADE_BELOW.
PLATE_MIN_CONFIDENCE = 0.45
PLATE_CASCADE_BELOW = 0.85
# YOLOv8-face boxes are tight; expand each side by this ratio before masking.
FACE_EXPAND_RATIO = 0.10
# OCR words below this recognition confidence are treated as noise.
OCR_MIN_CONFIDENCE = 0.20
# OCR languages. The project targets English documents; English-only recognition
# (english_g2) is markedly more accurate on Latin text than the Arabic-script model
# that EasyOCR selects when 'fa' is enabled. Override with e.g. OCR_LANGS="en,fa".
OCR_LANGS = [s.strip() for s in os.environ.get("OCR_LANGS", "en").split(",") if s.strip()]
# Transformer NER (PERSON / LOCATION) is optional; disable with ENABLE_NER=0.
ENABLE_NER = os.environ.get("ENABLE_NER", "1") != "0"

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
    """Safely downloads model weights with timeout and error handling."""
    if destination.exists() and destination.stat().st_size > 0:
        return True
    try:
        print(f"[ML-Pipeline] Downloading model weights to {destination}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(destination, "wb") as out_file:
            out_file.write(response.read())
        print(f"[ML-Pipeline] Successfully downloaded {destination.name}")
        return True
    except Exception as e:
        print(f"[ML-Pipeline] Warning: Could not download {destination.name}: {e}")
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
        return False


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
            cls._instance._face_failed = False
            cls._instance._plate_failed = False
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
            print(f"[ML-Pipeline] {name} weights unavailable; {name} detection disabled.")
            return None
        try:
            return YOLO(str(path))
        except Exception as e:
            print(f"[ML-Pipeline] Error loading {name} model: {e}")
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
        if self._ocr_reader is None:
            with self._lock:
                if self._ocr_reader is None:
                    import easyocr
                    try:
                        import torch
                        use_gpu = torch.cuda.is_available()
                    except Exception:
                        use_gpu = False
                    easyocr_storage = WEIGHTS_DIR / "easyocr"
                    easyocr_storage.mkdir(parents=True, exist_ok=True)
                    # verbose=False avoids Windows cp1252 charmap encoding errors with progress bar characters
                    self._ocr_reader = easyocr.Reader(
                        OCR_LANGS,
                        gpu=use_gpu,
                        model_storage_directory=str(easyocr_storage),
                        verbose=False
                    )
        return self._ocr_reader

    @property
    def ner(self):
        """Optional BERT NER recognizer; returns None when disabled or unavailable."""
        if not ENABLE_NER:
            return None
        if self._ner is None:
            self._ner = pii.NERRecognizer(cache_dir=str(WEIGHTS_DIR / "hf"))
        return self._ner if self._ner.available else None


# Global singleton instance
_model_manager = ModelManager()


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


def _iou(a: List[int], b: List[int]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _plate_text_is_valid(text: str) -> bool:
    """
    Real licence plates (US/EU/UK/...) contain at least one digit and 4-10
    alphanumerics. This rejects the classic false positive of the plate detector
    firing on a printed word or name (e.g. "Anthony Caldwell" in a letter).
    """
    alnum = re.sub(r"[^A-Za-z0-9\u0660-\u0669\u06F0-\u06F9\u0600-\u06FF]", "", text)
    has_digit = any(c.isdigit() for c in alnum)
    return has_digit and 4 <= len(alnum) <= 12


# --------------------------------------------------------------------------- #
# OCR line reconstruction and span -> polygon mapping
# --------------------------------------------------------------------------- #

class _OCRBox:
    __slots__ = ("pts", "text", "conf", "cx", "cy", "h", "x1", "x2", "y1", "y2", "start", "end")

    def __init__(self, bbox, text: str, conf: float):
        self.pts = [[float(p[0]), float(p[1])] for p in bbox]
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
    pad = 0.5
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


def _ocr_pii_detections(ocr_results, width: int, height: int) -> List[Dict[str, Any]]:
    boxes = []
    for bbox, text, conf in ocr_results:
        t = str(text).strip()
        if t and float(conf) >= OCR_MIN_CONFIDENCE:
            boxes.append(_OCRBox(bbox, t, float(conf)))
    if not boxes:
        return []

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

    detections: List[Dict[str, Any]] = []
    for span in spans:
        for line in lines:
            hit = [b for b in line if b.start < span.end and b.end > span.start]
            if not hit:
                continue
            first, last = hit[0], hit[-1]
            left = _sub_polygon(first, max(0, span.start - first.start), first.end - first.start)
            right = _sub_polygon(last, 0, min(span.end, last.end) - last.start)
            poly = [left[0], right[1], right[2], left[3]]
            poly = _expand_polygon(poly, ratio_h=0.08, ratio_w=0.01)
            polygon = [[_clamp(int(round(x)), 0, width), _clamp(int(round(y)), 0, height)] for x, y in poly]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
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


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Converts image to grayscale and applies CLAHE to enhance text contrast
    against complex security backgrounds and watermarks.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(gray)


def analyze_image_entities(image_input: Union[str, Path, np.ndarray, bytes], conf_threshold: float = 0.40) -> Dict[str, Any]:
    """
    Analyzes an input image to detect sensitive entities:
      1. Faces (YOLOv8-face)
      2. License plates (YOLOv8 plate detector + OCR cascade validation)
      3. PII inside text (EasyOCR -> line reconstruction -> anonymizer.pii recognizers)

    Args:
        image_input: Path to image file, raw bytes, or OpenCV numpy BGR frame.
        conf_threshold: Minimum confidence score to retain a face / plate detection.

    Returns:
        {
            "status": "success",
            "image_dimensions": {"width": w, "height": h},
            "detections": [
                {"id": 1, "type": "face", "bbox": [x1, y1, x2, y2], "polygon": [...], "score": 0.94},
                {"id": 2, "type": "text", "entity": "EMAIL", "label": "Email", ...},
                ...
            ]
        }
    """
    image = _load_image(image_input)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    height, width = image.shape[:2]

    detections: List[Dict[str, Any]] = []

    # 1. Faces -------------------------------------------------------------- #
    try:
        face_model = _model_manager.face_model
        if face_model is not None:
            for r in face_model.predict(source=image, conf=conf_threshold, verbose=False):
                for box in r.boxes:
                    score = float(box.conf[0].item())
                    if score < conf_threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    # Detector boxes are tight on the inner face; expand (never shrink)
                    # so hairline, ears and jaw are also covered.
                    w, h = x2 - x1, y2 - y1
                    x1 -= w * FACE_EXPAND_RATIO
                    x2 += w * FACE_EXPAND_RATIO
                    y1 -= h * FACE_EXPAND_RATIO * 1.5
                    y2 += h * FACE_EXPAND_RATIO
                    bx = [_clamp(int(x1), 0, width), _clamp(int(y1), 0, height),
                          _clamp(int(math.ceil(x2)), 0, width), _clamp(int(math.ceil(y2)), 0, height)]
                    if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
                        continue
                    detections.append({
                        "type": "face",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        print(f"[ML-Pipeline] Face detection warning: {e}")

    # 2. Licence plates with OCR cascade validation ------------------------- #
    try:
        plate_model = _model_manager.plate_model
        if plate_model is not None:
            plate_conf = max(conf_threshold, PLATE_MIN_CONFIDENCE)
            for r in plate_model.predict(source=image, conf=plate_conf, verbose=False):
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

                    if score < PLATE_CASCADE_BELOW:
                        crop = image[bx[1]:bx[3], bx[0]:bx[2]]
                        if crop.shape[0] < 32:
                            scale = 32.0 / crop.shape[0]
                            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        plate_ocr = _model_manager.ocr_reader.readtext(crop, detail=1, paragraph=False)
                        plate_text = "".join(str(item[1]) for item in plate_ocr)
                        if not _plate_text_is_valid(plate_text):
                            continue

                    detections.append({
                        "type": "plate",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        print(f"[ML-Pipeline] Plate detection warning: {e}")

    # 3. PII in text --------------------------------------------------------- #
    try:
        ocr_reader = _model_manager.ocr_reader
        longest = max(width, height)
        # Upscale small images so that small print is still legible to the recogniser.
        mag_ratio = 1.5 if longest < 1280 else 1.0
        ocr_image = preprocess_for_ocr(image)
        ocr_results = ocr_reader.readtext(
            ocr_image,
            detail=1,
            paragraph=False,
            canvas_size=2560,
            mag_ratio=mag_ratio,
            text_threshold=0.5,
            low_text=0.35,
            link_threshold=0.4,
            width_ths=0.5,      # keep boxes close to word level for precise span mapping
            add_margin=0.05,
        )
        text_dets = _ocr_pii_detections(ocr_results, width, height)

        # A plate already detected is masked with DP-Pix; avoid a duplicate text mask
        # over the same characters.
        plates = [d["bbox"] for d in detections if d["type"] == "plate"]
        for d in text_dets:
            if any(_iou(d["bbox"], p) > 0.5 for p in plates):
                continue
            detections.append(d)
    except Exception as e:
        print(f"[ML-Pipeline] Text OCR detection warning: {e}")

    for i, d in enumerate(detections, start=1):
        d["id"] = i

    return {
        "status": "success",
        "image_dimensions": {
            "width": int(width),
            "height": int(height)
        },
        "detections": detections
    }


if __name__ == "__main__":
    import json

    print("--- Running ML Pipeline Self-Test ---")

    # Synthetic document: PII lines + a prose line that must NOT be flagged.
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
