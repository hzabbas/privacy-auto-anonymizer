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

try:
    import imajev
    HAS_IMAJEV = True
except ImportError:
    HAS_IMAJEV = False

from . import pii

# Tunable thresholds -------------------------------------------------------- #
# Plate detector is noisy on documents (it fires on printed words), so it uses a
# stricter floor and OCR verification below PLATE_CASCADE_BELOW.
PLATE_MIN_CONFIDENCE = 0.45
PLATE_CASCADE_BELOW = 0.60
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


    @property
    def imajev_model(self):
        if not hasattr(self, '_imajev_model'):
            self._imajev_model = None
        if self._imajev_model is None and HAS_IMAJEV:
            with self._lock:
                if self._imajev_model is None:
                    try:
                        if hasattr(imajev, 'ImajevModel'):
                            self._imajev_model = imajev.ImajevModel()
                        elif hasattr(imajev, 'load_model'):
                            self._imajev_model = imajev.load_model()
                        elif hasattr(imajev, 'pipeline'):
                            self._imajev_model = imajev.pipeline()
                        else:
                            self._imajev_model = imajev
                        print("[ML-Pipeline] Loaded imajev VLM.")
                    except Exception as e:
                        print(f"[ML-Pipeline] Warning: Failed to initialize imajev ({e}).")
                        self._imajev_model = None
        return self._imajev_model

    @property
    def paddle_ocr_reader(self):
        if not hasattr(self, '_paddle_ocr_reader'):
            self._paddle_ocr_reader = None
        if self._paddle_ocr_reader is None:
            with self._lock:
                if self._paddle_ocr_reader is None:
                    try:
                        import inspect
                        from paddleocr import PaddleOCR
                        
                        paddle_kwargs = {
                            'use_textline_orientation': False,
                            'lang': 'en',
                            'enable_mkldnn': False,
                        }
                        sig = inspect.signature(PaddleOCR.__init__)
                        if 'use_doc_orientation_classify' in sig.parameters:
                            paddle_kwargs['use_doc_orientation_classify'] = False
                        if 'use_doc_unwarping' in sig.parameters:
                            paddle_kwargs['use_doc_unwarping'] = False

                        self._paddle_ocr_reader = PaddleOCR(**paddle_kwargs)
                        print("[ML-Pipeline] Loaded PaddleOCR (Persian/English).")
                    except ImportError:
                        print("[ML-Pipeline] Warning: PaddleOCR not installed.")
                        self._paddle_ocr_reader = None
                    except Exception as e:
                        print(f"[ML-Pipeline] Warning: Failed to initialize PaddleOCR ({e}).")
                        self._paddle_ocr_reader = None
        return self._paddle_ocr_reader

    @property
    def doctr_reader(self):
        if not hasattr(self, '_doctr_reader'):
            self._doctr_reader = None
        if self._doctr_reader is None:
            with self._lock:
                if self._doctr_reader is None:
                    try:
                        from doctr.models import ocr_predictor
                        self._doctr_reader = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True, assume_straight_pages=True)
                        print("[ML-Pipeline] Loaded DocTR. Note: Standard DocTR weights are Latin-only. Persian/Arabic text may not be recognized.")
                    except ImportError:
                        print("[ML-Pipeline] Warning: DocTR not installed.")
                        self._doctr_reader = None
        return self._doctr_reader
        
    @property
    def layout_parser(self):
        if not hasattr(self, '_layout_parser'):
            self._layout_parser = None
        if self._layout_parser is None:
            with self._lock:
                if self._layout_parser is None:
                    try:
                        import layoutparser as lp
                        self._layout_parser = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5])
                        print("[ML-Pipeline] Loaded LayoutParser.")
                    except ImportError:
                        print("[ML-Pipeline] Warning: LayoutParser not installed.")
                        self._layout_parser = None
        return self._layout_parser

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
    Real licence plates (US/EU/UK/Iran/...) contain alphanumeric characters,
    often with digits or uppercase letters (e.g. 'B 11 HOY', '21B34522').
    This rejects the classic false positive of the plate detector
    firing on printed words or names in text documents (e.g. 'Anthony Caldwell').
    """
    if not text.strip():
        # If crop is too blurry or distant for OCR to read, do not discard a detected vehicle plate
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
            poly = _expand_polygon(poly, ratio_h=0.02, ratio_w=0.005)
            polygon = [[_clamp(int(round(x)), 0, width), _clamp(int(round(y)), 0, height)] for x, y in poly]
            
            if M_inv is not None:
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                orig_pts = cv2.transform(pts, M_inv)
                polygon = orig_pts.reshape(-1, 2).astype(np.int32).tolist()

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


def preprocess_for_ocr(img_bgr: np.ndarray):
    """
    Attempts to deskew the image based on MRZ region.
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

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
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
        print("[ML-Pipeline] PaddleOCR unavailable, falling back to EasyOCR.")
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


def _run_doctr_pipeline(image: np.ndarray, width: int, height: int, is_crop: bool = False) -> List[Dict[str, Any]]:
    doctr = _model_manager.doctr_reader
    if doctr is None:
        print("[ML-Pipeline] DocTR unavailable, falling back to EasyOCR.")
        return _run_easyocr_pipeline(image, width, height, is_crop)
        
    from doctr.io import DocumentFile
    
    def _run_doctr_on_img(img_array):
        _, encoded_image = cv2.imencode('.png', img_array)
        doc = DocumentFile.from_images([encoded_image.tobytes()])
        result = doctr(doc)
        res_list = []
        img_h, img_w = img_array.shape[:2]
        for page in result.pages:
            page_h, page_w = page.dimensions
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(word.value for word in line.words)
                    if not text.strip(): continue
                    
                    if text.count('<') >= 2:
                        text = text.replace(" ", "")
                        
                    conf = sum(word.confidence for word in line.words) / len(line.words)
                    xmin, ymin = line.geometry[0]
                    xmax, ymax = line.geometry[1]
                    bbox = [[xmin*page_w, ymin*page_h], [xmax*page_w, ymin*page_h], [xmax*page_w, ymax*page_h], [xmin*page_w, ymax*page_h]]
                    res_list.append((bbox, text, conf))
        return res_list
        
    if is_crop:
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


def _validate_crop_with_imajev(crop_img: np.ndarray, entity_type: str, initial_score: float = None) -> bool:
    if not HAS_IMAJEV:
        if entity_type == "plate" and initial_score is not None and initial_score < 0.60:
            return False
        return True
    try:
        vlm = _model_manager.imajev_model
        if not vlm:
            if entity_type == "plate" and initial_score is not None and initial_score < 0.60:
                return False
            return True
            
        if entity_type == "plate":
            prompt = "Is this a clear, valid license plate? Answer YES or NO."
        elif entity_type == "face":
            prompt = "Is this a clear, valid human face? Answer YES or NO."
        else:
            prompt = "Is this a valid identification text? Answer YES or NO."
            
        if hasattr(vlm, 'generate'):
            resp = vlm.generate(crop_img, prompt)
        elif hasattr(vlm, '__call__'):
            resp = vlm(crop_img, prompt)
        elif hasattr(vlm, 'predict'):
            resp = vlm.predict(crop_img, prompt)
        else:
            resp = "yes"
            
        resp_text = str(resp).lower()
        if "no" in resp_text or "false" in resp_text or "negative" in resp_text:
            return False
        return True
    except Exception as e:
        print(f"[ML-Pipeline] Imajev validation failed: {e}. Falling back to safe handling.")
        if entity_type == "plate" and initial_score is not None and initial_score < 0.60:
            return False
        return True


def analyze_image_entities(
    image_input: Union[str, Path, np.ndarray, bytes], 
    conf_threshold: float = 0.40,
    ocr_engine: str = 'easyocr',
    layout_engine: str = 'regex',
    use_imajev: bool = False
) -> Dict[str, Any]:
    """
    Analyzes an input image to detect sensitive entities.
    Now supports modular dynamic routing based on requested OCR and Layout engines,
    and optional Imajev VLM reasoning and crop validation.
    """
    image = _load_image(image_input)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    height, width = image.shape[:2]

    detections: List[Dict[str, Any]] = []
    imajev_logs: List[str] = []

    skip_ocr = False
    if use_imajev:
        imajev_logs.append("🚀 مدل استدلالگر هوشمند Imajev VLM فعال شد.")
        if HAS_IMAJEV:
            try:
                vlm = _model_manager.imajev_model
                if vlm:
                    prompt = "Classify this image: 1. Document/Form/ID/Text, 2. Street/Landscape/People. Answer with 1 or 2."
                    if hasattr(vlm, 'generate'):
                        resp = vlm.generate(image, prompt)
                    elif hasattr(vlm, '__call__'):
                        resp = vlm(image, prompt)
                    elif hasattr(vlm, 'predict'):
                        resp = vlm.predict(image, prompt)
                    else:
                        resp = ""
                    resp_text = str(resp).lower()
                    
                    if "street" in resp_text or "landscape" in resp_text or "people" in resp_text or "2" in resp_text:
                        skip_ocr = True
                        log_msg = "⚡ Imajev: تصویر به عنوان 'فضای باز / عمومی' طبقه‌بندی شد (پردازش سنگین OCR لغو شد)"
                        imajev_logs.append(log_msg)
                        print(f"[ML-Pipeline] {log_msg}")
                    else:
                        log_msg = "🔍 Imajev: تصویر به عنوان 'سند / متن' طبقه‌بندی شد (پردازش OCR فعال است)"
                        imajev_logs.append(log_msg)
                        print(f"[ML-Pipeline] {log_msg}")
            except Exception as e:
                err_msg = f"⚠️ Imajev: خطا در مسیریابی هوشمند ({e})، استفاده از روال استاندارد"
                imajev_logs.append(err_msg)
                print(f"[ML-Pipeline] {err_msg}")
        else:
            imajev_logs.append("⚠️ ماژول Imajev روی سیستم بارگذاری نشد؛ پردازش به صورت استاندارد انجام می‌شود.")

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
                            if not _validate_crop_with_imajev(crop, "face", initial_score=score):
                                imajev_logs.append(f"❌ Imajev: کادر چهره با اطمینان مرزی ({score:.2f}) تایید نشد و حذف شد")
                                continue
                            else:
                                imajev_logs.append(f"✅ Imajev: کادر چهره با اطمینان ({score:.2f}) توسط مدل تایید شد")

                    detections.append({
                        "type": "face",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        print(f"[ML-Pipeline] Face detection warning: {e}")

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

                    if score < PLATE_CASCADE_BELOW:
                        crop = image[bx[1]:bx[3], bx[0]:bx[2]]
                        if crop.size > 0 and use_imajev:
                            if not _validate_crop_with_imajev(crop, "plate", initial_score=score):
                                imajev_logs.append(f"❌ Imajev: کادر پلاک با اطمینان مرزی ({score:.2f}) تایید نشد و حذف شد")
                                continue
                            else:
                                imajev_logs.append(f"✅ Imajev: کادر پلاک با اطمینان ({score:.2f}) توسط مدل تایید شد")
                        if crop.shape[0] < 32:
                            scale = 32.0 / crop.shape[0]
                            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        plate_ocr = _model_manager.ocr_reader.readtext(crop, detail=1, paragraph=False)
                        plate_text = "".join(str(item[1]) for item in plate_ocr)
                        if not _plate_text_is_valid(plate_text):
                            if use_imajev:
                                imajev_logs.append("❌ Imajev/OCR: کاراکترهای پلاک با الگوی معتبر همخوانی نداشت و حذف شد")
                            continue

                    detections.append({
                        "type": "plate",
                        "bbox": bx,
                        "polygon": _rect_polygon(*bx),
                        "score": round(score, 2),
                    })
    except Exception as e:
        print(f"[ML-Pipeline] Plate detection warning: {e}")

    # 3. Dynamic OCR & Layout strategy -------------------------------------- #
    try:
        lp_blocks = []
        if not skip_ocr and layout_engine == 'layoutparser':
            lp_model = _model_manager.layout_parser
            if lp_model is None:
                print("[ML-Pipeline] LayoutParser not available, falling back to regex.")
                layout_engine = 'regex'
            else:
                layout = lp_model.detect(image)
                # Filter for Text blocks
                for block in layout:
                    if block.type in ["Text", "Title", "List"]:
                        x1, y1, x2, y2 = block.coordinates
                        lp_blocks.append([int(x1), int(y1), int(x2), int(y2)])

        text_dets = []
        if skip_ocr:
            pass
        elif layout_engine == 'layoutparser' and lp_blocks:
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

        plates = [d["bbox"] for d in detections if d["type"] == "plate"]
        for d in text_dets:
            if any(_iou(d["bbox"], p) > 0.5 for p in plates):
                continue
            
            score = d.get("score", 1.0)
            if use_imajev and score < 0.5:
                bx = d["bbox"]
                crop = image[max(0, int(bx[1])):min(height, int(bx[3])), max(0, int(bx[0])):min(width, int(bx[2]))]
                if crop.size > 0:
                    if not _validate_crop_with_imajev(crop, "text", initial_score=score):
                        imajev_logs.append(f"❌ Imajev: قطعه متن مشکوک '{d.get('text', '')[:12]}' حذف شد")
                        continue
                    else:
                        imajev_logs.append(f"✅ Imajev: قطعه متن '{d.get('text', '')[:12]}' تایید شد")
                    
            detections.append(d)
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"[ML-Pipeline] Text OCR detection warning: {e}")
        # Allow other engines to fail gracefully unless it's a direct ValueError from our strict handling

    if use_imajev and len(imajev_logs) == 1:
        imajev_logs.append("✨ Imajev: همه عناصر شناسایی‌شده دارای ضریب اطمینان بالا بودند و نیازی به فیلتر ثانویه نبود.")

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

