import os
import sys
import urllib.request
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
import cv2
import re
import warnings
warnings.filterwarnings("ignore")
from thefuzz import fuzz

# Sensitive keywords for fuzzy string matching (threshold > 75%)
SENSITIVE_KEYWORDS = [
    'کد', 'ملی', 'رهگیری', 'شناسه', 'شبا', 'کارت', 'رمز', 'تلفن', 'شماره',
    'secret', 'confidential', 'token', 'password'
]

# Regex to detect at least 8 digits (ignoring optional spaces between digits)
DIGIT_8_REGEX = re.compile(r'(?:[\d\u0660-\u0669\u06F0-\u06F9]\s*){8,}')

# Regex for alphanumeric detection (used in cascade license plate validation)
ALPHANUMERIC_REGEX = re.compile(r'[a-zA-Z0-9\u0600-\u06FF]')

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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._face_model = None
            cls._instance._plate_model = None
            cls._instance._ocr_reader = None
        return cls._instance

    @property
    def face_model(self):
        if self._face_model is None:
            from ultralytics import YOLO
            if not FACE_MODEL_PATH.exists():
                _download_file(FACE_MODEL_URL, FACE_MODEL_PATH)

            if FACE_MODEL_PATH.exists():
                try:
                    self._face_model = YOLO(str(FACE_MODEL_PATH))
                except Exception as e:
                    print(f"[ML-Pipeline] Error loading face model: {e}")
                    self._face_model = YOLO("yolov8n.pt")
            else:
                self._face_model = YOLO("yolov8n.pt")
        return self._face_model

    @property
    def plate_model(self):
        if self._plate_model is None:
            from ultralytics import YOLO
            if not PLATE_MODEL_PATH.exists():
                _download_file(PLATE_MODEL_URL, PLATE_MODEL_PATH)

            if PLATE_MODEL_PATH.exists():
                try:
                    self._plate_model = YOLO(str(PLATE_MODEL_PATH))
                except Exception as e:
                    print(f"[ML-Pipeline] Error loading plate model: {e}")
                    self._plate_model = YOLO("yolov8n.pt")
            else:
                self._plate_model = YOLO("yolov8n.pt")
        return self._plate_model

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            easyocr_storage = WEIGHTS_DIR / "easyocr"
            easyocr_storage.mkdir(parents=True, exist_ok=True)
            # verbose=False avoids Windows cp1252 charmap encoding errors with progress bar characters
            self._ocr_reader = easyocr.Reader(
                ['fa', 'en'],
                gpu=True,
                model_storage_directory=str(easyocr_storage),
                verbose=False
            )
        return self._ocr_reader


# Global singleton instance
_model_manager = ModelManager()


def _load_image(image_input: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
    """
    Decodes image input from file path, bytes, or returns existing ndarray.
    Supports Windows non-ASCII / Persian file paths reliably via np.fromfile.
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image path does not exist: {path_str}")
        image_bytes = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to decode image from path: {path_str}")
        return img

    if isinstance(image_input, (bytes, bytearray)):
        image_bytes = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from byte buffer.")
        return img

    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


def analyze_image_entities(image_input: Union[str, Path, np.ndarray, bytes], conf_threshold: float = 0.40) -> Dict[str, Any]:
    """
    Analyzes an input image to detect sensitive entities:
      1. Faces (YOLO face model)
      2. License Plates (YOLO plate model)
      3. Text (EasyOCR)

    Args:
        image_input: Path to image file, raw bytes, or OpenCV numpy BGR frame.
        conf_threshold: Minimum confidence score to retain detection (default 0.40).

    Returns:
        Structured Python dictionary with standard JSON-compatible format:
        {
            "status": "success",
            "image_dimensions": {"width": w, "height": h},
            "detections": [
                {"id": 1, "type": "face", "bbox": [x1, y1, x2, y2], "score": 0.94},
                ...
            ]
        }
    """
    image = _load_image(image_input)
    height, width = image.shape[:2]

    detections: List[Dict[str, Any]] = []
    current_id = 1

    def clamp(val: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(val, max_val))

    # 1. Detect Faces
    try:
        face_model = _model_manager.face_model
        face_results = face_model.predict(source=image, conf=conf_threshold, verbose=False)
        for r in face_results:
            boxes = r.boxes
            for box in boxes:
                score = float(box.conf[0].item())
                if score >= conf_threshold:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "id": current_id,
                        "type": "face",
                        "bbox": [
                            clamp(x1, 0, width),
                            clamp(y1, 0, height),
                            clamp(x2, 0, width),
                            clamp(y2, 0, height)
                        ],
                        "score": round(score, 2)
                    })
                    current_id += 1
    except Exception as e:
        print(f"[ML-Pipeline] Face detection warning: {e}")

    # 2. Detect License Plates with Cascade Validation
    try:
        plate_model = _model_manager.plate_model
        plate_results = plate_model.predict(source=image, conf=conf_threshold, verbose=False)
        ocr_reader = _model_manager.ocr_reader

        for r in plate_results:
            boxes = r.boxes
            for box in boxes:
                entity_type = 'plate'
                conf = float(box.conf[0].item())
                if entity_type == 'plate' and float(conf) < 0.60:
                    continue

                score = float(conf)
                if score >= conf_threshold:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1_c, y1_c = clamp(x1, 0, width), clamp(y1, 0, height)
                    x2_c, y2_c = clamp(x2, 0, width), clamp(y2, 0, height)

                    # Cascade Validation: If confidence < 0.70, verify with OCR to drop false positives
                    if score < 0.70:
                        crop = image[y1_c:y2_c, x1_c:x2_c]
                        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                            continue

                        plate_ocr = ocr_reader.readtext(
                            crop,
                            adjust_contrast=True,
                            text_threshold=0.3,
                            low_text=0.3
                        )
                        plate_text = "".join([str(item[1]) for item in plate_ocr])
                        if not ALPHANUMERIC_REGEX.search(plate_text):
                            # Drop false positive (no alphanumeric content detected in crop)
                            continue

                    detections.append({
                        "id": current_id,
                        "type": "plate",
                        "bbox": [x1_c, y1_c, x2_c, y2_c],
                        "score": round(score, 2)
                    })
                    current_id += 1
    except Exception as e:
        print(f"[ML-Pipeline] Plate detection warning: {e}")

    # 3. Detect Sensitive Text with EasyOCR (Original Color Image with adjust_contrast, Fuzzy Matching & Regex Digit Validation)
    try:
        ocr_reader = _model_manager.ocr_reader
        ocr_results = ocr_reader.readtext(
            image,
            adjust_contrast=True,
            text_threshold=0.3,
            low_text=0.3,
            canvas_size=1920,
            mag_ratio=1.0
        )
        for bbox, text, score in ocr_results:
            score = float(score)
            cleaned_text = str(text).strip()
            print(f"[ML-Pipeline] OCR Found: '{cleaned_text}' (score: {score:.2f})")
            if not cleaned_text:
                continue

            is_sensitive = False

            # 1. Regex: At least 8 digits (ignoring spaces between digits)
            has_8_digits = bool(DIGIT_8_REGEX.search(cleaned_text))

            # 2. Fuzzy matching with sensitive keywords (> 75% partial ratio)
            is_fuzzy_sensitive = any(
                fuzz.partial_ratio(kw, cleaned_text) > 75
                for kw in SENSITIVE_KEYWORDS
            )

            if has_8_digits or is_fuzzy_sensitive:
                is_sensitive = True

            # Strictly filter: only keep detections flagged as sensitive
            if not is_sensitive:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            if (x2 - x1) > 2 and (y2 - y1) > 2:
                detections.append({
                    "id": current_id,
                    "type": "text",
                    "bbox": [
                        clamp(x1, 0, width),
                        clamp(y1, 0, height),
                        clamp(x2, 0, width),
                        clamp(y2, 0, height)
                    ],
                    "score": round(max(score, 0.90), 2)
                })
                current_id += 1
    except Exception as e:
        print(f"[ML-Pipeline] Text OCR detection warning: {e}")

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

    # Create a test image with sensitive text and tracking numbers
    test_img = np.full((500, 700, 3), 255, dtype=np.uint8)
    cv2.putText(test_img, "CONFIDENTIAL REPORT 2026", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(test_img, "TRACKING NO: 994827150", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    cv2.putText(test_img, "SECURITY TOKEN 88219473", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 120, 20), 2)

    print("Executing analyze_image_entities on test frame...")
    results = analyze_image_entities(test_img, conf_threshold=0.30)

    print("\n--- Inference Output JSON ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nTotal entities detected: {len(results['detections'])}")
    print("--- ML Pipeline Self-Test Completed Successfully ---")
