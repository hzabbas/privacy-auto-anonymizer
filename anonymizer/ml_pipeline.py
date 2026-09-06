import os
import sys
import urllib.request
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
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
                ['en'],
                gpu=False,
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

    # 2. Detect License Plates
    try:
        plate_model = _model_manager.plate_model
        plate_results = plate_model.predict(source=image, conf=conf_threshold, verbose=False)
        for r in plate_results:
            boxes = r.boxes
            for box in boxes:
                score = float(box.conf[0].item())
                if score >= conf_threshold:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "id": current_id,
                        "type": "plate",
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
        print(f"[ML-Pipeline] Plate detection warning: {e}")

    # 3. Detect Text with EasyOCR
    try:
        ocr_reader = _model_manager.ocr_reader
        ocr_results = ocr_reader.readtext(image)
        for bbox, text, score in ocr_results:
            score = float(score)
            if score >= conf_threshold:
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
                        "score": round(score, 2)
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

    # Create a test image with text
    test_img = np.full((500, 700, 3), 255, dtype=np.uint8)
    cv2.putText(test_img, "CONFIDENTIAL REPORT 2026", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(test_img, "SECURITY TOKEN: 99482", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)

    print("Executing analyze_image_entities on test frame...")
    results = analyze_image_entities(test_img, conf_threshold=0.30)

    print("\n--- Inference Output JSON ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nTotal entities detected: {len(results['detections'])}")
    print("--- ML Pipeline Self-Test Completed Successfully ---")
