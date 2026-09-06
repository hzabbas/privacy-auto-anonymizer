import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import numpy as np
import cv2

from .ml_pipeline import analyze_image_entities

logger = logging.getLogger(__name__)


def index(request):
    """
    Renders the main single-page interface for Privacy Auto-Anonymizer.
    """
    return render(request, 'anonymizer/index.html')


@csrf_exempt
@require_POST
def analyze_image_api(request):
    """
    RESTful API endpoint to analyze an uploaded image and extract sensitive
    entities (faces, license plates, texts) with their bounding boxes.

    Privacy Consideration:
    The image is read directly from memory (RAM) via np.frombuffer and cv2.imdecode.
    It is never written to persistent disk storage, ensuring zero artifact residue.

    Expects:
        - multipart/form-data POST request
        - File under key 'image' (or first uploaded file in request.FILES)
        - Optional 'conf_threshold' float parameter in request.POST (default 0.40)

    Returns:
        JSON response with structure:
        {
            "status": "success",
            "image_dimensions": {"width": int, "height": int},
            "detections": [
                {"id": int, "type": "face"|"plate"|"text", "bbox": [x1, y1, x2, y2], "score": float},
                ...
            ]
        }
    """
    if not request.FILES:
        return JsonResponse({
            "status": "error",
            "message": "No image file provided in request.FILES."
        }, status=400)

    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        # Fallback to the first file key if 'image' is not explicitly used
        first_key = next(iter(request.FILES))
        uploaded_file = request.FILES[first_key]

    try:
        # Read file bytes directly into memory buffer
        file_bytes = uploaded_file.read()
        if not file_bytes:
            return JsonResponse({
                "status": "error",
                "message": "Uploaded image file is empty."
            }, status=400)

        # Decode image from buffer into OpenCV BGR numpy ndarray
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return JsonResponse({
                "status": "error",
                "message": "Failed to decode image. Ensure file is a valid image format (JPEG, PNG, WEBP, etc.)."
            }, status=400)

        # Optional confidence threshold override
        conf_str = request.POST.get('conf_threshold', '0.40')
        try:
            conf_threshold = float(conf_str)
            conf_threshold = max(0.05, min(conf_threshold, 0.95))
        except ValueError:
            conf_threshold = 0.40

        # Run AI entity detection pipeline
        analysis_result = analyze_image_entities(frame, conf_threshold=conf_threshold)

        return JsonResponse(analysis_result, status=200)

    except Exception as exc:
        logger.exception("Error analyzing image entities: %s", exc)
        return JsonResponse({
            "status": "error",
            "message": f"Internal server error during image analysis: {str(exc)}"
        }, status=500)
