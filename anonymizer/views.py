"""HTTP layer: the single-page UI and the image-analysis JSON API."""

import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .ml_pipeline import analyze_image_entities, get_engine_status

logger = logging.getLogger(__name__)

# Hard cap on the accepted upload size (bytes). Enforced before decoding so a
# huge file cannot exhaust memory.
MAX_UPLOAD_SIZE = getattr(settings, "MAX_UPLOAD_SIZE", 30 * 1024 * 1024)

ALLOWED_OCR_ENGINES = {"easyocr", "paddleocr", "doctr"}
ALLOWED_LAYOUT_ENGINES = {"regex", "layoutparser", "doctr_layout"}

DEFAULT_CONF_THRESHOLD = 0.40


@ensure_csrf_cookie
def index(request):
    """
    Renders the main single-page interface.

    The engine capability map is injected into the template so engines that are
    not installed can be shown as disabled instead of silently falling back.
    """
    return render(request, "anonymizer/index.html", {
        "engine_status": get_engine_status(),
        "max_upload_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
    })


@require_POST
def analyze_image_api(request):
    """
    Analyzes an uploaded image and returns the sensitive entities (faces, licence
    plates, PII text) with their bounding boxes and polygons.

    Privacy consideration: the image is read straight from the in-memory upload
    (PIL / OpenCV decoding inside ``ml_pipeline``) and never written to disk, so no
    artifact of the original document is left behind. ``FILE_UPLOAD_MAX_MEMORY_SIZE``
    in settings is raised for the same reason: it keeps the whole upload in RAM.

    Expects (multipart/form-data):
        - file under key ``image`` (or the first uploaded file)
        - optional ``conf_threshold`` float  (default 0.40, clamped to [0.05, 0.95])
        - optional ``ocr_engine``:     easyocr | paddleocr | doctr
        - optional ``layout_engine``:  regex | doctr_layout | layoutparser
        - optional ``use_imajev``:     true | false

    Returns:
        JSON: {"status": "success", "image_dimensions": {...}, "detections": [...],
               "imajev_logs": [...]}
    """
    if not request.FILES:
        return JsonResponse({
            "status": "error",
            "message": "No image file provided in request.FILES."
        }, status=400)

    uploaded_file = request.FILES.get("image") or next(iter(request.FILES.values()))

    if uploaded_file.size == 0:
        return JsonResponse({
            "status": "error",
            "message": "Uploaded image file is empty."
        }, status=400)

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        return JsonResponse({
            "status": "error",
            "message": (
                f"Image is too large ({uploaded_file.size // (1024 * 1024)} MB). "
                f"The limit is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            )
        }, status=413)

    # Optional confidence-threshold override (clamped to a sane range).
    try:
        conf_threshold = float(request.POST.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
    except (TypeError, ValueError):
        conf_threshold = DEFAULT_CONF_THRESHOLD
    conf_threshold = max(0.05, min(conf_threshold, 0.95))

    ocr_engine = request.POST.get("ocr_engine", "easyocr")
    if ocr_engine not in ALLOWED_OCR_ENGINES:
        ocr_engine = "easyocr"

    layout_engine = request.POST.get("layout_engine", "regex")
    if layout_engine not in ALLOWED_LAYOUT_ENGINES:
        layout_engine = "regex"

    use_imajev = request.POST.get("use_imajev", "false").lower() in ("true", "1", "yes")

    try:
        # Read the upload into memory; ml_pipeline handles decoding and EXIF rotation.
        file_bytes = uploaded_file.read()

        analysis_result = analyze_image_entities(
            file_bytes,
            conf_threshold=conf_threshold,
            ocr_engine=ocr_engine,
            layout_engine=layout_engine,
            use_imajev=use_imajev
        )
        return JsonResponse(analysis_result, status=200)

    except ValueError as ve:
        # Undecodable / oversized / otherwise invalid input.
        logger.warning("Rejected upload: %s", ve)
        return JsonResponse({
            "status": "error",
            "message": str(ve)
        }, status=400)
    except Exception as exc:
        logger.exception("Error analyzing image entities: %s", exc)
        return JsonResponse({
            "status": "error",
            "message": f"Internal server error during image analysis: {str(exc)}"
        }, status=500)
