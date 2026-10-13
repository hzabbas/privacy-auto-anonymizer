import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AnonymizerConfig(AppConfig):
    name = 'anonymizer'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        """
        Optionally warms up the model cache inside the autoreloader child process
        so the first analysis request does not pay the YOLO + EasyOCR load cost.

        Disable with PRELOAD_MODELS=0 (useful on a memory-limited machine). A
        loading failure is logged but never fatal: every engine degrades
        independently inside ml_pipeline.
        """
        if os.environ.get('RUN_MAIN') != 'true':
            return
        if os.environ.get('PRELOAD_MODELS', '1') == '0':
            return
        try:
            from .ml_pipeline import _model_manager

            _ = _model_manager.face_model
            _ = _model_manager.plate_model
            _ = _model_manager.ocr_reader
        except Exception as exc:  # pragma: no cover - startup diagnostics only
            logger.warning("Model preload skipped: %s", exc)
