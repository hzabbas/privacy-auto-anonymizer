import os
from django.apps import AppConfig


class AnonymizerConfig(AppConfig):
    name = 'anonymizer'

    def ready(self):
        if os.environ.get('RUN_MAIN') == 'true':
            from .ml_pipeline import _model_manager
            _ = _model_manager.face_model
            _ = _model_manager.plate_model
            _ = _model_manager.ocr_reader
