from .evaluation_artifact_writer import (
    save_detection_evaluation_artifacts,
    save_evaluation_artifacts,
)
from .yolo_prediction_evaluator import (
    evaluate_coco_predictions,
    evaluate_detection_samples,
    evaluate_yolo_label_predictions,
    evaluate_yolo_txt_predictions,
)

__all__ = [
    "evaluate_coco_predictions",
    "evaluate_detection_samples",
    "evaluate_yolo_label_predictions",
    "evaluate_yolo_txt_predictions",
    "save_detection_evaluation_artifacts",
    "save_evaluation_artifacts",
    "evaluate_vlm_classification_predictions",
    "save_vlm_classification_artifacts",
]


def __getattr__(name: str):
    if name == "evaluate_vlm_classification_predictions":
        from .vlm_classification_evaluator import evaluate_vlm_classification_predictions

        return evaluate_vlm_classification_predictions
    if name == "save_vlm_classification_artifacts":
        from .vlm_classification_artifact_writer import save_vlm_classification_artifacts

        return save_vlm_classification_artifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
