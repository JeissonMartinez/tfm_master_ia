"""YOLO26 training pipeline utilities for ESP32-S3 deployment.

This package provides modular utilities for:
- Dataset conversion (COCO → YOLO format)
- Model training with YOLO26n
- Metrics extraction and visualization
- TFLite INT8 export for edge deployment
- Experiment tracking and comparison

YOLO26 Features:
- NMS-free (end-to-end) inference via end2end=True
- DFL-free architecture for simpler INT8 quantization
- MuSGD optimizer (auto-selected with optimizer='auto')
- ProgLoss + STAL for improved small object detection
- 43% faster CPU inference than YOLO11
"""

from src_yolo.utils_io import (
    log,
    safe_mkdir,
    safe_read_json,
    safe_write_json,
    safe_write_text,
    safe_copy,
    safe_exists,
    safe_filesize_mb,
)
from src_yolo.utils_yolo_data import (
    create_yolo_dataset,
    convert_single_set,
    get_class_distribution,
    get_class_distribution_yolo,
    calculate_class_weights,
    verify_yolo_labels,
    prepare_yolo_from_existing,
)
from src_yolo.utils_yolo_model import (
    load_yolo26_model,
    get_model_info,
    print_model_summary,
    check_ultralytics_version,
    check_yolo26_features,
    estimate_inference_time_esp32,
    YOLO26_SPECS,
)
from src_yolo.utils_yolo_train import (
    Yolo26TrainConfig,
    train_yolo26,
    validate_yolo26,
    get_training_results_path,
)
from src_yolo.utils_yolo_metrics import (
    extract_yolo_metrics,
    plot_yolo_history,
    compare_yolo_versions,
    get_best_metrics,
    load_history_csv,
    plot_per_class_metrics,
)
from src_yolo.utils_yolo_infer import (
    load_yolo_model,
    run_yolo_inference,
    run_inference_on_dataset,
    visualize_predictions,
    visualize_yolo_predictions_grid,
    BoundingBox,
    DetectionResult,
)
from src_yolo.utils_yolo_export import (
    export_tflite,
    export_tflite_int8,
    verify_tflite_model,
    estimate_model_size,
    compare_keras_vs_tflite,
)
from src_yolo.utils_yolo_eval import (
    evaluate_model,
    calculate_map50,
    plot_confusion_matrix,
    EvaluationResults,
    compute_iou,
    match_predictions_to_gt,
    calculate_ap,
    compare_models,
)
from src_yolo.utils_yolo_experiment import (
    Yolo26ExperimentConfig,
    Yolo26ExperimentResults,
    Yolo26Experiment,
    save_experiment,
    load_experiment,
    load_all_experiments,
    compare_experiments,
    print_experiments_table,
    export_experiments_csv,
)

__all__ = [
    # IO utilities
    "log",
    "safe_mkdir",
    "safe_read_json",
    "safe_write_json",
    "safe_write_text",
    "safe_copy",
    "safe_exists",
    "safe_filesize_mb",
    # Data utilities
    "create_yolo_dataset",
    "convert_single_set",
    "get_class_distribution",
    "get_class_distribution_yolo",
    "calculate_class_weights",
    "verify_yolo_labels",
    # Model utilities
    "load_yolo26_model",
    "get_model_info",
    "print_model_summary",
    "check_ultralytics_version",
    "check_yolo26_features",
    "estimate_inference_time_esp32",
    "YOLO26_SPECS",
    # Training utilities
    "Yolo26TrainConfig",
    "train_yolo26",
    "validate_yolo26",
    "get_training_results_path",
    # Metrics utilities
    "extract_yolo_metrics",
    "plot_yolo_history",
    "compare_yolo_versions",
    "get_best_metrics",
    "load_history_csv",
    "plot_per_class_metrics",
    # Inference utilities
    "load_yolo_model",
    "run_yolo_inference",
    "run_inference_on_dataset",
    "visualize_predictions",
    "visualize_yolo_predictions_grid",
    "BoundingBox",
    "DetectionResult",
    # Export utilities
    "export_tflite",
    "export_tflite_int8",
    "verify_tflite_model",
    "estimate_model_size",
    "compare_keras_vs_tflite",
    # Evaluation utilities
    "evaluate_model",
    "calculate_map50",
    "plot_confusion_matrix",
    "EvaluationResults",
    "compute_iou",
    "match_predictions_to_gt",
    "calculate_ap",
    "compare_models",
    # Experiment utilities
    "Yolo26ExperimentConfig",
    "Yolo26ExperimentResults",
    "Yolo26Experiment",
    "save_experiment",
    "load_experiment",
    "load_all_experiments",
    "compare_experiments",
    "print_experiments_table",
    "export_experiments_csv",
]
