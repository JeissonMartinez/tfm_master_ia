"""MobileNetV3 Small + SSD-Lite utilities for ESP32-S3 deployment."""
from .utils_mobilenet_model import (
    build_mobilenetv3_ssd_lite,
    build_mobilenetv2_ssd_lite,
    ssd_lite_head,
)
from .utils_mobilenet_data import (
    COCODataGenerator,
    compute_class_weights,
    load_coco_annotations,
    generate_anchors,
    compute_anchor_statistics,
)
from .utils_mobilenet_losses import (
    focal_loss,
    smooth_l1_loss,
    ssd_combined_loss,
)
from .utils_mobilenet_train import (
    create_callbacks,
    train_two_phase,
    freeze_backbone,
    unfreeze_backbone_layers,
)
from .utils_mobilenet_export import (
    export_tflite,
    export_tflite_int8,
    apply_quantization_aware_training,
)
from .utils_mobilenet_infer import (
    Detection,
    decode_ssd_predictions,
    apply_nms,
    postprocess_detections,
    run_inference_keras,
    run_inference_tflite,
    batch_inference_keras,
    visualize_detections,
)
from .utils_mobilenet_eval import (
    GroundTruth,
    EvaluationResults,
    compute_map50,
    compute_ap,
    build_confusion_matrix,
    plot_confusion_matrix,
    evaluate_model_full,
    compare_keras_vs_tflite,
)
from .utils_mobilenet_experiment import (
    ExperimentConfig,
    ExperimentResults,
    Experiment,
    save_experiment,
    load_experiment,
    load_all_experiments,
    save_experiment_history,
    load_training_history,
    experiments_to_dataframe,
    compare_experiments,
    plot_experiments_comparison,
    plot_training_histories,
    create_experiment_summary_table,
    create_default_config,
)

__all__ = [
    # Model
    "build_mobilenetv3_ssd_lite",
    "build_mobilenetv2_ssd_lite", 
    "ssd_lite_head",
    # Data
    "COCODataGenerator",
    "compute_class_weights",
    "load_coco_annotations",
    "generate_anchors",
    "compute_anchor_statistics",
    # Losses
    "focal_loss",
    "smooth_l1_loss",
    "ssd_combined_loss",
    # Training
    "create_callbacks",
    "train_two_phase",
    "freeze_backbone",
    "unfreeze_backbone_layers",
    # Export
    "export_tflite",
    "export_tflite_int8",
    "apply_quantization_aware_training",
    # Inference
    "Detection",
    "decode_ssd_predictions",
    "apply_nms",
    "postprocess_detections",
    "run_inference_keras",
    "run_inference_tflite",
    "batch_inference_keras",
    "visualize_detections",
    # Evaluation
    "GroundTruth",
    "EvaluationResults",
    "compute_map50",
    "compute_ap",
    "build_confusion_matrix",
    "plot_confusion_matrix",
    "evaluate_model_full",
    "compare_keras_vs_tflite",
    # Experiments
    "ExperimentConfig",
    "ExperimentResults",
    "Experiment",
    "save_experiment",
    "load_experiment",
    "load_all_experiments",
    "save_experiment_history",
    "load_training_history",
    "experiments_to_dataframe",
    "compare_experiments",
    "plot_experiments_comparison",
    "plot_training_histories",
    "create_experiment_summary_table",
    "create_default_config",
]
