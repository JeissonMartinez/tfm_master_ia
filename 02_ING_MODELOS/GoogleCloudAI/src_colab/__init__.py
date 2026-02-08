"""src_colab — unified training toolkit for Google Colab.

Supports YOLO11, YOLO26, MobileNetV2+SSD-Lite, MobileNetV3+SSD-Lite.

Usage::

    from src_colab import (
        setup_environment,
        create_model_selector,
        verify_dataset,
        build_mobilenet_ssd,
        train_yolo,
        export_tflite_int8,
    )
"""

# ── config ───────────────────────────────────────────────────────────
from .config import (
    MODEL_FAMILIES,
    DATASET_MASTER_CLASSES,
    ColabEnvironment,
    ProjectPaths,
    setup_environment,
    is_yolo_family,
    is_mobilenet_family,
    get_yolo_device,
)

# ── I/O ──────────────────────────────────────────────────────────────
from .utils_io import (
    log,
    safe_mkdir,
    read_json,
    write_json,
    write_text,
    write_yaml,
    safe_copy,
    file_exists,
    get_file_size_mb,
)

# ── widgets ──────────────────────────────────────────────────────────
from .utils_widgets import (
    ExperimentSetup,
    create_model_selector,
    create_manual_setup,
)

# ── data ─────────────────────────────────────────────────────────────
from .utils_data import (
    verify_dataset,
    verify_yolo_dataset,
    verify_tfrecord_dataset,
    generate_data_yaml,
    get_class_distribution,
    get_class_distribution_yolo,
    create_yolo_working_copy,
    calculate_class_weights,
    plot_class_distribution,
    split_yolo_dataset,
    delete_yolo_cache,
    write_tfrecord,
    read_tfrecord_dataset,
    generate_anchors,
    compute_anchor_statistics,
    encode_targets,
    create_mobilenet_pipeline,
)

# ── model ────────────────────────────────────────────────────────────
from .utils_model import (
    YOLO11_SPECS,
    YOLO26_SPECS,
    MOBILENET_SPECS,
    load_yolo_model,
    build_mobilenet_ssd,
    print_model_summary,
    estimate_model_size,
    estimate_esp32_inference,
)

# ── training ─────────────────────────────────────────────────────────
from .utils_train import (
    YoloTrainConfig,
    train_yolo,
    validate_yolo,
    freeze_backbone,
    unfreeze_backbone_layers,
    create_ssd_loss,
    create_callbacks,
    train_mobilenet_phase1,
    train_mobilenet_phase2,
    validate_mobilenet,
    save_training_history,
    combine_histories,
)

# ── metrics ──────────────────────────────────────────────────────────
from .utils_metrics import (
    TrainingHistory,
    extract_yolo_history,
    extract_mobilenet_history,
    plot_training_curves,
    plot_loss_comparison,
    print_training_summary,
)

# ── evaluation ───────────────────────────────────────────────────────
from .utils_eval import (
    EvaluationResults,
    evaluate_yolo_model,
    evaluate_mobilenet_model,
    evaluate_tflite_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    save_evaluation,
)

# ── inference ────────────────────────────────────────────────────────
from .utils_infer import (
    DetectedObject,
    predict_yolo,
    predict_mobilenet,
    predict_tflite,
    visualize_predictions,
    visualize_gt_samples_per_class,
    compare_predictions_side_by_side,
)

# ── export ───────────────────────────────────────────────────────────
from .utils_export import (
    TFLiteExportResult,
    TFLiteVerificationResult,
    export_tflite_int8,
    export_yolo_tflite,
    export_mobilenet_tflite,
    create_representative_dataset,
    print_export_report,
    save_export_result,
)

# ── comparison ───────────────────────────────────────────────────────
from .utils_compare import (
    compare_framework_vs_tflite,
    plot_iou_distribution,
    plot_confidence_scatter,
    visualize_comparison_grid,
    save_comparison_result,
    plot_fw_vs_tflite_metrics,
    visualize_fw_vs_tflite_samples,
)

# ── experiment ───────────────────────────────────────────────────────
from .utils_experiment import (
    UnifiedExperiment,
    UnifiedExperimentConfig,
    UnifiedExperimentResults,
    save_experiment,
    load_experiment,
    load_all_experiments,
    create_experiment_from_setup,
    experiments_to_dataframe,
    plot_experiments_comparison,
    print_experiments_table,
    save_comparison_csv,
)

__all__ = [
    # config
    "MODEL_FAMILIES", "DATASET_MASTER_CLASSES", "ColabEnvironment", "ProjectPaths",
    "setup_environment", "is_yolo_family", "is_mobilenet_family", "get_yolo_device",
    # io
    "log", "safe_mkdir", "read_json", "write_json", "write_text", "write_yaml",
    "safe_copy", "file_exists", "get_file_size_mb",
    # widgets
    "ExperimentSetup", "create_model_selector", "create_manual_setup",
    # data
    "verify_dataset", "verify_yolo_dataset", "verify_tfrecord_dataset",
    "generate_data_yaml", "get_class_distribution", "get_class_distribution_yolo",
    "create_yolo_working_copy",
    "calculate_class_weights", "plot_class_distribution",
    "split_yolo_dataset", "delete_yolo_cache",
    "write_tfrecord", "read_tfrecord_dataset",
    "generate_anchors", "compute_anchor_statistics", "encode_targets",
    "create_mobilenet_pipeline",
    # model
    "YOLO11_SPECS", "YOLO26_SPECS", "MOBILENET_SPECS",
    "load_yolo_model", "build_mobilenet_ssd",
    "print_model_summary", "estimate_model_size", "estimate_esp32_inference",
    # train
    "YoloTrainConfig", "train_yolo", "validate_yolo",
    "freeze_backbone", "unfreeze_backbone_layers",
    "create_ssd_loss", "create_callbacks",
    "train_mobilenet_phase1", "train_mobilenet_phase2", "validate_mobilenet",
    "save_training_history", "combine_histories",
    # metrics
    "TrainingHistory", "extract_yolo_history", "extract_mobilenet_history",
    "plot_training_curves", "plot_loss_comparison", "print_training_summary",
    # eval
    "EvaluationResults", "evaluate_yolo_model", "evaluate_mobilenet_model",
    "evaluate_tflite_model",
    "plot_confusion_matrix", "plot_per_class_metrics", "save_evaluation",
    # infer
    "DetectedObject", "predict_yolo", "predict_mobilenet", "predict_tflite",
    "visualize_predictions", "visualize_gt_samples_per_class",
    "compare_predictions_side_by_side",
    # export
    "TFLiteExportResult", "TFLiteVerificationResult",
    "export_tflite_int8", "export_yolo_tflite", "export_mobilenet_tflite",
    "create_representative_dataset", "print_export_report", "save_export_result",
    # compare
    "compare_framework_vs_tflite", "plot_iou_distribution",
    "plot_confidence_scatter", "visualize_comparison_grid", "save_comparison_result",
    "plot_fw_vs_tflite_metrics", "visualize_fw_vs_tflite_samples",
    # experiment
    "UnifiedExperiment", "UnifiedExperimentConfig", "UnifiedExperimentResults",
    "save_experiment", "load_experiment", "load_all_experiments",
    "create_experiment_from_setup", "experiments_to_dataframe",
    "plot_experiments_comparison", "print_experiments_table", "save_comparison_csv",
]
