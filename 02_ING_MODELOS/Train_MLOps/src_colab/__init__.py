"""src_colab — Cycle 2 public API (PyTorch only).

Re-exports the most commonly used symbols so that notebooks and
task entry-points can do::

    from src_colab import (
        build_fcos_model, train_two_phase, evaluate_pytorch_model, ...
    )
"""

# -- config & helpers ------------------------------------------------
from .config import (                                       # noqa: F401
    MODEL_FAMILIES,
    MODEL_VARIANTS,
    TRAINING_FRAMEWORKS,
    BASE_IMG_SIZE,
    DATASET_MASTER_CLASSES,
    is_fcos_family,
    is_yolo26_custom_family,
    is_espdet_family,
    is_export_family,
    is_pytorch_family,
    get_torch_device,
)

from .utils_io import (                                     # noqa: F401
    log,
    safe_mkdir,
    download_from_gcs,
    upload_to_gcs,
    setup_experiment_dirs,
)

from .utils_widgets import (                                # noqa: F401
    ExperimentSetup,
    build_experiment_setup,
    build_experiment_setup_from_yaml,
)

# -- data ------------------------------------------------------------
from .utils_data import (                                   # noqa: F401
    IODCDataset,
    iodc_collate_fn,
    create_dataloader,
    verify_yolo_dataset,
    generate_data_yaml,
    delete_yolo_cache,
    get_class_distribution_yolo,
    calculate_class_weights,
    plot_class_distribution,
    create_yolo_working_copy,
    split_yolo_dataset,
    visualize_gt_samples_per_class,
)

# -- models ----------------------------------------------------------
from .utils_model import (                                  # noqa: F401
    # FCOS
    FCOSModel,
    SimpleFPN,
    FCOSHead,
    build_fcos_model,
    FCOS_SPECS,
    # YOLO26 Custom
    build_yolo26_custom_model,
    get_yolo26_custom_torch_model,
    YOLO26_CUSTOM_SPECS,
    # ESPDet
    ESPDetPico,
    ESPDetPicoBackbone,
    ESPDetPicoHead,
    build_espdet_pico,
    ESPDET_SPECS,
    # Utilities
    freeze_backbone,
    unfreeze_all,
    print_model_summary,
    estimate_model_size,
)

# -- training --------------------------------------------------------
from .utils_train import (                                  # noqa: F401
    TwoPhaseConfig,
    TwoPhaseHistory,
    PhaseHistory,
    build_fcos_loss,
    build_espdet_loss,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    validate_one_epoch,
    train_two_phase,
    Yolo26CustomConfig,
    train_yolo26_custom,
    validate_yolo26_custom,
    save_two_phase_history,
)

# -- metrics ---------------------------------------------------------
from .utils_metrics import (                                # noqa: F401
    TrainingHistory,
    extract_two_phase_history,
    extract_yolo26_history,
    plot_training_curves,
    plot_loss_comparison,
    print_training_summary,
)

# -- evaluation ------------------------------------------------------
from .utils_eval import (                                   # noqa: F401
    Detection,
    EvaluationResults,
    evaluate_pytorch_model,
    evaluate_yolo26_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    save_evaluation,
)

# -- inference -------------------------------------------------------
from .utils_infer import (                                  # noqa: F401
    DetectedObject,
    predict_fcos,
    predict_yolo26_custom,
    predict_espdet,
    visualize_predictions,
    compare_predictions_side_by_side,
)

# -- export ----------------------------------------------------------
from .utils_export import (                                 # noqa: F401
    ExportResult,
    OnnxVerificationResult,
    export_pytorch_to_onnx,
    export_yolo26_to_onnx,
    verify_onnx_model,
    inspect_onnx_model,
)

# -- comparison ------------------------------------------------------
from .utils_compare import (                                # noqa: F401
    ComparisonResult,
    compare_detections,
    plot_iou_distribution,
    plot_confidence_scatter,
    plot_detection_count_comparison,
    print_comparison_summary,
    visualize_comparison_grid,
)

# -- experiment tracking ---------------------------------------------
from .utils_experiment import (                             # noqa: F401
    TwoPhaseTrainConf,
    Yolo26TrainConf,
    UnifiedExperimentConfig,
    UnifiedExperiment,
    create_experiment_from_setup,
    save_experiment,
    load_experiment,
    load_experiments,
    plot_experiments_comparison,
    plot_experiments_radar,
    print_experiments_table,
)
