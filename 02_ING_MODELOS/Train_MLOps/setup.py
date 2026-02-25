"""Package setup for Vertex AI Custom Training Job — Cycle 2.

Bundles ``src_colab`` (shared PyTorch training toolkit) and ``trainer``
(Vertex AI entry-points) into a single sdist that can be uploaded
to GCS and used with ``CustomPythonPackageTrainingJob``.

Build::

    cd 02_ING_MODELOS/Train_MLOps/
    python setup.py sdist --formats=gztar

The resulting ``dist/tfm_trainer-2.7.0.tar.gz`` is uploaded to GCS
by ``vertex_ai/build_and_launch.sh``.

Changes from v1.0.0:
    - All models are PyTorch-only (no TensorFlow/Keras)
    - Added torchvision, albumentations, opencv-python-headless
    - Removed protobuf <4 constraint (no longer needed without TF)
    - New entry-points: task_fcos, task_yolo26_custom, task_espdet, task_export

Changes from v2.0.0:
    - Added Sigmoid Focal Loss (γ, α params) for FCOS cls head
    - Added Smooth L1 → GIoU regression warmup in build_fcos_loss
    - Version bump forces pip cache invalidation on Vertex AI

Changes from v2.1.0:
    - Fixed config_loader.py whitelist bug: now passes ALL YAML keys
      (was silently dropping focal_gamma, reg_warmup_epochs, etc.)
    - conf_threshold raised to 0.35 for T8

Changes from v2.2.0:
    - Version bump for YOLO26_CUSTOM training cycle
    - Added missing Ultralytics fields to Yolo26CustomConfig
    - task_yolo26_custom: DEPLOY VERIFICATION + explicit YAML param pass

Changes from v2.3.0:
    - BUG FIX: CSV concatenation in Bloque 4/5 (was using only Phase 1 CSV)
    - BUG FIX: Phase 2 results.csv not uploaded to GCS
    - Train 2 config: MuSGD optimizer (YOLO26 native SGD+Muon hybrid)

Changes from v2.4.0:
    - ESPDet-Pico training cycle
    - task_espdet: DEPLOY VERIFICATION block (lesson from FCOS T8)
    - task_espdet: BUG FIX — aug_config was not passed to IODCDataset
    - Version bump forces pip cache invalidation on Vertex AI

Changes from v2.5.0:
    - ESPDet-Pico v2: Official Espressif architecture (0.36M params)
    - Replaced custom DepthwiseSeparableConv + SimpleFPN architecture
      with official DSConv/DSC3k2/ESPBlock/ESPDetectHead from esp-detection

Changes from v2.6.0:
    - BUG FIX: Added ultralytics>=8.2 to install_requires
      (ESPDet v2 imports from ultralytics.nn.modules but it was missing)
    - Version bump forces pip cache invalidation on Vertex AI
    - New espdet_modules/ sub-package with official blocks
    - Strides changed [4,8,16] → [8,16,32] (official)
    - Transfer learning from espdet_pico_224_224_cat.pt (cat detection, nc=1)
    - ONNX export: interleaved (box0,score0,box1,score1,box2,score2) for esp-ppq
    - Version bump forces pip cache invalidation on Vertex AI

Changes from v2.6.1:
    - Sigmoid Focal Loss for ESPDet cls head (γ, α params in YAML)
      build_espdet_loss() now accepts focal_gamma/focal_alpha;
      when γ > 0, replaces BCEWithLogitsLoss with _sigmoid_focal_loss
    - BUG FIX: ExperimentSetup tracking — experiment.json now records
      best_val_loss, best_epoch, final_train_loss, final_val_loss, duration_s
    - BUG FIX: batch_size in experiment.json reads from setup.batch_size
      (was defaulting to 16 from fc.get)
    - BUG FIX: aug field names in UnifiedExperimentConfig aligned with
      IODCDataset keys (aug_hflip_prob, aug_brightness_limit, etc.)
    - DEPLOY VERIFICATION updated with Focal Loss ON/OFF display
    - Version bump forces pip cache invalidation on Vertex AI

Changes from v2.6.2:
    - ESPDet-Pico v4: BCE + NMS tuning (conf=0.35, iou=0.40)
    - Focal Loss desactivada (γ=0.0) tras regresión en Train 3
    - NMS tuning: conf_threshold 0.25→0.35, iou_threshold 0.45→0.40
      para reducir FP de background sin reentrenar
    - DEPLOY VERIFICATION version string updated
    - Version bump forces pip cache invalidation on Vertex AI

Changes from v2.6.3:
    - YOLO26 v3: DFL Removal — pretrained_weights cambia de yolo11n.pt a yolo26n.pt
    - yolo26n.pt nativo: reg_max=1, dfl=Identity, cv2→4 channels (era 64)
    - Nuevo config: yolo26n_custom_v3.yaml
    - DEPLOY VERIFICATION: añadido reg_max display
    - Ultralytics 8.4.9 ya soporta DFL Removal (no requiere upgrade)
    - Version bump forces pip cache invalidation on Vertex AI
"""

from setuptools import setup, find_packages

setup(
    name="tfm-trainer",
    version="2.7.0",
    description=(
        "TFM — Entrenamiento de modelos de detección de objetos "
        "para ESP32-S3 en Vertex AI (Ciclo 2 — PyTorch)"
    ),
    author="TFM UNIR",
    packages=find_packages(),         # descubre src_colab/ y trainer/
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.26,<2.0",
        "pandas>=2.0",
        "matplotlib>=3.8",
        "scikit-learn>=1.4",
        "opencv-python-headless>=4.9",
        "albumentations>=2.0.0",
        "ultralytics>=8.2",
        "torchvision>=0.19",
        "google-cloud-storage>=2.14",
        "google-cloud-aiplatform>=1.40",
        "onnx>=1.14",
        "onnxruntime>=1.16",
    ],
    python_requires=">=3.10",
)
