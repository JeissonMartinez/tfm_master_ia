"""Package setup for Vertex AI Custom Training Job — Cycle 2.

Bundles ``src_colab`` (shared PyTorch training toolkit) and ``trainer``
(Vertex AI entry-points) into a single sdist that can be uploaded
to GCS and used with ``CustomPythonPackageTrainingJob``.

Build::

    cd 02_ING_MODELOS/Train_MLOps/
    python setup.py sdist --formats=gztar

The resulting ``dist/tfm_trainer-2.2.0.tar.gz`` is uploaded to GCS
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
"""

from setuptools import setup, find_packages

setup(
    name="tfm-trainer",
    version="2.2.0",
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
        "torchvision>=0.19",
        "google-cloud-storage>=2.14",
        "google-cloud-aiplatform>=1.40",
        "onnx>=1.14",
        "onnxruntime>=1.16",
    ],
    python_requires=">=3.10",
)
