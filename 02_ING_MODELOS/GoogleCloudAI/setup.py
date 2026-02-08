"""Package setup for Vertex AI Custom Training Job.

Bundles ``src_colab`` (shared training toolkit) and ``trainer``
(Vertex AI entry-points) into a single sdist that can be uploaded
to GCS and used with ``CustomPythonPackageTrainingJob``.

Build::

    cd 02_ING_MODELOS/GoogleCloudAI/
    python setup.py sdist --formats=gztar

The resulting ``dist/tfm-trainer-1.0.0.tar.gz`` is uploaded to GCS
by ``vertex_ai/build_and_launch.sh``.
"""

from setuptools import setup, find_packages

setup(
    name="tfm-trainer",
    version="1.0.0",
    description=(
        "TFM — Entrenamiento de modelos de detección de objetos "
        "para ESP32-S3 en Vertex AI"
    ),
    author="TFM UNIR",
    packages=find_packages(),         # descubre src_colab/ y trainer/
    install_requires=[
        "pyyaml>=6.0",
        "google-cloud-storage>=2.14",
        "google-cloud-aiplatform>=1.40",
    ],
    python_requires=">=3.10",
)
