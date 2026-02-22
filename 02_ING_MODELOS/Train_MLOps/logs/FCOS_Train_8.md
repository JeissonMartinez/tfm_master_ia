# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 8


## Logs de consola 
```bash
% ./vertex_ai/build_and_launch.sh fcos_v3s_v1
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.2.0' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.2.0.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.2.0.tar.gz [Content-Type=application/x-tar]...
- [1 files][ 67.3 KiB/ 67.3 KiB]                                                
Operation completed over 1 objects/67.3 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz

═══════════════════════════════════════════════════════════
🚀 Paso 3/3 — Lanzando Custom Job en Vertex AI
═══════════════════════════════════════════════════════════
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform_v1beta1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform_v1beta1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform_v1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.resourcemanager_v3 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.resourcemanager_v3 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1.schema.predict.instance_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1.schema.predict.instance_v1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1.schema.predict.params_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1.schema.predict.params_v1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1.schema.predict.prediction_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1.schema.predict.prediction_v1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1.schema.trainingjob.definition_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1.schema.trainingjob.definition_v1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1beta1.schema.predict.instance_v1beta1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1beta1.schema.predict.instance_v1beta1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1 past that date.
  warnings.warn(message, FutureWarning)
/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.19) which Google will stop supporting in new releases of google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1 past that date.
  warnings.warn(message, FutureWarning)
============================================================
🚀 VERTEX AI CUSTOM JOB — CONFIGURACIÓN
============================================================
  Proyecto:      project-18f58341-12cf-47bc-861
  Región:        us-central1
  Experimento:   tfm-deteccion-objetos
  Run:           fcos_v3s_v1-1771751066
  Familia:       FCOS
  Módulo:        trainer.task_fcos
  Contenedor:    us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
  Máquina:       n1-standard-8
  GPU:           NVIDIA_TESLA_T4 x1
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=fcos_v3s_v1-1771751066']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml

🚀 Lanzando Custom Job: tfm-fcos_v3s_v1-1771751066
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-22-04:04:28.037 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/1648613146346127360?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_PENDING
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/5942175266485829632?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob run completed. Resource name: projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360
Training did not produce a Managed Model returning None. Training Pipeline projects/608533675308/locations/us-central1/trainingPipelines/1648613146346127360 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR'].

============================================================
✅ Custom Job completado exitosamente
   Resultados en: gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1
   Experiments:   tfm-deteccion-objetos / fcos_v3s_v1-1771751066
============================================================

═══════════════════════════════════════════════════════════
✅ Proceso completado
═══════════════════════════════════════════════════════════
```

## Logs de entrenamiento

```zsh
% gcloud ai custom-jobs stream-logs 5942175266485829632 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-22 04:04:56 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-22 04:04:56 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-22 04:04:56 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-22 04:04:56 -0500	service	Waiting for training program to start.
INFO	2026-02-22 04:04:57 -0500	service	Job is preparing.
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-c8ecy7gt-2636ff80-pcxd
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-d3072ebba8-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771751066"]}
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:07:49 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz tfm_trainer-2.2.0.tar.gz
ERROR	2026-02-22 04:07:49 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-22 04:07:49 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-22 04:07:52 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-22 04:07:52 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-22 04:07:57 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:07:57 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:08:00 -0500	workerpool0-0	Processing /tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:08:00 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-22 04:08:04 -0500	service	Job is running.
INFO	2026-02-22 04:08:05 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-22 04:08:05 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-22 04:08:06 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-22 04:08:06 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-22 04:08:06 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-22 04:08:06 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-22 04:08:06 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.2.0-py3-none-any.whl size=82200 sha256=87f36b3f351e542bc85c37a90e1b26a426727214bae834886f230fd2d59bea62
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/59/08/30/48b717b2ba9bb8efd99f1cd08cdec96f012312dbb2422f6c5b
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	Successfully installed tfm-trainer-2.2.0
ERROR	2026-02-22 04:08:07 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-22 04:08:07 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-22 04:08:07 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-22 04:08:07 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:08:09 -0500	workerpool0-0	Processing /tfm_trainer-2.2.0.tar.gz
INFO	2026-02-22 04:08:09 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-22 04:08:12 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-22 04:08:12 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-22 04:08:13 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-22 04:08:13 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-22 04:08:13 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-22 04:08:13 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (6.0.2)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (2.2.3)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-22 04:08:14 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (1.6.1)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting albumentations>=2.0.0 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (0.19.0+cu124)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (2.19.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.2.0) (1.91.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.2.0) (1.11.4)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:15 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-22 04:08:17 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:17 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.34.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.40.3)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.26.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (3.20.3)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (25.0)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (3.34.0)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.14.2)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.1.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (4.14.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (0.16)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.70.0)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.32.4)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.73.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.48.2)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (5.5.2)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (0.4.2)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (4.9.1)
INFO	2026-02-22 04:08:18 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-22 04:08:19 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:19 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-22 04:08:19 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-22 04:08:19 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-22 04:08:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-22 04:08:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-22 04:08:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-22 04:08:22 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-22 04:08:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-22 04:08:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.4.3)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.7.2)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2.9.0.post0)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (0.14.2)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-22 04:08:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.2.0) (1.7.1)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:24 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.17.0)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (3.4.2)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (3.10)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (1.26.20)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (2025.6.15)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.2.0) (0.6.1)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (1.3.2)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (0.12.1)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (4.58.5)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (1.4.8)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (11.3.0)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.2.0) (3.2.3)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-22 04:08:27 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.2.0) (1.14.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.2.0) (2025.2)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.2.0) (2025.2)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.2.0) (1.5.1)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.2.0) (3.6.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.2.0) (2.4.0+cu124)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (3.18.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (3.4.2)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (3.1.6)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (2025.5.1)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (9.1.0.70)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.2.65)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (11.2.0.44)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (10.3.5.119)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (11.6.0.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.3.0.142)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (2.20.5)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (12.4.99)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (3.0.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.2.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.2.0) (3.0.2)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.2.0) (1.3.0)
INFO	2026-02-22 04:08:28 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 104.3 MB/s eta 0:00:00
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 92.1 MB/s eta 0:00:00
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 134.4 MB/s eta 0:00:00
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 131.6 MB/s eta 0:00:00
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 149.1 MB/s eta 0:00:00
INFO	2026-02-22 04:08:29 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 132.5 MB/s eta 0:00:00
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 26.9 MB/s eta 0:00:00
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 91.7 MB/s eta 0:00:00
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-22 04:08:30 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-22 04:08:31 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-22 04:08:31 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.2.0-py3-none-any.whl size=82200 sha256=d3e9ad78ec3d267801e65a97cfe85ff8c287b548fcf51bef849ff08b7d3683ed
INFO	2026-02-22 04:08:31 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/59/08/30/48b717b2ba9bb8efd99f1cd08cdec96f012312dbb2422f6c5b
INFO	2026-02-22 04:08:31 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-22 04:08:32 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-22 04:08:32 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:32 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-22 04:08:36 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:36 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-22 04:08:36 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:36 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-22 04:08:41 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:41 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-22 04:08:41 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:41 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-22 04:08:43 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-22 04:08:43 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-22 04:08:46 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-22 04:08:46 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.2.0
INFO	2026-02-22 04:08:46 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.2.0:
INFO	2026-02-22 04:08:46 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.2.0
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
INFO	2026-02-22 04:08:46 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.2.0 typing-inspection-0.4.2
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-22 04:08:46 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-22 04:08:47 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-22 04:08:47 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-22 04:08:48 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771751066
INFO	2026-02-22 04:08:51 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:08:51 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-22 04:08:51 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771751066.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Patience:     20
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  Conf Thresh:  0.35
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     Phase 2: 80 epochs @ LR=0.0001
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-22 04:09:17 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-22 04:09:18 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-22 04:09:18 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	metadata {
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  key: "method"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	}
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	metadata {
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  key: "service"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	}
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	]
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-22 04:09:20 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-22 04:09:22 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-22 04:09:23 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-22 04:09:23 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-22 04:09:25 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-22 04:09:25 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:25 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-22 04:09:25 -0500	workerpool0-0	============================================================
ERROR	2026-02-22 04:09:35 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-22 04:09:35 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-22 04:09:35 -0500	workerpool0-0	 58%|█████▊    | 5.75M/9.83M [00:00<00:00, 59.9MB/s]
ERROR	2026-02-22 04:09:35 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 78.3MB/s]
INFO	2026-02-22 04:09:36 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-22 04:09:36 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🎯 cls_loss: Sigmoid Focal Loss (γ=3.0, α=0.25)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	════════════════════════════════════════════════════════════
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🔍 DEPLOY VERIFICATION — package v2.2.0
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   focal_gamma  = 3.0  ✅ Focal Loss ACTIVE
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   reg_warmup   = 10  ✅ SL1→GIoU warmup ACTIVE
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   aug_hflip    = 0.5  ✅ HFlip ON
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   phase1_epochs= 30
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	════════════════════════════════════════════════════════════
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Phase 2: 80 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 20
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-22 04:09:37 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-22 04:10:06 -0500	workerpool0-0	  Epoch   0 | train=35.6860 [cls=0.3370 | reg=33.5084 | ctr=1.8406] | val=141.9633 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-22 04:10:27 -0500	workerpool0-0	  Epoch   1 | train=24.7332 [cls=0.1688 | reg=22.7813 | ctr=1.7831] | val=116.8983 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-22 04:10:49 -0500	workerpool0-0	  Epoch   2 | train=20.8505 [cls=0.1587 | reg=18.9172 | ctr=1.7745] | val=94.3391 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-22 04:11:10 -0500	workerpool0-0	  Epoch   3 | train=19.1886 [cls=0.1517 | reg=17.2728 | ctr=1.7641] | val=78.6198 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-22 04:11:31 -0500	workerpool0-0	  Epoch   4 | train=18.5078 [cls=0.1477 | reg=16.5998 | ctr=1.7604] | val=83.6588 | lr=9.97e-04 | img=640 | 
INFO	2026-02-22 04:11:51 -0500	workerpool0-0	  Epoch   5 | train=17.8676 [cls=0.1461 | reg=15.9658 | ctr=1.7556] | val=97.4164 | lr=9.87e-04 | img=640 | 
INFO	2026-02-22 04:12:11 -0500	workerpool0-0	  Epoch   6 | train=17.6714 [cls=0.1422 | reg=15.7782 | ctr=1.7510] | val=82.6674 | lr=9.70e-04 | img=640 | 
INFO	2026-02-22 04:12:28 -0500	workerpool0-0	  Epoch   7 | train=17.1914 [cls=0.1382 | reg=15.3029 | ctr=1.7503] | val=71.9893 | lr=9.47e-04 | img=640 | ★ best
INFO	2026-02-22 04:12:45 -0500	workerpool0-0	  Epoch   8 | train=17.1679 [cls=0.1352 | reg=15.2848 | ctr=1.7480] | val=105.9039 | lr=9.18e-04 | img=640 | 
INFO	2026-02-22 04:13:02 -0500	workerpool0-0	  Epoch   9 | train=16.7065 [cls=0.1338 | reg=14.8263 | ctr=1.7464] | val=70.1023 | lr=8.83e-04 | img=640 | ★ best
INFO	2026-02-22 04:13:02 -0500	workerpool0-0	🔄 reg_loss warmup complete at epoch 10: Smooth L1 → GIoU
INFO	2026-02-22 04:13:16 -0500	workerpool0-0	  Epoch  10 | train=3.9435 [cls=0.1255 | reg=2.0766 | ctr=1.7414] | val=36.6932 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-22 04:13:31 -0500	workerpool0-0	  Epoch  11 | train=3.7887 [cls=0.1205 | reg=1.9342 | ctr=1.7340] | val=28.1850 | lr=7.99e-04 | img=416 | ★ best
INFO	2026-02-22 04:13:45 -0500	workerpool0-0	  Epoch  12 | train=3.7562 [cls=0.1114 | reg=1.9140 | ctr=1.7308] | val=28.7545 | lr=7.50e-04 | img=416 | 
INFO	2026-02-22 04:13:59 -0500	workerpool0-0	  Epoch  13 | train=3.7275 [cls=0.1067 | reg=1.8925 | ctr=1.7284] | val=28.2046 | lr=6.98e-04 | img=416 | 
INFO	2026-02-22 04:14:13 -0500	workerpool0-0	  Epoch  14 | train=3.7031 [cls=0.1047 | reg=1.8722 | ctr=1.7263] | val=27.8838 | lr=6.43e-04 | img=416 | ★ best
INFO	2026-02-22 04:14:28 -0500	workerpool0-0	  Epoch  15 | train=3.7017 [cls=0.1025 | reg=1.8740 | ctr=1.7251] | val=39.1002 | lr=5.87e-04 | img=416 | 
INFO	2026-02-22 04:14:42 -0500	workerpool0-0	  Epoch  16 | train=3.6647 [cls=0.1014 | reg=1.8438 | ctr=1.7195] | val=29.3615 | lr=5.29e-04 | img=416 | 
INFO	2026-02-22 04:14:56 -0500	workerpool0-0	  Epoch  17 | train=3.6573 [cls=0.0971 | reg=1.8404 | ctr=1.7197] | val=34.8455 | lr=4.71e-04 | img=416 | 
INFO	2026-02-22 04:15:11 -0500	workerpool0-0	  Epoch  18 | train=3.6466 [cls=0.0964 | reg=1.8305 | ctr=1.7197] | val=21.5930 | lr=4.13e-04 | img=416 | ★ best
INFO	2026-02-22 04:15:25 -0500	workerpool0-0	  Epoch  19 | train=3.6271 [cls=0.0959 | reg=1.8127 | ctr=1.7185] | val=28.3977 | lr=3.57e-04 | img=416 | 
INFO	2026-02-22 04:15:39 -0500	workerpool0-0	  Epoch  20 | train=3.6668 [cls=0.0943 | reg=1.8519 | ctr=1.7206] | val=15.7986 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-22 04:15:52 -0500	workerpool0-0	  Epoch  21 | train=3.6195 [cls=0.0916 | reg=1.8101 | ctr=1.7178] | val=15.0828 | lr=2.50e-04 | img=320 | ★ best
INFO	2026-02-22 04:16:06 -0500	workerpool0-0	  Epoch  22 | train=3.5995 [cls=0.0905 | reg=1.7951 | ctr=1.7139] | val=19.4013 | lr=2.02e-04 | img=320 | 
INFO	2026-02-22 04:16:20 -0500	workerpool0-0	  Epoch  23 | train=3.5981 [cls=0.0898 | reg=1.7943 | ctr=1.7141] | val=13.8826 | lr=1.57e-04 | img=320 | ★ best
INFO	2026-02-22 04:16:34 -0500	workerpool0-0	  Epoch  24 | train=3.5864 [cls=0.0885 | reg=1.7827 | ctr=1.7152] | val=16.7900 | lr=1.17e-04 | img=320 | 
INFO	2026-02-22 04:16:48 -0500	workerpool0-0	  Epoch  25 | train=3.5553 [cls=0.0867 | reg=1.7587 | ctr=1.7099] | val=18.3213 | lr=8.23e-05 | img=320 | 
INFO	2026-02-22 04:17:01 -0500	workerpool0-0	  Epoch  26 | train=3.5694 [cls=0.0880 | reg=1.7689 | ctr=1.7125] | val=17.3739 | lr=5.33e-05 | img=320 | 
INFO	2026-02-22 04:17:15 -0500	workerpool0-0	  Epoch  27 | train=3.5526 [cls=0.0874 | reg=1.7564 | ctr=1.7088] | val=16.4683 | lr=3.03e-05 | img=320 | 
INFO	2026-02-22 04:17:29 -0500	workerpool0-0	  Epoch  28 | train=3.5543 [cls=0.0861 | reg=1.7571 | ctr=1.7111] | val=16.6918 | lr=1.36e-05 | img=320 | 
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	  Epoch  29 | train=3.5556 [cls=0.0868 | reg=1.7577 | ctr=1.7111] | val=16.7187 | lr=3.48e-06 | img=320 | 
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 8.1 min
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=0.0001
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-22 04:17:42 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=80)
INFO	2026-02-22 04:17:57 -0500	workerpool0-0	  Epoch  30 | train=3.7150 [cls=0.0938 | reg=1.8842 | ctr=1.7370] | val=11.4272 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-22 04:18:12 -0500	workerpool0-0	  Epoch  31 | train=3.5570 [cls=0.0883 | reg=1.7443 | ctr=1.7244] | val=10.4205 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-22 04:18:27 -0500	workerpool0-0	  Epoch  32 | train=3.4674 [cls=0.0841 | reg=1.6663 | ctr=1.7170] | val=9.3106 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-22 04:18:41 -0500	workerpool0-0	  Epoch  33 | train=3.4173 [cls=0.0820 | reg=1.6212 | ctr=1.7140] | val=10.8913 | lr=1.00e-04 | img=224 | 
INFO	2026-02-22 04:18:57 -0500	workerpool0-0	  Epoch  34 | train=3.3555 [cls=0.0804 | reg=1.5670 | ctr=1.7080] | val=9.0916 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-22 04:19:12 -0500	workerpool0-0	  Epoch  35 | train=3.3055 [cls=0.0775 | reg=1.5244 | ctr=1.7037] | val=9.4976 | lr=9.98e-05 | img=224 | 
INFO	2026-02-22 04:19:26 -0500	workerpool0-0	  Epoch  36 | train=3.2758 [cls=0.0776 | reg=1.4964 | ctr=1.7019] | val=10.1511 | lr=9.97e-05 | img=224 | 
INFO	2026-02-22 04:19:41 -0500	workerpool0-0	  Epoch  37 | train=3.2482 [cls=0.0755 | reg=1.4725 | ctr=1.7002] | val=8.8855 | lr=9.94e-05 | img=224 | ★ best
INFO	2026-02-22 04:19:56 -0500	workerpool0-0	  Epoch  38 | train=3.2195 [cls=0.0739 | reg=1.4459 | ctr=1.6998] | val=9.6585 | lr=9.90e-05 | img=224 | 
INFO	2026-02-22 04:20:11 -0500	workerpool0-0	  Epoch  39 | train=3.1764 [cls=0.0712 | reg=1.4132 | ctr=1.6920] | val=10.5625 | lr=9.86e-05 | img=224 | 
INFO	2026-02-22 04:20:26 -0500	workerpool0-0	  Epoch  40 | train=3.1652 [cls=0.0716 | reg=1.3999 | ctr=1.6937] | val=10.0664 | lr=9.81e-05 | img=224 | 
INFO	2026-02-22 04:20:41 -0500	workerpool0-0	  Epoch  41 | train=3.1368 [cls=0.0689 | reg=1.3759 | ctr=1.6920] | val=9.8724 | lr=9.76e-05 | img=224 | 
INFO	2026-02-22 04:20:55 -0500	workerpool0-0	  Epoch  42 | train=3.1086 [cls=0.0687 | reg=1.3514 | ctr=1.6884] | val=8.8087 | lr=9.69e-05 | img=224 | ★ best
INFO	2026-02-22 04:21:10 -0500	workerpool0-0	  Epoch  43 | train=3.0803 [cls=0.0674 | reg=1.3259 | ctr=1.6870] | val=8.9895 | lr=9.62e-05 | img=224 | 
INFO	2026-02-22 04:21:25 -0500	workerpool0-0	  Epoch  44 | train=3.0634 [cls=0.0676 | reg=1.3108 | ctr=1.6850] | val=10.7117 | lr=9.54e-05 | img=224 | 
INFO	2026-02-22 04:21:40 -0500	workerpool0-0	  Epoch  45 | train=3.0472 [cls=0.0659 | reg=1.2975 | ctr=1.6839] | val=7.8167 | lr=9.46e-05 | img=224 | ★ best
INFO	2026-02-22 04:21:55 -0500	workerpool0-0	  Epoch  46 | train=3.0536 [cls=0.0662 | reg=1.3065 | ctr=1.6809] | val=9.8561 | lr=9.36e-05 | img=224 | 
INFO	2026-02-22 04:22:09 -0500	workerpool0-0	  Epoch  47 | train=3.0050 [cls=0.0632 | reg=1.2613 | ctr=1.6804] | val=8.9878 | lr=9.26e-05 | img=224 | 
INFO	2026-02-22 04:22:24 -0500	workerpool0-0	  Epoch  48 | train=3.0091 [cls=0.0627 | reg=1.2659 | ctr=1.6806] | val=9.1211 | lr=9.16e-05 | img=224 | 
INFO	2026-02-22 04:22:39 -0500	workerpool0-0	  Epoch  49 | train=3.0048 [cls=0.0636 | reg=1.2620 | ctr=1.6791] | val=9.0287 | lr=9.05e-05 | img=224 | 
INFO	2026-02-22 04:22:54 -0500	workerpool0-0	  Epoch  50 | train=2.9780 [cls=0.0626 | reg=1.2361 | ctr=1.6793] | val=9.9995 | lr=8.93e-05 | img=224 | 
INFO	2026-02-22 04:23:08 -0500	workerpool0-0	  Epoch  51 | train=2.9509 [cls=0.0607 | reg=1.2144 | ctr=1.6758] | val=10.6083 | lr=8.80e-05 | img=224 | 
INFO	2026-02-22 04:23:23 -0500	workerpool0-0	  Epoch  52 | train=2.9533 [cls=0.0609 | reg=1.2152 | ctr=1.6772] | val=9.7902 | lr=8.67e-05 | img=224 | 
INFO	2026-02-22 04:23:38 -0500	workerpool0-0	  Epoch  53 | train=2.9414 [cls=0.0600 | reg=1.2058 | ctr=1.6756] | val=9.2261 | lr=8.54e-05 | img=224 | 
INFO	2026-02-22 04:23:52 -0500	workerpool0-0	  Epoch  54 | train=2.9258 [cls=0.0595 | reg=1.1919 | ctr=1.6744] | val=8.6277 | lr=8.40e-05 | img=224 | 
INFO	2026-02-22 04:24:07 -0500	workerpool0-0	  Epoch  55 | train=2.8828 [cls=0.0575 | reg=1.1558 | ctr=1.6695] | val=10.3762 | lr=8.25e-05 | img=224 | 
INFO	2026-02-22 04:24:22 -0500	workerpool0-0	  Epoch  56 | train=2.9014 [cls=0.0577 | reg=1.1718 | ctr=1.6719] | val=9.9794 | lr=8.10e-05 | img=224 | 
INFO	2026-02-22 04:24:37 -0500	workerpool0-0	  Epoch  57 | train=2.8914 [cls=0.0570 | reg=1.1635 | ctr=1.6709] | val=10.7737 | lr=7.94e-05 | img=224 | 
INFO	2026-02-22 04:24:51 -0500	workerpool0-0	  Epoch  58 | train=2.8613 [cls=0.0560 | reg=1.1370 | ctr=1.6682] | val=10.5100 | lr=7.78e-05 | img=224 | 
INFO	2026-02-22 04:25:06 -0500	workerpool0-0	  Epoch  59 | train=2.8793 [cls=0.0564 | reg=1.1523 | ctr=1.6706] | val=12.1451 | lr=7.61e-05 | img=224 | 
INFO	2026-02-22 04:25:20 -0500	workerpool0-0	  Epoch  60 | train=2.8470 [cls=0.0553 | reg=1.1226 | ctr=1.6691] | val=10.6927 | lr=7.45e-05 | img=224 | 
INFO	2026-02-22 04:25:35 -0500	workerpool0-0	  Epoch  61 | train=2.8311 [cls=0.0544 | reg=1.1098 | ctr=1.6669] | val=10.1774 | lr=7.27e-05 | img=224 | 
INFO	2026-02-22 04:25:49 -0500	workerpool0-0	  Epoch  62 | train=2.8387 [cls=0.0544 | reg=1.1175 | ctr=1.6669] | val=9.0428 | lr=7.10e-05 | img=224 | 
INFO	2026-02-22 04:26:04 -0500	workerpool0-0	  Epoch  63 | train=2.8341 [cls=0.0541 | reg=1.1113 | ctr=1.6686] | val=8.7334 | lr=6.92e-05 | img=224 | 
INFO	2026-02-22 04:26:19 -0500	workerpool0-0	  Epoch  64 | train=2.8373 [cls=0.0542 | reg=1.1156 | ctr=1.6676] | val=8.4018 | lr=6.73e-05 | img=224 | 
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	  Epoch  65 | train=2.8222 [cls=0.0539 | reg=1.1013 | ctr=1.6670] | val=9.3325 | lr=6.55e-05 | img=224 | 
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	⏹️  Early stopping at epoch 65 (patience=20)
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 8.9 min
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	📊 Entrenamiento completo: 66 epochs
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	   Mejor val_loss: 7.8167 (epoch 45)
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	   Tiempo total: 16.9 min
INFO	2026-02-22 04:26:34 -0500	workerpool0-0	⏱️  Entrenamiento completado en 17.0 min
INFO	2026-02-22 04:26:35 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-22 04:26:35 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:35 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-22 04:26:35 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	  Épocas: 66
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	  Mejor val_loss: 7.8167 (epoch 45)
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:37 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  mAP@50:    0.4104
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  mAP@50-95: 0.1754
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  Precision: 0.1868
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  Recall:    0.5338
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  F1-Score:  0.2767
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 2387 | GT: 762
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  Avg inference: 6.3 ms
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	    dog                   0.2546
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	    door                  0.4787
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	    obstacle              0.4410
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	    person                0.4287
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	    stair                 0.4489
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	📊 Val mAP@50: 0.4104
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	   dog: 0.2546
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	   door: 0.4787
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	   obstacle: 0.4410
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	   person: 0.4287
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	   stair: 0.4489
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-22 04:26:39 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  mAP@50:    0.5954
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  mAP@50-95: 0.2615
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  Precision: 0.2071
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  Recall:    0.7143
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  F1-Score:  0.3211
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 2106 | GT: 576
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  Avg inference: 5.9 ms
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	    dog                   0.4346
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	    door                  0.6193
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	    obstacle              0.5256
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	    person                0.6857
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	    stair                 0.7121
INFO	2026-02-22 04:26:41 -0500	workerpool0-0	📊 Test mAP@50: 0.5954
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	============================================================
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-22 04:26:42 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.8s)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     Latencia mediana: 6.9ms
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/config.yaml
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/training_history.csv
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/training_curves.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/class_distribution.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/gt_samples.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/val_confusion_matrix.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/val_per_class.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/val_evaluation.json
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/test_confusion_matrix.png
INFO	2026-02-22 04:26:43 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/test_evaluation.json
INFO	2026-02-22 04:26:44 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/experiment.json
INFO	2026-02-22 04:26:44 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/checkpoints/best_fcos.pt
INFO	2026-02-22 04:26:44 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771751066/export/fcos_v3s.onnx
INFO	2026-02-22 04:26:44 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-22 04:26:47 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-22 04:27:09 -0500	service	Tearing down training program.
INFO	2026-02-22 04:27:58 -0500	service	Finished tearing down training program.
INFO	2026-02-22 04:27:58 -0500	service	Job completed successfully.


```