# Registro de Entrenamiento - ESPDet-Pico (Custom PyTorch Loop)
**Corrida Exitosa #:** 1

Log de Monitor de Entrenamiento:

```zsh
% gcloud ai custom-jobs stream-logs 3958704668489547776 --region=us-centra
l1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO    2026-02-23 16:07:21 -0500       service Waiting for job to be provisioned.
INFO    2026-02-23 16:07:21 -0500       service Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO    2026-02-23 16:07:21 -0500       service Vertex AI is setting up this job.
INFO    2026-02-23 16:07:21 -0500       service Waiting for training program to start.
INFO    2026-02-23 16:07:22 -0500       service Job is preparing.
INFO    2026-02-23 16:11:09 -0500       service Job is running.
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-le4a7o3b-61c06b8d-x6fr
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-4442549cc4-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet_pico_v1-train1"]}
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:18 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz tfm_trainer-2.5.0.tar.gz
ERROR   2026-02-23 16:11:18 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 16:11:18 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 16:11:22 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 16:11:22 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 16:11:27 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:27 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:31 -0500       workerpool0-0   Processing /tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:31 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 16:11:36 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 16:11:36 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 16:11:37 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.5.0-py3-none-any.whl size=83612 sha256=1ca1614bd43a7ebac57adc307a5e83af85e2d7a1008f7da2c6699efd29756863
INFO    2026-02-23 16:11:37 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/74/6a/71/224e6efa5a321d16ecc9a2e15aeeb76bb23b1f8b4050cf5c14
INFO    2026-02-23 16:11:37 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 16:11:37 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 16:11:38 -0500       workerpool0-0   Successfully installed tfm-trainer-2.5.0
ERROR   2026-02-23 16:11:38 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 16:11:38 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 16:11:38 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 16:11:38 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:39 -0500       workerpool0-0   Processing /tfm_trainer-2.5.0.tar.gz
INFO    2026-02-23 16:11:39 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 16:11:43 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 16:11:43 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 16:11:44 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 16:11:44 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 16:11:45 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 16:11:45 -0500       workerpool0-0   Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (6.0.2)
INFO    2026-02-23 16:11:45 -0500       workerpool0-0   Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:46 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 16:11:46 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (2.2.3)
INFO    2026-02-23 16:11:46 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:46 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (1.6.1)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Collecting albumentations>=2.0.0 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0     Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (0.19.0+cu124)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (2.19.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.5.0) (1.91.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0     Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 16:11:47 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.5.0) (1.11.4)
INFO    2026-02-23 16:11:48 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:48 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 16:11:48 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:48 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 16:11:49 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:49 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 16:11:49 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:49 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.34.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.40.3)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.26.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (3.20.3)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (25.0)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (3.34.0)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.14.2)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.1.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (4.14.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (0.16)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.70.0)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.32.4)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.73.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.48.2)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (5.5.2)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (0.4.2)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (4.9.1)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 16:11:50 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.4.3)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.7.2)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2.9.0.post0)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (0.14.2)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.5.0) (1.7.1)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:51 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.17.0)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (3.4.2)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (3.10)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (1.26.20)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (2025.6.15)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.5.0) (0.6.1)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (1.3.2)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (0.12.1)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (4.58.5)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (1.4.8)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (11.3.0)
INFO    2026-02-23 16:11:53 -0500       workerpool0-0   Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.5.0) (3.2.3)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.5.0) (1.14.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0     Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.5.0) (2025.2)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.5.0) (2025.2)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.5.0) (1.5.1)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.5.0) (3.6.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.5.0) (2.4.0+cu124)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (3.18.0)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (3.4.2)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (3.1.6)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (2025.5.1)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (9.1.0.70)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.2.65)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (11.2.0.44)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (10.3.5.119)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (11.6.0.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.3.0.142)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (2.20.5)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (12.4.99)
INFO    2026-02-23 16:11:54 -0500       workerpool0-0   Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (3.0.0)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.5.0)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.5.0) (3.0.2)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.5.0) (1.3.0)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 113.0 MB/s eta 0:00:00
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 90.6 MB/s eta 0:00:00
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 126.3 MB/s eta 0:00:00
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 114.1 MB/s eta 0:00:00
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 16:11:55 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 126.9 MB/s eta 0:00:00
INFO    2026-02-23 16:11:55 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 126.8 MB/s eta 0:00:00
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 24.8 MB/s eta 0:00:00
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 74.8 MB/s eta 0:00:00
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 16:11:56 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 16:11:57 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 16:11:57 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 16:11:57 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 16:11:57 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.5.0-py3-none-any.whl size=83612 sha256=d4c031094ccae96969d670cbd94e88d744a656e8d66819044768b505a7dd7182
INFO    2026-02-23 16:11:57 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/74/6a/71/224e6efa5a321d16ecc9a2e15aeeb76bb23b1f8b4050cf5c14
INFO    2026-02-23 16:11:57 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 16:11:58 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 16:11:59 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:11:59 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 16:12:01 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:12:01 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 16:12:01 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:12:01 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 16:12:08 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:12:08 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 16:12:09 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:12:09 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 16:12:11 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 16:12:11 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 16:12:14 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 16:12:14 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.5.0
INFO    2026-02-23 16:12:14 -0500       workerpool0-0       Uninstalling tfm-trainer-2.5.0:
INFO    2026-02-23 16:12:14 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.5.0
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
INFO    2026-02-23 16:12:14 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.5.0 typing-inspection-0.4.2
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 16:12:14 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 16:12:15 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet_pico_v1-train1
INFO    2026-02-23 16:12:18 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:18 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 16:12:18 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml → /tmp/training/config.yaml (0.0 MB)
INFO    2026-02-23 16:12:48 -0500       workerpool0-0   🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Nombre:       espdet_pico_v1
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Familia:      ESPDet
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Variante:     espdet_pico
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Versión:      v1
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Descripción:  ESPDet-Pico anchor-free detector optimizado para ESP32-S3
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Dataset:      iodc_yolo
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Img Size:     224×224
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Batch Size:   32
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Patience:     20
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Seed:         42
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     Conf Thresh:  0.25
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     IoU Thresh:   0.45
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     📐 2-Phase Training:
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        Phase 1: 40 epochs @ LR=0.001
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        Phase 2: 80 epochs @ LR=5e-05
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        Resize Schedule: {0: 640, 15: 416, 30: 320, 40: 224}
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        Optimizer: AdamW | WD: 0.0005
INFO    2026-02-23 16:12:48 -0500       workerpool0-0     🟢 ESPDet Config:
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        Pretrained: None
INFO    2026-02-23 16:12:48 -0500       workerpool0-0        reg_max: 1
INFO    2026-02-23 16:12:48 -0500       workerpool0-0   ✅ Configuración aplicada correctamente
INFO    2026-02-23 16:12:48 -0500       workerpool0-0   📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   🖥️  Device: cuda
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   ⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO    2026-02-23 16:12:51 -0500       workerpool0-0      Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   domain: "googleapis.com"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   metadata {
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     key: "method"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   }
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   metadata {
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     key: "service"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     value: "aiplatform.googleapis.com"
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   }
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   ]
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   BLOQUE 2 — Verificación del Dataset
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   📂 Dataset YOLO: iodc_yolo
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO    2026-02-23 16:12:51 -0500       workerpool0-0     ✅  test:   187 imgs |   187 labels | 0 sin label
INFO    2026-02-23 16:12:51 -0500       workerpool0-0   📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO    2026-02-23 16:12:52 -0500       workerpool0-0   generated new fontManager
INFO    2026-02-23 16:12:53 -0500       workerpool0-0     📊 Guardado: /tmp/training/class_distribution.png
INFO    2026-02-23 16:12:53 -0500       workerpool0-0   ⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO    2026-02-23 16:12:55 -0500       workerpool0-0     🖼️  Guardado: /tmp/training/gt_samples.png
INFO    2026-02-23 16:12:55 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:55 -0500       workerpool0-0   BLOQUE 3 — Construcción del Modelo ESPDet-Pico
INFO    2026-02-23 16:12:55 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   ✅ ESPDet-Pico construido: 22,839 params (22,839 trainable)
INFO    2026-02-23 16:12:56 -0500       workerpool0-0      Width mult: 0.5 | reg_max: 1 | Classes: 5
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   🔒 Backbone congelado: 10,592 params frozen
INFO    2026-02-23 16:12:56 -0500       workerpool0-0      Trainable: 12,247 / 22,839 (53.6%)
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   📦 Modelo: ESPDet-Pico
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Total params:           22,839
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Trainable:              12,247
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Non-trainable:          10,592
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Est. float32:           0.09 MB
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Est. INT8:              0.02 MB
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   📐 Tamaño estimado: 0.09 MB (FP32), 0.02 MB (INT8)
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.5.0
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     width_mult:      0.5
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     reg_max:         1
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     pretrained:      None
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Phase 1:         40 ep, LR=0.001, WD=0.0001
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Phase 2:         80 ep, LR=5e-05, WD=1e-05
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Optimizer:       adamw
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     cls_weight:      1.0
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     reg_weight:      2.0
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Conf threshold:  0.25
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     IoU threshold:   0.45
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     AMP:             True
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Grad clip:       5.0
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Export imgsz:    224
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Batch size:      32
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Patience:        20
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', 'aug_hue_shift_limit', 'aug_sat_shift_limit', 'aug_val_shift_limit', 'aug_shift_limit', 'aug_scale_limit', 'aug_rotate_limit', 'aug_hflip_prob', 'aug_horizontal_flip', 'aug_brightness_contrast', 'aug_gaussian_noise', 'aug_rotation_limit', 'aug_hue_sat_val', 'aug_random_gamma', 'aug_clahe']
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   BLOQUE 4 — Entrenamiento (2 fases)
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   🔧 Two-Phase Training Config
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Phase 1: 40 epochs | LR=0.001 | WD=0.0001
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Phase 2: 80 epochs | LR=5e-05 | WD=1e-05
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Optimizer: adamw | Scheduler: cosine
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Batch: 32 | AMP: True | Patience: 20
INFO    2026-02-23 16:12:56 -0500       workerpool0-0     Resize schedule: [(0, 640), (15, 416), (30, 320), (40, 224)]
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   🔒 Backbone congelado: 10,592 params frozen
INFO    2026-02-23 16:12:56 -0500       workerpool0-0      Trainable: 12,247 / 22,839 (53.6%)
INFO    2026-02-23 16:12:56 -0500       workerpool0-0   🚀 Phase 1 (backbone frozen) — 40 epochs, LR=0.001
INFO    2026-02-23 16:13:06 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO    2026-02-23 16:13:06 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=37)
INFO    2026-02-23 16:13:34 -0500       workerpool0-0     Epoch   0 | train=10.6076 [cls=4.6072 | reg=6.0004] | val=10.6541 | lr=3.33e-04 | img=640 | ★ best
INFO    2026-02-23 16:13:54 -0500       workerpool0-0     Epoch   1 | train=10.3587 [cls=4.3613 | reg=5.9973] | val=9.9970 | lr=6.67e-04 | img=640 | ★ best
INFO    2026-02-23 16:14:14 -0500       workerpool0-0     Epoch   2 | train=10.0531 [cls=4.0771 | reg=5.9760] | val=9.6604 | lr=1.00e-03 | img=640 | ★ best
INFO    2026-02-23 16:14:34 -0500       workerpool0-0     Epoch   3 | train=9.6722 [cls=3.7784 | reg=5.8937] | val=9.2590 | lr=1.00e-03 | img=640 | ★ best
INFO    2026-02-23 16:14:53 -0500       workerpool0-0     Epoch   4 | train=9.3814 [cls=3.6289 | reg=5.7525] | val=8.9608 | lr=9.98e-04 | img=640 | ★ best
INFO    2026-02-23 16:15:12 -0500       workerpool0-0     Epoch   5 | train=9.1594 [cls=3.5605 | reg=5.5989] | val=8.8331 | lr=9.93e-04 | img=640 | ★ best
INFO    2026-02-23 16:15:29 -0500       workerpool0-0     Epoch   6 | train=8.9771 [cls=3.5270 | reg=5.4501] | val=8.7515 | lr=9.84e-04 | img=640 | ★ best
INFO    2026-02-23 16:15:47 -0500       workerpool0-0     Epoch   7 | train=8.8154 [cls=3.5157 | reg=5.2997] | val=8.6358 | lr=9.71e-04 | img=640 | ★ best
INFO    2026-02-23 16:16:04 -0500       workerpool0-0     Epoch   8 | train=8.6969 [cls=3.4966 | reg=5.2004] | val=8.5837 | lr=9.56e-04 | img=640 | ★ best
INFO    2026-02-23 16:16:21 -0500       workerpool0-0     Epoch   9 | train=8.5988 [cls=3.4822 | reg=5.1167] | val=8.5544 | lr=9.37e-04 | img=640 | ★ best
INFO    2026-02-23 16:16:38 -0500       workerpool0-0     Epoch  10 | train=8.5449 [cls=3.4865 | reg=5.0584] | val=8.5673 | lr=9.14e-04 | img=640 | 
INFO    2026-02-23 16:16:54 -0500       workerpool0-0     Epoch  11 | train=8.5172 [cls=3.5004 | reg=5.0169] | val=8.4966 | lr=8.89e-04 | img=640 | ★ best
INFO    2026-02-23 16:17:11 -0500       workerpool0-0     Epoch  12 | train=8.4702 [cls=3.4914 | reg=4.9788] | val=8.4712 | lr=8.61e-04 | img=640 | ★ best
INFO    2026-02-23 16:17:28 -0500       workerpool0-0     Epoch  13 | train=8.4251 [cls=3.4692 | reg=4.9559] | val=8.4383 | lr=8.30e-04 | img=640 | ★ best
INFO    2026-02-23 16:17:45 -0500       workerpool0-0     Epoch  14 | train=8.4120 [cls=3.4746 | reg=4.9374] | val=8.4916 | lr=7.97e-04 | img=640 | 
INFO    2026-02-23 16:17:58 -0500       workerpool0-0     Epoch  15 | train=8.3705 [cls=3.4373 | reg=4.9332] | val=8.4723 | lr=7.62e-04 | img=416 | 
INFO    2026-02-23 16:18:10 -0500       workerpool0-0     Epoch  16 | train=8.2582 [cls=3.4077 | reg=4.8505] | val=8.4391 | lr=7.25e-04 | img=416 | 
INFO    2026-02-23 16:18:23 -0500       workerpool0-0     Epoch  17 | train=8.2446 [cls=3.4089 | reg=4.8357] | val=8.4119 | lr=6.86e-04 | img=416 | ★ best
INFO    2026-02-23 16:18:36 -0500       workerpool0-0     Epoch  18 | train=8.2261 [cls=3.3982 | reg=4.8280] | val=8.3681 | lr=6.46e-04 | img=416 | ★ best
INFO    2026-02-23 16:18:48 -0500       workerpool0-0     Epoch  19 | train=8.1935 [cls=3.3747 | reg=4.8188] | val=8.4195 | lr=6.05e-04 | img=416 | 
INFO    2026-02-23 16:19:01 -0500       workerpool0-0     Epoch  20 | train=8.2032 [cls=3.3876 | reg=4.8156] | val=8.4075 | lr=5.64e-04 | img=416 | 
INFO    2026-02-23 16:19:13 -0500       workerpool0-0     Epoch  21 | train=8.1946 [cls=3.3842 | reg=4.8104] | val=8.3697 | lr=5.21e-04 | img=416 | 
INFO    2026-02-23 16:19:26 -0500       workerpool0-0     Epoch  22 | train=8.2053 [cls=3.3941 | reg=4.8111] | val=8.3826 | lr=4.79e-04 | img=416 | 
INFO    2026-02-23 16:19:39 -0500       workerpool0-0     Epoch  23 | train=8.1790 [cls=3.3699 | reg=4.8091] | val=8.3715 | lr=4.37e-04 | img=416 | 
INFO    2026-02-23 16:19:51 -0500       workerpool0-0     Epoch  24 | train=8.1993 [cls=3.3866 | reg=4.8128] | val=8.3768 | lr=3.95e-04 | img=416 | 
INFO    2026-02-23 16:20:04 -0500       workerpool0-0     Epoch  25 | train=8.1823 [cls=3.3777 | reg=4.8046] | val=8.3565 | lr=3.54e-04 | img=416 | ★ best
INFO    2026-02-23 16:20:17 -0500       workerpool0-0     Epoch  26 | train=8.2119 [cls=3.4061 | reg=4.8058] | val=8.3616 | lr=3.14e-04 | img=416 | 
INFO    2026-02-23 16:20:29 -0500       workerpool0-0     Epoch  27 | train=8.1891 [cls=3.3925 | reg=4.7966] | val=8.3500 | lr=2.75e-04 | img=416 | ★ best
INFO    2026-02-23 16:20:42 -0500       workerpool0-0     Epoch  28 | train=8.1689 [cls=3.3682 | reg=4.8007] | val=8.3577 | lr=2.38e-04 | img=416 | 
INFO    2026-02-23 16:20:54 -0500       workerpool0-0     Epoch  29 | train=8.1453 [cls=3.3519 | reg=4.7933] | val=8.3500 | lr=2.03e-04 | img=416 | ★ best
INFO    2026-02-23 16:21:06 -0500       workerpool0-0     Epoch  30 | train=8.2093 [cls=3.3591 | reg=4.8502] | val=8.4317 | lr=1.70e-04 | img=320 | 
INFO    2026-02-23 16:21:17 -0500       workerpool0-0     Epoch  31 | train=8.1500 [cls=3.3441 | reg=4.8059] | val=8.3919 | lr=1.39e-04 | img=320 | 
INFO    2026-02-23 16:21:28 -0500       workerpool0-0     Epoch  32 | train=8.1146 [cls=3.3209 | reg=4.7937] | val=8.3885 | lr=1.11e-04 | img=320 | 
INFO    2026-02-23 16:21:40 -0500       workerpool0-0     Epoch  33 | train=8.1259 [cls=3.3377 | reg=4.7882] | val=8.3691 | lr=8.58e-05 | img=320 | 
INFO    2026-02-23 16:21:51 -0500       workerpool0-0     Epoch  34 | train=8.1230 [cls=3.3359 | reg=4.7871] | val=8.3677 | lr=6.36e-05 | img=320 | 
INFO    2026-02-23 16:22:02 -0500       workerpool0-0     Epoch  35 | train=8.0955 [cls=3.3082 | reg=4.7873] | val=8.3711 | lr=4.45e-05 | img=320 | 
INFO    2026-02-23 16:22:13 -0500       workerpool0-0     Epoch  36 | train=8.1029 [cls=3.3229 | reg=4.7800] | val=8.3588 | lr=2.87e-05 | img=320 | 
INFO    2026-02-23 16:22:24 -0500       workerpool0-0     Epoch  37 | train=8.1276 [cls=3.3411 | reg=4.7865] | val=8.3589 | lr=1.62e-05 | img=320 | 
INFO    2026-02-23 16:22:36 -0500       workerpool0-0     Epoch  38 | train=8.1161 [cls=3.3323 | reg=4.7838] | val=8.3613 | lr=7.29e-06 | img=320 | 
INFO    2026-02-23 16:22:47 -0500       workerpool0-0     Epoch  39 | train=8.0903 [cls=3.3163 | reg=4.7739] | val=8.3572 | lr=1.90e-06 | img=320 | 
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   ✅ Phase 1 (backbone frozen) completada en 9.7 min
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   🔄 Mejor checkpoint de Phase 1 recargado
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   🔓 Todas las capas desbloqueadas: 10,592 params unfrozen
INFO    2026-02-23 16:22:47 -0500       workerpool0-0      Total trainable: 22,839
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=5e-05
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=5e-05, wd=1e-05)
INFO    2026-02-23 16:22:47 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=80)
INFO    2026-02-23 16:22:57 -0500       workerpool0-0     Epoch  40 | train=8.5775 [cls=3.3200 | reg=5.2576] | val=8.8846 | lr=5.00e-05 | img=224 | 
INFO    2026-02-23 16:23:09 -0500       workerpool0-0     Epoch  41 | train=8.5381 [cls=3.3151 | reg=5.2229] | val=8.8364 | lr=5.00e-05 | img=224 | 
INFO    2026-02-23 16:23:19 -0500       workerpool0-0     Epoch  42 | train=8.4723 [cls=3.3080 | reg=5.1643] | val=8.7621 | lr=5.00e-05 | img=224 | 
INFO    2026-02-23 16:23:29 -0500       workerpool0-0     Epoch  43 | train=8.4079 [cls=3.3056 | reg=5.1023] | val=8.6790 | lr=5.00e-05 | img=224 | 
INFO    2026-02-23 16:23:40 -0500       workerpool0-0     Epoch  44 | train=8.3044 [cls=3.2676 | reg=5.0367] | val=8.6304 | lr=5.00e-05 | img=224 | 
INFO    2026-02-23 16:23:50 -0500       workerpool0-0     Epoch  45 | train=8.2914 [cls=3.2812 | reg=5.0101] | val=8.5739 | lr=4.99e-05 | img=224 | 
INFO    2026-02-23 16:24:00 -0500       workerpool0-0     Epoch  46 | train=8.2397 [cls=3.2659 | reg=4.9738] | val=8.5375 | lr=4.98e-05 | img=224 | 
INFO    2026-02-23 16:24:11 -0500       workerpool0-0     Epoch  47 | train=8.2209 [cls=3.2663 | reg=4.9546] | val=8.5090 | lr=4.97e-05 | img=224 | 
INFO    2026-02-23 16:24:21 -0500       workerpool0-0     Epoch  48 | train=8.1788 [cls=3.2491 | reg=4.9297] | val=8.4811 | lr=4.95e-05 | img=224 | 
INFO    2026-02-23 16:24:32 -0500       workerpool0-0     Epoch  49 | train=8.1495 [cls=3.2376 | reg=4.9119] | val=8.4554 | lr=4.93e-05 | img=224 | 
INFO    2026-02-23 16:24:42 -0500       workerpool0-0     Epoch  50 | train=8.1137 [cls=3.2300 | reg=4.8837] | val=8.4407 | lr=4.91e-05 | img=224 | 
INFO    2026-02-23 16:24:53 -0500       workerpool0-0     Epoch  51 | train=8.1050 [cls=3.2281 | reg=4.8769] | val=8.4386 | lr=4.88e-05 | img=224 | 
INFO    2026-02-23 16:25:03 -0500       workerpool0-0     Epoch  52 | train=8.0942 [cls=3.2325 | reg=4.8617] | val=8.3990 | lr=4.85e-05 | img=224 | 
INFO    2026-02-23 16:25:14 -0500       workerpool0-0     Epoch  53 | train=8.0772 [cls=3.2267 | reg=4.8506] | val=8.3849 | lr=4.81e-05 | img=224 | 
INFO    2026-02-23 16:25:24 -0500       workerpool0-0     Epoch  54 | train=8.0440 [cls=3.2083 | reg=4.8357] | val=8.3625 | lr=4.77e-05 | img=224 | 
INFO    2026-02-23 16:25:34 -0500       workerpool0-0     Epoch  55 | train=8.0052 [cls=3.1898 | reg=4.8154] | val=8.3537 | lr=4.73e-05 | img=224 | 
INFO    2026-02-23 16:25:45 -0500       workerpool0-0     Epoch  56 | train=8.0019 [cls=3.1941 | reg=4.8078] | val=8.3475 | lr=4.68e-05 | img=224 | ★ best
INFO    2026-02-23 16:25:55 -0500       workerpool0-0     Epoch  57 | train=7.9813 [cls=3.1848 | reg=4.7965] | val=8.3257 | lr=4.63e-05 | img=224 | ★ best
INFO    2026-02-23 16:26:06 -0500       workerpool0-0     Epoch  58 | train=7.9662 [cls=3.1793 | reg=4.7869] | val=8.3212 | lr=4.58e-05 | img=224 | ★ best
INFO    2026-02-23 16:26:16 -0500       workerpool0-0     Epoch  59 | train=7.9542 [cls=3.1783 | reg=4.7760] | val=8.3016 | lr=4.52e-05 | img=224 | ★ best
INFO    2026-02-23 16:26:26 -0500       workerpool0-0     Epoch  60 | train=7.9315 [cls=3.1666 | reg=4.7648] | val=8.2981 | lr=4.46e-05 | img=224 | ★ best
INFO    2026-02-23 16:26:36 -0500       workerpool0-0     Epoch  61 | train=7.9312 [cls=3.1698 | reg=4.7614] | val=8.2803 | lr=4.40e-05 | img=224 | ★ best
INFO    2026-02-23 16:26:47 -0500       workerpool0-0     Epoch  62 | train=7.9206 [cls=3.1625 | reg=4.7581] | val=8.2826 | lr=4.34e-05 | img=224 | 
INFO    2026-02-23 16:26:57 -0500       workerpool0-0     Epoch  63 | train=7.8952 [cls=3.1514 | reg=4.7439] | val=8.2733 | lr=4.27e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:08 -0500       workerpool0-0     Epoch  64 | train=7.8995 [cls=3.1569 | reg=4.7426] | val=8.2619 | lr=4.20e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:18 -0500       workerpool0-0     Epoch  65 | train=7.8723 [cls=3.1411 | reg=4.7312] | val=8.2530 | lr=4.13e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:28 -0500       workerpool0-0     Epoch  66 | train=7.8665 [cls=3.1373 | reg=4.7292] | val=8.2433 | lr=4.05e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:39 -0500       workerpool0-0     Epoch  67 | train=7.8641 [cls=3.1340 | reg=4.7301] | val=8.2355 | lr=3.97e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:49 -0500       workerpool0-0     Epoch  68 | train=7.8399 [cls=3.1212 | reg=4.7187] | val=8.2233 | lr=3.89e-05 | img=224 | ★ best
INFO    2026-02-23 16:27:59 -0500       workerpool0-0     Epoch  69 | train=7.8391 [cls=3.1298 | reg=4.7094] | val=8.2231 | lr=3.81e-05 | img=224 | ★ best
INFO    2026-02-23 16:28:09 -0500       workerpool0-0     Epoch  70 | train=7.8397 [cls=3.1282 | reg=4.7114] | val=8.2096 | lr=3.72e-05 | img=224 | ★ best
INFO    2026-02-23 16:28:20 -0500       workerpool0-0     Epoch  71 | train=7.8331 [cls=3.1267 | reg=4.7064] | val=8.2113 | lr=3.64e-05 | img=224 | 
INFO    2026-02-23 16:28:30 -0500       workerpool0-0     Epoch  72 | train=7.8271 [cls=3.1260 | reg=4.7011] | val=8.2041 | lr=3.55e-05 | img=224 | ★ best
INFO    2026-02-23 16:28:41 -0500       workerpool0-0     Epoch  73 | train=7.8136 [cls=3.1106 | reg=4.7031] | val=8.2019 | lr=3.46e-05 | img=224 | ★ best
INFO    2026-02-23 16:28:51 -0500       workerpool0-0     Epoch  74 | train=7.8098 [cls=3.1123 | reg=4.6976] | val=8.1867 | lr=3.37e-05 | img=224 | ★ best
INFO    2026-02-23 16:29:01 -0500       workerpool0-0     Epoch  75 | train=7.7970 [cls=3.1071 | reg=4.6899] | val=8.1882 | lr=3.28e-05 | img=224 | 
INFO    2026-02-23 16:29:11 -0500       workerpool0-0     Epoch  76 | train=7.7829 [cls=3.1037 | reg=4.6792] | val=8.1807 | lr=3.18e-05 | img=224 | ★ best
INFO    2026-02-23 16:29:22 -0500       workerpool0-0     Epoch  77 | train=7.7954 [cls=3.1075 | reg=4.6879] | val=8.1781 | lr=3.09e-05 | img=224 | ★ best
INFO    2026-02-23 16:29:31 -0500       workerpool0-0     Epoch  78 | train=7.7574 [cls=3.0812 | reg=4.6762] | val=8.1737 | lr=2.99e-05 | img=224 | ★ best
INFO    2026-02-23 16:29:42 -0500       workerpool0-0     Epoch  79 | train=7.7819 [cls=3.1045 | reg=4.6775] | val=8.1650 | lr=2.90e-05 | img=224 | ★ best
INFO    2026-02-23 16:29:52 -0500       workerpool0-0     Epoch  80 | train=7.7495 [cls=3.0815 | reg=4.6679] | val=8.1650 | lr=2.80e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:02 -0500       workerpool0-0     Epoch  81 | train=7.7677 [cls=3.0964 | reg=4.6714] | val=8.1553 | lr=2.70e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:12 -0500       workerpool0-0     Epoch  82 | train=7.7510 [cls=3.0899 | reg=4.6611] | val=8.1505 | lr=2.60e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:23 -0500       workerpool0-0     Epoch  83 | train=7.7667 [cls=3.1045 | reg=4.6622] | val=8.1454 | lr=2.50e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:33 -0500       workerpool0-0     Epoch  84 | train=7.7372 [cls=3.0742 | reg=4.6630] | val=8.1415 | lr=2.41e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:43 -0500       workerpool0-0     Epoch  85 | train=7.7371 [cls=3.0804 | reg=4.6566] | val=8.1379 | lr=2.31e-05 | img=224 | ★ best
INFO    2026-02-23 16:30:54 -0500       workerpool0-0     Epoch  86 | train=7.7392 [cls=3.0836 | reg=4.6556] | val=8.1382 | lr=2.21e-05 | img=224 | 
INFO    2026-02-23 16:31:04 -0500       workerpool0-0     Epoch  87 | train=7.7277 [cls=3.0779 | reg=4.6497] | val=8.1370 | lr=2.11e-05 | img=224 | ★ best
INFO    2026-02-23 16:31:14 -0500       workerpool0-0     Epoch  88 | train=7.7207 [cls=3.0772 | reg=4.6436] | val=8.1317 | lr=2.02e-05 | img=224 | ★ best
INFO    2026-02-23 16:31:25 -0500       workerpool0-0     Epoch  89 | train=7.7319 [cls=3.0752 | reg=4.6566] | val=8.1229 | lr=1.92e-05 | img=224 | ★ best
INFO    2026-02-23 16:31:35 -0500       workerpool0-0     Epoch  90 | train=7.7274 [cls=3.0838 | reg=4.6436] | val=8.1236 | lr=1.83e-05 | img=224 | 
INFO    2026-02-23 16:31:45 -0500       workerpool0-0     Epoch  91 | train=7.6861 [cls=3.0541 | reg=4.6320] | val=8.1201 | lr=1.73e-05 | img=224 | ★ best
INFO    2026-02-23 16:31:55 -0500       workerpool0-0     Epoch  92 | train=7.7125 [cls=3.0709 | reg=4.6416] | val=8.1189 | lr=1.64e-05 | img=224 | ★ best
INFO    2026-02-23 16:32:06 -0500       workerpool0-0     Epoch  93 | train=7.7211 [cls=3.0757 | reg=4.6454] | val=8.1207 | lr=1.55e-05 | img=224 | 
INFO    2026-02-23 16:32:16 -0500       workerpool0-0     Epoch  94 | train=7.6921 [cls=3.0573 | reg=4.6347] | val=8.1188 | lr=1.46e-05 | img=224 | ★ best
INFO    2026-02-23 16:32:27 -0500       workerpool0-0     Epoch  95 | train=7.7131 [cls=3.0771 | reg=4.6360] | val=8.1113 | lr=1.37e-05 | img=224 | ★ best
INFO    2026-02-23 16:32:37 -0500       workerpool0-0     Epoch  96 | train=7.7176 [cls=3.0797 | reg=4.6380] | val=8.1234 | lr=1.29e-05 | img=224 | 
INFO    2026-02-23 16:32:47 -0500       workerpool0-0     Epoch  97 | train=7.6836 [cls=3.0585 | reg=4.6251] | val=8.1074 | lr=1.20e-05 | img=224 | ★ best
INFO    2026-02-23 16:32:58 -0500       workerpool0-0     Epoch  98 | train=7.6921 [cls=3.0633 | reg=4.6288] | val=8.1051 | lr=1.12e-05 | img=224 | ★ best
INFO    2026-02-23 16:33:08 -0500       workerpool0-0     Epoch  99 | train=7.6999 [cls=3.0703 | reg=4.6296] | val=8.1084 | lr=1.04e-05 | img=224 | 
INFO    2026-02-23 16:33:19 -0500       workerpool0-0     Epoch 100 | train=7.7177 [cls=3.0855 | reg=4.6322] | val=8.1029 | lr=9.60e-06 | img=224 | ★ best
INFO    2026-02-23 16:33:29 -0500       workerpool0-0     Epoch 101 | train=7.6981 [cls=3.0674 | reg=4.6307] | val=8.1078 | lr=8.85e-06 | img=224 | 
INFO    2026-02-23 16:33:39 -0500       workerpool0-0     Epoch 102 | train=7.6758 [cls=3.0480 | reg=4.6278] | val=8.1009 | lr=8.11e-06 | img=224 | ★ best
INFO    2026-02-23 16:33:49 -0500       workerpool0-0     Epoch 103 | train=7.6714 [cls=3.0510 | reg=4.6204] | val=8.1035 | lr=7.41e-06 | img=224 | 
INFO    2026-02-23 16:33:59 -0500       workerpool0-0     Epoch 104 | train=7.6884 [cls=3.0667 | reg=4.6217] | val=8.1081 | lr=6.73e-06 | img=224 | 
INFO    2026-02-23 16:34:09 -0500       workerpool0-0     Epoch 105 | train=7.6752 [cls=3.0588 | reg=4.6164] | val=8.1025 | lr=6.08e-06 | img=224 | 
INFO    2026-02-23 16:34:20 -0500       workerpool0-0     Epoch 106 | train=7.7060 [cls=3.0798 | reg=4.6262] | val=8.0987 | lr=5.46e-06 | img=224 | ★ best
INFO    2026-02-23 16:34:31 -0500       workerpool0-0     Epoch 107 | train=7.6541 [cls=3.0388 | reg=4.6153] | val=8.1060 | lr=4.87e-06 | img=224 | 
INFO    2026-02-23 16:34:41 -0500       workerpool0-0     Epoch 108 | train=7.6750 [cls=3.0536 | reg=4.6214] | val=8.0990 | lr=4.30e-06 | img=224 | 
INFO    2026-02-23 16:34:52 -0500       workerpool0-0     Epoch 109 | train=7.6640 [cls=3.0447 | reg=4.6193] | val=8.1035 | lr=3.78e-06 | img=224 | 
INFO    2026-02-23 16:35:02 -0500       workerpool0-0     Epoch 110 | train=7.6678 [cls=3.0485 | reg=4.6193] | val=8.1084 | lr=3.28e-06 | img=224 | 
INFO    2026-02-23 16:35:12 -0500       workerpool0-0     Epoch 111 | train=7.6968 [cls=3.0785 | reg=4.6183] | val=8.1108 | lr=2.82e-06 | img=224 | 
INFO    2026-02-23 16:35:22 -0500       workerpool0-0     Epoch 112 | train=7.6717 [cls=3.0566 | reg=4.6151] | val=8.1002 | lr=2.39e-06 | img=224 | 
INFO    2026-02-23 16:35:33 -0500       workerpool0-0     Epoch 113 | train=7.6580 [cls=3.0433 | reg=4.6146] | val=8.0962 | lr=2.00e-06 | img=224 | ★ best
INFO    2026-02-23 16:35:43 -0500       workerpool0-0     Epoch 114 | train=7.6831 [cls=3.0599 | reg=4.6232] | val=8.1034 | lr=1.64e-06 | img=224 | 
INFO    2026-02-23 16:35:53 -0500       workerpool0-0     Epoch 115 | train=7.6829 [cls=3.0649 | reg=4.6180] | val=8.0946 | lr=1.32e-06 | img=224 | ★ best
INFO    2026-02-23 16:36:03 -0500       workerpool0-0     Epoch 116 | train=7.6612 [cls=3.0452 | reg=4.6159] | val=8.0929 | lr=1.04e-06 | img=224 | ★ best
INFO    2026-02-23 16:36:13 -0500       workerpool0-0     Epoch 117 | train=7.6664 [cls=3.0501 | reg=4.6163] | val=8.0954 | lr=7.89e-07 | img=224 | 
INFO    2026-02-23 16:36:23 -0500       workerpool0-0     Epoch 118 | train=7.6796 [cls=3.0630 | reg=4.6166] | val=8.0985 | lr=5.79e-07 | img=224 | 
INFO    2026-02-23 16:36:34 -0500       workerpool0-0     Epoch 119 | train=7.6735 [cls=3.0549 | reg=4.6187] | val=8.0973 | lr=4.07e-07 | img=224 | 
INFO    2026-02-23 16:36:34 -0500       workerpool0-0   ✅ Phase 2 (full fine-tuning) completada en 13.8 min
INFO    2026-02-23 16:36:34 -0500       workerpool0-0   📊 Entrenamiento completo: 120 epochs
INFO    2026-02-23 16:36:34 -0500       workerpool0-0      Mejor val_loss: 8.0929 (epoch 116)
INFO    2026-02-23 16:36:34 -0500       workerpool0-0      Tiempo total: 23.5 min
INFO    2026-02-23 16:36:34 -0500       workerpool0-0   ⏱️  Entrenamiento completado en 23.6 min
INFO    2026-02-23 16:36:35 -0500       workerpool0-0   💾 Historial guardado: /tmp/training/training_history.csv
INFO    2026-02-23 16:36:35 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:36:35 -0500       workerpool0-0   BLOQUE 5 — Curvas de Entrenamiento
INFO    2026-02-23 16:36:35 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   📊 Curvas guardadas: /tmp/training/training_curves.png
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   📈 Resumen – PYTORCH 
INFO    2026-02-23 16:36:37 -0500       workerpool0-0     Épocas: 120
INFO    2026-02-23 16:36:37 -0500       workerpool0-0     Mejor val_loss: 8.0929 (epoch 116)
INFO    2026-02-23 16:36:37 -0500       workerpool0-0     Resoluciones: [640, 416, 320, 224]
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   BLOQUE 6 — Evaluación en Validación
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:36:37 -0500       workerpool0-0   ✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_espdet.pt
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=val
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     mAP@50:    0.0154
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     mAP@50-95: 0.0052
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     Precision: 0.0005
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     Recall:    0.1024
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     F1-Score:  0.0011
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     Imágenes: 188 | Detecciones: 107629 | GT: 762
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     Avg inference: 44.5 ms
INFO    2026-02-23 16:36:53 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 16:36:53 -0500       workerpool0-0       dog                   0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0       door                  0.0763
INFO    2026-02-23 16:36:53 -0500       workerpool0-0       obstacle              0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0       person                0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0       stair                 0.0004
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   📊 Val mAP@50: 0.0154
INFO    2026-02-23 16:36:53 -0500       workerpool0-0      dog: 0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0      door: 0.0763
INFO    2026-02-23 16:36:53 -0500       workerpool0-0      obstacle: 0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0      person: 0.0000
INFO    2026-02-23 16:36:53 -0500       workerpool0-0      stair: 0.0004
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   BLOQUE 7 — Evaluación en Test
INFO    2026-02-23 16:36:53 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=test
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     mAP@50:    0.0105
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     mAP@50-95: 0.0023
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     Precision: 0.0009
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     Recall:    0.1597
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     F1-Score:  0.0017
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     Imágenes: 187 | Detecciones: 96283 | GT: 576
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     Avg inference: 34.4 ms
INFO    2026-02-23 16:37:06 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 16:37:06 -0500       workerpool0-0       dog                   0.0008
INFO    2026-02-23 16:37:06 -0500       workerpool0-0       door                  0.0495
INFO    2026-02-23 16:37:06 -0500       workerpool0-0       obstacle              0.0000
INFO    2026-02-23 16:37:06 -0500       workerpool0-0       person                0.0002
INFO    2026-02-23 16:37:06 -0500       workerpool0-0       stair                 0.0018
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   📊 Test mAP@50: 0.0105
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   BLOQUE 8 — Guardado y subida a GCS
INFO    2026-02-23 16:37:06 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 16:37:07 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/test_confusion_matrix.png
INFO    2026-02-23 16:37:07 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/test_evaluation.json
INFO    2026-02-23 16:37:08 -0500       workerpool0-0     ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/experiment.json
INFO    2026-02-23 16:37:08 -0500       workerpool0-0     ☁️  Subido: /tmp/training/checkpoints/best_espdet.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/checkpoints/best_espdet.pt
INFO    2026-02-23 16:37:08 -0500       workerpool0-0     ☁️  Subido: /tmp/training/export/espdet_pico.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/export/espdet_pico.onnx
INFO    2026-02-23 16:37:08 -0500       workerpool0-0   ✅ Pipeline ESPDet-Pico completado exitosamente.
INFO    2026-02-23 16:37:11 -0500       workerpool0-0   Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO    2026-02-23 16:37:18 -0500       service Tearing down training program.
INFO    2026-02-23 16:37:55 -0500       service Finished tearing down training program.
INFO    2026-02-23 16:37:56 -0500       service Job completed successfully.
```


Log de Terminal de Lanzamiento:

```zsh
% bash vertex_ai/build_and_launch.sh espdet_pico_v1 --run-name espdet_pico_v1-train1
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.5.0' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.5.0.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.5.0.tar.gz [Content-Type=application/x-tar]...
- [1 files][ 68.9 KiB/ 68.9 KiB]                                                
Operation completed over 1 objects/68.9 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz

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
  Run:           espdet_pico_v1-train1
  Familia:       ESPDet
  Módulo:        trainer.task_espdet
  Contenedor:    us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
  Máquina:       n1-standard-8
  GPU:           NVIDIA_TESLA_T4 x1
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.5.0.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet_pico_v1-train1']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1-train1.yaml

🚀 Lanzando Custom Job: tfm-espdet_pico_v1-1771880828
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-23-16:07:09.903 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/1278128305320493056?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/3958704668489547776?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob run completed. Resource name: projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056
Training did not produce a Managed Model returning None. Training Pipeline projects/608533675308/locations/us-central1/trainingPipelines/1278128305320493056 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR'].

============================================================
✅ Custom Job completado exitosamente
   Resultados en: gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1
   Experiments:   tfm-deteccion-objetos / espdet_pico_v1-train1
============================================================

═══════════════════════════════════════════════════════════
✅ Proceso completado
═══════════════════════════════════════════════════════════
```