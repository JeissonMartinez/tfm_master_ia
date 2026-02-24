# Registro de Entrenamiento - ESPDet-Pico (v2 - Espressif Official)
**Corrida Exitosa #:** 2 (Fail)

## Logs de Monitor de Entrenamiento:
```zsh
% gcloud ai custom-jobs stream-logs 8501620632646057984 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO    2026-02-23 18:48:37 -0500       service Waiting for job to be provisioned.
INFO    2026-02-23 18:48:37 -0500       service Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO    2026-02-23 18:48:37 -0500       service Vertex AI is setting up this job.
INFO    2026-02-23 18:48:37 -0500       service Waiting for training program to start.
INFO    2026-02-23 18:48:38 -0500       service Job is preparing.
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-dcdsd89h-a8e84b5d-8rl2
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-782452dcb4-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet-pico-v2-t2"]}
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:03 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:04 -0500       service Job is running.
ERROR   2026-02-23 18:52:04 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 18:52:04 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 18:52:07 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 18:52:07 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 18:52:11 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:11 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:14 -0500       workerpool0-0   Processing /tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:14 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 18:52:19 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 18:52:19 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 18:52:19 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 18:52:19 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 18:52:20 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:52:20 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 18:52:20 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 18:52:20 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:52:20 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.0-py3-none-any.whl size=90902 sha256=9d12c067f16e0b5c8f3d0a69acdd532f98dff90c0b10234a45c11ae8699c7978
INFO    2026-02-23 18:52:20 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/7f/36/7b/b824d28beab63f13b30c80bf3768d6e455873551726f4e5c53
INFO    2026-02-23 18:52:20 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 18:52:20 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 18:52:20 -0500       workerpool0-0   Successfully installed tfm-trainer-2.6.0
ERROR   2026-02-23 18:52:20 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 18:52:21 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 18:52:21 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 18:52:21 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:22 -0500       workerpool0-0   Processing /tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:52:22 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 18:52:25 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 18:52:25 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 18:52:26 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 18:52:26 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 18:52:27 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:52:27 -0500       workerpool0-0   Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (6.0.2)
INFO    2026-02-23 18:52:27 -0500       workerpool0-0   Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:27 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 18:52:27 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (2.2.3)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (1.6.1)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0   Collecting albumentations>=2.0.0 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:28 -0500       workerpool0-0     Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO    2026-02-23 18:52:29 -0500       workerpool0-0   Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (0.19.0+cu124)
INFO    2026-02-23 18:52:29 -0500       workerpool0-0   Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (2.19.0)
INFO    2026-02-23 18:52:29 -0500       workerpool0-0   Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (1.91.0)
INFO    2026-02-23 18:52:29 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:29 -0500       workerpool0-0     Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 18:52:30 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:30 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 18:52:30 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.6.0) (1.11.4)
INFO    2026-02-23 18:52:31 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:31 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 18:52:31 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:31 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 18:52:32 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:33 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 18:52:34 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:34 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.34.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.40.3)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.26.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.20.3)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (25.0)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.34.0)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.14.2)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.1.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (4.14.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.16)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.70.0)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.32.4)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.73.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.48.2)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (5.5.2)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.4.2)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (4.9.1)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:35 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.4.3)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.7.2)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.9.0.post0)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.14.2)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.6.0) (1.7.1)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:36 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.17.0)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.4.2)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.10)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.26.20)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2025.6.15)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.6.1)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (1.3.2)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (0.12.1)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (4.58.5)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (1.4.8)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (11.3.0)
INFO    2026-02-23 18:52:38 -0500       workerpool0-0   Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (3.2.3)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.6.0) (1.14.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0     Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.0) (2025.2)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.0) (2025.2)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.0) (1.5.1)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.0) (3.6.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.6.0) (2.4.0+cu124)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.18.0)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.4.2)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.1.6)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (2025.5.1)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (9.1.0.70)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.2.65)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (11.2.0.44)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (10.3.5.119)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (11.6.0.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.3.0.142)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (2.20.5)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (12.4.99)
INFO    2026-02-23 18:52:39 -0500       workerpool0-0   Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.0.0)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.6.0)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.0.2)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.6.0) (1.3.0)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 121.4 MB/s eta 0:00:00
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 108.0 MB/s eta 0:00:00
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 125.7 MB/s eta 0:00:00
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 123.6 MB/s eta 0:00:00
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 18:52:40 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 124.7 MB/s eta 0:00:00
INFO    2026-02-23 18:52:40 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 124.1 MB/s eta 0:00:00
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 27.3 MB/s eta 0:00:00
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 105.0 MB/s eta 0:00:00
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 18:52:41 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 18:52:42 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 18:52:42 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 18:52:42 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:52:42 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.0-py3-none-any.whl size=90902 sha256=41349fde1ab05b6ed81cd6e7818872a4c02a543ca3b259a80d9973b44e3d335a
INFO    2026-02-23 18:52:42 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/7f/36/7b/b824d28beab63f13b30c80bf3768d6e455873551726f4e5c53
INFO    2026-02-23 18:52:42 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 18:52:43 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 18:52:44 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:44 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:52:47 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:47 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:52:47 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:47 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:52:52 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:52 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:52:52 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:52 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:52:53 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:52:53 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 18:52:59 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 18:52:59 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.6.0
INFO    2026-02-23 18:52:59 -0500       workerpool0-0       Uninstalling tfm-trainer-2.6.0:
INFO    2026-02-23 18:52:59 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.6.0
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
INFO    2026-02-23 18:52:59 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.6.0 typing-inspection-0.4.2
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 18:52:59 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 18:53:00 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet-pico-v2-t2
INFO    2026-02-23 18:53:03 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:03 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 18:53:03 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml → /tmp/training/config.yaml (0.0 MB)
INFO    2026-02-23 18:53:29 -0500       workerpool0-0   🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Nombre:       espdet_pico_v2
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Familia:      ESPDet
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Variante:     espdet_pico
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Versión:      v2
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Descripción:  ESPDet-Pico oficial Espressif + transfer learning cat-detection
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Dataset:      iodc_yolo
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Img Size:     224×224
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Batch Size:   32
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Patience:     25
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Seed:         42
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     Conf Thresh:  0.25
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     IoU Thresh:   0.45
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     📐 2-Phase Training:
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        Phase 1: 50 epochs @ LR=0.001
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        Phase 2: 100 epochs @ LR=0.0001
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        Resize Schedule: {0: 224}
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        Optimizer: AdamW | WD: 0.0005
INFO    2026-02-23 18:53:29 -0500       workerpool0-0     🟢 ESPDet Config:
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        Pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 18:53:29 -0500       workerpool0-0        reg_max: 1
INFO    2026-02-23 18:53:29 -0500       workerpool0-0   ✅ Configuración aplicada correctamente
INFO    2026-02-23 18:53:29 -0500       workerpool0-0   📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO    2026-02-23 18:53:30 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO    2026-02-23 18:53:30 -0500       workerpool0-0     📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     ✅ Descomprimido correctamente
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   🖥️  Device: cuda
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   ⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO    2026-02-23 18:53:32 -0500       workerpool0-0      Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   domain: "googleapis.com"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   metadata {
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     key: "method"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   }
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   metadata {
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     key: "service"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     value: "aiplatform.googleapis.com"
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   }
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   ]
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   BLOQUE 2 — Verificación del Dataset
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   📂 Dataset YOLO: iodc_yolo
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO    2026-02-23 18:53:32 -0500       workerpool0-0     ✅  test:   187 imgs |   187 labels | 0 sin label
INFO    2026-02-23 18:53:32 -0500       workerpool0-0   📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO    2026-02-23 18:53:33 -0500       workerpool0-0   generated new fontManager
INFO    2026-02-23 18:53:34 -0500       workerpool0-0     📊 Guardado: /tmp/training/class_distribution.png
INFO    2026-02-23 18:53:34 -0500       workerpool0-0   ⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO    2026-02-23 18:53:36 -0500       workerpool0-0     🖼️  Guardado: /tmp/training/gt_samples.png
INFO    2026-02-23 18:53:36 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:36 -0500       workerpool0-0   BLOQUE 3 — Construcción del Modelo ESPDet-Pico
INFO    2026-02-23 18:53:36 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:53:36 -0500       workerpool0-0   ⬇️  Descargando pesos pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 18:53:36 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt → /tmp/training/pretrained_weights.pt (1.0 MB)
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0   Traceback (most recent call last):
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/opt/python/3.10/lib/python3.10/runpy.py", line 196, in _run_module_as_main
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       return _run_code(code, main_globals, None,
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/opt/python/3.10/lib/python3.10/runpy.py", line 86, in _run_code
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       exec(code, run_globals)
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py", line 620, in <module>
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       main()
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py", line 293, in main
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       model = build_espdet_pico(
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py", line 532, in build_espdet_pico
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       model = ESPDetPico(nc=num_classes)
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0     File "/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py", line 261, in __init__
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0       from ultralytics.nn.modules.conv import Conv
ERROR   2026-02-23 18:53:36 -0500       workerpool0-0   ModuleNotFoundError: No module named 'ultralytics'
ERROR   2026-02-23 18:53:39 -0500       workerpool0-0   Command '['python3', '-m', 'trainer.task_espdet', '--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet-pico-v2-t2']' returned non-zero exit status 1.
INFO    2026-02-23 18:53:39 -0500       workerpool0-0   Task completed. Exit code (1). Exit reason (SUBPROCESS_EXCEPTION)
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-dcdsd89h-a8e84b5d-8rl2
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-782452dcb4-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet-pico-v2-t2"]}
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:53:56 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz tfm_trainer-2.6.0.tar.gz
ERROR   2026-02-23 18:53:56 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 18:53:56 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 18:53:57 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 18:53:57 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 18:54:00 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:54:00 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:54:02 -0500       workerpool0-0   Processing /tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:54:02 -0500       workerpool0-0     Installing build dependencies: started
ERROR   2026-02-23 18:54:05 -0500       service The replica workerpool0-0 exited with a non-zero status of 1. Termination reason: Error. 
ERROR   2026-02-23 18:54:05 -0500       service Traceback (most recent call last):
ERROR   2026-02-23 18:54:05 -0500       service   File "/opt/python/3.10/lib/python3.10/runpy.py", line 196, in _run_module_as_main
ERROR   2026-02-23 18:54:05 -0500       service     return _run_code(code, main_globals, None,
ERROR   2026-02-23 18:54:05 -0500       service   File "/opt/python/3.10/lib/python3.10/runpy.py", line 86, in _run_code
ERROR   2026-02-23 18:54:05 -0500       service     exec(code, run_globals)
ERROR   2026-02-23 18:54:05 -0500       service   File "/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py", line 620, in <module>
ERROR   2026-02-23 18:54:05 -0500       service     main()
ERROR   2026-02-23 18:54:05 -0500       service   File "/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py", line 293, in main
ERROR   2026-02-23 18:54:05 -0500       service     model = build_espdet_pico(
ERROR   2026-02-23 18:54:05 -0500       service   File "/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py", line 532, in build_espdet_pico
ERROR   2026-02-23 18:54:05 -0500       service     model = ESPDetPico(nc=num_classes)
ERROR   2026-02-23 18:54:05 -0500       service   File "/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py", line 261, in __init__
ERROR   2026-02-23 18:54:05 -0500       service     from ultralytics.nn.modules.conv import Conv
ERROR   2026-02-23 18:54:05 -0500       service ModuleNotFoundError: No module named 'ultralytics'
ERROR   2026-02-23 18:54:05 -0500       service
ERROR   2026-02-23 18:54:05 -0500       service To find out more about why your job exited please check the logs: https://console.cloud.google.com/logs/viewer?project=608533675308&resource=ml_job%2Fjob_id%2F8501620632646057984&advancedFilter=resource.type%3D%22ml_job%22%0Aresource.labels.job_id%3D%228501620632646057984%22
INFO    2026-02-23 18:54:06 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 18:54:06 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 18:54:07 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 18:54:07 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 18:54:07 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:54:07 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 18:54:07 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 18:54:08 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:54:08 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.0-py3-none-any.whl size=90902 sha256=6aa88cacf0a706064380a8f336a8eb08f2755133abc13c605de1d2abf6a4d966
INFO    2026-02-23 18:54:08 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/7f/36/7b/b824d28beab63f13b30c80bf3768d6e455873551726f4e5c53
INFO    2026-02-23 18:54:08 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 18:54:08 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 18:54:08 -0500       workerpool0-0   Successfully installed tfm-trainer-2.6.0
ERROR   2026-02-23 18:54:08 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 18:54:08 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 18:54:08 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 18:54:08 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:54:09 -0500       workerpool0-0   Processing /tfm_trainer-2.6.0.tar.gz
INFO    2026-02-23 18:54:09 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 18:54:13 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 18:54:13 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 18:54:13 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 18:54:13 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 18:54:14 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:54:14 -0500       workerpool0-0   Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (6.0.2)
INFO    2026-02-23 18:54:14 -0500       workerpool0-0   Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:14 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (2.2.3)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (1.6.1)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Collecting albumentations>=2.0.0 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0     Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (0.19.0+cu124)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (2.19.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.0) (1.91.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0     Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:15 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 18:54:16 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.6.0) (1.11.4)
INFO    2026-02-23 18:54:17 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:17 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 18:54:18 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:18 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 18:54:19 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:19 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 18:54:21 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:21 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.34.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.40.3)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.26.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.20.3)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (25.0)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.34.0)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.14.2)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.1.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (4.14.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.16)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.70.0)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.32.4)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.73.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.48.2)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (5.5.2)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.4.2)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (4.9.1)
INFO    2026-02-23 18:54:22 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:54:23 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:23 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:54:23 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 18:54:24 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 18:54:24 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.4.3)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.7.2)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2.9.0.post0)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.14.2)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.6.0) (1.7.1)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:25 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.17.0)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.4.2)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (3.10)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (1.26.20)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (2025.6.15)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.0) (0.6.1)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (1.3.2)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (0.12.1)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (4.58.5)
INFO    2026-02-23 18:54:26 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.0) (1.4.8)
INFO    2026-02-23 18:54:27 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 18:54:27 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 18:54:27 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 18:54:27 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 18:54:27 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:28 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.6.0)
INFO    2026-02-23 18:54:28 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 18:54:28 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.0) (3.0.2)
INFO    2026-02-23 18:54:28 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.6.0) (1.3.0)
INFO    2026-02-23 18:54:28 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 116.2 MB/s eta 0:00:00
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 101.6 MB/s eta 0:00:00
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 119.7 MB/s eta 0:00:00
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 148.6 MB/s eta 0:00:00
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 126.2 MB/s eta 0:00:00
INFO    2026-02-23 18:54:29 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 18:54:29 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 135.2 MB/s eta 0:00:00
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 16.9 MB/s eta 0:00:00
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 61.3 MB/s eta 0:00:00
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 18:54:30 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 18:54:30 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 18:54:31 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 18:54:31 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.0-py3-none-any.whl size=90902 sha256=ec28b8be687749ed718b4af94b536f55064744c4c41a797fc01c3693eab885b3
INFO    2026-02-23 18:54:31 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/7f/36/7b/b824d28beab63f13b30c80bf3768d6e455873551726f4e5c53
INFO    2026-02-23 18:54:31 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 18:54:32 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 18:54:32 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:32 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:54:35 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:35 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:54:35 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:35 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:54:40 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:40 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:54:40 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:40 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 18:54:41 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 18:54:41 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 18:54:47 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 18:54:47 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.6.0
INFO    2026-02-23 18:54:47 -0500       workerpool0-0       Uninstalling tfm-trainer-2.6.0:
INFO    2026-02-23 18:54:47 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.6.0
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
INFO    2026-02-23 18:54:48 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.6.0 typing-inspection-0.4.2
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 18:54:48 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 18:54:49 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet-pico-v2-t2
INFO    2026-02-23 18:54:53 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:54:53 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 18:54:53 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 18:55:55 -0500       service Finished tearing down training program.
INFO    2026-02-23 18:55:55 -0500       service Job failed.
```

## Logs de Terminal de Lanzamiento del Job

```bash
% ./vertex_ai/build_and_launch.sh espdet_pico_v2 --run-name espdet-pico-v2-t2
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.6.0' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.6.0.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.6.0.tar.gz [Content-Type=application/x-tar]...
/ [1 files][ 75.2 KiB/ 75.2 KiB]                                                
Operation completed over 1 objects/75.2 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz

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
  Run:           espdet-pico-v2-t2
  Familia:       ESPDet
  Módulo:        trainer.task_espdet
  Contenedor:    us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
  Máquina:       n1-standard-8
  GPU:           NVIDIA_TESLA_T4 x1
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.0.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet-pico-v2-t2']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml

🚀 Lanzando Custom Job: tfm-espdet_pico_v2-1771890482
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-23-18:48:04.010 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/8560413718406496256?project=608533675308
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/8501620632646057984?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/8560413718406496256 current state:
PipelineState.PIPELINE_STATE_RUNNING
Traceback (most recent call last):
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps/vertex_ai/launch_job.py", line 214, in <module>
    main()
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps/vertex_ai/launch_job.py", line 197, in main
    job.run(
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 8155, in run
    return self._run(
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/base.py", line 862, in wrapper
    return method(*args, **kwargs)
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 8485, in _run
    model = self._run_job(
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 855, in _run_job
    model = self._get_model(block=block)
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 942, in _get_model
    self._block_until_complete()
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 985, in _block_until_complete
    self._raise_failure()
  File "/Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/env/lib/python3.10/site-packages/google/cloud/aiplatform/training_jobs.py", line 1002, in _raise_failure
    raise RuntimeError("Training failed with:\n%s" % self._gca_resource.error)
RuntimeError: Training failed with:
code: 3
message: "The replica workerpool0-0 exited with a non-zero status of 1. Termination reason: Error. \nTraceback (most recent call last):\n  File \"/opt/python/3.10/lib/python3.10/runpy.py\", line 196, in _run_module_as_main\n    return _run_code(code, main_globals, None,\n  File \"/opt/python/3.10/lib/python3.10/runpy.py\", line 86, in _run_code\n    exec(code, run_globals)\n  File \"/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py\", line 620, in <module>\n    main()\n  File \"/root/.local/lib/python3.10/site-packages/trainer/task_espdet.py\", line 293, in main\n    model = build_espdet_pico(\n  File \"/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py\", line 532, in build_espdet_pico\n    model = ESPDetPico(nc=num_classes)\n  File \"/root/.local/lib/python3.10/site-packages/src_colab/utils_model.py\", line 261, in __init__\n    from ultralytics.nn.modules.conv import Conv\nModuleNotFoundError: No module named \'ultralytics\'\n\nTo find out more about why your job exited please check the logs: https://console.cloud.google.com/logs/viewer?project=608533675308&resource=ml_job%2Fjob_id%2F8501620632646057984&advancedFilter=resource.type%3D%22ml_job%22%0Aresource.labels.job_id%3D%228501620632646057984%22"
```