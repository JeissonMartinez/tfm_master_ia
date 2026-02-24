# Registro de Entrenamiento - ESPDet-Pico (v2 - Espressif Official)
**Corrida Exitosa #:** 4 (exitoso)

## Logs de Monitor de Entrenamiento:
```zsh
% gcloud ai custom-jobs stream-logs 5636733135312912384 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO    2026-02-23 21:47:42 -0500       service Waiting for job to be provisioned.
INFO    2026-02-23 21:47:42 -0500       service Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO    2026-02-23 21:47:42 -0500       service Vertex AI is setting up this job.
INFO    2026-02-23 21:47:42 -0500       service Waiting for training program to start.
INFO    2026-02-23 21:47:43 -0500       service Job is preparing.
INFO    2026-02-23 21:51:20 -0500       service Job is running.
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-psoa4xvz-72e3b1b1-7r7r
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-08cec8cfa1-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet-pico-v4-t4"]}
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:27 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz tfm_trainer-2.6.3.tar.gz
ERROR   2026-02-23 21:51:28 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 21:51:28 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 21:51:31 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 21:51:31 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 21:51:35 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:35 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:38 -0500       workerpool0-0   Processing /tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:39 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 21:51:44 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 21:51:44 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 21:51:44 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 21:51:44 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 21:51:45 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 21:51:45 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 21:51:45 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 21:51:45 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 21:51:45 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.3-py3-none-any.whl size=91307 sha256=4a80209a930c3a244a2e09ba299b699b17b2f6d3ea8b5fec1c66c238b96564ff
INFO    2026-02-23 21:51:45 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/97/3f/6f/742b4ab67323475bcd0f3a83b5e91f621145e40896690b3ffc
INFO    2026-02-23 21:51:45 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 21:51:45 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 21:51:46 -0500       workerpool0-0   Successfully installed tfm-trainer-2.6.3
ERROR   2026-02-23 21:51:46 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 21:51:46 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 21:51:46 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 21:51:46 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:47 -0500       workerpool0-0   Processing /tfm_trainer-2.6.3.tar.gz
INFO    2026-02-23 21:51:47 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 21:51:51 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 21:51:51 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 21:51:52 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 21:51:52 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 21:51:52 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 21:51:52 -0500       workerpool0-0   Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (6.0.2)
INFO    2026-02-23 21:51:53 -0500       workerpool0-0   Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:53 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 21:51:53 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (2.2.3)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (1.6.1)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Collecting albumentations>=2.0.0 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0     Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Collecting ultralytics>=8.2 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0     Downloading ultralytics-8.4.15-py3-none-any.whl.metadata (39 kB)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (0.19.0+cu124)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (2.19.0)
INFO    2026-02-23 21:51:54 -0500       workerpool0-0   Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.3) (1.91.0)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0     Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.6.3) (1.11.4)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:55 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 21:51:56 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:56 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.34.1)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.40.3)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.26.1)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (3.20.3)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (25.0)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (3.34.0)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.14.2)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.1.1)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (4.14.1)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (0.16)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.70.0)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.32.4)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.73.1)
INFO    2026-02-23 21:51:57 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.48.2)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (5.5.2)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (0.4.2)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (4.9.1)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 21:51:58 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.4.3)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.7.2)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2.9.0.post0)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (0.14.2)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.6.3) (1.7.1)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:51:59 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 21:52:01 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:01 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.17.0)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (3.4.2)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (3.10)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (1.26.20)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (2025.6.15)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.3) (0.6.1)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (1.3.2)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (0.12.1)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (4.58.5)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (1.4.8)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (11.3.0)
INFO    2026-02-23 21:52:02 -0500       workerpool0-0   Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.3) (3.2.3)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0   INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:52:03 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0     Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 21:52:03 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0     Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0   Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0     Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0   Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.6.3) (1.14.0)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:52:04 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:52:04 -0500       workerpool0-0     Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0     Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.3) (2025.2)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.3) (2025.2)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.3) (1.5.1)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.3) (3.6.0)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.6.3) (2.4.0+cu124)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (3.18.0)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (3.4.2)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (3.1.6)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (2025.5.1)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (9.1.0.70)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.2.65)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (11.2.0.44)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (10.3.5.119)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (11.6.0.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.3.0.142)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (2.20.5)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (12.4.99)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (3.0.0)
INFO    2026-02-23 21:52:05 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:06 -0500       workerpool0-0     Downloading opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:52:06 -0500       workerpool0-0   Requirement already satisfied: psutil>=5.8.0 in /opt/python/3.10/lib/python3.10/site-packages (from ultralytics>=8.2->tfm-trainer==2.6.3) (5.9.3)
INFO    2026-02-23 21:52:06 -0500       workerpool0-0   Collecting polars>=0.20.0 (from ultralytics>=8.2->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:06 -0500       workerpool0-0     Downloading polars-1.38.1-py3-none-any.whl.metadata (10 kB)
INFO    2026-02-23 21:52:07 -0500       workerpool0-0   Collecting ultralytics-thop>=2.0.18 (from ultralytics>=8.2->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:07 -0500       workerpool0-0     Downloading ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
INFO    2026-02-23 21:52:07 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 21:52:07 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:07 -0500       workerpool0-0     Downloading opencv_python-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:52:07 -0500       workerpool0-0     Downloading opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 21:52:08 -0500       workerpool0-0     Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 21:52:08 -0500       workerpool0-0   Collecting polars-runtime-32==1.38.1 (from polars>=0.20.0->ultralytics>=8.2->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:08 -0500       workerpool0-0     Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.6.3)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.3) (3.0.2)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.6.3) (1.3.0)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 122.2 MB/s eta 0:00:00
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 88.6 MB/s eta 0:00:00
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 120.3 MB/s eta 0:00:00
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 21:52:09 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 80.0 MB/s eta 0:00:00
INFO    2026-02-23 21:52:09 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 125.0 MB/s eta 0:00:00
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 119.0 MB/s eta 0:00:00
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 27.3 MB/s eta 0:00:00
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 54.2 MB/s eta 0:00:00
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading ultralytics-8.4.15-py3-none-any.whl (1.2 MB)
INFO    2026-02-23 21:52:10 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 55.0 MB/s eta 0:00:00
INFO    2026-02-23 21:52:10 -0500       workerpool0-0   Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 MB 119.7 MB/s eta 0:00:00
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading polars-1.38.1-py3-none-any.whl (810 kB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 810.4/810.4 kB 28.1 MB/s eta 0:00:00
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (45.8 MB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.8/45.8 MB 122.1 MB/s eta 0:00:00
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 21:52:11 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 21:52:12 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 21:52:12 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 21:52:13 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 21:52:13 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.3-py3-none-any.whl size=91307 sha256=f17996a9bac171a35bea9559ef09b848bd2901debfa221635fe82752a39bf8c5
INFO    2026-02-23 21:52:13 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/97/3f/6f/742b4ab67323475bcd0f3a83b5e91f621145e40896690b3ffc
INFO    2026-02-23 21:52:13 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 21:52:14 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, polars-runtime-32, numpy, humanfriendly, annotated-types, pydantic, polars, opencv-python-headless, opencv-python, onnx, coloredlogs, onnxruntime, matplotlib, albucore, ultralytics-thop, albumentations, ultralytics, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 21:52:15 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:15 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:19 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:19 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:19 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:19 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:28 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:28 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:29 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:29 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:31 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:31 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 21:52:35 -0500       workerpool0-0     WARNING: The scripts ultralytics and yolo are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 21:52:35 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 21:52:36 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 21:52:36 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.6.3
INFO    2026-02-23 21:52:36 -0500       workerpool0-0       Uninstalling tfm-trainer-2.6.3:
INFO    2026-02-23 21:52:36 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.6.3
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
INFO    2026-02-23 21:52:36 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-4.11.0.86 opencv-python-headless-4.11.0.86 polars-1.38.1 polars-runtime-32-1.38.1 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.6.3 typing-inspection-0.4.2 ultralytics-8.4.15 ultralytics-thop-2.0.18
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 21:52:36 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 21:52:37 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet-pico-v4-t4
INFO    2026-02-23 21:52:41 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:52:41 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 21:52:41 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml → /tmp/training/config.yaml (0.0 MB)
INFO    2026-02-23 21:53:06 -0500       workerpool0-0   🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Nombre:       espdet_pico_v4
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Familia:      ESPDet
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Variante:     espdet_pico
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Versión:      v4
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Descripción:  ESPDet-Pico BCE + NMS tuning (conf=0.35, iou=0.40)
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Dataset:      iodc_yolo
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Img Size:     224×224
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Batch Size:   32
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Patience:     25
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Seed:         42
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     Conf Thresh:  0.35
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     IoU Thresh:   0.4
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     📐 2-Phase Training:
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        Phase 1: 50 epochs @ LR=0.001
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        Phase 2: 100 epochs @ LR=0.0001
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        Resize Schedule: {0: 224}
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        Optimizer: AdamW | WD: 0.0005
INFO    2026-02-23 21:53:06 -0500       workerpool0-0     🟢 ESPDet Config:
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        Pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 21:53:06 -0500       workerpool0-0        reg_max: 1
INFO    2026-02-23 21:53:06 -0500       workerpool0-0   ✅ Configuración aplicada correctamente
INFO    2026-02-23 21:53:06 -0500       workerpool0-0   📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO    2026-02-23 21:53:08 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO    2026-02-23 21:53:08 -0500       workerpool0-0     📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO    2026-02-23 21:53:09 -0500       workerpool0-0     ✅ Descomprimido correctamente
INFO    2026-02-23 21:53:09 -0500       workerpool0-0     🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO    2026-02-23 21:53:09 -0500       workerpool0-0     📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO    2026-02-23 21:53:09 -0500       workerpool0-0   🖥️  Device: cuda
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   ⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO    2026-02-23 21:53:10 -0500       workerpool0-0      Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   domain: "googleapis.com"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   metadata {
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     key: "method"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   }
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   metadata {
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     key: "service"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     value: "aiplatform.googleapis.com"
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   }
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   ]
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   BLOQUE 2 — Verificación del Dataset
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   📂 Dataset YOLO: iodc_yolo
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO    2026-02-23 21:53:10 -0500       workerpool0-0     ✅  test:   187 imgs |   187 labels | 0 sin label
INFO    2026-02-23 21:53:10 -0500       workerpool0-0   📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO    2026-02-23 21:53:11 -0500       workerpool0-0   generated new fontManager
INFO    2026-02-23 21:53:11 -0500       workerpool0-0     📊 Guardado: /tmp/training/class_distribution.png
INFO    2026-02-23 21:53:11 -0500       workerpool0-0   ⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO    2026-02-23 21:53:14 -0500       workerpool0-0     🖼️  Guardado: /tmp/training/gt_samples.png
INFO    2026-02-23 21:53:14 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:14 -0500       workerpool0-0   BLOQUE 3 — Construcción del Modelo ESPDet-Pico
INFO    2026-02-23 21:53:14 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:14 -0500       workerpool0-0   ⬇️  Descargando pesos pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 21:53:14 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt → /tmp/training/pretrained_weights.pt (1.0 MB)
INFO    2026-02-23 21:53:15 -0500       workerpool0-0   Creating new Ultralytics Settings v0.0.6 file ✅ 
INFO    2026-02-23 21:53:15 -0500       workerpool0-0   View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
INFO    2026-02-23 21:53:15 -0500       workerpool0-0   Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
INFO    2026-02-23 21:53:15 -0500       workerpool0-0   🔄 Cargando pesos pretrained: /tmp/training/pretrained_weights.pt
INFO    2026-02-23 21:53:15 -0500       workerpool0-0     ✅ Transfer learning: 622 param groups cargados
INFO    2026-02-23 21:53:15 -0500       workerpool0-0     ℹ️  Shape mismatch (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 21:53:15 -0500       workerpool0-0     ℹ️  Missing keys (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 21:53:15 -0500       workerpool0-0   ✅ ESPDet-Pico (oficial) construido: 361,563 params (361,563 trainable)
INFO    2026-02-23 21:53:15 -0500       workerpool0-0      Strides: [8, 16, 32] | Classes: 5
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 21:53:16 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   📦 Modelo: ESPDet-Pico
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Total params:          361,563
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Trainable:             148,123
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Non-trainable:         213,440
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Est. float32:           1.38 MB
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Est. INT8:              0.34 MB
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   📐 Tamaño estimado: 1.38 MB (FP32), 0.34 MB (INT8)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.6.3 (BCE + NMS tuning)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Architecture:    Official Espressif (esp-detection repo)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Strides:         [8, 16, 32]
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     pretrained:      gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Phase 1:         50 ep, LR=0.001, WD=0.0001
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Phase 2:         100 ep, LR=0.0001, WD=1e-05
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Optimizer:       adamw
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     cls_weight:      1.0
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     reg_weight:      2.0
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Focal Loss:      OFF (γ=0.0, α=0.25)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Conf threshold:  0.35
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     IoU threshold:   0.4
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     AMP:             True
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Grad clip:       5.0
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Export imgsz:    224
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Batch size:      32
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Patience:        25
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', 'aug_hue_shift_limit', 'aug_sat_shift_limit', 'aug_val_shift_limit', 'aug_shift_limit', 'aug_scale_limit', 'aug_rotate_limit', 'aug_hflip_prob', 'aug_gaussian_noise']
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   BLOQUE 4 — Entrenamiento (2 fases)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🎯 ESPDet cls_loss: BCEWithLogitsLoss (standard)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🔧 Two-Phase Training Config
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Phase 1: 50 epochs | LR=0.001 | WD=0.0001
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Phase 2: 100 epochs | LR=0.0001 | WD=1e-05
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Optimizer: adamw | Scheduler: cosine
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Batch: 32 | AMP: True | Patience: 25
INFO    2026-02-23 21:53:16 -0500       workerpool0-0     Resize schedule: [(0, 224)]
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 21:53:16 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 21:53:16 -0500       workerpool0-0   🚀 Phase 1 (backbone frozen) — 50 epochs, LR=0.001
INFO    2026-02-23 21:53:24 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO    2026-02-23 21:53:24 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=47)
INFO    2026-02-23 21:53:46 -0500       workerpool0-0     Epoch   0 | train=14.3979 [cls=10.5796 | reg=3.8183] | val=12.5994 | lr=3.33e-04 | img=224 | ★ best
INFO    2026-02-23 21:54:03 -0500       workerpool0-0     Epoch   1 | train=9.9862 [cls=6.7288 | reg=3.2574] | val=9.0033 | lr=6.67e-04 | img=224 | ★ best
INFO    2026-02-23 21:54:19 -0500       workerpool0-0     Epoch   2 | train=6.7973 [cls=3.9243 | reg=2.8729] | val=6.8110 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 21:54:35 -0500       workerpool0-0     Epoch   3 | train=5.3479 [cls=2.7151 | reg=2.6328] | val=6.0260 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 21:54:51 -0500       workerpool0-0     Epoch   4 | train=4.8178 [cls=2.3407 | reg=2.4772] | val=5.7046 | lr=9.99e-04 | img=224 | ★ best
INFO    2026-02-23 21:55:06 -0500       workerpool0-0     Epoch   5 | train=4.4856 [cls=2.1457 | reg=2.3399] | val=5.5546 | lr=9.96e-04 | img=224 | ★ best
INFO    2026-02-23 21:55:21 -0500       workerpool0-0     Epoch   6 | train=4.2875 [cls=2.0268 | reg=2.2608] | val=5.4775 | lr=9.90e-04 | img=224 | ★ best
INFO    2026-02-23 21:55:35 -0500       workerpool0-0     Epoch   7 | train=4.1271 [cls=1.9257 | reg=2.2014] | val=5.2986 | lr=9.82e-04 | img=224 | ★ best
INFO    2026-02-23 21:55:49 -0500       workerpool0-0     Epoch   8 | train=4.0530 [cls=1.8749 | reg=2.1782] | val=5.4744 | lr=9.72e-04 | img=224 | 
INFO    2026-02-23 21:56:01 -0500       workerpool0-0     Epoch   9 | train=3.9501 [cls=1.8188 | reg=2.1313] | val=5.1866 | lr=9.60e-04 | img=224 | ★ best
INFO    2026-02-23 21:56:13 -0500       workerpool0-0     Epoch  10 | train=3.8992 [cls=1.7712 | reg=2.1280] | val=5.1368 | lr=9.46e-04 | img=224 | ★ best
INFO    2026-02-23 21:56:25 -0500       workerpool0-0     Epoch  11 | train=3.8100 [cls=1.7224 | reg=2.0876] | val=5.0963 | lr=9.30e-04 | img=224 | ★ best
INFO    2026-02-23 21:56:37 -0500       workerpool0-0     Epoch  12 | train=3.7142 [cls=1.6672 | reg=2.0469] | val=5.1439 | lr=9.12e-04 | img=224 | 
INFO    2026-02-23 21:56:49 -0500       workerpool0-0     Epoch  13 | train=3.7161 [cls=1.6722 | reg=2.0439] | val=5.0555 | lr=8.92e-04 | img=224 | ★ best
INFO    2026-02-23 21:57:01 -0500       workerpool0-0     Epoch  14 | train=3.6427 [cls=1.6288 | reg=2.0139] | val=5.1318 | lr=8.71e-04 | img=224 | 
INFO    2026-02-23 21:57:14 -0500       workerpool0-0     Epoch  15 | train=3.5881 [cls=1.5960 | reg=1.9921] | val=5.1720 | lr=8.48e-04 | img=224 | 
INFO    2026-02-23 21:57:26 -0500       workerpool0-0     Epoch  16 | train=3.6030 [cls=1.5918 | reg=2.0113] | val=5.1857 | lr=8.23e-04 | img=224 | 
INFO    2026-02-23 21:57:39 -0500       workerpool0-0     Epoch  17 | train=3.5667 [cls=1.5918 | reg=1.9749] | val=4.9069 | lr=7.97e-04 | img=224 | ★ best
INFO    2026-02-23 21:57:51 -0500       workerpool0-0     Epoch  18 | train=3.5080 [cls=1.5552 | reg=1.9528] | val=4.8497 | lr=7.69e-04 | img=224 | ★ best
INFO    2026-02-23 21:58:03 -0500       workerpool0-0     Epoch  19 | train=3.4974 [cls=1.5377 | reg=1.9597] | val=4.9651 | lr=7.40e-04 | img=224 | 
INFO    2026-02-23 21:58:15 -0500       workerpool0-0     Epoch  20 | train=3.4574 [cls=1.5097 | reg=1.9477] | val=4.8751 | lr=7.10e-04 | img=224 | 
INFO    2026-02-23 21:58:27 -0500       workerpool0-0     Epoch  21 | train=3.4323 [cls=1.5023 | reg=1.9300] | val=4.9967 | lr=6.80e-04 | img=224 | 
INFO    2026-02-23 21:58:39 -0500       workerpool0-0     Epoch  22 | train=3.3645 [cls=1.4556 | reg=1.9088] | val=4.8487 | lr=6.48e-04 | img=224 | ★ best
INFO    2026-02-23 21:58:51 -0500       workerpool0-0     Epoch  23 | train=3.4279 [cls=1.4826 | reg=1.9453] | val=4.9421 | lr=6.16e-04 | img=224 | 
INFO    2026-02-23 21:59:03 -0500       workerpool0-0     Epoch  24 | train=3.3588 [cls=1.4569 | reg=1.9019] | val=4.8766 | lr=5.83e-04 | img=224 | 
INFO    2026-02-23 21:59:15 -0500       workerpool0-0     Epoch  25 | train=3.3538 [cls=1.4398 | reg=1.9139] | val=4.7848 | lr=5.50e-04 | img=224 | ★ best
INFO    2026-02-23 21:59:27 -0500       workerpool0-0     Epoch  26 | train=3.3013 [cls=1.4366 | reg=1.8647] | val=4.7789 | lr=5.17e-04 | img=224 | ★ best
INFO    2026-02-23 21:59:40 -0500       workerpool0-0     Epoch  27 | train=3.2500 [cls=1.3985 | reg=1.8515] | val=4.7373 | lr=4.83e-04 | img=224 | ★ best
INFO    2026-02-23 21:59:52 -0500       workerpool0-0     Epoch  28 | train=3.3101 [cls=1.4308 | reg=1.8793] | val=4.7968 | lr=4.50e-04 | img=224 | 
INFO    2026-02-23 22:00:04 -0500       workerpool0-0     Epoch  29 | train=3.2257 [cls=1.3953 | reg=1.8304] | val=4.6963 | lr=4.17e-04 | img=224 | ★ best
INFO    2026-02-23 22:00:16 -0500       workerpool0-0     Epoch  30 | train=3.2491 [cls=1.4177 | reg=1.8314] | val=4.8082 | lr=3.84e-04 | img=224 | 
INFO    2026-02-23 22:00:28 -0500       workerpool0-0     Epoch  31 | train=3.1555 [cls=1.3418 | reg=1.8137] | val=4.6996 | lr=3.52e-04 | img=224 | 
INFO    2026-02-23 22:00:40 -0500       workerpool0-0     Epoch  32 | train=3.2125 [cls=1.3823 | reg=1.8303] | val=4.7509 | lr=3.20e-04 | img=224 | 
INFO    2026-02-23 22:00:52 -0500       workerpool0-0     Epoch  33 | train=3.1701 [cls=1.3572 | reg=1.8128] | val=4.7446 | lr=2.90e-04 | img=224 | 
INFO    2026-02-23 22:01:04 -0500       workerpool0-0     Epoch  34 | train=3.1246 [cls=1.3388 | reg=1.7857] | val=4.7343 | lr=2.60e-04 | img=224 | 
INFO    2026-02-23 22:01:16 -0500       workerpool0-0     Epoch  35 | train=3.0924 [cls=1.3189 | reg=1.7735] | val=4.7427 | lr=2.31e-04 | img=224 | 
INFO    2026-02-23 22:01:28 -0500       workerpool0-0     Epoch  36 | train=3.1456 [cls=1.3397 | reg=1.8059] | val=4.7074 | lr=2.03e-04 | img=224 | 
INFO    2026-02-23 22:01:40 -0500       workerpool0-0     Epoch  37 | train=3.1245 [cls=1.3288 | reg=1.7957] | val=4.7184 | lr=1.77e-04 | img=224 | 
INFO    2026-02-23 22:01:53 -0500       workerpool0-0     Epoch  38 | train=3.1671 [cls=1.3566 | reg=1.8105] | val=4.7211 | lr=1.52e-04 | img=224 | 
INFO    2026-02-23 22:02:05 -0500       workerpool0-0     Epoch  39 | train=3.1346 [cls=1.3472 | reg=1.7874] | val=4.6858 | lr=1.29e-04 | img=224 | ★ best
INFO    2026-02-23 22:02:17 -0500       workerpool0-0     Epoch  40 | train=3.1319 [cls=1.3317 | reg=1.8002] | val=4.7052 | lr=1.08e-04 | img=224 | 
INFO    2026-02-23 22:02:29 -0500       workerpool0-0     Epoch  41 | train=3.1231 [cls=1.3325 | reg=1.7906] | val=4.6788 | lr=8.79e-05 | img=224 | ★ best
INFO    2026-02-23 22:02:41 -0500       workerpool0-0     Epoch  42 | train=3.0789 [cls=1.3164 | reg=1.7625] | val=4.6666 | lr=6.99e-05 | img=224 | ★ best
INFO    2026-02-23 22:02:53 -0500       workerpool0-0     Epoch  43 | train=3.1100 [cls=1.3330 | reg=1.7770] | val=4.6526 | lr=5.38e-05 | img=224 | ★ best
INFO    2026-02-23 22:03:05 -0500       workerpool0-0     Epoch  44 | train=3.1029 [cls=1.3263 | reg=1.7766] | val=4.6898 | lr=3.98e-05 | img=224 | 
INFO    2026-02-23 22:03:17 -0500       workerpool0-0     Epoch  45 | train=3.1267 [cls=1.3406 | reg=1.7862] | val=4.6727 | lr=2.78e-05 | img=224 | 
INFO    2026-02-23 22:03:29 -0500       workerpool0-0     Epoch  46 | train=3.0702 [cls=1.3082 | reg=1.7620] | val=4.6792 | lr=1.79e-05 | img=224 | 
INFO    2026-02-23 22:03:41 -0500       workerpool0-0     Epoch  47 | train=3.0696 [cls=1.3105 | reg=1.7591] | val=4.6997 | lr=1.01e-05 | img=224 | 
INFO    2026-02-23 22:03:53 -0500       workerpool0-0     Epoch  48 | train=3.0986 [cls=1.3136 | reg=1.7849] | val=4.6755 | lr=4.56e-06 | img=224 | 
INFO    2026-02-23 22:04:05 -0500       workerpool0-0     Epoch  49 | train=3.0808 [cls=1.3188 | reg=1.7620] | val=4.6951 | lr=1.22e-06 | img=224 | 
INFO    2026-02-23 22:04:05 -0500       workerpool0-0   ✅ Phase 1 (backbone frozen) completada en 10.7 min
INFO    2026-02-23 22:04:06 -0500       workerpool0-0   🔄 Mejor checkpoint de Phase 1 recargado
INFO    2026-02-23 22:04:06 -0500       workerpool0-0   🔓 Todas las capas desbloqueadas: 213,440 params unfrozen
INFO    2026-02-23 22:04:06 -0500       workerpool0-0      Total trainable: 361,563
INFO    2026-02-23 22:04:06 -0500       workerpool0-0   🚀 Phase 2 (full fine-tuning) — 100 epochs, LR=0.0001
INFO    2026-02-23 22:04:06 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO    2026-02-23 22:04:06 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=100)
INFO    2026-02-23 22:04:19 -0500       workerpool0-0     Epoch  50 | train=3.0820 [cls=1.3093 | reg=1.7727] | val=4.6322 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 22:04:31 -0500       workerpool0-0     Epoch  51 | train=2.9695 [cls=1.2442 | reg=1.7253] | val=4.5600 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 22:04:44 -0500       workerpool0-0     Epoch  52 | train=2.9776 [cls=1.2556 | reg=1.7220] | val=4.5718 | lr=1.00e-04 | img=224 | 
INFO    2026-02-23 22:04:57 -0500       workerpool0-0     Epoch  53 | train=2.9032 [cls=1.2064 | reg=1.6968] | val=4.5317 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 22:05:10 -0500       workerpool0-0     Epoch  54 | train=2.8520 [cls=1.1743 | reg=1.6777] | val=4.5379 | lr=1.00e-04 | img=224 | 
INFO    2026-02-23 22:05:23 -0500       workerpool0-0     Epoch  55 | train=2.8882 [cls=1.1916 | reg=1.6967] | val=4.5267 | lr=9.99e-05 | img=224 | ★ best
INFO    2026-02-23 22:05:36 -0500       workerpool0-0     Epoch  56 | train=2.7998 [cls=1.1480 | reg=1.6519] | val=4.5106 | lr=9.98e-05 | img=224 | ★ best
INFO    2026-02-23 22:05:49 -0500       workerpool0-0     Epoch  57 | train=2.8181 [cls=1.1425 | reg=1.6755] | val=4.5071 | lr=9.96e-05 | img=224 | ★ best
INFO    2026-02-23 22:06:02 -0500       workerpool0-0     Epoch  58 | train=2.8350 [cls=1.1484 | reg=1.6866] | val=4.4723 | lr=9.94e-05 | img=224 | ★ best
INFO    2026-02-23 22:06:15 -0500       workerpool0-0     Epoch  59 | train=2.7245 [cls=1.0822 | reg=1.6423] | val=4.4909 | lr=9.91e-05 | img=224 | 
INFO    2026-02-23 22:06:28 -0500       workerpool0-0     Epoch  60 | train=2.7028 [cls=1.0783 | reg=1.6245] | val=4.5110 | lr=9.88e-05 | img=224 | 
INFO    2026-02-23 22:06:41 -0500       workerpool0-0     Epoch  61 | train=2.6963 [cls=1.0637 | reg=1.6326] | val=4.5246 | lr=9.84e-05 | img=224 | 
INFO    2026-02-23 22:06:53 -0500       workerpool0-0     Epoch  62 | train=2.6833 [cls=1.0639 | reg=1.6194] | val=4.5636 | lr=9.80e-05 | img=224 | 
INFO    2026-02-23 22:07:06 -0500       workerpool0-0     Epoch  63 | train=2.6460 [cls=1.0384 | reg=1.6075] | val=4.4995 | lr=9.76e-05 | img=224 | 
INFO    2026-02-23 22:07:19 -0500       workerpool0-0     Epoch  64 | train=2.6283 [cls=1.0186 | reg=1.6097] | val=4.4932 | lr=9.70e-05 | img=224 | 
INFO    2026-02-23 22:07:32 -0500       workerpool0-0     Epoch  65 | train=2.5997 [cls=1.0126 | reg=1.5872] | val=4.4661 | lr=9.65e-05 | img=224 | ★ best
INFO    2026-02-23 22:07:45 -0500       workerpool0-0     Epoch  66 | train=2.5846 [cls=1.0033 | reg=1.5814] | val=4.4594 | lr=9.59e-05 | img=224 | ★ best
INFO    2026-02-23 22:07:58 -0500       workerpool0-0     Epoch  67 | train=2.5731 [cls=0.9923 | reg=1.5808] | val=4.4680 | lr=9.52e-05 | img=224 | 
INFO    2026-02-23 22:08:11 -0500       workerpool0-0     Epoch  68 | train=2.5610 [cls=0.9861 | reg=1.5750] | val=4.4477 | lr=9.46e-05 | img=224 | ★ best
INFO    2026-02-23 22:08:24 -0500       workerpool0-0     Epoch  69 | train=2.5395 [cls=0.9682 | reg=1.5713] | val=4.5211 | lr=9.38e-05 | img=224 | 
INFO    2026-02-23 22:08:37 -0500       workerpool0-0     Epoch  70 | train=2.5285 [cls=0.9677 | reg=1.5609] | val=4.4907 | lr=9.30e-05 | img=224 | 
INFO    2026-02-23 22:08:50 -0500       workerpool0-0     Epoch  71 | train=2.4655 [cls=0.9297 | reg=1.5358] | val=4.4718 | lr=9.22e-05 | img=224 | 
INFO    2026-02-23 22:09:03 -0500       workerpool0-0     Epoch  72 | train=2.5854 [cls=1.0072 | reg=1.5782] | val=4.4690 | lr=9.14e-05 | img=224 | 
INFO    2026-02-23 22:09:15 -0500       workerpool0-0     Epoch  73 | train=2.4463 [cls=0.9256 | reg=1.5207] | val=4.4824 | lr=9.05e-05 | img=224 | 
INFO    2026-02-23 22:09:28 -0500       workerpool0-0     Epoch  74 | train=2.5212 [cls=0.9580 | reg=1.5632] | val=4.4820 | lr=8.95e-05 | img=224 | 
INFO    2026-02-23 22:09:41 -0500       workerpool0-0     Epoch  75 | train=2.4179 [cls=0.8948 | reg=1.5231] | val=4.4972 | lr=8.85e-05 | img=224 | 
INFO    2026-02-23 22:09:54 -0500       workerpool0-0     Epoch  76 | train=2.4272 [cls=0.9015 | reg=1.5257] | val=4.4904 | lr=8.75e-05 | img=224 | 
INFO    2026-02-23 22:10:07 -0500       workerpool0-0     Epoch  77 | train=2.4043 [cls=0.8969 | reg=1.5074] | val=4.4545 | lr=8.65e-05 | img=224 | 
INFO    2026-02-23 22:10:19 -0500       workerpool0-0     Epoch  78 | train=2.4235 [cls=0.9101 | reg=1.5134] | val=4.5293 | lr=8.54e-05 | img=224 | 
INFO    2026-02-23 22:10:32 -0500       workerpool0-0     Epoch  79 | train=2.3469 [cls=0.8637 | reg=1.4832] | val=4.4550 | lr=8.42e-05 | img=224 | 
INFO    2026-02-23 22:10:45 -0500       workerpool0-0     Epoch  80 | train=2.3408 [cls=0.8594 | reg=1.4815] | val=4.5492 | lr=8.31e-05 | img=224 | 
INFO    2026-02-23 22:10:58 -0500       workerpool0-0     Epoch  81 | train=2.3958 [cls=0.8826 | reg=1.5132] | val=4.4923 | lr=8.19e-05 | img=224 | 
INFO    2026-02-23 22:11:11 -0500       workerpool0-0     Epoch  82 | train=2.3560 [cls=0.8763 | reg=1.4797] | val=4.5585 | lr=8.07e-05 | img=224 | 
INFO    2026-02-23 22:11:24 -0500       workerpool0-0     Epoch  83 | train=2.2939 [cls=0.8374 | reg=1.4566] | val=4.5254 | lr=7.94e-05 | img=224 | 
INFO    2026-02-23 22:11:37 -0500       workerpool0-0     Epoch  84 | train=2.2707 [cls=0.8233 | reg=1.4475] | val=4.5069 | lr=7.81e-05 | img=224 | 
INFO    2026-02-23 22:11:49 -0500       workerpool0-0     Epoch  85 | train=2.3144 [cls=0.8425 | reg=1.4719] | val=4.5437 | lr=7.68e-05 | img=224 | 
INFO    2026-02-23 22:12:03 -0500       workerpool0-0     Epoch  86 | train=2.3204 [cls=0.8509 | reg=1.4695] | val=4.5410 | lr=7.55e-05 | img=224 | 
INFO    2026-02-23 22:12:16 -0500       workerpool0-0     Epoch  87 | train=2.3189 [cls=0.8461 | reg=1.4728] | val=4.4845 | lr=7.41e-05 | img=224 | 
INFO    2026-02-23 22:12:28 -0500       workerpool0-0     Epoch  88 | train=2.2924 [cls=0.8376 | reg=1.4548] | val=4.5756 | lr=7.27e-05 | img=224 | 
INFO    2026-02-23 22:12:41 -0500       workerpool0-0     Epoch  89 | train=2.2623 [cls=0.8139 | reg=1.4484] | val=4.5773 | lr=7.13e-05 | img=224 | 
INFO    2026-02-23 22:12:54 -0500       workerpool0-0     Epoch  90 | train=2.2227 [cls=0.7952 | reg=1.4276] | val=4.5259 | lr=6.99e-05 | img=224 | 
INFO    2026-02-23 22:13:07 -0500       workerpool0-0     Epoch  91 | train=2.2436 [cls=0.8126 | reg=1.4310] | val=4.5576 | lr=6.84e-05 | img=224 | 
INFO    2026-02-23 22:13:20 -0500       workerpool0-0     Epoch  92 | train=2.1991 [cls=0.7860 | reg=1.4130] | val=4.5113 | lr=6.70e-05 | img=224 | 
INFO    2026-02-23 22:13:33 -0500       workerpool0-0     Epoch  93 | train=2.2709 [cls=0.8247 | reg=1.4463] | val=4.5894 | lr=6.55e-05 | img=224 | 
INFO    2026-02-23 22:13:33 -0500       workerpool0-0   ⏹️  Early stopping at epoch 93 (patience=25)
INFO    2026-02-23 22:13:33 -0500       workerpool0-0   ✅ Phase 2 (full fine-tuning) completada en 9.5 min
INFO    2026-02-23 22:13:33 -0500       workerpool0-0   📊 Entrenamiento completo: 94 epochs
INFO    2026-02-23 22:13:33 -0500       workerpool0-0      Mejor val_loss: 4.4477 (epoch 68)
INFO    2026-02-23 22:13:33 -0500       workerpool0-0      Tiempo total: 20.2 min
INFO    2026-02-23 22:13:33 -0500       workerpool0-0   ⏱️  Entrenamiento completado en 20.3 min
INFO    2026-02-23 22:13:34 -0500       workerpool0-0   💾 Historial guardado: /tmp/training/training_history.csv
INFO    2026-02-23 22:13:34 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:34 -0500       workerpool0-0   BLOQUE 5 — Curvas de Entrenamiento
INFO    2026-02-23 22:13:34 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   📊 Curvas guardadas: /tmp/training/training_curves.png
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   📈 Resumen – PYTORCH 
INFO    2026-02-23 22:13:36 -0500       workerpool0-0     Épocas: 94
INFO    2026-02-23 22:13:36 -0500       workerpool0-0     Mejor val_loss: 4.4477 (epoch 68)
INFO    2026-02-23 22:13:36 -0500       workerpool0-0     Resoluciones: [224]
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   BLOQUE 6 — Evaluación en Validación
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:36 -0500       workerpool0-0   ✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_espdet.pt
INFO    2026-02-23 22:13:39 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=val
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     mAP@50:    0.4184
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     mAP@50-95: 0.1825
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     Precision: 0.2986
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     Recall:    0.5358
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     F1-Score:  0.3835
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     Imágenes: 188 | Detecciones: 1447 | GT: 762
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     Avg inference: 12.2 ms
INFO    2026-02-23 22:13:39 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 22:13:39 -0500       workerpool0-0       dog                   0.3942
INFO    2026-02-23 22:13:39 -0500       workerpool0-0       door                  0.4576
INFO    2026-02-23 22:13:39 -0500       workerpool0-0       obstacle              0.3235
INFO    2026-02-23 22:13:39 -0500       workerpool0-0       person                0.4596
INFO    2026-02-23 22:13:39 -0500       workerpool0-0       stair                 0.4573
INFO    2026-02-23 22:13:39 -0500       workerpool0-0   📊 Val mAP@50: 0.4184
INFO    2026-02-23 22:13:39 -0500       workerpool0-0      dog: 0.3942
INFO    2026-02-23 22:13:39 -0500       workerpool0-0      door: 0.4576
INFO    2026-02-23 22:13:39 -0500       workerpool0-0      obstacle: 0.3235
INFO    2026-02-23 22:13:39 -0500       workerpool0-0      person: 0.4596
INFO    2026-02-23 22:13:39 -0500       workerpool0-0      stair: 0.4573
INFO    2026-02-23 22:13:39 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO    2026-02-23 22:13:40 -0500       workerpool0-0   📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO    2026-02-23 22:13:40 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO    2026-02-23 22:13:40 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:40 -0500       workerpool0-0   BLOQUE 7 — Evaluación en Test
INFO    2026-02-23 22:13:40 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:41 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=test
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     mAP@50:    0.6052
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     mAP@50-95: 0.2701
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     Precision: 0.3298
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     Recall:    0.7278
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     F1-Score:  0.4539
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     Imágenes: 187 | Detecciones: 1296 | GT: 576
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     Avg inference: 5.0 ms
INFO    2026-02-23 22:13:41 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 22:13:41 -0500       workerpool0-0       dog                   0.5305
INFO    2026-02-23 22:13:41 -0500       workerpool0-0       door                  0.5572
INFO    2026-02-23 22:13:41 -0500       workerpool0-0       obstacle              0.4618
INFO    2026-02-23 22:13:41 -0500       workerpool0-0       person                0.7059
INFO    2026-02-23 22:13:41 -0500       workerpool0-0       stair                 0.7708
INFO    2026-02-23 22:13:41 -0500       workerpool0-0   📊 Test mAP@50: 0.6052
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   BLOQUE 8 — Guardado y subida a GCS
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 22:13:42 -0500       workerpool0-0   📦 Exportando ESPDet → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=6)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0     ℹ️ onnxsim not installed, skipping simplification
INFO    2026-02-23 22:13:43 -0500       workerpool0-0     ✅ Exportado: /tmp/training/export/espdet_pico.onnx (1.41 MB, 1.4s)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0     ✅ ONNX verificado: /tmp/training/export/espdet_pico.onnx
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        Latencia mediana: 3.8ms
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        box0: (1, 4, 28, 28)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        score0: (1, 5, 28, 28)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        box1: (1, 4, 14, 14)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        score1: (1, 5, 14, 14)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        box2: (1, 4, 7, 7)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0        score2: (1, 5, 7, 7)
INFO    2026-02-23 22:13:43 -0500       workerpool0-0   💾 Experimento guardado: /tmp/training/experiment.json
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/config.yaml
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/training_history.csv
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/training_curves.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/class_distribution.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/gt_samples.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/val_confusion_matrix.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/val_per_class.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/val_evaluation.json
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/test_confusion_matrix.png
INFO    2026-02-23 22:13:44 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/test_evaluation.json
INFO    2026-02-23 22:13:45 -0500       workerpool0-0     ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/experiment.json
INFO    2026-02-23 22:13:45 -0500       workerpool0-0     ☁️  Subido: /tmp/training/checkpoints/best_espdet.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/checkpoints/best_espdet.pt
INFO    2026-02-23 22:13:45 -0500       workerpool0-0     ☁️  Subido: /tmp/training/export/espdet_pico.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v4-t4/export/espdet_pico.onnx
INFO    2026-02-23 22:13:45 -0500       workerpool0-0   ✅ Pipeline ESPDet-Pico completado exitosamente.
INFO    2026-02-23 22:13:47 -0500       workerpool0-0   Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO    2026-02-23 22:14:03 -0500       service Tearing down training program.
INFO    2026-02-23 22:14:48 -0500       service Finished tearing down training program.
INFO    2026-02-23 22:14:48 -0500       service Job completed successfully.

```

## Logs de Terminal de Lanzamiento:
```zsh
% ./vertex_ai/build_and_launch.sh espdet_pico_v4 --run-name espdet-pico-v4-t4
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.6.3' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.6.3.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.6.3.tar.gz [Content-Type=application/x-tar]...
/ [1 files][ 76.0 KiB/ 76.0 KiB]                                                
Operation completed over 1 objects/76.0 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz

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
  Run:           espdet-pico-v4-t4
  Familia:       ESPDet
  Módulo:        trainer.task_espdet
  Contenedor:    us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
  Máquina:       n1-standard-8
  GPU:           NVIDIA_TESLA_T4 x1
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.3.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet-pico-v4-t4']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v4-t4.yaml

🚀 Lanzando Custom Job: tfm-espdet_pico_v4-1771901256
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-23-21:47:38.328 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/7265276981816590336?project=608533675308
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/5636733135312912384?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob run completed. Resource name: projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336
Training did not produce a Managed Model returning None. Training Pipeline projects/608533675308/locations/us-central1/trainingPipelines/7265276981816590336 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR'].

============================================================
✅ Custom Job completado exitosamente
   Resultados en: gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v4
   Experiments:   tfm-deteccion-objetos / espdet-pico-v4-t4
============================================================

═══════════════════════════════════════════════════════════
✅ Proceso completado
═══════════════════════════════════════════════════════════
```