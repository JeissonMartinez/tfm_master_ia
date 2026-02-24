# Registro de Entrenamiento - ESPDet-Pico (v2 - Espressif Official)
**Corrida Exitosa #:** 3 (exitoso)

## Log de Monitor de Entrenamiento:
```zsh
% gcloud ai custom-jobs stream-logs 2124347638428991488 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO    2026-02-23 20:27:52 -0500       service Waiting for job to be provisioned.
INFO    2026-02-23 20:27:52 -0500       service Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO    2026-02-23 20:27:52 -0500       service Vertex AI is setting up this job.
INFO    2026-02-23 20:27:52 -0500       service Waiting for training program to start.
INFO    2026-02-23 20:27:52 -0500       service Job is preparing.
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-1v0c5ns3-a2237120-fwpp
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-e5b4592e5d-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet-pico-v3-t3"]}
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:30:46 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz tfm_trainer-2.6.2.tar.gz
ERROR   2026-02-23 20:30:47 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 20:30:47 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 20:30:50 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 20:30:50 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 20:30:54 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:30:54 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:30:56 -0500       service Job is running.
INFO    2026-02-23 20:30:57 -0500       workerpool0-0   Processing /tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:30:57 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 20:31:02 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 20:31:02 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 20:31:02 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 20:31:02 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 20:31:03 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 20:31:03 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 20:31:03 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 20:31:03 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 20:31:03 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.2-py3-none-any.whl size=91294 sha256=723905576a3f377e4b10e6038f7667b1e910dead58c04f7c547a5a6391db6f3b
INFO    2026-02-23 20:31:03 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/c5/de/94/8c369de60fec08fa46f7247e0e661abc80fb2d433fa958f70f
INFO    2026-02-23 20:31:03 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 20:31:03 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 20:31:03 -0500       workerpool0-0   Successfully installed tfm-trainer-2.6.2
ERROR   2026-02-23 20:31:03 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 20:31:03 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 20:31:03 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 20:31:04 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:31:05 -0500       workerpool0-0   Processing /tfm_trainer-2.6.2.tar.gz
INFO    2026-02-23 20:31:05 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 20:31:08 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 20:31:08 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 20:31:09 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 20:31:09 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 20:31:09 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 20:31:09 -0500       workerpool0-0   Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (6.0.2)
INFO    2026-02-23 20:31:10 -0500       workerpool0-0   Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:10 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 20:31:10 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (2.2.3)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (1.6.1)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0   Collecting albumentations>=2.0.0 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:11 -0500       workerpool0-0     Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Collecting ultralytics>=8.2 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0     Downloading ultralytics-8.4.15-py3-none-any.whl.metadata (39 kB)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (0.19.0+cu124)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (2.19.0)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.2) (1.91.0)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0     Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.6.2) (1.11.4)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:12 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 20:31:14 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:14 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.34.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.40.3)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.26.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (3.20.3)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (25.0)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (3.34.0)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.14.2)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.1.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (4.14.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (0.16)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.70.0)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.32.4)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.73.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.48.2)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (5.5.2)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (0.4.2)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (4.9.1)
INFO    2026-02-23 20:31:15 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:16 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 20:31:16 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.4.3)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.7.2)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2.9.0.post0)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (0.14.2)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:17 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 20:31:17 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 20:31:18 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 20:31:18 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.6.2) (1.7.1)
INFO    2026-02-23 20:31:18 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:18 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 20:31:19 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:19 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.17.0)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (3.4.2)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (3.10)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (1.26.20)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (2025.6.15)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.2) (0.6.1)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (1.3.2)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (0.12.1)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (4.58.5)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (1.4.8)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (11.3.0)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.2) (3.2.3)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:20 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 20:31:20 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.6.2) (1.14.0)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.2) (2025.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.2) (2025.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.2) (1.5.1)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.2) (3.6.0)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.6.2) (2.4.0+cu124)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (3.18.0)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (3.4.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (3.1.6)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (2025.5.1)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (9.1.0.70)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.2.65)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (11.2.0.44)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (10.3.5.119)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (11.6.0.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.3.0.142)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (2.20.5)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (12.4.99)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (3.0.0)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0     Downloading opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:21 -0500       workerpool0-0   Requirement already satisfied: psutil>=5.8.0 in /opt/python/3.10/lib/python3.10/site-packages (from ultralytics>=8.2->tfm-trainer==2.6.2) (5.9.3)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Collecting polars>=0.20.0 (from ultralytics>=8.2->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading polars-1.38.1-py3-none-any.whl.metadata (10 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Collecting ultralytics-thop>=2.0.18 (from ultralytics>=8.2->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading opencv_python-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Collecting polars-runtime-32==1.38.1 (from polars>=0.20.0->ultralytics>=8.2->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.6.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.2) (3.0.2)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.6.2) (1.3.0)
INFO    2026-02-23 20:31:22 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 79.5 MB/s eta 0:00:00
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 59.8 MB/s eta 0:00:00
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 63.9 MB/s eta 0:00:00
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 114.5 MB/s eta 0:00:00
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 20:31:23 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 93.5 MB/s eta 0:00:00
INFO    2026-02-23 20:31:23 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 20:31:24 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 88.6 MB/s eta 0:00:00
INFO    2026-02-23 20:31:24 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 20:31:24 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 29.0 MB/s eta 0:00:00
INFO    2026-02-23 20:31:24 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 20:31:24 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 91.5 MB/s eta 0:00:00
INFO    2026-02-23 20:31:24 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 20:31:24 -0500       workerpool0-0   Downloading ultralytics-8.4.15-py3-none-any.whl (1.2 MB)
INFO    2026-02-23 20:31:24 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 61.2 MB/s eta 0:00:00
INFO    2026-02-23 20:31:24 -0500       workerpool0-0   Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 MB 102.9 MB/s eta 0:00:00
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading polars-1.38.1-py3-none-any.whl (810 kB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 810.4/810.4 kB 26.2 MB/s eta 0:00:00
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (45.8 MB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.8/45.8 MB 103.3 MB/s eta 0:00:00
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 20:31:25 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 20:31:27 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 20:31:27 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 20:31:27 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 20:31:27 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.2-py3-none-any.whl size=91294 sha256=7945ebf38784b20533ea81a06e8f5a6bfe13825db3fb6d394ae5b59888b6913e
INFO    2026-02-23 20:31:27 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/c5/de/94/8c369de60fec08fa46f7247e0e661abc80fb2d433fa958f70f
INFO    2026-02-23 20:31:27 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 20:31:28 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, polars-runtime-32, numpy, humanfriendly, annotated-types, pydantic, polars, opencv-python-headless, opencv-python, onnx, coloredlogs, onnxruntime, matplotlib, albucore, ultralytics-thop, albumentations, ultralytics, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 20:31:29 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:29 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:32 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:32 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:33 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:33 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:40 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:40 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:40 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:40 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:42 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:42 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 20:31:44 -0500       workerpool0-0     WARNING: The scripts ultralytics and yolo are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 20:31:44 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 20:31:45 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 20:31:45 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.6.2
INFO    2026-02-23 20:31:45 -0500       workerpool0-0       Uninstalling tfm-trainer-2.6.2:
INFO    2026-02-23 20:31:45 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.6.2
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
INFO    2026-02-23 20:31:45 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-4.11.0.86 opencv-python-headless-4.11.0.86 polars-1.38.1 polars-runtime-32-1.38.1 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.6.2 typing-inspection-0.4.2 ultralytics-8.4.15 ultralytics-thop-2.0.18
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 20:31:45 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 20:31:46 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet-pico-v3-t3
INFO    2026-02-23 20:31:49 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:31:49 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 20:31:49 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml → /tmp/training/config.yaml (0.0 MB)
INFO    2026-02-23 20:32:15 -0500       workerpool0-0   🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Nombre:       espdet_pico_v3
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Familia:      ESPDet
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Variante:     espdet_pico
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Versión:      v3
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Descripción:  ESPDet-Pico + Focal Loss (γ=2.0, α=0.25) para mejorar Precision
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Dataset:      iodc_yolo
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Img Size:     224×224
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Batch Size:   32
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Patience:     25
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Seed:         42
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     Conf Thresh:  0.25
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     IoU Thresh:   0.45
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     📐 2-Phase Training:
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        Phase 1: 50 epochs @ LR=0.001
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        Phase 2: 100 epochs @ LR=0.0001
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        Resize Schedule: {0: 224}
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        Optimizer: AdamW | WD: 0.0005
INFO    2026-02-23 20:32:15 -0500       workerpool0-0     🟢 ESPDet Config:
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        Pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 20:32:15 -0500       workerpool0-0        reg_max: 1
INFO    2026-02-23 20:32:15 -0500       workerpool0-0   ✅ Configuración aplicada correctamente
INFO    2026-02-23 20:32:15 -0500       workerpool0-0   📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO    2026-02-23 20:32:16 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO    2026-02-23 20:32:16 -0500       workerpool0-0     📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     ✅ Descomprimido correctamente
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   🖥️  Device: cuda
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   ⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO    2026-02-23 20:32:18 -0500       workerpool0-0      Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   domain: "googleapis.com"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   metadata {
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     key: "method"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   }
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   metadata {
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     key: "service"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     value: "aiplatform.googleapis.com"
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   }
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   ]
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   BLOQUE 2 — Verificación del Dataset
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   📂 Dataset YOLO: iodc_yolo
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO    2026-02-23 20:32:18 -0500       workerpool0-0     ✅  test:   187 imgs |   187 labels | 0 sin label
INFO    2026-02-23 20:32:18 -0500       workerpool0-0   📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO    2026-02-23 20:32:19 -0500       workerpool0-0   generated new fontManager
INFO    2026-02-23 20:32:20 -0500       workerpool0-0     📊 Guardado: /tmp/training/class_distribution.png
INFO    2026-02-23 20:32:20 -0500       workerpool0-0   ⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO    2026-02-23 20:32:22 -0500       workerpool0-0     🖼️  Guardado: /tmp/training/gt_samples.png
INFO    2026-02-23 20:32:22 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:22 -0500       workerpool0-0   BLOQUE 3 — Construcción del Modelo ESPDet-Pico
INFO    2026-02-23 20:32:22 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:22 -0500       workerpool0-0   ⬇️  Descargando pesos pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 20:32:22 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt → /tmp/training/pretrained_weights.pt (1.0 MB)
INFO    2026-02-23 20:32:23 -0500       workerpool0-0   Creating new Ultralytics Settings v0.0.6 file ✅ 
INFO    2026-02-23 20:32:23 -0500       workerpool0-0   View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
INFO    2026-02-23 20:32:23 -0500       workerpool0-0   Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
INFO    2026-02-23 20:32:23 -0500       workerpool0-0   🔄 Cargando pesos pretrained: /tmp/training/pretrained_weights.pt
INFO    2026-02-23 20:32:23 -0500       workerpool0-0     ✅ Transfer learning: 622 param groups cargados
INFO    2026-02-23 20:32:23 -0500       workerpool0-0     ℹ️  Shape mismatch (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 20:32:23 -0500       workerpool0-0     ℹ️  Missing keys (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 20:32:23 -0500       workerpool0-0   ✅ ESPDet-Pico (oficial) construido: 361,563 params (361,563 trainable)
INFO    2026-02-23 20:32:23 -0500       workerpool0-0      Strides: [8, 16, 32] | Classes: 5
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 20:32:24 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   📦 Modelo: ESPDet-Pico
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Total params:          361,563
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Trainable:             148,123
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Non-trainable:         213,440
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Est. float32:           1.38 MB
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Est. INT8:              0.34 MB
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   📐 Tamaño estimado: 1.38 MB (FP32), 0.34 MB (INT8)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.6.2 (Focal Loss)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Architecture:    Official Espressif (esp-detection repo)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Strides:         [8, 16, 32]
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     pretrained:      gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Phase 1:         50 ep, LR=0.001, WD=0.0001
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Phase 2:         100 ep, LR=0.0001, WD=1e-05
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Optimizer:       adamw
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     cls_weight:      1.0
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     reg_weight:      2.0
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Focal Loss:      ON (γ=2.0, α=0.25)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Conf threshold:  0.25
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     IoU threshold:   0.45
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     AMP:             True
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Grad clip:       5.0
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Export imgsz:    224
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Batch size:      32
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Patience:        25
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', 'aug_hue_shift_limit', 'aug_sat_shift_limit', 'aug_val_shift_limit', 'aug_shift_limit', 'aug_scale_limit', 'aug_rotate_limit', 'aug_hflip_prob', 'aug_gaussian_noise']
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   BLOQUE 4 — Entrenamiento (2 fases)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🎯 ESPDet cls_loss: Sigmoid Focal Loss (γ=2.0, α=0.25)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🔧 Two-Phase Training Config
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Phase 1: 50 epochs | LR=0.001 | WD=0.0001
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Phase 2: 100 epochs | LR=0.0001 | WD=1e-05
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Optimizer: adamw | Scheduler: cosine
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Batch: 32 | AMP: True | Patience: 25
INFO    2026-02-23 20:32:24 -0500       workerpool0-0     Resize schedule: [(0, 224)]
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 20:32:24 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 20:32:24 -0500       workerpool0-0   🚀 Phase 1 (backbone frozen) — 50 epochs, LR=0.001
INFO    2026-02-23 20:32:31 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO    2026-02-23 20:32:31 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=47)
INFO    2026-02-23 20:32:53 -0500       workerpool0-0     Epoch   0 | train=6.4625 [cls=2.6649 | reg=3.7976] | val=5.6537 | lr=3.33e-04 | img=224 | ★ best
INFO    2026-02-23 20:33:08 -0500       workerpool0-0     Epoch   1 | train=4.2547 [cls=0.9875 | reg=3.2673] | val=4.0623 | lr=6.67e-04 | img=224 | ★ best
INFO    2026-02-23 20:33:24 -0500       workerpool0-0     Epoch   2 | train=3.2962 [cls=0.4008 | reg=2.8954] | val=3.6521 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 20:33:39 -0500       workerpool0-0     Epoch   3 | train=2.9056 [cls=0.3050 | reg=2.6006] | val=3.5499 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 20:33:55 -0500       workerpool0-0     Epoch   4 | train=2.7442 [cls=0.2737 | reg=2.4705] | val=3.3143 | lr=9.99e-04 | img=224 | ★ best
INFO    2026-02-23 20:34:09 -0500       workerpool0-0     Epoch   5 | train=2.6093 [cls=0.2582 | reg=2.3511] | val=3.2201 | lr=9.96e-04 | img=224 | ★ best
INFO    2026-02-23 20:34:23 -0500       workerpool0-0     Epoch   6 | train=2.5123 [cls=0.2472 | reg=2.2651] | val=3.3301 | lr=9.90e-04 | img=224 | 
INFO    2026-02-23 20:34:36 -0500       workerpool0-0     Epoch   7 | train=2.4384 [cls=0.2398 | reg=2.1986] | val=3.1471 | lr=9.82e-04 | img=224 | ★ best
INFO    2026-02-23 20:34:49 -0500       workerpool0-0     Epoch   8 | train=2.3500 [cls=0.2295 | reg=2.1205] | val=3.2027 | lr=9.72e-04 | img=224 | 
INFO    2026-02-23 20:35:01 -0500       workerpool0-0     Epoch   9 | train=2.3364 [cls=0.2260 | reg=2.1104] | val=3.1226 | lr=9.60e-04 | img=224 | ★ best
INFO    2026-02-23 20:35:12 -0500       workerpool0-0     Epoch  10 | train=2.2903 [cls=0.2240 | reg=2.0663] | val=3.1283 | lr=9.46e-04 | img=224 | 
INFO    2026-02-23 20:35:23 -0500       workerpool0-0     Epoch  11 | train=2.2600 [cls=0.2193 | reg=2.0407] | val=3.0245 | lr=9.30e-04 | img=224 | ★ best
INFO    2026-02-23 20:35:35 -0500       workerpool0-0     Epoch  12 | train=2.2398 [cls=0.2156 | reg=2.0242] | val=3.1093 | lr=9.12e-04 | img=224 | 
INFO    2026-02-23 20:35:46 -0500       workerpool0-0     Epoch  13 | train=2.2334 [cls=0.2179 | reg=2.0155] | val=3.1314 | lr=8.92e-04 | img=224 | 
INFO    2026-02-23 20:35:57 -0500       workerpool0-0     Epoch  14 | train=2.1922 [cls=0.2121 | reg=1.9801] | val=2.9407 | lr=8.71e-04 | img=224 | ★ best
INFO    2026-02-23 20:36:09 -0500       workerpool0-0     Epoch  15 | train=2.1511 [cls=0.2099 | reg=1.9412] | val=3.0586 | lr=8.48e-04 | img=224 | 
INFO    2026-02-23 20:36:20 -0500       workerpool0-0     Epoch  16 | train=2.1290 [cls=0.2074 | reg=1.9216] | val=2.9519 | lr=8.23e-04 | img=224 | 
INFO    2026-02-23 20:36:32 -0500       workerpool0-0     Epoch  17 | train=2.1190 [cls=0.2059 | reg=1.9131] | val=2.9190 | lr=7.97e-04 | img=224 | ★ best
INFO    2026-02-23 20:36:43 -0500       workerpool0-0     Epoch  18 | train=2.1073 [cls=0.2049 | reg=1.9024] | val=2.9442 | lr=7.69e-04 | img=224 | 
INFO    2026-02-23 20:36:55 -0500       workerpool0-0     Epoch  19 | train=2.1081 [cls=0.2028 | reg=1.9052] | val=2.9816 | lr=7.40e-04 | img=224 | 
INFO    2026-02-23 20:37:06 -0500       workerpool0-0     Epoch  20 | train=2.0675 [cls=0.2002 | reg=1.8673] | val=3.0007 | lr=7.10e-04 | img=224 | 
INFO    2026-02-23 20:37:18 -0500       workerpool0-0     Epoch  21 | train=2.0783 [cls=0.2020 | reg=1.8763] | val=2.8866 | lr=6.80e-04 | img=224 | ★ best
INFO    2026-02-23 20:37:29 -0500       workerpool0-0     Epoch  22 | train=2.0668 [cls=0.2011 | reg=1.8657] | val=2.8874 | lr=6.48e-04 | img=224 | 
INFO    2026-02-23 20:37:41 -0500       workerpool0-0     Epoch  23 | train=2.0588 [cls=0.2007 | reg=1.8581] | val=2.9102 | lr=6.16e-04 | img=224 | 
INFO    2026-02-23 20:37:52 -0500       workerpool0-0     Epoch  24 | train=2.0282 [cls=0.1967 | reg=1.8314] | val=2.9577 | lr=5.83e-04 | img=224 | 
INFO    2026-02-23 20:38:04 -0500       workerpool0-0     Epoch  25 | train=2.0125 [cls=0.1955 | reg=1.8170] | val=2.8224 | lr=5.50e-04 | img=224 | ★ best
INFO    2026-02-23 20:38:15 -0500       workerpool0-0     Epoch  26 | train=2.0159 [cls=0.1949 | reg=1.8210] | val=2.8650 | lr=5.17e-04 | img=224 | 
INFO    2026-02-23 20:38:26 -0500       workerpool0-0     Epoch  27 | train=1.9900 [cls=0.1919 | reg=1.7981] | val=2.8820 | lr=4.83e-04 | img=224 | 
INFO    2026-02-23 20:38:38 -0500       workerpool0-0     Epoch  28 | train=1.9769 [cls=0.1921 | reg=1.7848] | val=2.7980 | lr=4.50e-04 | img=224 | ★ best
INFO    2026-02-23 20:38:49 -0500       workerpool0-0     Epoch  29 | train=1.9536 [cls=0.1907 | reg=1.7629] | val=2.7965 | lr=4.17e-04 | img=224 | ★ best
INFO    2026-02-23 20:39:01 -0500       workerpool0-0     Epoch  30 | train=1.9729 [cls=0.1904 | reg=1.7825] | val=2.8311 | lr=3.84e-04 | img=224 | 
INFO    2026-02-23 20:39:13 -0500       workerpool0-0     Epoch  31 | train=1.9370 [cls=0.1899 | reg=1.7471] | val=2.8475 | lr=3.52e-04 | img=224 | 
INFO    2026-02-23 20:39:24 -0500       workerpool0-0     Epoch  32 | train=1.9552 [cls=0.1876 | reg=1.7677] | val=2.8425 | lr=3.20e-04 | img=224 | 
INFO    2026-02-23 20:39:36 -0500       workerpool0-0     Epoch  33 | train=1.9353 [cls=0.1864 | reg=1.7489] | val=2.8476 | lr=2.90e-04 | img=224 | 
INFO    2026-02-23 20:39:48 -0500       workerpool0-0     Epoch  34 | train=1.9361 [cls=0.1876 | reg=1.7485] | val=2.8426 | lr=2.60e-04 | img=224 | 
INFO    2026-02-23 20:39:59 -0500       workerpool0-0     Epoch  35 | train=1.9431 [cls=0.1875 | reg=1.7556] | val=2.8275 | lr=2.31e-04 | img=224 | 
INFO    2026-02-23 20:40:11 -0500       workerpool0-0     Epoch  36 | train=1.8952 [cls=0.1838 | reg=1.7114] | val=2.7864 | lr=2.03e-04 | img=224 | ★ best
INFO    2026-02-23 20:40:22 -0500       workerpool0-0     Epoch  37 | train=1.9127 [cls=0.1851 | reg=1.7275] | val=2.8080 | lr=1.77e-04 | img=224 | 
INFO    2026-02-23 20:40:33 -0500       workerpool0-0     Epoch  38 | train=1.8872 [cls=0.1825 | reg=1.7047] | val=2.8034 | lr=1.52e-04 | img=224 | 
INFO    2026-02-23 20:40:45 -0500       workerpool0-0     Epoch  39 | train=1.8819 [cls=0.1815 | reg=1.7004] | val=2.8034 | lr=1.29e-04 | img=224 | 
INFO    2026-02-23 20:40:57 -0500       workerpool0-0     Epoch  40 | train=1.8811 [cls=0.1832 | reg=1.6979] | val=2.8070 | lr=1.08e-04 | img=224 | 
INFO    2026-02-23 20:41:08 -0500       workerpool0-0     Epoch  41 | train=1.9104 [cls=0.1845 | reg=1.7259] | val=2.7785 | lr=8.79e-05 | img=224 | ★ best
INFO    2026-02-23 20:41:20 -0500       workerpool0-0     Epoch  42 | train=1.8809 [cls=0.1807 | reg=1.7002] | val=2.8080 | lr=6.99e-05 | img=224 | 
INFO    2026-02-23 20:41:31 -0500       workerpool0-0     Epoch  43 | train=1.8804 [cls=0.1838 | reg=1.6967] | val=2.7760 | lr=5.38e-05 | img=224 | ★ best
INFO    2026-02-23 20:41:43 -0500       workerpool0-0     Epoch  44 | train=1.8632 [cls=0.1822 | reg=1.6811] | val=2.7748 | lr=3.98e-05 | img=224 | ★ best
INFO    2026-02-23 20:41:55 -0500       workerpool0-0     Epoch  45 | train=1.9006 [cls=0.1833 | reg=1.7172] | val=2.7773 | lr=2.78e-05 | img=224 | 
INFO    2026-02-23 20:42:06 -0500       workerpool0-0     Epoch  46 | train=1.8623 [cls=0.1814 | reg=1.6809] | val=2.7736 | lr=1.79e-05 | img=224 | ★ best
INFO    2026-02-23 20:42:17 -0500       workerpool0-0     Epoch  47 | train=1.8572 [cls=0.1835 | reg=1.6737] | val=2.7743 | lr=1.01e-05 | img=224 | 
INFO    2026-02-23 20:42:30 -0500       workerpool0-0     Epoch  48 | train=1.8791 [cls=0.1836 | reg=1.6955] | val=2.7633 | lr=4.56e-06 | img=224 | ★ best
INFO    2026-02-23 20:42:41 -0500       workerpool0-0     Epoch  49 | train=1.8720 [cls=0.1825 | reg=1.6895] | val=2.7779 | lr=1.22e-06 | img=224 | 
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   ✅ Phase 1 (backbone frozen) completada en 10.2 min
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   🔄 Mejor checkpoint de Phase 1 recargado
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   🔓 Todas las capas desbloqueadas: 213,440 params unfrozen
INFO    2026-02-23 20:42:41 -0500       workerpool0-0      Total trainable: 361,563
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   🚀 Phase 2 (full fine-tuning) — 100 epochs, LR=0.0001
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO    2026-02-23 20:42:41 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=100)
INFO    2026-02-23 20:42:54 -0500       workerpool0-0     Epoch  50 | train=1.8778 [cls=0.1829 | reg=1.6949] | val=2.7311 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 20:43:06 -0500       workerpool0-0     Epoch  51 | train=1.8532 [cls=0.1780 | reg=1.6751] | val=2.7076 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 20:43:18 -0500       workerpool0-0     Epoch  52 | train=1.8351 [cls=0.1791 | reg=1.6559] | val=2.6745 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 20:43:30 -0500       workerpool0-0     Epoch  53 | train=1.7970 [cls=0.1755 | reg=1.6215] | val=2.6861 | lr=1.00e-04 | img=224 | 
INFO    2026-02-23 20:43:43 -0500       workerpool0-0     Epoch  54 | train=1.7817 [cls=0.1725 | reg=1.6093] | val=2.6941 | lr=1.00e-04 | img=224 | 
INFO    2026-02-23 20:43:55 -0500       workerpool0-0     Epoch  55 | train=1.7543 [cls=0.1724 | reg=1.5819] | val=2.6934 | lr=9.99e-05 | img=224 | 
INFO    2026-02-23 20:44:08 -0500       workerpool0-0     Epoch  56 | train=1.7631 [cls=0.1704 | reg=1.5927] | val=2.6908 | lr=9.98e-05 | img=224 | 
INFO    2026-02-23 20:44:20 -0500       workerpool0-0     Epoch  57 | train=1.7334 [cls=0.1695 | reg=1.5639] | val=2.6945 | lr=9.96e-05 | img=224 | 
INFO    2026-02-23 20:44:33 -0500       workerpool0-0     Epoch  58 | train=1.7259 [cls=0.1685 | reg=1.5574] | val=2.6698 | lr=9.94e-05 | img=224 | ★ best
INFO    2026-02-23 20:44:45 -0500       workerpool0-0     Epoch  59 | train=1.7218 [cls=0.1639 | reg=1.5579] | val=2.6617 | lr=9.91e-05 | img=224 | ★ best
INFO    2026-02-23 20:44:58 -0500       workerpool0-0     Epoch  60 | train=1.7124 [cls=0.1661 | reg=1.5463] | val=2.6696 | lr=9.88e-05 | img=224 | 
INFO    2026-02-23 20:45:10 -0500       workerpool0-0     Epoch  61 | train=1.6809 [cls=0.1668 | reg=1.5141] | val=2.6589 | lr=9.84e-05 | img=224 | ★ best
INFO    2026-02-23 20:45:23 -0500       workerpool0-0     Epoch  62 | train=1.6764 [cls=0.1644 | reg=1.5120] | val=2.6443 | lr=9.80e-05 | img=224 | ★ best
INFO    2026-02-23 20:45:36 -0500       workerpool0-0     Epoch  63 | train=1.6510 [cls=0.1615 | reg=1.4895] | val=2.6480 | lr=9.76e-05 | img=224 | 
INFO    2026-02-23 20:45:48 -0500       workerpool0-0     Epoch  64 | train=1.6534 [cls=0.1620 | reg=1.4913] | val=2.6435 | lr=9.70e-05 | img=224 | ★ best
INFO    2026-02-23 20:46:00 -0500       workerpool0-0     Epoch  65 | train=1.6782 [cls=0.1652 | reg=1.5130] | val=2.6467 | lr=9.65e-05 | img=224 | 
INFO    2026-02-23 20:46:13 -0500       workerpool0-0     Epoch  66 | train=1.6150 [cls=0.1582 | reg=1.4568] | val=2.6645 | lr=9.59e-05 | img=224 | 
INFO    2026-02-23 20:46:25 -0500       workerpool0-0     Epoch  67 | train=1.6543 [cls=0.1594 | reg=1.4949] | val=2.6005 | lr=9.52e-05 | img=224 | ★ best
INFO    2026-02-23 20:46:38 -0500       workerpool0-0     Epoch  68 | train=1.6277 [cls=0.1605 | reg=1.4672] | val=2.6382 | lr=9.46e-05 | img=224 | 
INFO    2026-02-23 20:46:50 -0500       workerpool0-0     Epoch  69 | train=1.6028 [cls=0.1577 | reg=1.4450] | val=2.6411 | lr=9.38e-05 | img=224 | 
INFO    2026-02-23 20:47:02 -0500       workerpool0-0     Epoch  70 | train=1.6145 [cls=0.1562 | reg=1.4583] | val=2.6363 | lr=9.30e-05 | img=224 | 
INFO    2026-02-23 20:47:15 -0500       workerpool0-0     Epoch  71 | train=1.6031 [cls=0.1567 | reg=1.4464] | val=2.6042 | lr=9.22e-05 | img=224 | 
INFO    2026-02-23 20:47:27 -0500       workerpool0-0     Epoch  72 | train=1.5796 [cls=0.1518 | reg=1.4278] | val=2.6023 | lr=9.14e-05 | img=224 | 
INFO    2026-02-23 20:47:39 -0500       workerpool0-0     Epoch  73 | train=1.5618 [cls=0.1515 | reg=1.4104] | val=2.6075 | lr=9.05e-05 | img=224 | 
INFO    2026-02-23 20:47:52 -0500       workerpool0-0     Epoch  74 | train=1.5721 [cls=0.1525 | reg=1.4197] | val=2.5907 | lr=8.95e-05 | img=224 | ★ best
INFO    2026-02-23 20:48:04 -0500       workerpool0-0     Epoch  75 | train=1.5672 [cls=0.1540 | reg=1.4132] | val=2.6008 | lr=8.85e-05 | img=224 | 
INFO    2026-02-23 20:48:16 -0500       workerpool0-0     Epoch  76 | train=1.5860 [cls=0.1558 | reg=1.4303] | val=2.6152 | lr=8.75e-05 | img=224 | 
INFO    2026-02-23 20:48:29 -0500       workerpool0-0     Epoch  77 | train=1.5382 [cls=0.1502 | reg=1.3880] | val=2.5748 | lr=8.65e-05 | img=224 | ★ best
INFO    2026-02-23 20:48:41 -0500       workerpool0-0     Epoch  78 | train=1.5330 [cls=0.1511 | reg=1.3819] | val=2.5963 | lr=8.54e-05 | img=224 | 
INFO    2026-02-23 20:48:53 -0500       workerpool0-0     Epoch  79 | train=1.5447 [cls=0.1535 | reg=1.3912] | val=2.5789 | lr=8.42e-05 | img=224 | 
INFO    2026-02-23 20:49:05 -0500       workerpool0-0     Epoch  80 | train=1.5429 [cls=0.1526 | reg=1.3903] | val=2.5829 | lr=8.31e-05 | img=224 | 
INFO    2026-02-23 20:49:18 -0500       workerpool0-0     Epoch  81 | train=1.5406 [cls=0.1499 | reg=1.3907] | val=2.6014 | lr=8.19e-05 | img=224 | 
INFO    2026-02-23 20:49:30 -0500       workerpool0-0     Epoch  82 | train=1.5122 [cls=0.1477 | reg=1.3645] | val=2.6127 | lr=8.07e-05 | img=224 | 
INFO    2026-02-23 20:49:43 -0500       workerpool0-0     Epoch  83 | train=1.5047 [cls=0.1446 | reg=1.3601] | val=2.5712 | lr=7.94e-05 | img=224 | ★ best
INFO    2026-02-23 20:49:55 -0500       workerpool0-0     Epoch  84 | train=1.5118 [cls=0.1472 | reg=1.3646] | val=2.5748 | lr=7.81e-05 | img=224 | 
INFO    2026-02-23 20:50:08 -0500       workerpool0-0     Epoch  85 | train=1.4778 [cls=0.1454 | reg=1.3324] | val=2.5704 | lr=7.68e-05 | img=224 | ★ best
INFO    2026-02-23 20:50:20 -0500       workerpool0-0     Epoch  86 | train=1.5006 [cls=0.1461 | reg=1.3545] | val=2.5970 | lr=7.55e-05 | img=224 | 
INFO    2026-02-23 20:50:33 -0500       workerpool0-0     Epoch  87 | train=1.4944 [cls=0.1434 | reg=1.3510] | val=2.5844 | lr=7.41e-05 | img=224 | 
INFO    2026-02-23 20:50:45 -0500       workerpool0-0     Epoch  88 | train=1.4645 [cls=0.1415 | reg=1.3231] | val=2.5809 | lr=7.27e-05 | img=224 | 
INFO    2026-02-23 20:50:58 -0500       workerpool0-0     Epoch  89 | train=1.5049 [cls=0.1442 | reg=1.3608] | val=2.5863 | lr=7.13e-05 | img=224 | 
INFO    2026-02-23 20:51:11 -0500       workerpool0-0     Epoch  90 | train=1.4978 [cls=0.1469 | reg=1.3509] | val=2.6055 | lr=6.99e-05 | img=224 | 
INFO    2026-02-23 20:51:23 -0500       workerpool0-0     Epoch  91 | train=1.4705 [cls=0.1424 | reg=1.3281] | val=2.5957 | lr=6.84e-05 | img=224 | 
INFO    2026-02-23 20:51:36 -0500       workerpool0-0     Epoch  92 | train=1.4636 [cls=0.1435 | reg=1.3201] | val=2.6034 | lr=6.70e-05 | img=224 | 
INFO    2026-02-23 20:51:48 -0500       workerpool0-0     Epoch  93 | train=1.4792 [cls=0.1444 | reg=1.3348] | val=2.5882 | lr=6.55e-05 | img=224 | 
INFO    2026-02-23 20:52:01 -0500       workerpool0-0     Epoch  94 | train=1.4856 [cls=0.1416 | reg=1.3440] | val=2.6152 | lr=6.40e-05 | img=224 | 
INFO    2026-02-23 20:52:14 -0500       workerpool0-0     Epoch  95 | train=1.4499 [cls=0.1402 | reg=1.3097] | val=2.6011 | lr=6.25e-05 | img=224 | 
INFO    2026-02-23 20:52:26 -0500       workerpool0-0     Epoch  96 | train=1.4530 [cls=0.1417 | reg=1.3113] | val=2.5996 | lr=6.09e-05 | img=224 | 
INFO    2026-02-23 20:52:38 -0500       workerpool0-0     Epoch  97 | train=1.4564 [cls=0.1423 | reg=1.3140] | val=2.5775 | lr=5.94e-05 | img=224 | 
INFO    2026-02-23 20:52:51 -0500       workerpool0-0     Epoch  98 | train=1.4400 [cls=0.1381 | reg=1.3018] | val=2.6004 | lr=5.79e-05 | img=224 | 
INFO    2026-02-23 20:53:04 -0500       workerpool0-0     Epoch  99 | train=1.4366 [cls=0.1388 | reg=1.2978] | val=2.5741 | lr=5.63e-05 | img=224 | 
INFO    2026-02-23 20:53:17 -0500       workerpool0-0     Epoch 100 | train=1.4259 [cls=0.1390 | reg=1.2869] | val=2.5673 | lr=5.48e-05 | img=224 | ★ best
INFO    2026-02-23 20:53:29 -0500       workerpool0-0     Epoch 101 | train=1.4120 [cls=0.1352 | reg=1.2768] | val=2.5838 | lr=5.32e-05 | img=224 | 
INFO    2026-02-23 20:53:42 -0500       workerpool0-0     Epoch 102 | train=1.4327 [cls=0.1384 | reg=1.2942] | val=2.5712 | lr=5.16e-05 | img=224 | 
INFO    2026-02-23 20:53:54 -0500       workerpool0-0     Epoch 103 | train=1.4083 [cls=0.1363 | reg=1.2720] | val=2.5598 | lr=5.01e-05 | img=224 | ★ best
INFO    2026-02-23 20:54:07 -0500       workerpool0-0     Epoch 104 | train=1.4096 [cls=0.1373 | reg=1.2722] | val=2.5973 | lr=4.85e-05 | img=224 | 
INFO    2026-02-23 20:54:20 -0500       workerpool0-0     Epoch 105 | train=1.4410 [cls=0.1387 | reg=1.3023] | val=2.5571 | lr=4.69e-05 | img=224 | ★ best
INFO    2026-02-23 20:54:32 -0500       workerpool0-0     Epoch 106 | train=1.3967 [cls=0.1386 | reg=1.2581] | val=2.5769 | lr=4.53e-05 | img=224 | 
INFO    2026-02-23 20:54:45 -0500       workerpool0-0     Epoch 107 | train=1.3934 [cls=0.1343 | reg=1.2590] | val=2.5695 | lr=4.38e-05 | img=224 | 
INFO    2026-02-23 20:54:57 -0500       workerpool0-0     Epoch 108 | train=1.3854 [cls=0.1354 | reg=1.2501] | val=2.5985 | lr=4.22e-05 | img=224 | 
INFO    2026-02-23 20:55:10 -0500       workerpool0-0     Epoch 109 | train=1.4012 [cls=0.1349 | reg=1.2663] | val=2.5634 | lr=4.07e-05 | img=224 | 
INFO    2026-02-23 20:55:22 -0500       workerpool0-0     Epoch 110 | train=1.4053 [cls=0.1363 | reg=1.2690] | val=2.5819 | lr=3.92e-05 | img=224 | 
INFO    2026-02-23 20:55:34 -0500       workerpool0-0     Epoch 111 | train=1.3591 [cls=0.1327 | reg=1.2263] | val=2.5838 | lr=3.76e-05 | img=224 | 
INFO    2026-02-23 20:55:46 -0500       workerpool0-0     Epoch 112 | train=1.3672 [cls=0.1326 | reg=1.2347] | val=2.5914 | lr=3.61e-05 | img=224 | 
INFO    2026-02-23 20:55:58 -0500       workerpool0-0     Epoch 113 | train=1.3810 [cls=0.1347 | reg=1.2463] | val=2.5675 | lr=3.46e-05 | img=224 | 
INFO    2026-02-23 20:56:10 -0500       workerpool0-0     Epoch 114 | train=1.3864 [cls=0.1365 | reg=1.2498] | val=2.5641 | lr=3.31e-05 | img=224 | 
INFO    2026-02-23 20:56:22 -0500       workerpool0-0     Epoch 115 | train=1.3531 [cls=0.1300 | reg=1.2231] | val=2.5634 | lr=3.17e-05 | img=224 | 
INFO    2026-02-23 20:56:34 -0500       workerpool0-0     Epoch 116 | train=1.3705 [cls=0.1303 | reg=1.2402] | val=2.5779 | lr=3.02e-05 | img=224 | 
INFO    2026-02-23 20:56:46 -0500       workerpool0-0     Epoch 117 | train=1.3597 [cls=0.1293 | reg=1.2304] | val=2.5669 | lr=2.88e-05 | img=224 | 
INFO    2026-02-23 20:56:58 -0500       workerpool0-0     Epoch 118 | train=1.3538 [cls=0.1326 | reg=1.2212] | val=2.5786 | lr=2.74e-05 | img=224 | 
INFO    2026-02-23 20:57:10 -0500       workerpool0-0     Epoch 119 | train=1.3897 [cls=0.1345 | reg=1.2553] | val=2.5694 | lr=2.60e-05 | img=224 | 
INFO    2026-02-23 20:57:23 -0500       workerpool0-0     Epoch 120 | train=1.3590 [cls=0.1321 | reg=1.2269] | val=2.5750 | lr=2.46e-05 | img=224 | 
INFO    2026-02-23 20:57:35 -0500       workerpool0-0     Epoch 121 | train=1.3777 [cls=0.1332 | reg=1.2445] | val=2.5799 | lr=2.33e-05 | img=224 | 
INFO    2026-02-23 20:57:47 -0500       workerpool0-0     Epoch 122 | train=1.3607 [cls=0.1328 | reg=1.2279] | val=2.5863 | lr=2.20e-05 | img=224 | 
INFO    2026-02-23 20:58:00 -0500       workerpool0-0     Epoch 123 | train=1.3498 [cls=0.1328 | reg=1.2170] | val=2.5746 | lr=2.07e-05 | img=224 | 
INFO    2026-02-23 20:58:12 -0500       workerpool0-0     Epoch 124 | train=1.3708 [cls=0.1317 | reg=1.2391] | val=2.5646 | lr=1.94e-05 | img=224 | 
INFO    2026-02-23 20:58:25 -0500       workerpool0-0     Epoch 125 | train=1.3470 [cls=0.1289 | reg=1.2181] | val=2.5708 | lr=1.82e-05 | img=224 | 
INFO    2026-02-23 20:58:37 -0500       workerpool0-0     Epoch 126 | train=1.3412 [cls=0.1315 | reg=1.2097] | val=2.5755 | lr=1.70e-05 | img=224 | 
INFO    2026-02-23 20:58:49 -0500       workerpool0-0     Epoch 127 | train=1.3332 [cls=0.1298 | reg=1.2035] | val=2.5712 | lr=1.59e-05 | img=224 | 
INFO    2026-02-23 20:59:02 -0500       workerpool0-0     Epoch 128 | train=1.3631 [cls=0.1314 | reg=1.2317] | val=2.5833 | lr=1.47e-05 | img=224 | 
INFO    2026-02-23 20:59:14 -0500       workerpool0-0     Epoch 129 | train=1.3570 [cls=0.1302 | reg=1.2267] | val=2.5675 | lr=1.36e-05 | img=224 | 
INFO    2026-02-23 20:59:26 -0500       workerpool0-0     Epoch 130 | train=1.3433 [cls=0.1295 | reg=1.2138] | val=2.5723 | lr=1.26e-05 | img=224 | 
INFO    2026-02-23 20:59:26 -0500       workerpool0-0   ⏹️  Early stopping at epoch 130 (patience=25)
INFO    2026-02-23 20:59:26 -0500       workerpool0-0   ✅ Phase 2 (full fine-tuning) completada en 16.7 min
INFO    2026-02-23 20:59:26 -0500       workerpool0-0   📊 Entrenamiento completo: 131 epochs
INFO    2026-02-23 20:59:26 -0500       workerpool0-0      Mejor val_loss: 2.5571 (epoch 105)
INFO    2026-02-23 20:59:26 -0500       workerpool0-0      Tiempo total: 26.9 min
INFO    2026-02-23 20:59:26 -0500       workerpool0-0   ⏱️  Entrenamiento completado en 27.0 min
INFO    2026-02-23 20:59:28 -0500       workerpool0-0   💾 Historial guardado: /tmp/training/training_history.csv
INFO    2026-02-23 20:59:28 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:28 -0500       workerpool0-0   BLOQUE 5 — Curvas de Entrenamiento
INFO    2026-02-23 20:59:28 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   📊 Curvas guardadas: /tmp/training/training_curves.png
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   📈 Resumen – PYTORCH 
INFO    2026-02-23 20:59:29 -0500       workerpool0-0     Épocas: 131
INFO    2026-02-23 20:59:29 -0500       workerpool0-0     Mejor val_loss: 2.5571 (epoch 105)
INFO    2026-02-23 20:59:29 -0500       workerpool0-0     Resoluciones: [224]
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   BLOQUE 6 — Evaluación en Validación
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:29 -0500       workerpool0-0   ✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_espdet.pt
INFO    2026-02-23 20:59:32 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=val
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     mAP@50:    0.4230
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     mAP@50-95: 0.2164
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     Precision: 0.2111
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     Recall:    0.5936
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     F1-Score:  0.3114
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     Imágenes: 188 | Detecciones: 2367 | GT: 762
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     Avg inference: 12.1 ms
INFO    2026-02-23 20:59:32 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 20:59:32 -0500       workerpool0-0       dog                   0.3913
INFO    2026-02-23 20:59:32 -0500       workerpool0-0       door                  0.4683
INFO    2026-02-23 20:59:32 -0500       workerpool0-0       obstacle              0.3309
INFO    2026-02-23 20:59:32 -0500       workerpool0-0       person                0.4956
INFO    2026-02-23 20:59:32 -0500       workerpool0-0       stair                 0.4286
INFO    2026-02-23 20:59:32 -0500       workerpool0-0   📊 Val mAP@50: 0.4230
INFO    2026-02-23 20:59:32 -0500       workerpool0-0      dog: 0.3913
INFO    2026-02-23 20:59:32 -0500       workerpool0-0      door: 0.4683
INFO    2026-02-23 20:59:32 -0500       workerpool0-0      obstacle: 0.3309
INFO    2026-02-23 20:59:32 -0500       workerpool0-0      person: 0.4956
INFO    2026-02-23 20:59:32 -0500       workerpool0-0      stair: 0.4286
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   BLOQUE 7 — Evaluación en Test
INFO    2026-02-23 20:59:33 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=test
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     mAP@50:    0.5993
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     mAP@50-95: 0.3220
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     Precision: 0.2156
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     Recall:    0.7663
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     F1-Score:  0.3365
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     Imágenes: 187 | Detecciones: 2119 | GT: 576
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     Avg inference: 5.1 ms
INFO    2026-02-23 20:59:35 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 20:59:35 -0500       workerpool0-0       dog                   0.5989
INFO    2026-02-23 20:59:35 -0500       workerpool0-0       door                  0.5589
INFO    2026-02-23 20:59:35 -0500       workerpool0-0       obstacle              0.4418
INFO    2026-02-23 20:59:35 -0500       workerpool0-0       person                0.7078
INFO    2026-02-23 20:59:35 -0500       workerpool0-0       stair                 0.6893
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   📊 Test mAP@50: 0.5993
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   BLOQUE 8 — Guardado y subida a GCS
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 20:59:35 -0500       workerpool0-0   📦 Exportando ESPDet → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=6)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0     ℹ️ onnxsim not installed, skipping simplification
INFO    2026-02-23 20:59:36 -0500       workerpool0-0     ✅ Exportado: /tmp/training/export/espdet_pico.onnx (1.41 MB, 0.9s)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0     ✅ ONNX verificado: /tmp/training/export/espdet_pico.onnx
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        Latencia mediana: 3.6ms
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        box0: (1, 4, 28, 28)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        score0: (1, 5, 28, 28)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        box1: (1, 4, 14, 14)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        score1: (1, 5, 14, 14)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        box2: (1, 4, 7, 7)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0        score2: (1, 5, 7, 7)
INFO    2026-02-23 20:59:36 -0500       workerpool0-0   💾 Experimento guardado: /tmp/training/experiment.json
INFO    2026-02-23 20:59:36 -0500       workerpool0-0     ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/config.yaml
INFO    2026-02-23 20:59:36 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/training_history.csv
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/training_curves.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/class_distribution.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/gt_samples.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/val_confusion_matrix.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/val_per_class.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/val_evaluation.json
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/test_confusion_matrix.png
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/test_evaluation.json
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/experiment.json
INFO    2026-02-23 20:59:37 -0500       workerpool0-0     ☁️  Subido: /tmp/training/checkpoints/best_espdet.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/checkpoints/best_espdet.pt
INFO    2026-02-23 20:59:38 -0500       workerpool0-0     ☁️  Subido: /tmp/training/export/espdet_pico.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/export/espdet_pico.onnx
INFO    2026-02-23 20:59:38 -0500       workerpool0-0   ✅ Pipeline ESPDet-Pico completado exitosamente.
INFO    2026-02-23 20:59:40 -0500       workerpool0-0   Task completed. Exit code (0). Exit reason (SUCCEEDED)
```

## Log de Terminal de Lanzamiento
```zsh
% ./vertex_ai/build_and_launch.sh espdet_pico_v3 --run-name espdet-pico-v3-t3
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.6.2' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.6.2.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.6.2.tar.gz [Content-Type=application/x-tar]...
/ [1 files][ 75.9 KiB/ 75.9 KiB]                                                
Operation completed over 1 objects/75.9 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz

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
  Run:           espdet-pico-v3-t3
  Familia:       ESPDet
  Módulo:        trainer.task_espdet
  Contenedor:    us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
  Máquina:       n1-standard-8
  GPU:           NVIDIA_TESLA_T4 x1
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.2.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet-pico-v3-t3']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v3-t3.yaml

🚀 Lanzando Custom Job: tfm-espdet_pico_v3-1771896463
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-23-20:27:44.759 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/579401860008378368?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/2124347638428991488?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob run completed. Resource name: projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368
Training did not produce a Managed Model returning None. Training Pipeline projects/608533675308/locations/us-central1/trainingPipelines/579401860008378368 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR'].

============================================================
✅ Custom Job completado exitosamente
   Resultados en: gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v3
   Experiments:   tfm-deteccion-objetos / espdet-pico-v3-t3
============================================================

═══════════════════════════════════════════════════════════
✅ Proceso completado
═══════════════════════════════════════════════════════════
```