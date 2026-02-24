# Registro de Entrenamiento - ESPDet-Pico (v2 - Espressif Official)
**Corrida Exitosa #:** 2 (exitoso)

## Logs de Monitor de Entrenamiento:
```zsh
% gcloud ai custom-jobs stream-logs 3793775725299892224 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO    2026-02-23 19:10:22 -0500       service Waiting for job to be provisioned.
INFO    2026-02-23 19:10:22 -0500       service Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO    2026-02-23 19:10:22 -0500       service Vertex AI is setting up this job.
INFO    2026-02-23 19:10:22 -0500       service Waiting for training program to start.
INFO    2026-02-23 19:10:22 -0500       service Job is preparing.
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Updating master address to local address gk3-cml-0221-054317-2fe2-nap-1c73f772-8119d3ee-9bxk
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Running run_module.py
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-33d7222dd4-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_espdet","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003despdet-pico-v2-t2"]}
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Running module trainer.task_espdet.
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:18 -0500       workerpool0-0   Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:19 -0500       service Job is running.
ERROR   2026-02-23 19:13:19 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 19:13:19 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR   2026-02-23 19:13:22 -0500       workerpool0-0   Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR   2026-02-23 19:13:22 -0500       workerpool0-0   ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO    2026-02-23 19:13:27 -0500       workerpool0-0   Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:27 -0500       workerpool0-0   Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:30 -0500       workerpool0-0   Processing /tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:30 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 19:13:35 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 19:13:35 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 19:13:36 -0500       workerpool0-0     Getting requirements to build wheel: finished with status 'done'
INFO    2026-02-23 19:13:36 -0500       workerpool0-0     Preparing metadata (pyproject.toml): started
INFO    2026-02-23 19:13:36 -0500       workerpool0-0     Preparing metadata (pyproject.toml): finished with status 'done'
INFO    2026-02-23 19:13:36 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 19:13:36 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 19:13:37 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 19:13:37 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.1-py3-none-any.whl size=90915 sha256=444afb6501d12a9a47937688447ac0f98e6f9647062e89fc6b9860d36de0a4de
INFO    2026-02-23 19:13:37 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/23/51/c5/5cfc0ce757ea2aa78758be36234e6d5e82daaf6a7f75af2f9c
INFO    2026-02-23 19:13:37 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 19:13:37 -0500       workerpool0-0   Installing collected packages: tfm-trainer
INFO    2026-02-23 19:13:37 -0500       workerpool0-0   Successfully installed tfm-trainer-2.6.1
ERROR   2026-02-23 19:13:37 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 19:13:37 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 19:13:37 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 19:13:37 -0500       workerpool0-0   Running command: pip3 install --user tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:38 -0500       workerpool0-0   Processing /tfm_trainer-2.6.1.tar.gz
INFO    2026-02-23 19:13:38 -0500       workerpool0-0     Installing build dependencies: started
INFO    2026-02-23 19:13:42 -0500       workerpool0-0     Installing build dependencies: finished with status 'done'
INFO    2026-02-23 19:13:42 -0500       workerpool0-0     Getting requirements to build wheel: started
INFO    2026-02-23 19:13:44 -0500       workerpool0-0     Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO    2026-02-23 19:13:45 -0500       workerpool0-0   Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.1) (2.2.3)
INFO    2026-02-23 19:13:45 -0500       workerpool0-0   Collecting matplotlib>=3.8 (from tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:45 -0500       workerpool0-0     Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO    2026-02-23 19:13:45 -0500       workerpool0-0   Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.6.1) (1.6.1)
INFO    2026-02-23 19:13:49 -0500       workerpool0-0   Collecting onnxruntime>=1.16 (from tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:49 -0500       workerpool0-0     Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO    2026-02-23 19:13:49 -0500       workerpool0-0   Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.6.1) (1.11.4)
INFO    2026-02-23 19:13:50 -0500       workerpool0-0   Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:50 -0500       workerpool0-0     Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO    2026-02-23 19:13:50 -0500       workerpool0-0   Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:50 -0500       workerpool0-0     Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO    2026-02-23 19:13:51 -0500       workerpool0-0   Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:51 -0500       workerpool0-0     Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.34.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.40.3)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.26.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (3.20.3)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (25.0)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (3.34.0)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.14.2)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.1.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (4.14.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (0.16)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.70.0)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.32.4)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.73.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.48.2)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (5.5.2)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (0.4.2)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (4.9.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:52 -0500       workerpool0-0   Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:52 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.4.3)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.7.2)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2.9.0.post0)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (0.14.2)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.6.1) (1.7.1)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0   Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:53 -0500       workerpool0-0     Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0     Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0     Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.17.0)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (3.4.2)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (3.10)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (1.26.20)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (2025.6.15)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.6.1) (0.6.1)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (1.3.2)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (0.12.1)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (4.58.5)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (1.4.8)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (11.3.0)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.6.1) (3.2.3)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:55 -0500       workerpool0-0   Collecting onnx>=1.14 (from tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:55 -0500       workerpool0-0     Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.6.1) (1.14.0)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0     Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.1) (2025.2)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.6.1) (2025.2)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.1) (1.5.1)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.6.1) (3.6.0)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.6.1) (2.4.0+cu124)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (3.18.0)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (3.4.2)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (3.1.6)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (2025.5.1)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (9.1.0.70)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.2.65)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (11.2.0.44)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (10.3.5.119)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (11.6.0.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.3.0.142)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (2.20.5)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (12.4.99)
INFO    2026-02-23 19:13:56 -0500       workerpool0-0   Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (3.0.0)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Requirement already satisfied: psutil>=5.8.0 in /opt/python/3.10/lib/python3.10/site-packages (from ultralytics>=8.2->tfm-trainer==2.6.1) (5.9.3)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting polars>=0.20.0 (from ultralytics>=8.2->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading polars-1.38.1-py3-none-any.whl.metadata (10 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting ultralytics-thop>=2.0.18 (from ultralytics>=8.2->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   INFO: pip is looking at multiple versions of opencv-python to determine which version is compatible with other requirements. This could take a while.
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting opencv-python>=4.6.0 (from ultralytics>=8.2->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading opencv_python-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting polars-runtime-32==1.38.1 (from polars>=0.20.0->ultralytics>=8.2->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.6.1)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0     Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.6.1) (3.0.2)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.6.1) (1.3.0)
INFO    2026-02-23 19:13:57 -0500       workerpool0-0   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 113.8 MB/s eta 0:00:00
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 85.8 MB/s eta 0:00:00
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 114.3 MB/s eta 0:00:00
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 122.4 MB/s eta 0:00:00
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO    2026-02-23 19:13:58 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 125.5 MB/s eta 0:00:00
INFO    2026-02-23 19:13:58 -0500       workerpool0-0   Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 117.4 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 28.2 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 74.4 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading ultralytics-8.4.14-py3-none-any.whl (1.2 MB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 61.3 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 MB 114.8 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading polars-1.38.1-py3-none-any.whl (810 kB)
INFO    2026-02-23 19:13:59 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 810.4/810.4 kB 35.1 MB/s eta 0:00:00
INFO    2026-02-23 19:13:59 -0500       workerpool0-0   Downloading polars_runtime_32-1.38.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (45.8 MB)
INFO    2026-02-23 19:14:00 -0500       workerpool0-0      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.8/45.8 MB 130.2 MB/s eta 0:00:00
INFO    2026-02-23 19:14:00 -0500       workerpool0-0   Downloading ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
INFO    2026-02-23 19:14:00 -0500       workerpool0-0   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO    2026-02-23 19:14:00 -0500       workerpool0-0   Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO    2026-02-23 19:14:01 -0500       workerpool0-0   Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO    2026-02-23 19:14:02 -0500       workerpool0-0   Building wheels for collected packages: tfm-trainer
INFO    2026-02-23 19:14:02 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): started
INFO    2026-02-23 19:14:02 -0500       workerpool0-0     Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO    2026-02-23 19:14:02 -0500       workerpool0-0     Created wheel for tfm-trainer: filename=tfm_trainer-2.6.1-py3-none-any.whl size=90915 sha256=0d26bf520efff351df606d6533646c5507a2ee3bac8a2f1f24f7c280298f01d3
INFO    2026-02-23 19:14:02 -0500       workerpool0-0     Stored in directory: /root/.cache/pip/wheels/23/51/c5/5cfc0ce757ea2aa78758be36234e6d5e82daaf6a7f75af2f9c
INFO    2026-02-23 19:14:02 -0500       workerpool0-0   Successfully built tfm-trainer
INFO    2026-02-23 19:14:04 -0500       workerpool0-0   Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, polars-runtime-32, numpy, humanfriendly, annotated-types, pydantic, polars, opencv-python-headless, opencv-python, onnx, coloredlogs, onnxruntime, matplotlib, albucore, ultralytics-thop, albumentations, ultralytics, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR   2026-02-23 19:14:04 -0500       workerpool0-0     WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:04 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:08 -0500       workerpool0-0     WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:08 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:09 -0500       workerpool0-0     WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:09 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:17 -0500       workerpool0-0     WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:17 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:17 -0500       workerpool0-0     WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:17 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:19 -0500       workerpool0-0     WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:19 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR   2026-02-23 19:14:21 -0500       workerpool0-0     WARNING: The scripts ultralytics and yolo are installed in '/root/.local/bin' which is not on PATH.
ERROR   2026-02-23 19:14:21 -0500       workerpool0-0     Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO    2026-02-23 19:14:22 -0500       workerpool0-0     Attempting uninstall: tfm-trainer
INFO    2026-02-23 19:14:22 -0500       workerpool0-0       Found existing installation: tfm-trainer 2.6.1
INFO    2026-02-23 19:14:22 -0500       workerpool0-0       Uninstalling tfm-trainer-2.6.1:
INFO    2026-02-23 19:14:22 -0500       workerpool0-0         Successfully uninstalled tfm-trainer-2.6.1
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
INFO    2026-02-23 19:14:22 -0500       workerpool0-0   Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-4.11.0.86 opencv-python-headless-4.11.0.86 polars-1.38.1 polars-runtime-32-1.38.1 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.6.1 typing-inspection-0.4.2 ultralytics-8.4.14 ultralytics-thop-2.0.18
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   [notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR   2026-02-23 19:14:22 -0500       workerpool0-0   [notice] To update, run: pip install --upgrade pip
INFO    2026-02-23 19:14:23 -0500       workerpool0-0   Running command: python3 -m trainer.task_espdet --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=espdet-pico-v2-t2
INFO    2026-02-23 19:14:26 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:26 -0500       workerpool0-0   BLOQUE 1 — Setup y descarga de datos
INFO    2026-02-23 19:14:26 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml → /tmp/training/config.yaml (0.0 MB)
INFO    2026-02-23 19:14:51 -0500       workerpool0-0   🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Nombre:       espdet_pico_v2
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Familia:      ESPDet
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Variante:     espdet_pico
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Versión:      v2
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Descripción:  ESPDet-Pico oficial Espressif + transfer learning cat-detection
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Dataset:      iodc_yolo
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Img Size:     224×224
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Batch Size:   32
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Patience:     25
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Seed:         42
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     Conf Thresh:  0.25
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     IoU Thresh:   0.45
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     📐 2-Phase Training:
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        Phase 1: 50 epochs @ LR=0.001
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        Phase 2: 100 epochs @ LR=0.0001
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        Resize Schedule: {0: 224}
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        Optimizer: AdamW | WD: 0.0005
INFO    2026-02-23 19:14:51 -0500       workerpool0-0     🟢 ESPDet Config:
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        Pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 19:14:51 -0500       workerpool0-0        reg_max: 1
INFO    2026-02-23 19:14:51 -0500       workerpool0-0   ✅ Configuración aplicada correctamente
INFO    2026-02-23 19:14:51 -0500       workerpool0-0   📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO    2026-02-23 19:14:52 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO    2026-02-23 19:14:52 -0500       workerpool0-0     📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     ✅ Descomprimido correctamente
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   🖥️  Device: cuda
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   ⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO    2026-02-23 19:14:54 -0500       workerpool0-0      Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   domain: "googleapis.com"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   metadata {
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     key: "method"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   }
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   metadata {
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     key: "service"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     value: "aiplatform.googleapis.com"
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   }
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   ]
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   BLOQUE 2 — Verificación del Dataset
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   📂 Dataset YOLO: iodc_yolo
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO    2026-02-23 19:14:54 -0500       workerpool0-0     ✅  test:   187 imgs |   187 labels | 0 sin label
INFO    2026-02-23 19:14:54 -0500       workerpool0-0   📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO    2026-02-23 19:14:55 -0500       workerpool0-0   generated new fontManager
INFO    2026-02-23 19:14:56 -0500       workerpool0-0     📊 Guardado: /tmp/training/class_distribution.png
INFO    2026-02-23 19:14:56 -0500       workerpool0-0   ⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO    2026-02-23 19:14:58 -0500       workerpool0-0     🖼️  Guardado: /tmp/training/gt_samples.png
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   BLOQUE 3 — Construcción del Modelo ESPDet-Pico
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   ⬇️  Descargando pesos pretrained: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 19:14:58 -0500       workerpool0-0     ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt → /tmp/training/pretrained_weights.pt (1.0 MB)
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   Creating new Ultralytics Settings v0.0.6 file ✅ 
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
INFO    2026-02-23 19:14:58 -0500       workerpool0-0   Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
INFO    2026-02-23 19:14:59 -0500       workerpool0-0   🔄 Cargando pesos pretrained: /tmp/training/pretrained_weights.pt
INFO    2026-02-23 19:14:59 -0500       workerpool0-0     ✅ Transfer learning: 622 param groups cargados
INFO    2026-02-23 19:14:59 -0500       workerpool0-0     ℹ️  Shape mismatch (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 19:14:59 -0500       workerpool0-0     ℹ️  Missing keys (random init): ['head.cv3.0.2.weight', 'head.cv3.0.2.bias', 'head.cv3.1.2.weight', 'head.cv3.1.2.bias', 'head.cv3.2.2.weight', 'head.cv3.2.2.bias']
INFO    2026-02-23 19:14:59 -0500       workerpool0-0   ✅ ESPDet-Pico (oficial) construido: 361,563 params (361,563 trainable)
INFO    2026-02-23 19:14:59 -0500       workerpool0-0      Strides: [8, 16, 32] | Classes: 5
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 19:15:00 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   📦 Modelo: ESPDet-Pico
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Total params:          361,563
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Trainable:             148,123
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Non-trainable:         213,440
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Est. float32:           1.38 MB
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Est. INT8:              0.34 MB
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   📐 Tamaño estimado: 1.38 MB (FP32), 0.34 MB (INT8)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.6.1 (Official Architecture)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Architecture:    Official Espressif (esp-detection repo)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Strides:         [8, 16, 32]
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     pretrained:      gs://project-18f58341-12cf-47bc-861-tfm-data/pretrained/espdet_pico_224_224_cat.pt
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Phase 1:         50 ep, LR=0.001, WD=0.0001
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Phase 2:         100 ep, LR=0.0001, WD=1e-05
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Optimizer:       adamw
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     cls_weight:      1.0
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     reg_weight:      2.0
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Conf threshold:  0.25
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     IoU threshold:   0.45
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     AMP:             True
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Grad clip:       5.0
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Export imgsz:    224
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Batch size:      32
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Patience:        25
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', 'aug_hue_shift_limit', 'aug_sat_shift_limit', 'aug_val_shift_limit', 'aug_shift_limit', 'aug_scale_limit', 'aug_rotate_limit', 'aug_hflip_prob', 'aug_gaussian_noise']
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   BLOQUE 4 — Entrenamiento (2 fases)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   🔧 Two-Phase Training Config
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Phase 1: 50 epochs | LR=0.001 | WD=0.0001
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Phase 2: 100 epochs | LR=0.0001 | WD=1e-05
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Optimizer: adamw | Scheduler: cosine
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Batch: 32 | AMP: True | Patience: 25
INFO    2026-02-23 19:15:00 -0500       workerpool0-0     Resize schedule: [(0, 224)]
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   🔒 Backbone congelado: 213,440 params frozen
INFO    2026-02-23 19:15:00 -0500       workerpool0-0      Trainable: 148,123 / 361,563 (41.0%)
INFO    2026-02-23 19:15:00 -0500       workerpool0-0   🚀 Phase 1 (backbone frozen) — 50 epochs, LR=0.001
INFO    2026-02-23 19:15:08 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO    2026-02-23 19:15:08 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=47)
INFO    2026-02-23 19:15:30 -0500       workerpool0-0     Epoch   0 | train=16.4756 [cls=12.6544 | reg=3.8211] | val=14.6752 | lr=3.33e-04 | img=224 | ★ best
INFO    2026-02-23 19:15:45 -0500       workerpool0-0     Epoch   1 | train=11.2120 [cls=7.9131 | reg=3.2989] | val=9.6318 | lr=6.67e-04 | img=224 | ★ best
INFO    2026-02-23 19:16:00 -0500       workerpool0-0     Epoch   2 | train=7.1496 [cls=4.2392 | reg=2.9104] | val=6.7810 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 19:16:14 -0500       workerpool0-0     Epoch   3 | train=5.4298 [cls=2.7891 | reg=2.6407] | val=6.0913 | lr=1.00e-03 | img=224 | ★ best
INFO    2026-02-23 19:16:28 -0500       workerpool0-0     Epoch   4 | train=4.8373 [cls=2.3730 | reg=2.4644] | val=5.7335 | lr=9.99e-04 | img=224 | ★ best
INFO    2026-02-23 19:16:41 -0500       workerpool0-0     Epoch   5 | train=4.5383 [cls=2.1759 | reg=2.3624] | val=5.5781 | lr=9.96e-04 | img=224 | ★ best
INFO    2026-02-23 19:16:54 -0500       workerpool0-0     Epoch   6 | train=4.3182 [cls=2.0317 | reg=2.2865] | val=5.4298 | lr=9.90e-04 | img=224 | ★ best
INFO    2026-02-23 19:17:06 -0500       workerpool0-0     Epoch   7 | train=4.1865 [cls=1.9613 | reg=2.2253] | val=5.4106 | lr=9.82e-04 | img=224 | ★ best
INFO    2026-02-23 19:17:17 -0500       workerpool0-0     Epoch   8 | train=4.0306 [cls=1.8741 | reg=2.1564] | val=5.2752 | lr=9.72e-04 | img=224 | ★ best
INFO    2026-02-23 19:17:28 -0500       workerpool0-0     Epoch   9 | train=3.9796 [cls=1.8322 | reg=2.1474] | val=5.2990 | lr=9.60e-04 | img=224 | 
INFO    2026-02-23 19:17:38 -0500       workerpool0-0     Epoch  10 | train=3.8642 [cls=1.7502 | reg=2.1141] | val=5.0796 | lr=9.46e-04 | img=224 | ★ best
INFO    2026-02-23 19:17:49 -0500       workerpool0-0     Epoch  11 | train=3.7880 [cls=1.7083 | reg=2.0797] | val=5.0246 | lr=9.30e-04 | img=224 | ★ best
INFO    2026-02-23 19:18:00 -0500       workerpool0-0     Epoch  12 | train=3.8105 [cls=1.7272 | reg=2.0833] | val=5.0929 | lr=9.12e-04 | img=224 | 
INFO    2026-02-23 19:18:11 -0500       workerpool0-0     Epoch  13 | train=3.7144 [cls=1.6691 | reg=2.0453] | val=5.0374 | lr=8.92e-04 | img=224 | 
INFO    2026-02-23 19:18:22 -0500       workerpool0-0     Epoch  14 | train=3.6587 [cls=1.6384 | reg=2.0203] | val=5.2902 | lr=8.71e-04 | img=224 | 
INFO    2026-02-23 19:18:32 -0500       workerpool0-0     Epoch  15 | train=3.6170 [cls=1.6078 | reg=2.0092] | val=5.1082 | lr=8.48e-04 | img=224 | 
INFO    2026-02-23 19:18:43 -0500       workerpool0-0     Epoch  16 | train=3.5665 [cls=1.5657 | reg=2.0008] | val=4.9382 | lr=8.23e-04 | img=224 | ★ best
INFO    2026-02-23 19:18:54 -0500       workerpool0-0     Epoch  17 | train=3.5500 [cls=1.5573 | reg=1.9927] | val=5.2681 | lr=7.97e-04 | img=224 | 
INFO    2026-02-23 19:19:05 -0500       workerpool0-0     Epoch  18 | train=3.5654 [cls=1.5677 | reg=1.9977] | val=5.1272 | lr=7.69e-04 | img=224 | 
INFO    2026-02-23 19:19:16 -0500       workerpool0-0     Epoch  19 | train=3.4665 [cls=1.5183 | reg=1.9481] | val=4.8809 | lr=7.40e-04 | img=224 | ★ best
INFO    2026-02-23 19:19:27 -0500       workerpool0-0     Epoch  20 | train=3.3840 [cls=1.4733 | reg=1.9107] | val=5.0354 | lr=7.10e-04 | img=224 | 
INFO    2026-02-23 19:19:38 -0500       workerpool0-0     Epoch  21 | train=3.4715 [cls=1.5246 | reg=1.9469] | val=4.9482 | lr=6.80e-04 | img=224 | 
INFO    2026-02-23 19:19:49 -0500       workerpool0-0     Epoch  22 | train=3.3861 [cls=1.4793 | reg=1.9068] | val=5.0780 | lr=6.48e-04 | img=224 | 
INFO    2026-02-23 19:20:00 -0500       workerpool0-0     Epoch  23 | train=3.4698 [cls=1.5206 | reg=1.9492] | val=4.8576 | lr=6.16e-04 | img=224 | ★ best
INFO    2026-02-23 19:20:10 -0500       workerpool0-0     Epoch  24 | train=3.3823 [cls=1.4688 | reg=1.9135] | val=4.8010 | lr=5.83e-04 | img=224 | ★ best
INFO    2026-02-23 19:20:21 -0500       workerpool0-0     Epoch  25 | train=3.3151 [cls=1.4301 | reg=1.8850] | val=4.7722 | lr=5.50e-04 | img=224 | ★ best
INFO    2026-02-23 19:20:32 -0500       workerpool0-0     Epoch  26 | train=3.2730 [cls=1.4174 | reg=1.8556] | val=4.8781 | lr=5.17e-04 | img=224 | 
INFO    2026-02-23 19:20:42 -0500       workerpool0-0     Epoch  27 | train=3.2780 [cls=1.4303 | reg=1.8478] | val=4.7208 | lr=4.83e-04 | img=224 | ★ best
INFO    2026-02-23 19:20:53 -0500       workerpool0-0     Epoch  28 | train=3.2548 [cls=1.3950 | reg=1.8598] | val=4.7462 | lr=4.50e-04 | img=224 | 
INFO    2026-02-23 19:21:04 -0500       workerpool0-0     Epoch  29 | train=3.2717 [cls=1.4170 | reg=1.8547] | val=4.7353 | lr=4.17e-04 | img=224 | 
INFO    2026-02-23 19:21:15 -0500       workerpool0-0     Epoch  30 | train=3.2301 [cls=1.3931 | reg=1.8370] | val=4.8703 | lr=3.84e-04 | img=224 | 
INFO    2026-02-23 19:21:25 -0500       workerpool0-0     Epoch  31 | train=3.1987 [cls=1.3706 | reg=1.8281] | val=4.7846 | lr=3.52e-04 | img=224 | 
INFO    2026-02-23 19:21:36 -0500       workerpool0-0     Epoch  32 | train=3.2170 [cls=1.3782 | reg=1.8388] | val=4.7391 | lr=3.20e-04 | img=224 | 
INFO    2026-02-23 19:21:47 -0500       workerpool0-0     Epoch  33 | train=3.1910 [cls=1.3698 | reg=1.8212] | val=4.7613 | lr=2.90e-04 | img=224 | 
INFO    2026-02-23 19:21:58 -0500       workerpool0-0     Epoch  34 | train=3.1436 [cls=1.3392 | reg=1.8044] | val=4.7108 | lr=2.60e-04 | img=224 | ★ best
INFO    2026-02-23 19:22:08 -0500       workerpool0-0     Epoch  35 | train=3.1390 [cls=1.3292 | reg=1.8099] | val=4.7463 | lr=2.31e-04 | img=224 | 
INFO    2026-02-23 19:22:19 -0500       workerpool0-0     Epoch  36 | train=3.1052 [cls=1.3149 | reg=1.7902] | val=4.7131 | lr=2.03e-04 | img=224 | 
INFO    2026-02-23 19:22:30 -0500       workerpool0-0     Epoch  37 | train=3.1534 [cls=1.3435 | reg=1.8098] | val=4.7208 | lr=1.77e-04 | img=224 | 
INFO    2026-02-23 19:22:41 -0500       workerpool0-0     Epoch  38 | train=3.1090 [cls=1.3251 | reg=1.7839] | val=4.7231 | lr=1.52e-04 | img=224 | 
INFO    2026-02-23 19:22:52 -0500       workerpool0-0     Epoch  39 | train=3.1013 [cls=1.3165 | reg=1.7848] | val=4.7488 | lr=1.29e-04 | img=224 | 
INFO    2026-02-23 19:23:03 -0500       workerpool0-0     Epoch  40 | train=3.1207 [cls=1.3283 | reg=1.7925] | val=4.7095 | lr=1.08e-04 | img=224 | ★ best
INFO    2026-02-23 19:23:14 -0500       workerpool0-0     Epoch  41 | train=2.9996 [cls=1.2630 | reg=1.7365] | val=4.7053 | lr=8.79e-05 | img=224 | ★ best
INFO    2026-02-23 19:23:25 -0500       workerpool0-0     Epoch  42 | train=3.1133 [cls=1.3285 | reg=1.7848] | val=4.6902 | lr=6.99e-05 | img=224 | ★ best
INFO    2026-02-23 19:23:35 -0500       workerpool0-0     Epoch  43 | train=3.0719 [cls=1.3062 | reg=1.7657] | val=4.6903 | lr=5.38e-05 | img=224 | 
INFO    2026-02-23 19:23:46 -0500       workerpool0-0     Epoch  44 | train=3.0445 [cls=1.2947 | reg=1.7498] | val=4.6887 | lr=3.98e-05 | img=224 | ★ best
INFO    2026-02-23 19:23:57 -0500       workerpool0-0     Epoch  45 | train=3.0978 [cls=1.3161 | reg=1.7817] | val=4.6850 | lr=2.78e-05 | img=224 | ★ best
INFO    2026-02-23 19:24:08 -0500       workerpool0-0     Epoch  46 | train=3.0668 [cls=1.3104 | reg=1.7564] | val=4.7071 | lr=1.79e-05 | img=224 | 
INFO    2026-02-23 19:24:18 -0500       workerpool0-0     Epoch  47 | train=3.0939 [cls=1.3168 | reg=1.7772] | val=4.6923 | lr=1.01e-05 | img=224 | 
INFO    2026-02-23 19:24:30 -0500       workerpool0-0     Epoch  48 | train=3.0496 [cls=1.3030 | reg=1.7466] | val=4.6824 | lr=4.56e-06 | img=224 | ★ best
INFO    2026-02-23 19:24:41 -0500       workerpool0-0     Epoch  49 | train=3.0493 [cls=1.2828 | reg=1.7664] | val=4.6898 | lr=1.22e-06 | img=224 | 
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   ✅ Phase 1 (backbone frozen) completada en 9.5 min
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   🔄 Mejor checkpoint de Phase 1 recargado
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   🔓 Todas las capas desbloqueadas: 213,440 params unfrozen
INFO    2026-02-23 19:24:41 -0500       workerpool0-0      Total trainable: 361,563
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   🚀 Phase 2 (full fine-tuning) — 100 epochs, LR=0.0001
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   ⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO    2026-02-23 19:24:41 -0500       workerpool0-0   📈 Scheduler: CosineAnnealing (T_max=100)
INFO    2026-02-23 19:24:53 -0500       workerpool0-0     Epoch  50 | train=3.0614 [cls=1.3052 | reg=1.7562] | val=4.6453 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 19:25:05 -0500       workerpool0-0     Epoch  51 | train=3.0698 [cls=1.2982 | reg=1.7716] | val=4.5828 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 19:25:16 -0500       workerpool0-0     Epoch  52 | train=2.9553 [cls=1.2280 | reg=1.7272] | val=4.5783 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 19:25:28 -0500       workerpool0-0     Epoch  53 | train=2.8887 [cls=1.1852 | reg=1.7034] | val=4.5938 | lr=1.00e-04 | img=224 | 
INFO    2026-02-23 19:25:40 -0500       workerpool0-0     Epoch  54 | train=2.8870 [cls=1.1771 | reg=1.7099] | val=4.5443 | lr=1.00e-04 | img=224 | ★ best
INFO    2026-02-23 19:25:51 -0500       workerpool0-0     Epoch  55 | train=2.8354 [cls=1.1496 | reg=1.6858] | val=4.5125 | lr=9.99e-05 | img=224 | ★ best
INFO    2026-02-23 19:26:03 -0500       workerpool0-0     Epoch  56 | train=2.8158 [cls=1.1491 | reg=1.6667] | val=4.5569 | lr=9.98e-05 | img=224 | 
INFO    2026-02-23 19:26:15 -0500       workerpool0-0     Epoch  57 | train=2.8332 [cls=1.1416 | reg=1.6916] | val=4.5795 | lr=9.96e-05 | img=224 | 
INFO    2026-02-23 19:26:26 -0500       workerpool0-0     Epoch  58 | train=2.8250 [cls=1.1458 | reg=1.6792] | val=4.5386 | lr=9.94e-05 | img=224 | 
INFO    2026-02-23 19:26:38 -0500       workerpool0-0     Epoch  59 | train=2.7109 [cls=1.0701 | reg=1.6408] | val=4.4730 | lr=9.91e-05 | img=224 | ★ best
INFO    2026-02-23 19:26:50 -0500       workerpool0-0     Epoch  60 | train=2.8160 [cls=1.1291 | reg=1.6869] | val=4.4645 | lr=9.88e-05 | img=224 | ★ best
INFO    2026-02-23 19:27:01 -0500       workerpool0-0     Epoch  61 | train=2.6575 [cls=1.0424 | reg=1.6150] | val=4.4716 | lr=9.84e-05 | img=224 | 
INFO    2026-02-23 19:27:13 -0500       workerpool0-0     Epoch  62 | train=2.6972 [cls=1.0627 | reg=1.6345] | val=4.4563 | lr=9.80e-05 | img=224 | ★ best
INFO    2026-02-23 19:27:25 -0500       workerpool0-0     Epoch  63 | train=2.6636 [cls=1.0318 | reg=1.6318] | val=4.4976 | lr=9.76e-05 | img=224 | 
INFO    2026-02-23 19:27:36 -0500       workerpool0-0     Epoch  64 | train=2.6037 [cls=1.0028 | reg=1.6010] | val=4.4979 | lr=9.70e-05 | img=224 | 
INFO    2026-02-23 19:27:48 -0500       workerpool0-0     Epoch  65 | train=2.6338 [cls=1.0266 | reg=1.6072] | val=4.4913 | lr=9.65e-05 | img=224 | 
INFO    2026-02-23 19:28:00 -0500       workerpool0-0     Epoch  66 | train=2.5939 [cls=1.0035 | reg=1.5905] | val=4.4562 | lr=9.59e-05 | img=224 | ★ best
INFO    2026-02-23 19:28:11 -0500       workerpool0-0     Epoch  67 | train=2.5745 [cls=0.9934 | reg=1.5811] | val=4.4888 | lr=9.52e-05 | img=224 | 
INFO    2026-02-23 19:28:23 -0500       workerpool0-0     Epoch  68 | train=2.5754 [cls=0.9875 | reg=1.5879] | val=4.5033 | lr=9.46e-05 | img=224 | 
INFO    2026-02-23 19:28:34 -0500       workerpool0-0     Epoch  69 | train=2.4980 [cls=0.9459 | reg=1.5521] | val=4.5113 | lr=9.38e-05 | img=224 | 
INFO    2026-02-23 19:28:46 -0500       workerpool0-0     Epoch  70 | train=2.5063 [cls=0.9463 | reg=1.5600] | val=4.4672 | lr=9.30e-05 | img=224 | 
INFO    2026-02-23 19:28:58 -0500       workerpool0-0     Epoch  71 | train=2.4896 [cls=0.9376 | reg=1.5521] | val=4.4181 | lr=9.22e-05 | img=224 | ★ best
INFO    2026-02-23 19:29:09 -0500       workerpool0-0     Epoch  72 | train=2.4505 [cls=0.9155 | reg=1.5350] | val=4.5042 | lr=9.14e-05 | img=224 | 
INFO    2026-02-23 19:29:21 -0500       workerpool0-0     Epoch  73 | train=2.4528 [cls=0.9187 | reg=1.5341] | val=4.4477 | lr=9.05e-05 | img=224 | 
INFO    2026-02-23 19:29:32 -0500       workerpool0-0     Epoch  74 | train=2.4736 [cls=0.9475 | reg=1.5260] | val=4.5228 | lr=8.95e-05 | img=224 | 
INFO    2026-02-23 19:29:43 -0500       workerpool0-0     Epoch  75 | train=2.4432 [cls=0.9042 | reg=1.5390] | val=4.4305 | lr=8.85e-05 | img=224 | 
INFO    2026-02-23 19:29:55 -0500       workerpool0-0     Epoch  76 | train=2.4025 [cls=0.8850 | reg=1.5175] | val=4.4628 | lr=8.75e-05 | img=224 | 
INFO    2026-02-23 19:30:06 -0500       workerpool0-0     Epoch  77 | train=2.4304 [cls=0.9220 | reg=1.5084] | val=4.5073 | lr=8.65e-05 | img=224 | 
INFO    2026-02-23 19:30:18 -0500       workerpool0-0     Epoch  78 | train=2.4048 [cls=0.8972 | reg=1.5076] | val=4.4723 | lr=8.54e-05 | img=224 | 
INFO    2026-02-23 19:30:29 -0500       workerpool0-0     Epoch  79 | train=2.4203 [cls=0.9077 | reg=1.5126] | val=4.5019 | lr=8.42e-05 | img=224 | 
INFO    2026-02-23 19:30:41 -0500       workerpool0-0     Epoch  80 | train=2.3764 [cls=0.8777 | reg=1.4987] | val=4.4569 | lr=8.31e-05 | img=224 | 
INFO    2026-02-23 19:30:52 -0500       workerpool0-0     Epoch  81 | train=2.3375 [cls=0.8485 | reg=1.4890] | val=4.4388 | lr=8.19e-05 | img=224 | 
INFO    2026-02-23 19:31:04 -0500       workerpool0-0     Epoch  82 | train=2.3588 [cls=0.8735 | reg=1.4853] | val=4.4848 | lr=8.07e-05 | img=224 | 
INFO    2026-02-23 19:31:15 -0500       workerpool0-0     Epoch  83 | train=2.3441 [cls=0.8595 | reg=1.4846] | val=4.4642 | lr=7.94e-05 | img=224 | 
INFO    2026-02-23 19:31:26 -0500       workerpool0-0     Epoch  84 | train=2.2888 [cls=0.8326 | reg=1.4562] | val=4.4511 | lr=7.81e-05 | img=224 | 
INFO    2026-02-23 19:31:38 -0500       workerpool0-0     Epoch  85 | train=2.2960 [cls=0.8361 | reg=1.4599] | val=4.4710 | lr=7.68e-05 | img=224 | 
INFO    2026-02-23 19:31:49 -0500       workerpool0-0     Epoch  86 | train=2.3158 [cls=0.8461 | reg=1.4697] | val=4.4653 | lr=7.55e-05 | img=224 | 
INFO    2026-02-23 19:32:01 -0500       workerpool0-0     Epoch  87 | train=2.2911 [cls=0.8334 | reg=1.4577] | val=4.4100 | lr=7.41e-05 | img=224 | ★ best
INFO    2026-02-23 19:32:12 -0500       workerpool0-0     Epoch  88 | train=2.2435 [cls=0.8032 | reg=1.4403] | val=4.4446 | lr=7.27e-05 | img=224 | 
INFO    2026-02-23 19:32:24 -0500       workerpool0-0     Epoch  89 | train=2.2062 [cls=0.7889 | reg=1.4173] | val=4.4541 | lr=7.13e-05 | img=224 | 
INFO    2026-02-23 19:32:35 -0500       workerpool0-0     Epoch  90 | train=2.2763 [cls=0.8236 | reg=1.4527] | val=4.4534 | lr=6.99e-05 | img=224 | 
INFO    2026-02-23 19:32:46 -0500       workerpool0-0     Epoch  91 | train=2.2395 [cls=0.7949 | reg=1.4446] | val=4.4628 | lr=6.84e-05 | img=224 | 
INFO    2026-02-23 19:32:58 -0500       workerpool0-0     Epoch  92 | train=2.2307 [cls=0.7891 | reg=1.4416] | val=4.5194 | lr=6.70e-05 | img=224 | 
INFO    2026-02-23 19:33:10 -0500       workerpool0-0     Epoch  93 | train=2.2823 [cls=0.8178 | reg=1.4645] | val=4.4807 | lr=6.55e-05 | img=224 | 
INFO    2026-02-23 19:33:21 -0500       workerpool0-0     Epoch  94 | train=2.2271 [cls=0.7983 | reg=1.4288] | val=4.5175 | lr=6.40e-05 | img=224 | 
INFO    2026-02-23 19:33:33 -0500       workerpool0-0     Epoch  95 | train=2.2209 [cls=0.7883 | reg=1.4326] | val=4.4855 | lr=6.25e-05 | img=224 | 
INFO    2026-02-23 19:33:44 -0500       workerpool0-0     Epoch  96 | train=2.2112 [cls=0.7871 | reg=1.4241] | val=4.4983 | lr=6.09e-05 | img=224 | 
INFO    2026-02-23 19:33:56 -0500       workerpool0-0     Epoch  97 | train=2.1840 [cls=0.7593 | reg=1.4247] | val=4.4954 | lr=5.94e-05 | img=224 | 
INFO    2026-02-23 19:34:07 -0500       workerpool0-0     Epoch  98 | train=2.1892 [cls=0.7747 | reg=1.4145] | val=4.4383 | lr=5.79e-05 | img=224 | 
INFO    2026-02-23 19:34:19 -0500       workerpool0-0     Epoch  99 | train=2.1699 [cls=0.7523 | reg=1.4176] | val=4.4652 | lr=5.63e-05 | img=224 | 
INFO    2026-02-23 19:34:30 -0500       workerpool0-0     Epoch 100 | train=2.1422 [cls=0.7393 | reg=1.4029] | val=4.5049 | lr=5.48e-05 | img=224 | 
INFO    2026-02-23 19:34:42 -0500       workerpool0-0     Epoch 101 | train=2.1548 [cls=0.7552 | reg=1.3996] | val=4.5054 | lr=5.32e-05 | img=224 | 
INFO    2026-02-23 19:34:54 -0500       workerpool0-0     Epoch 102 | train=2.1633 [cls=0.7592 | reg=1.4041] | val=4.5104 | lr=5.16e-05 | img=224 | 
INFO    2026-02-23 19:35:05 -0500       workerpool0-0     Epoch 103 | train=2.1415 [cls=0.7360 | reg=1.4055] | val=4.4566 | lr=5.01e-05 | img=224 | 
INFO    2026-02-23 19:35:17 -0500       workerpool0-0     Epoch 104 | train=2.1839 [cls=0.7716 | reg=1.4123] | val=4.5278 | lr=4.85e-05 | img=224 | 
INFO    2026-02-23 19:35:29 -0500       workerpool0-0     Epoch 105 | train=2.1036 [cls=0.7227 | reg=1.3809] | val=4.4719 | lr=4.69e-05 | img=224 | 
INFO    2026-02-23 19:35:40 -0500       workerpool0-0     Epoch 106 | train=2.1060 [cls=0.7184 | reg=1.3876] | val=4.5147 | lr=4.53e-05 | img=224 | 
INFO    2026-02-23 19:35:52 -0500       workerpool0-0     Epoch 107 | train=2.1116 [cls=0.7400 | reg=1.3716] | val=4.5118 | lr=4.38e-05 | img=224 | 
INFO    2026-02-23 19:36:03 -0500       workerpool0-0     Epoch 108 | train=2.1030 [cls=0.7248 | reg=1.3782] | val=4.5087 | lr=4.22e-05 | img=224 | 
INFO    2026-02-23 19:36:15 -0500       workerpool0-0     Epoch 109 | train=2.1351 [cls=0.7472 | reg=1.3879] | val=4.4906 | lr=4.07e-05 | img=224 | 
INFO    2026-02-23 19:36:27 -0500       workerpool0-0     Epoch 110 | train=2.1111 [cls=0.7310 | reg=1.3800] | val=4.5560 | lr=3.92e-05 | img=224 | 
INFO    2026-02-23 19:36:38 -0500       workerpool0-0     Epoch 111 | train=2.1118 [cls=0.7381 | reg=1.3737] | val=4.5310 | lr=3.76e-05 | img=224 | 
INFO    2026-02-23 19:36:50 -0500       workerpool0-0     Epoch 112 | train=2.0887 [cls=0.7224 | reg=1.3663] | val=4.5325 | lr=3.61e-05 | img=224 | 
INFO    2026-02-23 19:36:50 -0500       workerpool0-0   ⏹️  Early stopping at epoch 112 (patience=25)
INFO    2026-02-23 19:36:50 -0500       workerpool0-0   ✅ Phase 2 (full fine-tuning) completada en 12.1 min
INFO    2026-02-23 19:36:50 -0500       workerpool0-0   📊 Entrenamiento completo: 113 epochs
INFO    2026-02-23 19:36:50 -0500       workerpool0-0      Mejor val_loss: 4.4100 (epoch 87)
INFO    2026-02-23 19:36:50 -0500       workerpool0-0      Tiempo total: 21.7 min
INFO    2026-02-23 19:36:50 -0500       workerpool0-0   ⏱️  Entrenamiento completado en 21.8 min
INFO    2026-02-23 19:36:51 -0500       workerpool0-0   💾 Historial guardado: /tmp/training/training_history.csv
INFO    2026-02-23 19:36:51 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:51 -0500       workerpool0-0   BLOQUE 5 — Curvas de Entrenamiento
INFO    2026-02-23 19:36:51 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:52 -0500       workerpool0-0   📊 Curvas guardadas: /tmp/training/training_curves.png
INFO    2026-02-23 19:36:52 -0500       workerpool0-0   📈 Resumen – PYTORCH 
INFO    2026-02-23 19:36:52 -0500       workerpool0-0     Épocas: 113
INFO    2026-02-23 19:36:52 -0500       workerpool0-0     Mejor val_loss: 4.4100 (epoch 87)
INFO    2026-02-23 19:36:52 -0500       workerpool0-0     Resoluciones: [224]
INFO    2026-02-23 19:36:52 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:52 -0500       workerpool0-0   BLOQUE 6 — Evaluación en Validación
INFO    2026-02-23 19:36:52 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:53 -0500       workerpool0-0   ✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_espdet.pt
INFO    2026-02-23 19:36:55 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=val
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     mAP@50:    0.4543
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     mAP@50-95: 0.2100
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     Precision: 0.2848
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     Recall:    0.5628
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     F1-Score:  0.3782
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     Imágenes: 188 | Detecciones: 1538 | GT: 762
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     Avg inference: 11.2 ms
INFO    2026-02-23 19:36:55 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 19:36:55 -0500       workerpool0-0       dog                   0.4100
INFO    2026-02-23 19:36:55 -0500       workerpool0-0       door                  0.5313
INFO    2026-02-23 19:36:55 -0500       workerpool0-0       obstacle              0.3614
INFO    2026-02-23 19:36:55 -0500       workerpool0-0       person                0.4893
INFO    2026-02-23 19:36:55 -0500       workerpool0-0       stair                 0.4796
INFO    2026-02-23 19:36:55 -0500       workerpool0-0   📊 Val mAP@50: 0.4543
INFO    2026-02-23 19:36:55 -0500       workerpool0-0      dog: 0.4100
INFO    2026-02-23 19:36:55 -0500       workerpool0-0      door: 0.5313
INFO    2026-02-23 19:36:55 -0500       workerpool0-0      obstacle: 0.3614
INFO    2026-02-23 19:36:55 -0500       workerpool0-0      person: 0.4893
INFO    2026-02-23 19:36:55 -0500       workerpool0-0      stair: 0.4796
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   BLOQUE 7 — Evaluación en Test
INFO    2026-02-23 19:36:56 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:58 -0500       workerpool0-0   📊 Evaluación: espdet_pico (ESPDet) – split=test
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     mAP@50:    0.6203
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     mAP@50-95: 0.3078
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     Precision: 0.2956
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     Recall:    0.7235
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     F1-Score:  0.4197
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     Imágenes: 187 | Detecciones: 1416 | GT: 576
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     Avg inference: 4.6 ms
INFO    2026-02-23 19:36:58 -0500       workerpool0-0     Per-class AP@50:
INFO    2026-02-23 19:36:58 -0500       workerpool0-0       dog                   0.6085
INFO    2026-02-23 19:36:58 -0500       workerpool0-0       door                  0.5846
INFO    2026-02-23 19:36:58 -0500       workerpool0-0       obstacle              0.5043
INFO    2026-02-23 19:36:58 -0500       workerpool0-0       person                0.6742
INFO    2026-02-23 19:36:58 -0500       workerpool0-0       stair                 0.7299
INFO    2026-02-23 19:36:58 -0500       workerpool0-0   📊 Test mAP@50: 0.6203
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   BLOQUE 8 — Guardado y subida a GCS
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   ============================================================
INFO    2026-02-23 19:36:59 -0500       workerpool0-0   📦 Exportando ESPDet → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=6)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ℹ️ onnxsim not installed, skipping simplification
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ✅ Exportado: /tmp/training/export/espdet_pico.onnx (1.41 MB, 0.9s)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ✅ ONNX verificado: /tmp/training/export/espdet_pico.onnx
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        Latencia mediana: 2.5ms
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        box0: (1, 4, 28, 28)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        score0: (1, 5, 28, 28)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        box1: (1, 4, 14, 14)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        score1: (1, 5, 14, 14)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        box2: (1, 4, 7, 7)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0        score2: (1, 5, 7, 7)
INFO    2026-02-23 19:37:00 -0500       workerpool0-0   💾 Experimento guardado: /tmp/training/experiment.json
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/config.yaml
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/training_history.csv
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/training_curves.png
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/class_distribution.png
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/gt_samples.png
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/val_confusion_matrix.png
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/val_per_class.png
INFO    2026-02-23 19:37:00 -0500       workerpool0-0     ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/val_evaluation.json
INFO    2026-02-23 19:37:01 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/test_confusion_matrix.png
INFO    2026-02-23 19:37:01 -0500       workerpool0-0     ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/test_evaluation.json
INFO    2026-02-23 19:37:01 -0500       workerpool0-0     ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/experiment.json
INFO    2026-02-23 19:37:01 -0500       workerpool0-0     ☁️  Subido: /tmp/training/checkpoints/best_espdet.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/checkpoints/best_espdet.pt
INFO    2026-02-23 19:37:01 -0500       workerpool0-0     ☁️  Subido: /tmp/training/export/espdet_pico.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/export/espdet_pico.onnx
INFO    2026-02-23 19:37:01 -0500       workerpool0-0   ✅ Pipeline ESPDet-Pico completado exitosamente.
INFO    2026-02-23 19:37:03 -0500       workerpool0-0   Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO    2026-02-23 19:37:28 -0500       service Tearing down training program.
INFO    2026-02-23 19:38:15 -0500       service Finished tearing down training program.
INFO    2026-02-23 19:38:15 -0500       service Job completed successfully.
```


## Logs de Terminal de Lanzamiento:
```zsh
% ./vertex_ai/build_and_launch.sh espdet_pico_v2 --run-name espdet-pico-v2-t2
═══════════════════════════════════════════════════════════
📦 Paso 1/3 — Empaquetando código fuente
═══════════════════════════════════════════════════════════
creating dist
Creating tar archive
removing 'tfm_trainer-2.6.1' (and everything under it)
  ✅ Paquete: dist/tfm_trainer-2.6.1.tar.gz

═══════════════════════════════════════════════════════════
☁️  Paso 2/3 — Subiendo paquete a GCS
═══════════════════════════════════════════════════════════
Copying file://dist/tfm_trainer-2.6.1.tar.gz [Content-Type=application/x-tar]...
/ [1 files][ 75.3 KiB/ 75.3 KiB]                                                
Operation completed over 1 objects/75.3 KiB.                                     
  ✅ Subido: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz

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
  Paquete:       gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.6.1.tar.gz
  Config GCS:    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml
  Job Dir:       gs://project-18f58341-12cf-47bc-861-tfm-data/output
  Args:          ['--config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml', '--job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output', '--project-id=project-18f58341-12cf-47bc-861', '--region=us-central1', '--experiment-name=tfm-deteccion-objetos', '--run-name=espdet-pico-v2-t2']
============================================================

☁️  Config subido: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet-pico-v2-t2.yaml

🚀 Lanzando Custom Job: tfm-espdet_pico_v2-1771891782
   Revisa el progreso en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861
Training Output directory:
gs://project-18f58341-12cf-47bc-861-tfm-data/aiplatform-custom-training-2026-02-23-19:09:43.752 
View Training:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/1216450101048770560?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_PENDING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_PENDING
View backing custom job:
https://console.cloud.google.com/ai/platform/locations/us-central1/training/3793775725299892224?project=608533675308
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 current state:
PipelineState.PIPELINE_STATE_RUNNING
CustomPythonPackageTrainingJob run completed. Resource name: projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560
Training did not produce a Managed Model returning None. Training Pipeline projects/608533675308/locations/us-central1/trainingPipelines/1216450101048770560 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR'].

============================================================
✅ Custom Job completado exitosamente
   Resultados en: gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v2
   Experiments:   tfm-deteccion-objetos / espdet-pico-v2-t2
============================================================

═══════════════════════════════════════════════════════════
✅ Proceso completado
═══════════════════════════════════════════════════════════
```