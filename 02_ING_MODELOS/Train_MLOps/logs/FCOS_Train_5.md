# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 5

```zsh
% gcloud ai custom-jobs stream-logs 3110976805427740672 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 16:53:24 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 16:53:24 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 16:53:24 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 16:53:24 -0500	service	Waiting for training program to start.
INFO	2026-02-21 16:53:25 -0500	service	Job is preparing.
INFO	2026-02-21 16:56:08 -0500	service	Job is running.
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-1rjxb9dd-973a632a-mzt7
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-c8815a27bb-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771710798.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771710798"]}
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:13 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 16:56:14 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 16:56:14 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 16:56:17 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 16:56:17 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 16:56:21 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:21 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:24 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:24 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 16:56:28 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 16:56:28 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 16:56:29 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 16:56:29 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 16:56:29 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 16:56:29 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 16:56:29 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=81246 sha256=379ccb4c3becc76ee7f99944c3158c52bfd0a0c564c7ea5db2b748e13df90214
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 16:56:30 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 16:56:30 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 16:56:30 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 16:56:30 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:32 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 16:56:32 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 16:56:35 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 16:56:35 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 16:56:36 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 16:56:36 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 16:56:36 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 16:56:36 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 16:56:37 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:37 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 16:56:37 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 16:56:37 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:37 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Collecting albumentations>=2.0.0 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 16:56:38 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 16:56:39 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:39 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 16:56:39 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:39 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 16:56:39 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 16:56:40 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:40 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 16:56:40 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:40 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 16:56:41 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:41 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 16:56:42 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:42 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 16:56:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 16:56:45 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:46 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 16:56:47 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 16:56:48 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 119.0 MB/s eta 0:00:00
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 90.1 MB/s eta 0:00:00
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 122.8 MB/s eta 0:00:00
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 126.9 MB/s eta 0:00:00
INFO	2026-02-21 16:56:49 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 149.0 MB/s eta 0:00:00
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 121.0 MB/s eta 0:00:00
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 30.3 MB/s eta 0:00:00
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 67.5 MB/s eta 0:00:00
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 16:56:50 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=81246 sha256=6df400c7b941a6f679c86d7c256e0ad28bf28010ca8ad21e7b70fd0456d9423d
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 16:56:51 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 16:56:53 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 16:56:53 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:56:53 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 16:56:55 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:56:55 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 16:56:56 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:56:56 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 16:57:02 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:57:02 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 16:57:02 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:57:02 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 16:57:03 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 16:57:03 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 16:57:06 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 16:57:06 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 16:57:06 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 16:57:06 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
INFO	2026-02-21 16:57:06 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 16:57:06 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 16:57:07 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771710798.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771710798
INFO	2026-02-21 16:57:11 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:11 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 16:57:11 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771710798.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Patience:     20
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  Conf Thresh:  0.25
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     Phase 2: 80 epochs @ LR=0.0001
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 16:57:36 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 16:57:37 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 16:57:37 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	metadata {
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	}
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	metadata {
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	}
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	]
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 16:57:38 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:39 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 16:57:39 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 16:57:39 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 16:57:39 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 16:57:39 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 16:57:40 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 16:57:40 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 16:57:40 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 16:57:43 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 16:57:43 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:43 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 16:57:43 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 16:57:53 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 16:57:53 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 16:57:53 -0500	workerpool0-0	 52%|█████▏    | 5.12M/9.83M [00:00<00:00, 53.2MB/s]
ERROR	2026-02-21 16:57:53 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 74.9MB/s]
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	============================================================
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Phase 2: 80 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 20
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 16:57:54 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-21 16:58:23 -0500	workerpool0-0	  Epoch   0 | train=8.6322 [cls=3.1234 | reg=3.6730 | ctr=1.8358] | val=2078.4352 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 16:58:44 -0500	workerpool0-0	  Epoch   1 | train=7.1881 [cls=2.6110 | reg=2.7806 | ctr=1.7965] | val=1493.3792 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 16:59:04 -0500	workerpool0-0	  Epoch   2 | train=6.7332 [cls=2.4250 | reg=2.5259 | ctr=1.7823] | val=1660.8001 | lr=1.00e-03 | img=640 | 
INFO	2026-02-21 16:59:25 -0500	workerpool0-0	  Epoch   3 | train=6.4339 [cls=2.2501 | reg=2.4089 | ctr=1.7749] | val=1970.7406 | lr=1.00e-03 | img=640 | 
INFO	2026-02-21 16:59:45 -0500	workerpool0-0	  Epoch   4 | train=6.2287 [cls=2.1176 | reg=2.3413 | ctr=1.7698] | val=938.8713 | lr=9.97e-04 | img=640 | ★ best
INFO	2026-02-21 17:00:04 -0500	workerpool0-0	  Epoch   5 | train=6.0867 [cls=2.0427 | reg=2.2778 | ctr=1.7662] | val=1319.3960 | lr=9.87e-04 | img=640 | 
INFO	2026-02-21 17:00:21 -0500	workerpool0-0	  Epoch   6 | train=5.9091 [cls=1.9235 | reg=2.2220 | ctr=1.7636] | val=803.4206 | lr=9.70e-04 | img=640 | ★ best
INFO	2026-02-21 17:00:38 -0500	workerpool0-0	  Epoch   7 | train=5.8511 [cls=1.9042 | reg=2.1850 | ctr=1.7620] | val=1188.7796 | lr=9.47e-04 | img=640 | 
INFO	2026-02-21 17:00:54 -0500	workerpool0-0	  Epoch   8 | train=5.7846 [cls=1.8574 | reg=2.1671 | ctr=1.7600] | val=975.5597 | lr=9.18e-04 | img=640 | 
INFO	2026-02-21 17:01:11 -0500	workerpool0-0	  Epoch   9 | train=5.7109 [cls=1.8126 | reg=2.1398 | ctr=1.7585] | val=860.7581 | lr=8.83e-04 | img=640 | 
INFO	2026-02-21 17:01:24 -0500	workerpool0-0	  Epoch  10 | train=5.4563 [cls=1.6695 | reg=2.0336 | ctr=1.7532] | val=440.9044 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-21 17:01:38 -0500	workerpool0-0	  Epoch  11 | train=5.3458 [cls=1.6481 | reg=1.9512 | ctr=1.7465] | val=564.4336 | lr=7.99e-04 | img=416 | 
INFO	2026-02-21 17:01:51 -0500	workerpool0-0	  Epoch  12 | train=5.2322 [cls=1.5843 | reg=1.9036 | ctr=1.7443] | val=425.5308 | lr=7.50e-04 | img=416 | ★ best
INFO	2026-02-21 17:02:05 -0500	workerpool0-0	  Epoch  13 | train=5.1441 [cls=1.5171 | reg=1.8839 | ctr=1.7431] | val=437.3130 | lr=6.98e-04 | img=416 | 
INFO	2026-02-21 17:02:18 -0500	workerpool0-0	  Epoch  14 | train=5.1403 [cls=1.5162 | reg=1.8829 | ctr=1.7412] | val=561.4354 | lr=6.43e-04 | img=416 | 
INFO	2026-02-21 17:02:31 -0500	workerpool0-0	  Epoch  15 | train=5.0985 [cls=1.4941 | reg=1.8643 | ctr=1.7401] | val=351.3526 | lr=5.87e-04 | img=416 | ★ best
INFO	2026-02-21 17:02:44 -0500	workerpool0-0	  Epoch  16 | train=5.0339 [cls=1.4667 | reg=1.8305 | ctr=1.7368] | val=455.2403 | lr=5.29e-04 | img=416 | 
INFO	2026-02-21 17:02:58 -0500	workerpool0-0	  Epoch  17 | train=4.9972 [cls=1.4309 | reg=1.8291 | ctr=1.7373] | val=392.0969 | lr=4.71e-04 | img=416 | 
INFO	2026-02-21 17:03:11 -0500	workerpool0-0	  Epoch  18 | train=4.9569 [cls=1.4087 | reg=1.8146 | ctr=1.7336] | val=297.2469 | lr=4.13e-04 | img=416 | ★ best
INFO	2026-02-21 17:03:24 -0500	workerpool0-0	  Epoch  19 | train=4.9055 [cls=1.3801 | reg=1.7935 | ctr=1.7319] | val=290.2418 | lr=3.57e-04 | img=416 | ★ best
INFO	2026-02-21 17:03:37 -0500	workerpool0-0	  Epoch  20 | train=4.9367 [cls=1.4135 | reg=1.7896 | ctr=1.7336] | val=197.0997 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-21 17:03:49 -0500	workerpool0-0	  Epoch  21 | train=4.8147 [cls=1.3467 | reg=1.7380 | ctr=1.7300] | val=237.0032 | lr=2.50e-04 | img=320 | 
INFO	2026-02-21 17:04:01 -0500	workerpool0-0	  Epoch  22 | train=4.7888 [cls=1.3367 | reg=1.7227 | ctr=1.7294] | val=186.4797 | lr=2.02e-04 | img=320 | ★ best
INFO	2026-02-21 17:04:13 -0500	workerpool0-0	  Epoch  23 | train=4.7769 [cls=1.3324 | reg=1.7170 | ctr=1.7274] | val=167.3489 | lr=1.57e-04 | img=320 | ★ best
INFO	2026-02-21 17:04:26 -0500	workerpool0-0	  Epoch  24 | train=4.7557 [cls=1.3118 | reg=1.7166 | ctr=1.7273] | val=233.1300 | lr=1.17e-04 | img=320 | 
INFO	2026-02-21 17:04:38 -0500	workerpool0-0	  Epoch  25 | train=4.7444 [cls=1.3214 | reg=1.6973 | ctr=1.7256] | val=190.6250 | lr=8.23e-05 | img=320 | 
INFO	2026-02-21 17:04:50 -0500	workerpool0-0	  Epoch  26 | train=4.7044 [cls=1.2867 | reg=1.6921 | ctr=1.7255] | val=185.9942 | lr=5.33e-05 | img=320 | 
INFO	2026-02-21 17:05:02 -0500	workerpool0-0	  Epoch  27 | train=4.6693 [cls=1.2673 | reg=1.6783 | ctr=1.7237] | val=181.2907 | lr=3.03e-05 | img=320 | 
INFO	2026-02-21 17:05:15 -0500	workerpool0-0	  Epoch  28 | train=4.6736 [cls=1.2762 | reg=1.6746 | ctr=1.7227] | val=177.5932 | lr=1.36e-05 | img=320 | 
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	  Epoch  29 | train=4.7263 [cls=1.2980 | reg=1.7001 | ctr=1.7282] | val=182.5350 | lr=3.48e-06 | img=320 | 
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 7.5 min
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=0.0001
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 17:05:27 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=80)
INFO	2026-02-21 17:05:41 -0500	workerpool0-0	  Epoch  30 | train=5.0215 [cls=1.4323 | reg=1.8344 | ctr=1.7548] | val=98.3060 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 17:05:54 -0500	workerpool0-0	  Epoch  31 | train=4.7465 [cls=1.2952 | reg=1.7072 | ctr=1.7441] | val=95.6233 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 17:06:08 -0500	workerpool0-0	  Epoch  32 | train=4.6016 [cls=1.2228 | reg=1.6419 | ctr=1.7369] | val=72.0612 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 17:06:21 -0500	workerpool0-0	  Epoch  33 | train=4.4771 [cls=1.1444 | reg=1.6014 | ctr=1.7313] | val=62.0700 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 17:06:35 -0500	workerpool0-0	  Epoch  34 | train=4.4243 [cls=1.1270 | reg=1.5678 | ctr=1.7295] | val=60.6716 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 17:06:48 -0500	workerpool0-0	  Epoch  35 | train=4.3414 [cls=1.0832 | reg=1.5335 | ctr=1.7247] | val=70.4845 | lr=9.98e-05 | img=224 | 
INFO	2026-02-21 17:07:01 -0500	workerpool0-0	  Epoch  36 | train=4.2747 [cls=1.0437 | reg=1.5089 | ctr=1.7221] | val=67.6932 | lr=9.97e-05 | img=224 | 
INFO	2026-02-21 17:07:15 -0500	workerpool0-0	  Epoch  37 | train=4.2157 [cls=1.0013 | reg=1.4946 | ctr=1.7198] | val=69.2810 | lr=9.94e-05 | img=224 | 
INFO	2026-02-21 17:07:28 -0500	workerpool0-0	  Epoch  38 | train=4.1900 [cls=0.9936 | reg=1.4783 | ctr=1.7181] | val=71.9120 | lr=9.90e-05 | img=224 | 
INFO	2026-02-21 17:07:41 -0500	workerpool0-0	  Epoch  39 | train=4.1434 [cls=0.9709 | reg=1.4564 | ctr=1.7161] | val=62.9884 | lr=9.86e-05 | img=224 | 
INFO	2026-02-21 17:07:55 -0500	workerpool0-0	  Epoch  40 | train=4.0663 [cls=0.9279 | reg=1.4247 | ctr=1.7137] | val=42.8335 | lr=9.81e-05 | img=224 | ★ best
INFO	2026-02-21 17:08:08 -0500	workerpool0-0	  Epoch  41 | train=4.0231 [cls=0.9014 | reg=1.4123 | ctr=1.7095] | val=39.3344 | lr=9.76e-05 | img=224 | ★ best
INFO	2026-02-21 17:08:22 -0500	workerpool0-0	  Epoch  42 | train=4.0361 [cls=0.9175 | reg=1.4084 | ctr=1.7102] | val=46.0707 | lr=9.69e-05 | img=224 | 
INFO	2026-02-21 17:08:35 -0500	workerpool0-0	  Epoch  43 | train=3.9676 [cls=0.8715 | reg=1.3874 | ctr=1.7087] | val=56.6328 | lr=9.62e-05 | img=224 | 
INFO	2026-02-21 17:08:49 -0500	workerpool0-0	  Epoch  44 | train=3.9297 [cls=0.8601 | reg=1.3664 | ctr=1.7031] | val=62.9439 | lr=9.54e-05 | img=224 | 
INFO	2026-02-21 17:09:02 -0500	workerpool0-0	  Epoch  45 | train=3.8892 [cls=0.8361 | reg=1.3505 | ctr=1.7027] | val=47.8988 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 17:09:15 -0500	workerpool0-0	  Epoch  46 | train=3.8642 [cls=0.8252 | reg=1.3390 | ctr=1.7000] | val=52.0776 | lr=9.36e-05 | img=224 | 
INFO	2026-02-21 17:09:29 -0500	workerpool0-0	  Epoch  47 | train=3.8276 [cls=0.8078 | reg=1.3185 | ctr=1.7013] | val=56.1770 | lr=9.26e-05 | img=224 | 
INFO	2026-02-21 17:09:42 -0500	workerpool0-0	  Epoch  48 | train=3.8105 [cls=0.7975 | reg=1.3125 | ctr=1.7004] | val=65.1247 | lr=9.16e-05 | img=224 | 
INFO	2026-02-21 17:09:55 -0500	workerpool0-0	  Epoch  49 | train=3.7774 [cls=0.7863 | reg=1.2927 | ctr=1.6984] | val=38.3292 | lr=9.05e-05 | img=224 | ★ best
INFO	2026-02-21 17:10:09 -0500	workerpool0-0	  Epoch  50 | train=3.8136 [cls=0.7925 | reg=1.3214 | ctr=1.6997] | val=84.9254 | lr=8.93e-05 | img=224 | 
INFO	2026-02-21 17:10:23 -0500	workerpool0-0	  Epoch  51 | train=3.7224 [cls=0.7555 | reg=1.2705 | ctr=1.6964] | val=41.4157 | lr=8.80e-05 | img=224 | 
INFO	2026-02-21 17:10:36 -0500	workerpool0-0	  Epoch  52 | train=3.7094 [cls=0.7385 | reg=1.2752 | ctr=1.6957] | val=39.5016 | lr=8.67e-05 | img=224 | 
INFO	2026-02-21 17:10:49 -0500	workerpool0-0	  Epoch  53 | train=3.7127 [cls=0.7487 | reg=1.2692 | ctr=1.6948] | val=49.4668 | lr=8.54e-05 | img=224 | 
INFO	2026-02-21 17:11:03 -0500	workerpool0-0	  Epoch  54 | train=3.6580 [cls=0.7223 | reg=1.2430 | ctr=1.6928] | val=39.0443 | lr=8.40e-05 | img=224 | 
INFO	2026-02-21 17:11:16 -0500	workerpool0-0	  Epoch  55 | train=3.6916 [cls=0.7488 | reg=1.2483 | ctr=1.6945] | val=28.7534 | lr=8.25e-05 | img=224 | ★ best
INFO	2026-02-21 17:11:30 -0500	workerpool0-0	  Epoch  56 | train=3.6461 [cls=0.7157 | reg=1.2395 | ctr=1.6909] | val=38.9865 | lr=8.10e-05 | img=224 | 
INFO	2026-02-21 17:11:43 -0500	workerpool0-0	  Epoch  57 | train=3.6120 [cls=0.6935 | reg=1.2296 | ctr=1.6889] | val=53.6738 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 17:11:56 -0500	workerpool0-0	  Epoch  58 | train=3.5746 [cls=0.6903 | reg=1.1957 | ctr=1.6886] | val=46.7964 | lr=7.78e-05 | img=224 | 
INFO	2026-02-21 17:12:10 -0500	workerpool0-0	  Epoch  59 | train=3.5778 [cls=0.6827 | reg=1.2081 | ctr=1.6869] | val=82.9297 | lr=7.61e-05 | img=224 | 
INFO	2026-02-21 17:12:23 -0500	workerpool0-0	  Epoch  60 | train=3.5873 [cls=0.6932 | reg=1.2047 | ctr=1.6894] | val=42.4611 | lr=7.45e-05 | img=224 | 
INFO	2026-02-21 17:12:36 -0500	workerpool0-0	  Epoch  61 | train=3.5685 [cls=0.6858 | reg=1.1934 | ctr=1.6892] | val=37.4635 | lr=7.27e-05 | img=224 | 
INFO	2026-02-21 17:12:50 -0500	workerpool0-0	  Epoch  62 | train=3.5146 [cls=0.6464 | reg=1.1806 | ctr=1.6876] | val=41.1670 | lr=7.10e-05 | img=224 | 
INFO	2026-02-21 17:13:03 -0500	workerpool0-0	  Epoch  63 | train=3.5240 [cls=0.6595 | reg=1.1775 | ctr=1.6870] | val=38.7352 | lr=6.92e-05 | img=224 | 
INFO	2026-02-21 17:13:17 -0500	workerpool0-0	  Epoch  64 | train=3.5100 [cls=0.6470 | reg=1.1803 | ctr=1.6827] | val=44.8669 | lr=6.73e-05 | img=224 | 
INFO	2026-02-21 17:13:30 -0500	workerpool0-0	  Epoch  65 | train=3.4937 [cls=0.6414 | reg=1.1691 | ctr=1.6832] | val=42.6080 | lr=6.55e-05 | img=224 | 
INFO	2026-02-21 17:13:43 -0500	workerpool0-0	  Epoch  66 | train=3.5108 [cls=0.6554 | reg=1.1707 | ctr=1.6847] | val=32.3693 | lr=6.36e-05 | img=224 | 
INFO	2026-02-21 17:13:57 -0500	workerpool0-0	  Epoch  67 | train=3.4737 [cls=0.6376 | reg=1.1540 | ctr=1.6820] | val=43.0762 | lr=6.17e-05 | img=224 | 
INFO	2026-02-21 17:14:11 -0500	workerpool0-0	  Epoch  68 | train=3.4649 [cls=0.6307 | reg=1.1494 | ctr=1.6847] | val=58.3808 | lr=5.98e-05 | img=224 | 
INFO	2026-02-21 17:14:25 -0500	workerpool0-0	  Epoch  69 | train=3.4373 [cls=0.6183 | reg=1.1391 | ctr=1.6800] | val=50.3709 | lr=5.79e-05 | img=224 | 
INFO	2026-02-21 17:14:39 -0500	workerpool0-0	  Epoch  70 | train=3.4833 [cls=0.6402 | reg=1.1585 | ctr=1.6846] | val=39.7114 | lr=5.59e-05 | img=224 | 
INFO	2026-02-21 17:14:52 -0500	workerpool0-0	  Epoch  71 | train=3.4660 [cls=0.6339 | reg=1.1514 | ctr=1.6806] | val=38.2524 | lr=5.40e-05 | img=224 | 
INFO	2026-02-21 17:15:06 -0500	workerpool0-0	  Epoch  72 | train=3.4024 [cls=0.5974 | reg=1.1250 | ctr=1.6800] | val=57.7168 | lr=5.20e-05 | img=224 | 
INFO	2026-02-21 17:15:19 -0500	workerpool0-0	  Epoch  73 | train=3.3776 [cls=0.5848 | reg=1.1129 | ctr=1.6799] | val=47.9325 | lr=5.01e-05 | img=224 | 
INFO	2026-02-21 17:15:32 -0500	workerpool0-0	  Epoch  74 | train=3.3937 [cls=0.5969 | reg=1.1161 | ctr=1.6808] | val=49.0249 | lr=4.81e-05 | img=224 | 
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	  Epoch  75 | train=3.3926 [cls=0.6012 | reg=1.1121 | ctr=1.6793] | val=30.0461 | lr=4.61e-05 | img=224 | 
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	⏹️  Early stopping at epoch 75 (patience=20)
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 10.3 min
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	📊 Entrenamiento completo: 76 epochs
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	   Mejor val_loss: 28.7534 (epoch 55)
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	   Tiempo total: 17.9 min
INFO	2026-02-21 17:15:46 -0500	workerpool0-0	⏱️  Entrenamiento completado en 17.9 min
INFO	2026-02-21 17:15:47 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 17:15:47 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:47 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 17:15:47 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	  Épocas: 76
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	  Mejor val_loss: 28.7534 (epoch 55)
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:49 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  mAP@50:    0.4134
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  mAP@50-95: 0.1810
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  Precision: 0.3417
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  Recall:    0.5135
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  F1-Score:  0.4103
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 1190 | GT: 762
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  Avg inference: 5.5 ms
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	    dog                   0.3300
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	    door                  0.4496
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	    obstacle              0.4189
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	    person                0.4218
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	    stair                 0.4469
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	📊 Val mAP@50: 0.4134
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	   dog: 0.3300
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	   door: 0.4496
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	   obstacle: 0.4189
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	   person: 0.4218
INFO	2026-02-21 17:15:50 -0500	workerpool0-0	   stair: 0.4469
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 17:15:51 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  mAP@50:    0.5887
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  mAP@50-95: 0.2703
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  Precision: 0.3505
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  Recall:    0.6845
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  F1-Score:  0.4636
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 1114 | GT: 576
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  Avg inference: 4.9 ms
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	    dog                   0.5223
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	    door                  0.5436
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	    obstacle              0.4644
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	    person                0.6663
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	    stair                 0.7467
INFO	2026-02-21 17:15:52 -0500	workerpool0-0	📊 Test mAP@50: 0.5887
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 17:15:53 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.7s)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     Latencia mediana: 4.4ms
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/config.yaml
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/training_history.csv
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/training_curves.png
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/class_distribution.png
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/gt_samples.png
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/val_confusion_matrix.png
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/val_per_class.png
INFO	2026-02-21 17:15:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/val_evaluation.json
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/test_confusion_matrix.png
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/test_evaluation.json
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/experiment.json
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/checkpoints/best_fcos.pt
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/export/fcos_v3s.onnx
INFO	2026-02-21 17:15:55 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 17:15:58 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 17:16:14 -0500	service	Tearing down training program.
INFO	2026-02-21 17:16:57 -0500	service	Finished tearing down training program.
INFO	2026-02-21 17:16:57 -0500	service	Job completed successfully.
```