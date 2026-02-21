# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 2

```zsh
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 10:29:12 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 10:29:12 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 10:29:12 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 10:29:12 -0500	service	Waiting for training program to start.
INFO	2026-02-21 10:29:12 -0500	service	Job is preparing.
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-d5cl6ko2-97f7ebda-9szg
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 10:33:34 -0500	service	Job is running.
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-f121e736d8-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771687747.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771687747"]}
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:34 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 10:33:35 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 10:33:35 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 10:33:38 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 10:33:38 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 10:33:43 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:43 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:46 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:46 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 10:33:51 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 10:33:51 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 10:33:51 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 10:33:51 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=79001 sha256=03b7d31f516a2908d0f004443d717596a01b8e43206f808c1719afc475b5934f
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 10:33:52 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 10:33:52 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 10:33:53 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 10:33:53 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 10:33:53 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:54 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 10:33:54 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 10:33:58 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 10:33:58 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 10:33:59 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 10:33:59 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 10:33:59 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 10:33:59 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 10:34:00 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:00 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 10:34:00 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	Collecting albumentations>=1.4 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:01 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 10:34:02 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 10:34:02 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 10:34:02 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 10:34:02 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:02 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 10:34:03 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:03 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 10:34:03 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=1.4->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 10:34:03 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:03 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 10:34:04 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:04 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 10:34:06 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:06 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 10:34:07 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 10:34:08 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:09 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 10:34:10 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 10:34:11 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 117.4 MB/s eta 0:00:00
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 89.2 MB/s eta 0:00:00
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 123.8 MB/s eta 0:00:00
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 134.5 MB/s eta 0:00:00
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 136.3 MB/s eta 0:00:00
INFO	2026-02-21 10:34:12 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 141.3 MB/s eta 0:00:00
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 29.6 MB/s eta 0:00:00
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 79.0 MB/s eta 0:00:00
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 10:34:13 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 10:34:14 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 10:34:14 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=79001 sha256=8e28ca99b23b7aba1887a2fcb854ada5f27bae60e6243770650d8c58ca87708b
INFO	2026-02-21 10:34:14 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 10:34:14 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 10:34:15 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 10:34:15 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:15 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 10:34:18 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:18 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 10:34:19 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:19 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 10:34:24 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:24 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 10:34:25 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:25 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 10:34:27 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 10:34:27 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 10:34:30 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 10:34:30 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 10:34:30 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 10:34:30 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
INFO	2026-02-21 10:34:30 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 10:34:30 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 10:34:31 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771687747.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771687747
INFO	2026-02-21 10:34:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:34:34 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 10:34:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771687747.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Patience:     15
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  Conf Thresh:  0.25
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     Phase 2: 60 epochs @ LR=0.0001
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 10:34:59 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 10:35:00 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 10:35:00 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 10:35:01 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 10:35:01 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 10:35:01 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	metadata {
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	}
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	metadata {
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	}
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	]
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 10:35:02 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 10:35:03 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 10:35:03 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 10:35:06 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 10:35:06 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:35:06 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 10:35:06 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 10:35:17 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 10:35:17 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 10:35:17 -0500	workerpool0-0	 70%|██████▉   | 6.88M/9.83M [00:00<00:00, 71.8MB/s]
ERROR	2026-02-21 10:35:17 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 90.5MB/s]
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Phase 2: 60 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 15
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 10:35:18 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-21 10:35:44 -0500	workerpool0-0	  Epoch   0 | train=38.8863 [cls=3.2595 | reg=33.7630 | ctr=1.8639] | val=3222.6225 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 10:36:04 -0500	workerpool0-0	  Epoch   1 | train=27.2689 [cls=2.7573 | reg=22.7113 | ctr=1.8003] | val=2647.4351 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 10:36:23 -0500	workerpool0-0	  Epoch   2 | train=23.3087 [cls=2.6026 | reg=18.9221 | ctr=1.7840] | val=1941.3388 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 10:36:42 -0500	workerpool0-0	  Epoch   3 | train=21.6700 [cls=2.4922 | reg=17.4051 | ctr=1.7727] | val=1198.3517 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 10:37:00 -0500	workerpool0-0	  Epoch   4 | train=20.5980 [cls=2.3338 | reg=16.4942 | ctr=1.7699] | val=1293.7705 | lr=9.97e-04 | img=640 | 
INFO	2026-02-21 10:37:17 -0500	workerpool0-0	  Epoch   5 | train=19.7817 [cls=2.2446 | reg=15.7684 | ctr=1.7687] | val=1938.5391 | lr=9.87e-04 | img=640 | 
INFO	2026-02-21 10:37:35 -0500	workerpool0-0	  Epoch   6 | train=19.6672 [cls=2.1452 | reg=15.7574 | ctr=1.7647] | val=1196.9003 | lr=9.70e-04 | img=640 | ★ best
INFO	2026-02-21 10:37:50 -0500	workerpool0-0	  Epoch   7 | train=19.1737 [cls=2.0562 | reg=15.3529 | ctr=1.7647] | val=1498.7786 | lr=9.47e-04 | img=640 | 
INFO	2026-02-21 10:38:06 -0500	workerpool0-0	  Epoch   8 | train=18.8218 [cls=1.9982 | reg=15.0608 | ctr=1.7628] | val=1849.1440 | lr=9.18e-04 | img=640 | 
INFO	2026-02-21 10:38:21 -0500	workerpool0-0	  Epoch   9 | train=18.5654 [cls=1.9756 | reg=14.8281 | ctr=1.7618] | val=902.2592 | lr=8.83e-04 | img=640 | ★ best
INFO	2026-02-21 10:38:34 -0500	workerpool0-0	  Epoch  10 | train=11.7346 [cls=1.7984 | reg=8.1794 | ctr=1.7568] | val=672.6255 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-21 10:38:46 -0500	workerpool0-0	  Epoch  11 | train=11.1380 [cls=1.7175 | reg=7.6712 | ctr=1.7493] | val=848.3952 | lr=7.99e-04 | img=416 | 
INFO	2026-02-21 10:38:58 -0500	workerpool0-0	  Epoch  12 | train=10.8765 [cls=1.6884 | reg=7.4402 | ctr=1.7479] | val=771.9897 | lr=7.50e-04 | img=416 | 
INFO	2026-02-21 10:39:10 -0500	workerpool0-0	  Epoch  13 | train=10.7034 [cls=1.6111 | reg=7.3510 | ctr=1.7413] | val=541.6673 | lr=6.98e-04 | img=416 | ★ best
INFO	2026-02-21 10:39:22 -0500	workerpool0-0	  Epoch  14 | train=10.5393 [cls=1.5588 | reg=7.2403 | ctr=1.7402] | val=607.8572 | lr=6.43e-04 | img=416 | 
INFO	2026-02-21 10:39:34 -0500	workerpool0-0	  Epoch  15 | train=10.3642 [cls=1.5845 | reg=7.0403 | ctr=1.7394] | val=469.2689 | lr=5.87e-04 | img=416 | ★ best
INFO	2026-02-21 10:39:46 -0500	workerpool0-0	  Epoch  16 | train=10.2969 [cls=1.5417 | reg=7.0176 | ctr=1.7376] | val=311.9240 | lr=5.29e-04 | img=416 | ★ best
INFO	2026-02-21 10:39:59 -0500	workerpool0-0	  Epoch  17 | train=10.1859 [cls=1.4699 | reg=6.9813 | ctr=1.7347] | val=445.6521 | lr=4.71e-04 | img=416 | 
INFO	2026-02-21 10:40:11 -0500	workerpool0-0	  Epoch  18 | train=10.1023 [cls=1.4791 | reg=6.8893 | ctr=1.7338] | val=419.5113 | lr=4.13e-04 | img=416 | 
INFO	2026-02-21 10:40:23 -0500	workerpool0-0	  Epoch  19 | train=10.0692 [cls=1.4506 | reg=6.8845 | ctr=1.7341] | val=428.8200 | lr=3.57e-04 | img=416 | 
INFO	2026-02-21 10:40:34 -0500	workerpool0-0	  Epoch  20 | train=8.1179 [cls=1.4677 | reg=4.9162 | ctr=1.7339] | val=273.8959 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-21 10:40:46 -0500	workerpool0-0	  Epoch  21 | train=7.7692 [cls=1.4391 | reg=4.5979 | ctr=1.7322] | val=225.3405 | lr=2.50e-04 | img=320 | ★ best
INFO	2026-02-21 10:40:57 -0500	workerpool0-0	  Epoch  22 | train=7.6661 [cls=1.4037 | reg=4.5321 | ctr=1.7302] | val=221.8418 | lr=2.02e-04 | img=320 | ★ best
INFO	2026-02-21 10:41:08 -0500	workerpool0-0	  Epoch  23 | train=7.5852 [cls=1.3890 | reg=4.4673 | ctr=1.7288] | val=192.6260 | lr=1.57e-04 | img=320 | ★ best
INFO	2026-02-21 10:41:20 -0500	workerpool0-0	  Epoch  24 | train=7.5240 [cls=1.3486 | reg=4.4488 | ctr=1.7266] | val=199.8557 | lr=1.17e-04 | img=320 | 
INFO	2026-02-21 10:41:31 -0500	workerpool0-0	  Epoch  25 | train=7.4925 [cls=1.3488 | reg=4.4165 | ctr=1.7272] | val=186.9726 | lr=8.23e-05 | img=320 | ★ best
INFO	2026-02-21 10:41:42 -0500	workerpool0-0	  Epoch  26 | train=7.5005 [cls=1.3466 | reg=4.4274 | ctr=1.7266] | val=202.0225 | lr=5.33e-05 | img=320 | 
INFO	2026-02-21 10:41:54 -0500	workerpool0-0	  Epoch  27 | train=7.4272 [cls=1.3224 | reg=4.3784 | ctr=1.7265] | val=164.1382 | lr=3.03e-05 | img=320 | ★ best
INFO	2026-02-21 10:42:05 -0500	workerpool0-0	  Epoch  28 | train=7.3633 [cls=1.3200 | reg=4.3198 | ctr=1.7235] | val=194.0059 | lr=1.36e-05 | img=320 | 
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	  Epoch  29 | train=7.3154 [cls=1.3076 | reg=4.2823 | ctr=1.7255] | val=189.3078 | lr=3.48e-06 | img=320 | 
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 7.0 min
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 60 epochs, LR=0.0001
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 10:42:16 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=60)
INFO	2026-02-21 10:42:28 -0500	workerpool0-0	  Epoch  30 | train=6.3223 [cls=1.4618 | reg=3.1079 | ctr=1.7525] | val=153.3625 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 10:42:41 -0500	workerpool0-0	  Epoch  31 | train=5.6042 [cls=1.3377 | reg=2.5268 | ctr=1.7397] | val=128.7185 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 10:42:53 -0500	workerpool0-0	  Epoch  32 | train=5.3156 [cls=1.2321 | reg=2.3505 | ctr=1.7330] | val=89.3257 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 10:43:05 -0500	workerpool0-0	  Epoch  33 | train=5.2108 [cls=1.1935 | reg=2.2869 | ctr=1.7303] | val=73.0919 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 10:43:17 -0500	workerpool0-0	  Epoch  34 | train=5.0625 [cls=1.1407 | reg=2.1935 | ctr=1.7284] | val=83.0626 | lr=9.99e-05 | img=224 | 
INFO	2026-02-21 10:43:29 -0500	workerpool0-0	  Epoch  35 | train=4.9421 [cls=1.1145 | reg=2.1030 | ctr=1.7246] | val=102.3352 | lr=9.97e-05 | img=224 | 
INFO	2026-02-21 10:43:42 -0500	workerpool0-0	  Epoch  36 | train=4.8601 [cls=1.0920 | reg=2.0486 | ctr=1.7195] | val=104.5999 | lr=9.94e-05 | img=224 | 
INFO	2026-02-21 10:43:54 -0500	workerpool0-0	  Epoch  37 | train=4.7125 [cls=1.0309 | reg=1.9647 | ctr=1.7170] | val=90.5316 | lr=9.89e-05 | img=224 | 
INFO	2026-02-21 10:44:06 -0500	workerpool0-0	  Epoch  38 | train=4.6730 [cls=1.0353 | reg=1.9210 | ctr=1.7167] | val=90.4968 | lr=9.83e-05 | img=224 | 
INFO	2026-02-21 10:44:18 -0500	workerpool0-0	  Epoch  39 | train=4.5813 [cls=0.9987 | reg=1.8701 | ctr=1.7125] | val=104.0796 | lr=9.76e-05 | img=224 | 
INFO	2026-02-21 10:44:30 -0500	workerpool0-0	  Epoch  40 | train=4.5021 [cls=0.9659 | reg=1.8263 | ctr=1.7100] | val=81.9096 | lr=9.67e-05 | img=224 | 
INFO	2026-02-21 10:44:42 -0500	workerpool0-0	  Epoch  41 | train=4.4805 [cls=0.9555 | reg=1.8149 | ctr=1.7101] | val=89.5363 | lr=9.57e-05 | img=224 | 
INFO	2026-02-21 10:44:55 -0500	workerpool0-0	  Epoch  42 | train=4.3454 [cls=0.9229 | reg=1.7178 | ctr=1.7048] | val=81.8893 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 10:45:07 -0500	workerpool0-0	  Epoch  43 | train=4.2637 [cls=0.8987 | reg=1.6621 | ctr=1.7029] | val=74.0100 | lr=9.33e-05 | img=224 | 
INFO	2026-02-21 10:45:19 -0500	workerpool0-0	  Epoch  44 | train=4.2582 [cls=0.8880 | reg=1.6687 | ctr=1.7015] | val=53.7381 | lr=9.19e-05 | img=224 | ★ best
INFO	2026-02-21 10:45:31 -0500	workerpool0-0	  Epoch  45 | train=4.2064 [cls=0.8783 | reg=1.6259 | ctr=1.7022] | val=68.7864 | lr=9.05e-05 | img=224 | 
INFO	2026-02-21 10:45:44 -0500	workerpool0-0	  Epoch  46 | train=4.1969 [cls=0.8551 | reg=1.6414 | ctr=1.7004] | val=90.8422 | lr=8.89e-05 | img=224 | 
INFO	2026-02-21 10:45:56 -0500	workerpool0-0	  Epoch  47 | train=4.1674 [cls=0.8481 | reg=1.6204 | ctr=1.6989] | val=53.3978 | lr=8.72e-05 | img=224 | ★ best
INFO	2026-02-21 10:46:08 -0500	workerpool0-0	  Epoch  48 | train=4.1123 [cls=0.8443 | reg=1.5700 | ctr=1.6981] | val=41.6047 | lr=8.54e-05 | img=224 | ★ best
INFO	2026-02-21 10:46:20 -0500	workerpool0-0	  Epoch  49 | train=4.0120 [cls=0.8136 | reg=1.5018 | ctr=1.6966] | val=47.3468 | lr=8.35e-05 | img=224 | 
INFO	2026-02-21 10:46:32 -0500	workerpool0-0	  Epoch  50 | train=4.0467 [cls=0.8132 | reg=1.5359 | ctr=1.6976] | val=86.6128 | lr=8.15e-05 | img=224 | 
INFO	2026-02-21 10:46:44 -0500	workerpool0-0	  Epoch  51 | train=4.0657 [cls=0.8135 | reg=1.5562 | ctr=1.6959] | val=69.0442 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 10:46:56 -0500	workerpool0-0	  Epoch  52 | train=3.9283 [cls=0.7812 | reg=1.4551 | ctr=1.6920] | val=36.6910 | lr=7.73e-05 | img=224 | ★ best
INFO	2026-02-21 10:47:08 -0500	workerpool0-0	  Epoch  53 | train=3.9118 [cls=0.7675 | reg=1.4541 | ctr=1.6902] | val=45.4714 | lr=7.50e-05 | img=224 | 
INFO	2026-02-21 10:47:20 -0500	workerpool0-0	  Epoch  54 | train=3.9126 [cls=0.7708 | reg=1.4492 | ctr=1.6926] | val=41.6447 | lr=7.27e-05 | img=224 | 
INFO	2026-02-21 10:47:33 -0500	workerpool0-0	  Epoch  55 | train=3.8725 [cls=0.7537 | reg=1.4272 | ctr=1.6916] | val=37.9038 | lr=7.04e-05 | img=224 | 
INFO	2026-02-21 10:47:45 -0500	workerpool0-0	  Epoch  56 | train=3.8292 [cls=0.7558 | reg=1.3844 | ctr=1.6890] | val=38.3161 | lr=6.80e-05 | img=224 | 
INFO	2026-02-21 10:47:57 -0500	workerpool0-0	  Epoch  57 | train=3.7888 [cls=0.7312 | reg=1.3705 | ctr=1.6871] | val=30.7335 | lr=6.55e-05 | img=224 | ★ best
INFO	2026-02-21 10:48:09 -0500	workerpool0-0	  Epoch  58 | train=3.7853 [cls=0.7311 | reg=1.3667 | ctr=1.6875] | val=24.0068 | lr=6.30e-05 | img=224 | ★ best
INFO	2026-02-21 10:48:21 -0500	workerpool0-0	  Epoch  59 | train=3.7269 [cls=0.7147 | reg=1.3256 | ctr=1.6866] | val=39.8302 | lr=6.04e-05 | img=224 | 
INFO	2026-02-21 10:48:33 -0500	workerpool0-0	  Epoch  60 | train=3.7569 [cls=0.7076 | reg=1.3622 | ctr=1.6871] | val=40.4950 | lr=5.79e-05 | img=224 | 
INFO	2026-02-21 10:48:45 -0500	workerpool0-0	  Epoch  61 | train=3.7788 [cls=0.7275 | reg=1.3632 | ctr=1.6881] | val=34.8189 | lr=5.53e-05 | img=224 | 
INFO	2026-02-21 10:48:57 -0500	workerpool0-0	  Epoch  62 | train=3.7001 [cls=0.7028 | reg=1.3135 | ctr=1.6838] | val=41.5574 | lr=5.27e-05 | img=224 | 
INFO	2026-02-21 10:49:09 -0500	workerpool0-0	  Epoch  63 | train=3.6903 [cls=0.6974 | reg=1.3068 | ctr=1.6862] | val=35.6381 | lr=5.01e-05 | img=224 | 
INFO	2026-02-21 10:49:22 -0500	workerpool0-0	  Epoch  64 | train=3.6613 [cls=0.6932 | reg=1.2852 | ctr=1.6829] | val=30.4072 | lr=4.74e-05 | img=224 | 
INFO	2026-02-21 10:49:34 -0500	workerpool0-0	  Epoch  65 | train=3.6593 [cls=0.6822 | reg=1.2920 | ctr=1.6852] | val=43.8740 | lr=4.48e-05 | img=224 | 
INFO	2026-02-21 10:49:46 -0500	workerpool0-0	  Epoch  66 | train=3.6169 [cls=0.6769 | reg=1.2555 | ctr=1.6845] | val=36.2154 | lr=4.22e-05 | img=224 | 
INFO	2026-02-21 10:49:58 -0500	workerpool0-0	  Epoch  67 | train=3.5886 [cls=0.6573 | reg=1.2492 | ctr=1.6821] | val=32.4139 | lr=3.97e-05 | img=224 | 
INFO	2026-02-21 10:50:11 -0500	workerpool0-0	  Epoch  68 | train=3.6362 [cls=0.6696 | reg=1.2820 | ctr=1.6846] | val=37.3412 | lr=3.71e-05 | img=224 | 
INFO	2026-02-21 10:50:23 -0500	workerpool0-0	  Epoch  69 | train=3.6086 [cls=0.6655 | reg=1.2585 | ctr=1.6847] | val=28.6531 | lr=3.46e-05 | img=224 | 
INFO	2026-02-21 10:50:35 -0500	workerpool0-0	  Epoch  70 | train=3.5539 [cls=0.6522 | reg=1.2194 | ctr=1.6824] | val=39.2512 | lr=3.21e-05 | img=224 | 
INFO	2026-02-21 10:50:47 -0500	workerpool0-0	  Epoch  71 | train=3.5586 [cls=0.6545 | reg=1.2233 | ctr=1.6808] | val=36.0808 | lr=2.97e-05 | img=224 | 
INFO	2026-02-21 10:50:59 -0500	workerpool0-0	  Epoch  72 | train=3.5872 [cls=0.6533 | reg=1.2536 | ctr=1.6802] | val=33.4989 | lr=2.74e-05 | img=224 | 
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	  Epoch  73 | train=3.5411 [cls=0.6465 | reg=1.2147 | ctr=1.6799] | val=26.7820 | lr=2.51e-05 | img=224 | 
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	⏹️  Early stopping at epoch 73 (patience=15)
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 8.9 min
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	📊 Entrenamiento completo: 74 epochs
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	   Mejor val_loss: 24.0068 (epoch 58)
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	   Tiempo total: 15.9 min
INFO	2026-02-21 10:51:11 -0500	workerpool0-0	⏱️  Entrenamiento completado en 15.9 min
INFO	2026-02-21 10:51:12 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 10:51:12 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:12 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 10:51:12 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	  Épocas: 74
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	  Mejor val_loss: 24.0068 (epoch 58)
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:14 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  mAP@50:    0.3792
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  mAP@50-95: 0.0000
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  Precision: 0.5345
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  Recall:    0.4312
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  F1-Score:  0.4773
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 623 | GT: 762
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  Avg inference: 5.0 ms
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	    dog                   0.2743
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	    door                  0.4329
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	    obstacle              0.3935
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	    person                0.3871
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	    stair                 0.4083
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	📊 Val mAP@50: 0.3792
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	   dog: 0.2743
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	   door: 0.4329
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	   obstacle: 0.3935
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	   person: 0.3871
INFO	2026-02-21 10:51:15 -0500	workerpool0-0	   stair: 0.4083
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 10:51:16 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  mAP@50:    0.5600
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  mAP@50-95: 0.0000
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  Precision: 0.6049
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  Recall:    0.6271
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  F1-Score:  0.6158
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 585 | GT: 576
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  Avg inference: 4.6 ms
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	    dog                   0.4627
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	    door                  0.5194
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	    obstacle              0.4051
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	    person                0.6355
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	    stair                 0.7774
INFO	2026-02-21 10:51:17 -0500	workerpool0-0	📊 Test mAP@50: 0.5600
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	============================================================
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.7s)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     Latencia mediana: 5.9ms
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 10:51:18 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/config.yaml
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/training_history.csv
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/training_curves.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/class_distribution.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/gt_samples.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/val_confusion_matrix.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/val_per_class.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/val_evaluation.json
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/test_confusion_matrix.png
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/test_evaluation.json
INFO	2026-02-21 10:51:19 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/experiment.json
INFO	2026-02-21 10:51:20 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/checkpoints/best_fcos.pt
INFO	2026-02-21 10:51:20 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/export/fcos_v3s.onnx
INFO	2026-02-21 10:51:20 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 10:51:22 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 10:51:39 -0500	service	Tearing down training program.
INFO	2026-02-21 10:52:24 -0500	service	Finished tearing down training program.
INFO	2026-02-21 10:52:24 -0500	service	Job completed successfully.
```