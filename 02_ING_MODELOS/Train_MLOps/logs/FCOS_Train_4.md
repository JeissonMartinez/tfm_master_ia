# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 4

```zsh
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 12:44:20 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 12:44:20 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 12:44:20 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 12:44:20 -0500	service	Waiting for training program to start.
INFO	2026-02-21 12:44:21 -0500	service	Job is preparing.
INFO	2026-02-21 12:46:56 -0500	service	Job is running.
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-11gtgrmk-209b0071-s9xm
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-c1d886e9de-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771695807.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771695807"]}
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:08 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 12:47:08 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 12:47:08 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 12:47:12 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 12:47:12 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 12:47:16 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:16 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:19 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:19 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 12:47:25 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=80437 sha256=8d8bbebc734c1d02d1c94e2e052bb438b772c39b76f14f45ad1f4c6cf6500392
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 12:47:26 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 12:47:26 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 12:47:27 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 12:47:27 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 12:47:27 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:28 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 12:47:28 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 12:47:31 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 12:47:31 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 12:47:32 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 12:47:32 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 12:47:32 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 12:47:32 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Collecting albumentations>=1.4 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:33 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=1.4->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:34 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 12:47:35 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:35 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 12:47:36 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 12:47:37 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 12:47:38 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 12:47:38 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 12:47:38 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 12:47:38 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 12:47:38 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 12:47:39 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 12:47:39 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:39 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 12:47:41 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:41 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 12:47:41 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:41 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:42 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:43 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 12:47:44 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 12:47:45 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 12:47:46 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 12:47:46 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 12:47:46 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 12:47:46 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 12:47:46 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 140.3 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 92.8 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 113.0 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 171.8 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 157.5 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 141.2 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 22.4 MB/s eta 0:00:00
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 12:47:47 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 92.8 MB/s eta 0:00:00
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 12:47:48 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 12:47:49 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 12:47:49 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=80437 sha256=e211432ad033232cfe6668a4c169e1b3506f88aefa4338762a3390f7837e71a8
INFO	2026-02-21 12:47:49 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 12:47:49 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 12:47:50 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 12:47:50 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:50 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 12:47:52 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:52 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 12:47:52 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:52 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 12:47:57 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:57 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 12:47:57 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:57 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 12:47:59 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 12:47:59 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 12:48:02 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 12:48:02 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 12:48:02 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 12:48:02 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
INFO	2026-02-21 12:48:02 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 12:48:02 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 12:48:03 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771695807.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771695807
INFO	2026-02-21 12:48:06 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:06 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 12:48:06 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771695807.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Patience:     20
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  Conf Thresh:  0.15
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     Phase 2: 80 epochs @ LR=0.0001
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 12:48:31 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 12:48:32 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 12:48:32 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	metadata {
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	}
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	metadata {
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	}
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	]
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 12:48:34 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 12:48:35 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 12:48:36 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 12:48:36 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 12:48:38 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 12:48:38 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:38 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 12:48:38 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 12:48:49 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 12:48:49 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 12:48:49 -0500	workerpool0-0	  5%|▌         | 512k/9.83M [00:00<00:01, 4.91MB/s]
ERROR	2026-02-21 12:48:49 -0500	workerpool0-0	 72%|███████▏  | 7.12M/9.83M [00:00<00:00, 41.7MB/s]
ERROR	2026-02-21 12:48:49 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 45.0MB/s]
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	============================================================
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Phase 2: 80 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 20
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 12:48:50 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-21 12:49:18 -0500	workerpool0-0	  Epoch   0 | train=8.9468 [cls=3.1172 | reg=3.9605 | ctr=1.8691] | val=2659.4966 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 12:49:38 -0500	workerpool0-0	  Epoch   1 | train=7.1962 [cls=2.5659 | reg=2.8338 | ctr=1.7965] | val=995.0554 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 12:49:58 -0500	workerpool0-0	  Epoch   2 | train=6.7399 [cls=2.4100 | reg=2.5456 | ctr=1.7843] | val=954.2536 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 12:50:18 -0500	workerpool0-0	  Epoch   3 | train=6.4143 [cls=2.2371 | reg=2.4023 | ctr=1.7748] | val=771.0601 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 12:50:37 -0500	workerpool0-0	  Epoch   4 | train=6.1661 [cls=2.0503 | reg=2.3435 | ctr=1.7723] | val=1418.8974 | lr=9.97e-04 | img=640 | 
INFO	2026-02-21 12:50:56 -0500	workerpool0-0	  Epoch   5 | train=5.9963 [cls=1.9695 | reg=2.2590 | ctr=1.7677] | val=903.7661 | lr=9.87e-04 | img=640 | 
INFO	2026-02-21 12:51:13 -0500	workerpool0-0	  Epoch   6 | train=5.8968 [cls=1.8873 | reg=2.2433 | ctr=1.7662] | val=1086.1063 | lr=9.70e-04 | img=640 | 
INFO	2026-02-21 12:51:29 -0500	workerpool0-0	  Epoch   7 | train=5.8236 [cls=1.8592 | reg=2.2027 | ctr=1.7618] | val=741.2226 | lr=9.47e-04 | img=640 | ★ best
INFO	2026-02-21 12:51:45 -0500	workerpool0-0	  Epoch   8 | train=5.7514 [cls=1.8203 | reg=2.1725 | ctr=1.7586] | val=972.6061 | lr=9.18e-04 | img=640 | 
INFO	2026-02-21 12:52:02 -0500	workerpool0-0	  Epoch   9 | train=5.7275 [cls=1.7939 | reg=2.1763 | ctr=1.7573] | val=954.0550 | lr=8.83e-04 | img=640 | 
INFO	2026-02-21 12:52:15 -0500	workerpool0-0	  Epoch  10 | train=5.5168 [cls=1.6953 | reg=2.0678 | ctr=1.7537] | val=434.6506 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-21 12:52:28 -0500	workerpool0-0	  Epoch  11 | train=5.2902 [cls=1.6065 | reg=1.9395 | ctr=1.7442] | val=391.9179 | lr=7.99e-04 | img=416 | ★ best
INFO	2026-02-21 12:52:41 -0500	workerpool0-0	  Epoch  12 | train=5.1680 [cls=1.5327 | reg=1.8949 | ctr=1.7405] | val=639.6647 | lr=7.50e-04 | img=416 | 
INFO	2026-02-21 12:52:55 -0500	workerpool0-0	  Epoch  13 | train=5.1280 [cls=1.5163 | reg=1.8738 | ctr=1.7379] | val=386.0489 | lr=6.98e-04 | img=416 | ★ best
INFO	2026-02-21 12:53:09 -0500	workerpool0-0	  Epoch  14 | train=5.1304 [cls=1.4993 | reg=1.8888 | ctr=1.7423] | val=423.6888 | lr=6.43e-04 | img=416 | 
INFO	2026-02-21 12:53:22 -0500	workerpool0-0	  Epoch  15 | train=5.0445 [cls=1.4431 | reg=1.8654 | ctr=1.7361] | val=327.0653 | lr=5.87e-04 | img=416 | ★ best
INFO	2026-02-21 12:53:35 -0500	workerpool0-0	  Epoch  16 | train=4.9853 [cls=1.4317 | reg=1.8188 | ctr=1.7348] | val=562.2242 | lr=5.29e-04 | img=416 | 
INFO	2026-02-21 12:53:48 -0500	workerpool0-0	  Epoch  17 | train=4.9229 [cls=1.3841 | reg=1.8045 | ctr=1.7342] | val=303.9218 | lr=4.71e-04 | img=416 | ★ best
INFO	2026-02-21 12:54:01 -0500	workerpool0-0	  Epoch  18 | train=4.8903 [cls=1.3646 | reg=1.7947 | ctr=1.7310] | val=356.5001 | lr=4.13e-04 | img=416 | 
INFO	2026-02-21 12:54:15 -0500	workerpool0-0	  Epoch  19 | train=4.8504 [cls=1.3369 | reg=1.7832 | ctr=1.7303] | val=378.1592 | lr=3.57e-04 | img=416 | 
INFO	2026-02-21 12:54:28 -0500	workerpool0-0	  Epoch  20 | train=4.8696 [cls=1.3623 | reg=1.7758 | ctr=1.7314] | val=209.3018 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-21 12:54:40 -0500	workerpool0-0	  Epoch  21 | train=4.8082 [cls=1.3418 | reg=1.7375 | ctr=1.7289] | val=160.3295 | lr=2.50e-04 | img=320 | ★ best
INFO	2026-02-21 12:54:53 -0500	workerpool0-0	  Epoch  22 | train=4.7305 [cls=1.2928 | reg=1.7117 | ctr=1.7260] | val=131.9033 | lr=2.02e-04 | img=320 | ★ best
INFO	2026-02-21 12:55:05 -0500	workerpool0-0	  Epoch  23 | train=4.7308 [cls=1.3008 | reg=1.7038 | ctr=1.7263] | val=170.0520 | lr=1.57e-04 | img=320 | 
INFO	2026-02-21 12:55:17 -0500	workerpool0-0	  Epoch  24 | train=4.7048 [cls=1.2762 | reg=1.7021 | ctr=1.7265] | val=103.4694 | lr=1.17e-04 | img=320 | ★ best
INFO	2026-02-21 12:55:30 -0500	workerpool0-0	  Epoch  25 | train=4.6485 [cls=1.2497 | reg=1.6769 | ctr=1.7219] | val=138.5846 | lr=8.23e-05 | img=320 | 
INFO	2026-02-21 12:55:42 -0500	workerpool0-0	  Epoch  26 | train=4.6485 [cls=1.2492 | reg=1.6751 | ctr=1.7242] | val=143.8620 | lr=5.33e-05 | img=320 | 
INFO	2026-02-21 12:55:54 -0500	workerpool0-0	  Epoch  27 | train=4.6358 [cls=1.2510 | reg=1.6630 | ctr=1.7217] | val=137.3299 | lr=3.03e-05 | img=320 | 
INFO	2026-02-21 12:56:07 -0500	workerpool0-0	  Epoch  28 | train=4.6143 [cls=1.2333 | reg=1.6591 | ctr=1.7220] | val=132.0357 | lr=1.36e-05 | img=320 | 
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	  Epoch  29 | train=4.6102 [cls=1.2169 | reg=1.6702 | ctr=1.7232] | val=134.6712 | lr=3.48e-06 | img=320 | 
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 7.5 min
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=0.0001
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 12:56:19 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=80)
INFO	2026-02-21 12:56:33 -0500	workerpool0-0	  Epoch  30 | train=4.9683 [cls=1.4012 | reg=1.8168 | ctr=1.7503] | val=86.5997 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 12:56:46 -0500	workerpool0-0	  Epoch  31 | train=4.6648 [cls=1.2442 | reg=1.6826 | ctr=1.7381] | val=68.1492 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 12:57:00 -0500	workerpool0-0	  Epoch  32 | train=4.5585 [cls=1.1859 | reg=1.6366 | ctr=1.7360] | val=69.2322 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 12:57:13 -0500	workerpool0-0	  Epoch  33 | train=4.4698 [cls=1.1381 | reg=1.6005 | ctr=1.7311] | val=61.2703 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 12:57:27 -0500	workerpool0-0	  Epoch  34 | train=4.3828 [cls=1.1022 | reg=1.5530 | ctr=1.7276] | val=38.1871 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 12:57:40 -0500	workerpool0-0	  Epoch  35 | train=4.3160 [cls=1.0550 | reg=1.5383 | ctr=1.7226] | val=42.6694 | lr=9.98e-05 | img=224 | 
INFO	2026-02-21 12:57:54 -0500	workerpool0-0	  Epoch  36 | train=4.2492 [cls=1.0282 | reg=1.5009 | ctr=1.7200] | val=68.3459 | lr=9.97e-05 | img=224 | 
INFO	2026-02-21 12:58:08 -0500	workerpool0-0	  Epoch  37 | train=4.1928 [cls=0.9955 | reg=1.4803 | ctr=1.7170] | val=48.6567 | lr=9.94e-05 | img=224 | 
INFO	2026-02-21 12:58:21 -0500	workerpool0-0	  Epoch  38 | train=4.1199 [cls=0.9506 | reg=1.4535 | ctr=1.7158] | val=63.5153 | lr=9.90e-05 | img=224 | 
INFO	2026-02-21 12:58:34 -0500	workerpool0-0	  Epoch  39 | train=4.0938 [cls=0.9411 | reg=1.4382 | ctr=1.7145] | val=43.7897 | lr=9.86e-05 | img=224 | 
INFO	2026-02-21 12:58:48 -0500	workerpool0-0	  Epoch  40 | train=4.0403 [cls=0.9159 | reg=1.4148 | ctr=1.7097] | val=58.3370 | lr=9.81e-05 | img=224 | 
INFO	2026-02-21 12:59:01 -0500	workerpool0-0	  Epoch  41 | train=3.9907 [cls=0.8865 | reg=1.3964 | ctr=1.7077] | val=51.5132 | lr=9.76e-05 | img=224 | 
INFO	2026-02-21 12:59:14 -0500	workerpool0-0	  Epoch  42 | train=3.9442 [cls=0.8625 | reg=1.3763 | ctr=1.7055] | val=54.2747 | lr=9.69e-05 | img=224 | 
INFO	2026-02-21 12:59:27 -0500	workerpool0-0	  Epoch  43 | train=3.9159 [cls=0.8522 | reg=1.3605 | ctr=1.7032] | val=49.7117 | lr=9.62e-05 | img=224 | 
INFO	2026-02-21 12:59:40 -0500	workerpool0-0	  Epoch  44 | train=3.8899 [cls=0.8347 | reg=1.3497 | ctr=1.7055] | val=38.4884 | lr=9.54e-05 | img=224 | 
INFO	2026-02-21 12:59:54 -0500	workerpool0-0	  Epoch  45 | train=3.8361 [cls=0.8076 | reg=1.3266 | ctr=1.7019] | val=43.2143 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 13:00:07 -0500	workerpool0-0	  Epoch  46 | train=3.7885 [cls=0.7811 | reg=1.3092 | ctr=1.6982] | val=44.7987 | lr=9.36e-05 | img=224 | 
INFO	2026-02-21 13:00:20 -0500	workerpool0-0	  Epoch  47 | train=3.7867 [cls=0.7805 | reg=1.3071 | ctr=1.6990] | val=46.5041 | lr=9.26e-05 | img=224 | 
INFO	2026-02-21 13:00:33 -0500	workerpool0-0	  Epoch  48 | train=3.7802 [cls=0.7760 | reg=1.3057 | ctr=1.6986] | val=50.6399 | lr=9.16e-05 | img=224 | 
INFO	2026-02-21 13:00:47 -0500	workerpool0-0	  Epoch  49 | train=3.7219 [cls=0.7469 | reg=1.2794 | ctr=1.6956] | val=39.5812 | lr=9.05e-05 | img=224 | 
INFO	2026-02-21 13:01:00 -0500	workerpool0-0	  Epoch  50 | train=3.7091 [cls=0.7411 | reg=1.2738 | ctr=1.6943] | val=36.6089 | lr=8.93e-05 | img=224 | ★ best
INFO	2026-02-21 13:01:13 -0500	workerpool0-0	  Epoch  51 | train=3.6972 [cls=0.7308 | reg=1.2733 | ctr=1.6930] | val=56.3001 | lr=8.80e-05 | img=224 | 
INFO	2026-02-21 13:01:26 -0500	workerpool0-0	  Epoch  52 | train=3.6656 [cls=0.7169 | reg=1.2563 | ctr=1.6924] | val=41.7767 | lr=8.67e-05 | img=224 | 
INFO	2026-02-21 13:01:40 -0500	workerpool0-0	  Epoch  53 | train=3.6577 [cls=0.7141 | reg=1.2504 | ctr=1.6932] | val=60.6186 | lr=8.54e-05 | img=224 | 
INFO	2026-02-21 13:01:54 -0500	workerpool0-0	  Epoch  54 | train=3.6185 [cls=0.6975 | reg=1.2296 | ctr=1.6914] | val=55.9250 | lr=8.40e-05 | img=224 | 
INFO	2026-02-21 13:02:07 -0500	workerpool0-0	  Epoch  55 | train=3.6075 [cls=0.6877 | reg=1.2308 | ctr=1.6891] | val=50.8381 | lr=8.25e-05 | img=224 | 
INFO	2026-02-21 13:02:20 -0500	workerpool0-0	  Epoch  56 | train=3.5682 [cls=0.6663 | reg=1.2142 | ctr=1.6877] | val=36.4576 | lr=8.10e-05 | img=224 | ★ best
INFO	2026-02-21 13:02:33 -0500	workerpool0-0	  Epoch  57 | train=3.5548 [cls=0.6632 | reg=1.2052 | ctr=1.6864] | val=70.1327 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 13:02:47 -0500	workerpool0-0	  Epoch  58 | train=3.5538 [cls=0.6681 | reg=1.1986 | ctr=1.6871] | val=61.7622 | lr=7.78e-05 | img=224 | 
INFO	2026-02-21 13:03:00 -0500	workerpool0-0	  Epoch  59 | train=3.5267 [cls=0.6575 | reg=1.1825 | ctr=1.6867] | val=46.0785 | lr=7.61e-05 | img=224 | 
INFO	2026-02-21 13:03:14 -0500	workerpool0-0	  Epoch  60 | train=3.4963 [cls=0.6439 | reg=1.1667 | ctr=1.6857] | val=42.7658 | lr=7.45e-05 | img=224 | 
INFO	2026-02-21 13:03:27 -0500	workerpool0-0	  Epoch  61 | train=3.4850 [cls=0.6320 | reg=1.1684 | ctr=1.6846] | val=59.1642 | lr=7.27e-05 | img=224 | 
INFO	2026-02-21 13:03:40 -0500	workerpool0-0	  Epoch  62 | train=3.4876 [cls=0.6365 | reg=1.1653 | ctr=1.6858] | val=46.1354 | lr=7.10e-05 | img=224 | 
INFO	2026-02-21 13:03:53 -0500	workerpool0-0	  Epoch  63 | train=3.4843 [cls=0.6388 | reg=1.1629 | ctr=1.6826] | val=50.2727 | lr=6.92e-05 | img=224 | 
INFO	2026-02-21 13:04:07 -0500	workerpool0-0	  Epoch  64 | train=3.4631 [cls=0.6246 | reg=1.1591 | ctr=1.6794] | val=57.8364 | lr=6.73e-05 | img=224 | 
INFO	2026-02-21 13:04:20 -0500	workerpool0-0	  Epoch  65 | train=3.4287 [cls=0.6102 | reg=1.1382 | ctr=1.6803] | val=53.5584 | lr=6.55e-05 | img=224 | 
INFO	2026-02-21 13:04:33 -0500	workerpool0-0	  Epoch  66 | train=3.4334 [cls=0.6095 | reg=1.1429 | ctr=1.6811] | val=51.7146 | lr=6.36e-05 | img=224 | 
INFO	2026-02-21 13:04:46 -0500	workerpool0-0	  Epoch  67 | train=3.4199 [cls=0.6039 | reg=1.1349 | ctr=1.6812] | val=60.2615 | lr=6.17e-05 | img=224 | 
INFO	2026-02-21 13:04:59 -0500	workerpool0-0	  Epoch  68 | train=3.4045 [cls=0.5976 | reg=1.1278 | ctr=1.6791] | val=53.0390 | lr=5.98e-05 | img=224 | 
INFO	2026-02-21 13:05:12 -0500	workerpool0-0	  Epoch  69 | train=3.3890 [cls=0.5923 | reg=1.1181 | ctr=1.6787] | val=51.3936 | lr=5.79e-05 | img=224 | 
INFO	2026-02-21 13:05:25 -0500	workerpool0-0	  Epoch  70 | train=3.3875 [cls=0.5866 | reg=1.1236 | ctr=1.6773] | val=54.1834 | lr=5.59e-05 | img=224 | 
INFO	2026-02-21 13:05:39 -0500	workerpool0-0	  Epoch  71 | train=3.3765 [cls=0.5827 | reg=1.1146 | ctr=1.6793] | val=67.2205 | lr=5.40e-05 | img=224 | 
INFO	2026-02-21 13:05:52 -0500	workerpool0-0	  Epoch  72 | train=3.3594 [cls=0.5754 | reg=1.1076 | ctr=1.6764] | val=66.0198 | lr=5.20e-05 | img=224 | 
INFO	2026-02-21 13:06:05 -0500	workerpool0-0	  Epoch  73 | train=3.3596 [cls=0.5729 | reg=1.1100 | ctr=1.6766] | val=55.4425 | lr=5.01e-05 | img=224 | 
INFO	2026-02-21 13:06:19 -0500	workerpool0-0	  Epoch  74 | train=3.3318 [cls=0.5673 | reg=1.0889 | ctr=1.6756] | val=53.4914 | lr=4.81e-05 | img=224 | 
INFO	2026-02-21 13:06:32 -0500	workerpool0-0	  Epoch  75 | train=3.3334 [cls=0.5601 | reg=1.0960 | ctr=1.6774] | val=38.3067 | lr=4.61e-05 | img=224 | 
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	  Epoch  76 | train=3.3235 [cls=0.5608 | reg=1.0869 | ctr=1.6758] | val=45.3457 | lr=4.42e-05 | img=224 | 
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	⏹️  Early stopping at epoch 76 (patience=20)
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 10.4 min
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	📊 Entrenamiento completo: 77 epochs
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	   Mejor val_loss: 36.4576 (epoch 56)
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	   Tiempo total: 17.9 min
INFO	2026-02-21 13:06:45 -0500	workerpool0-0	⏱️  Entrenamiento completado en 17.9 min
INFO	2026-02-21 13:06:46 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 13:06:46 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:46 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 13:06:46 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	  Épocas: 77
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	  Mejor val_loss: 36.4576 (epoch 56)
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:48 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  mAP@50:    0.4178
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  mAP@50-95: 0.1843
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  Precision: 0.3189
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  Recall:    0.5115
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  F1-Score:  0.3928
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 1224 | GT: 762
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  Avg inference: 5.9 ms
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	    dog                   0.3138
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	    door                  0.4459
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	    obstacle              0.4088
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	    person                0.4531
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	    stair                 0.4673
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	📊 Val mAP@50: 0.4178
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	   dog: 0.3138
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	   door: 0.4459
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	   obstacle: 0.4088
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	   person: 0.4531
INFO	2026-02-21 13:06:49 -0500	workerpool0-0	   stair: 0.4673
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 13:06:50 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  mAP@50:    0.5936
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  mAP@50-95: 0.2644
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  Precision: 0.3462
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  Recall:    0.6886
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  F1-Score:  0.4607
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 1120 | GT: 576
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  Avg inference: 5.4 ms
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	    dog                   0.5021
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	    door                  0.5334
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	    obstacle              0.5116
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	    person                0.6816
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	    stair                 0.7394
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	📊 Test mAP@50: 0.5936
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	============================================================
INFO	2026-02-21 13:06:52 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.7s)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     Latencia mediana: 5.9ms
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/config.yaml
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/training_history.csv
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/training_curves.png
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/class_distribution.png
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/gt_samples.png
INFO	2026-02-21 13:06:53 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/val_confusion_matrix.png
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/val_per_class.png
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/val_evaluation.json
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/test_confusion_matrix.png
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/test_evaluation.json
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/experiment.json
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/checkpoints/best_fcos.pt
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/export/fcos_v3s.onnx
INFO	2026-02-21 13:06:54 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 13:06:57 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 13:07:02 -0500	service	Tearing down training program.
INFO	2026-02-21 13:07:40 -0500	service	Finished tearing down training program.
INFO	2026-02-21 13:07:40 -0500	service	Job completed successfully.
```