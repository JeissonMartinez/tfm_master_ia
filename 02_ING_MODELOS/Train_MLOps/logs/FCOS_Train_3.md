# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 3

```zsh
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 11:20:14 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 11:20:14 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 11:20:14 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 11:20:14 -0500	service	Waiting for training program to start.
INFO	2026-02-21 11:20:15 -0500	service	Job is preparing.
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-9qb9pr6v-65c711e3-lc9r
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-e5b9df8b1f-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771690809.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771690809"]}
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:11 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 11:23:12 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 11:23:12 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 11:23:15 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 11:23:15 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 11:23:19 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:19 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:20 -0500	service	Job is running.
INFO	2026-02-21 11:23:22 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:22 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 11:23:27 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 11:23:27 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 11:23:27 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 11:23:27 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=79902 sha256=885d5c380d46a1a42a147b5a143707a1cab1f2c801fae088ac19f169e19a54ae
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 11:23:28 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 11:23:29 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 11:23:29 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 11:23:29 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 11:23:29 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 11:23:29 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:30 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 11:23:30 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 11:23:33 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 11:23:33 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 11:23:34 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 11:23:34 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 11:23:34 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 11:23:34 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 11:23:35 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:35 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:36 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 11:23:37 -0500	workerpool0-0	Collecting albumentations>=1.4 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:37 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 11:23:37 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 11:23:37 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 11:23:37 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=1.4->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:38 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 11:23:39 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:39 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 11:23:40 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:40 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 11:23:41 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 11:23:42 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:43 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:44 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 11:23:46 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 11:23:47 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 109.0 MB/s eta 0:00:00
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 95.3 MB/s eta 0:00:00
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 100.6 MB/s eta 0:00:00
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 126.4 MB/s eta 0:00:00
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 146.4 MB/s eta 0:00:00
INFO	2026-02-21 11:23:48 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 133.8 MB/s eta 0:00:00
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 31.7 MB/s eta 0:00:00
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 103.1 MB/s eta 0:00:00
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 11:23:49 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 11:23:50 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 11:23:50 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=79902 sha256=36acbaa30cca427981f85e4c0377fb6340552941e82ba85e731f4ccaf2e61328
INFO	2026-02-21 11:23:50 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 11:23:50 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 11:23:51 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 11:23:51 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:23:51 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 11:23:54 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:23:54 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 11:23:54 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:23:54 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 11:23:59 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:23:59 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 11:23:59 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:23:59 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 11:24:01 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 11:24:01 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 11:24:06 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 11:24:06 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 11:24:06 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 11:24:06 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
INFO	2026-02-21 11:24:07 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 11:24:07 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 11:24:09 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771690809.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771690809
INFO	2026-02-21 11:24:12 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:12 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 11:24:12 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771690809.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Patience:     20
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  Conf Thresh:  0.25
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     Phase 2: 80 epochs @ LR=0.0001
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 11:24:35 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 11:24:36 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 11:24:36 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	metadata {
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	}
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	metadata {
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	}
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	]
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 11:24:38 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 11:24:39 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 11:24:40 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 11:24:40 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 11:24:42 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 11:24:42 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:42 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 11:24:42 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 11:24:51 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 11:24:52 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 11:24:52 -0500	workerpool0-0	  5%|▌         | 512k/9.83M [00:00<00:02, 4.81MB/s]
ERROR	2026-02-21 11:24:52 -0500	workerpool0-0	 69%|██████▊   | 6.75M/9.83M [00:00<00:00, 39.2MB/s]
ERROR	2026-02-21 11:24:52 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 43.9MB/s]
INFO	2026-02-21 11:24:52 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 11:24:52 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Phase 2: 80 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 20
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 11:24:53 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-21 11:25:21 -0500	workerpool0-0	  Epoch   0 | train=9.4117 [cls=3.0551 | reg=4.5001 | ctr=1.8565] | val=1434.1345 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 11:25:41 -0500	workerpool0-0	  Epoch   1 | train=8.8412 [cls=2.5454 | reg=4.5000 | ctr=1.7958] | val=1247.7807 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 11:26:01 -0500	workerpool0-0	  Epoch   2 | train=8.6620 [cls=2.3715 | reg=4.5000 | ctr=1.7905] | val=1322.4485 | lr=1.00e-03 | img=640 | 
INFO	2026-02-21 11:26:21 -0500	workerpool0-0	  Epoch   3 | train=8.5238 [cls=2.2419 | reg=4.5000 | ctr=1.7819] | val=898.7998 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 11:26:41 -0500	workerpool0-0	  Epoch   4 | train=8.4404 [cls=2.1643 | reg=4.5000 | ctr=1.7761] | val=1090.4647 | lr=9.97e-04 | img=640 | 
INFO	2026-02-21 11:27:00 -0500	workerpool0-0	  Epoch   5 | train=8.3039 [cls=2.0324 | reg=4.5000 | ctr=1.7715] | val=1311.8631 | lr=9.87e-04 | img=640 | 
INFO	2026-02-21 11:27:18 -0500	workerpool0-0	  Epoch   6 | train=8.2042 [cls=1.9338 | reg=4.5000 | ctr=1.7703] | val=1632.2675 | lr=9.70e-04 | img=640 | 
INFO	2026-02-21 11:27:35 -0500	workerpool0-0	  Epoch   7 | train=8.1554 [cls=1.8892 | reg=4.5000 | ctr=1.7662] | val=857.3488 | lr=9.47e-04 | img=640 | ★ best
INFO	2026-02-21 11:27:51 -0500	workerpool0-0	  Epoch   8 | train=8.0691 [cls=1.8050 | reg=4.5000 | ctr=1.7641] | val=974.5837 | lr=9.18e-04 | img=640 | 
INFO	2026-02-21 11:28:08 -0500	workerpool0-0	  Epoch   9 | train=8.0308 [cls=1.7674 | reg=4.5000 | ctr=1.7634] | val=1378.4736 | lr=8.83e-04 | img=640 | 
INFO	2026-02-21 11:28:21 -0500	workerpool0-0	  Epoch  10 | train=7.9104 [cls=1.6534 | reg=4.5000 | ctr=1.7571] | val=556.2519 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-21 11:28:34 -0500	workerpool0-0	  Epoch  11 | train=7.8480 [cls=1.5946 | reg=4.5000 | ctr=1.7535] | val=674.9317 | lr=7.99e-04 | img=416 | 
INFO	2026-02-21 11:28:47 -0500	workerpool0-0	  Epoch  12 | train=7.8127 [cls=1.5634 | reg=4.5000 | ctr=1.7493] | val=425.2703 | lr=7.50e-04 | img=416 | ★ best
INFO	2026-02-21 11:29:01 -0500	workerpool0-0	  Epoch  13 | train=7.7850 [cls=1.5366 | reg=4.5000 | ctr=1.7484] | val=480.9552 | lr=6.98e-04 | img=416 | 
INFO	2026-02-21 11:29:14 -0500	workerpool0-0	  Epoch  14 | train=6.3705 [cls=1.4928 | reg=3.1323 | ctr=1.7454] | val=304.8273 | lr=6.43e-04 | img=416 | ★ best
INFO	2026-02-21 11:29:27 -0500	workerpool0-0	  Epoch  15 | train=5.5106 [cls=1.4471 | reg=2.3204 | ctr=1.7431] | val=394.0569 | lr=5.87e-04 | img=416 | 
INFO	2026-02-21 11:29:40 -0500	workerpool0-0	  Epoch  16 | train=5.3439 [cls=1.4395 | reg=2.1628 | ctr=1.7416] | val=461.1721 | lr=5.29e-04 | img=416 | 
INFO	2026-02-21 11:29:54 -0500	workerpool0-0	  Epoch  17 | train=5.2661 [cls=1.4132 | reg=2.1127 | ctr=1.7403] | val=461.0953 | lr=4.71e-04 | img=416 | 
INFO	2026-02-21 11:30:07 -0500	workerpool0-0	  Epoch  18 | train=5.2097 [cls=1.4182 | reg=2.0541 | ctr=1.7374] | val=578.4338 | lr=4.13e-04 | img=416 | 
INFO	2026-02-21 11:30:20 -0500	workerpool0-0	  Epoch  19 | train=5.1114 [cls=1.3658 | reg=2.0093 | ctr=1.7362] | val=286.2885 | lr=3.57e-04 | img=416 | ★ best
INFO	2026-02-21 11:30:33 -0500	workerpool0-0	  Epoch  20 | train=5.0911 [cls=1.3901 | reg=1.9641 | ctr=1.7369] | val=218.6010 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-21 11:30:45 -0500	workerpool0-0	  Epoch  21 | train=4.9615 [cls=1.3316 | reg=1.8991 | ctr=1.7307] | val=198.1118 | lr=2.50e-04 | img=320 | ★ best
INFO	2026-02-21 11:30:58 -0500	workerpool0-0	  Epoch  22 | train=4.9368 [cls=1.3131 | reg=1.8916 | ctr=1.7322] | val=182.3396 | lr=2.02e-04 | img=320 | ★ best
INFO	2026-02-21 11:31:11 -0500	workerpool0-0	  Epoch  23 | train=4.8935 [cls=1.3038 | reg=1.8607 | ctr=1.7290] | val=181.4920 | lr=1.57e-04 | img=320 | ★ best
INFO	2026-02-21 11:31:23 -0500	workerpool0-0	  Epoch  24 | train=4.8494 [cls=1.2833 | reg=1.8374 | ctr=1.7288] | val=134.2776 | lr=1.17e-04 | img=320 | ★ best
INFO	2026-02-21 11:31:36 -0500	workerpool0-0	  Epoch  25 | train=4.8367 [cls=1.2748 | reg=1.8323 | ctr=1.7297] | val=174.0702 | lr=8.23e-05 | img=320 | 
INFO	2026-02-21 11:31:48 -0500	workerpool0-0	  Epoch  26 | train=4.8297 [cls=1.2612 | reg=1.8398 | ctr=1.7287] | val=183.7018 | lr=5.33e-05 | img=320 | 
INFO	2026-02-21 11:32:01 -0500	workerpool0-0	  Epoch  27 | train=4.8004 [cls=1.2415 | reg=1.8293 | ctr=1.7296] | val=142.2836 | lr=3.03e-05 | img=320 | 
INFO	2026-02-21 11:32:13 -0500	workerpool0-0	  Epoch  28 | train=4.8255 [cls=1.2686 | reg=1.8294 | ctr=1.7274] | val=155.8250 | lr=1.36e-05 | img=320 | 
INFO	2026-02-21 11:32:25 -0500	workerpool0-0	  Epoch  29 | train=4.7954 [cls=1.2513 | reg=1.8158 | ctr=1.7284] | val=150.3285 | lr=3.48e-06 | img=320 | 
INFO	2026-02-21 11:32:25 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 7.5 min
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=0.0001
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 11:32:26 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=80)
INFO	2026-02-21 11:32:39 -0500	workerpool0-0	  Epoch  30 | train=5.0865 [cls=1.4075 | reg=1.9222 | ctr=1.7568] | val=99.1772 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 11:32:53 -0500	workerpool0-0	  Epoch  31 | train=4.8180 [cls=1.2787 | reg=1.7911 | ctr=1.7482] | val=108.5478 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 11:33:06 -0500	workerpool0-0	  Epoch  32 | train=4.6615 [cls=1.1887 | reg=1.7338 | ctr=1.7391] | val=87.0127 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 11:33:20 -0500	workerpool0-0	  Epoch  33 | train=4.5569 [cls=1.1324 | reg=1.6873 | ctr=1.7372] | val=63.4802 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 11:33:33 -0500	workerpool0-0	  Epoch  34 | train=4.4675 [cls=1.0825 | reg=1.6536 | ctr=1.7314] | val=59.0888 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 11:33:47 -0500	workerpool0-0	  Epoch  35 | train=4.4385 [cls=1.0719 | reg=1.6402 | ctr=1.7263] | val=62.0535 | lr=9.98e-05 | img=224 | 
INFO	2026-02-21 11:34:00 -0500	workerpool0-0	  Epoch  36 | train=4.3576 [cls=1.0290 | reg=1.6037 | ctr=1.7249] | val=75.0883 | lr=9.97e-05 | img=224 | 
INFO	2026-02-21 11:34:14 -0500	workerpool0-0	  Epoch  37 | train=4.3132 [cls=1.0113 | reg=1.5785 | ctr=1.7234] | val=66.6089 | lr=9.94e-05 | img=224 | 
INFO	2026-02-21 11:34:27 -0500	workerpool0-0	  Epoch  38 | train=4.2478 [cls=0.9668 | reg=1.5609 | ctr=1.7201] | val=81.9033 | lr=9.90e-05 | img=224 | 
INFO	2026-02-21 11:34:41 -0500	workerpool0-0	  Epoch  39 | train=4.1703 [cls=0.9244 | reg=1.5282 | ctr=1.7178] | val=102.1955 | lr=9.86e-05 | img=224 | 
INFO	2026-02-21 11:34:54 -0500	workerpool0-0	  Epoch  40 | train=4.1389 [cls=0.9176 | reg=1.5068 | ctr=1.7145] | val=98.3034 | lr=9.81e-05 | img=224 | 
INFO	2026-02-21 11:35:08 -0500	workerpool0-0	  Epoch  41 | train=4.1116 [cls=0.8951 | reg=1.5032 | ctr=1.7133] | val=70.5338 | lr=9.76e-05 | img=224 | 
INFO	2026-02-21 11:35:21 -0500	workerpool0-0	  Epoch  42 | train=4.0837 [cls=0.8892 | reg=1.4829 | ctr=1.7116] | val=55.1901 | lr=9.69e-05 | img=224 | ★ best
INFO	2026-02-21 11:35:35 -0500	workerpool0-0	  Epoch  43 | train=4.0064 [cls=0.8432 | reg=1.4527 | ctr=1.7104] | val=48.0324 | lr=9.62e-05 | img=224 | ★ best
INFO	2026-02-21 11:35:48 -0500	workerpool0-0	  Epoch  44 | train=3.9887 [cls=0.8448 | reg=1.4366 | ctr=1.7073] | val=62.2091 | lr=9.54e-05 | img=224 | 
INFO	2026-02-21 11:36:02 -0500	workerpool0-0	  Epoch  45 | train=3.9317 [cls=0.8062 | reg=1.4216 | ctr=1.7039] | val=79.9831 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 11:36:15 -0500	workerpool0-0	  Epoch  46 | train=3.8842 [cls=0.7885 | reg=1.3947 | ctr=1.7010] | val=60.7627 | lr=9.36e-05 | img=224 | 
INFO	2026-02-21 11:36:29 -0500	workerpool0-0	  Epoch  47 | train=3.9018 [cls=0.8048 | reg=1.3955 | ctr=1.7016] | val=51.6777 | lr=9.26e-05 | img=224 | 
INFO	2026-02-21 11:36:42 -0500	workerpool0-0	  Epoch  48 | train=3.8465 [cls=0.7716 | reg=1.3766 | ctr=1.6983] | val=47.8699 | lr=9.16e-05 | img=224 | ★ best
INFO	2026-02-21 11:36:56 -0500	workerpool0-0	  Epoch  49 | train=3.8006 [cls=0.7481 | reg=1.3523 | ctr=1.7002] | val=38.4928 | lr=9.05e-05 | img=224 | ★ best
INFO	2026-02-21 11:37:10 -0500	workerpool0-0	  Epoch  50 | train=3.8221 [cls=0.7624 | reg=1.3602 | ctr=1.6995] | val=34.6629 | lr=8.93e-05 | img=224 | ★ best
INFO	2026-02-21 11:37:23 -0500	workerpool0-0	  Epoch  51 | train=3.7732 [cls=0.7331 | reg=1.3424 | ctr=1.6976] | val=51.6776 | lr=8.80e-05 | img=224 | 
INFO	2026-02-21 11:37:37 -0500	workerpool0-0	  Epoch  52 | train=3.7573 [cls=0.7203 | reg=1.3405 | ctr=1.6966] | val=45.4824 | lr=8.67e-05 | img=224 | 
INFO	2026-02-21 11:37:50 -0500	workerpool0-0	  Epoch  53 | train=3.7259 [cls=0.7134 | reg=1.3182 | ctr=1.6943] | val=65.6085 | lr=8.54e-05 | img=224 | 
INFO	2026-02-21 11:38:04 -0500	workerpool0-0	  Epoch  54 | train=3.7334 [cls=0.7194 | reg=1.3200 | ctr=1.6940] | val=49.2591 | lr=8.40e-05 | img=224 | 
INFO	2026-02-21 11:38:17 -0500	workerpool0-0	  Epoch  55 | train=3.7010 [cls=0.7084 | reg=1.3008 | ctr=1.6918] | val=35.7827 | lr=8.25e-05 | img=224 | 
INFO	2026-02-21 11:38:31 -0500	workerpool0-0	  Epoch  56 | train=3.6793 [cls=0.6957 | reg=1.2912 | ctr=1.6925] | val=66.3978 | lr=8.10e-05 | img=224 | 
INFO	2026-02-21 11:38:44 -0500	workerpool0-0	  Epoch  57 | train=3.6285 [cls=0.6667 | reg=1.2717 | ctr=1.6902] | val=44.4211 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 11:38:57 -0500	workerpool0-0	  Epoch  58 | train=3.6550 [cls=0.6834 | reg=1.2789 | ctr=1.6927] | val=39.5562 | lr=7.78e-05 | img=224 | 
INFO	2026-02-21 11:39:11 -0500	workerpool0-0	  Epoch  59 | train=3.6272 [cls=0.6742 | reg=1.2648 | ctr=1.6882] | val=36.4294 | lr=7.61e-05 | img=224 | 
INFO	2026-02-21 11:39:24 -0500	workerpool0-0	  Epoch  60 | train=3.6048 [cls=0.6572 | reg=1.2589 | ctr=1.6887] | val=54.0157 | lr=7.45e-05 | img=224 | 
INFO	2026-02-21 11:39:38 -0500	workerpool0-0	  Epoch  61 | train=3.5869 [cls=0.6544 | reg=1.2462 | ctr=1.6864] | val=38.4052 | lr=7.27e-05 | img=224 | 
INFO	2026-02-21 11:39:51 -0500	workerpool0-0	  Epoch  62 | train=3.5508 [cls=0.6377 | reg=1.2288 | ctr=1.6842] | val=34.2935 | lr=7.10e-05 | img=224 | ★ best
INFO	2026-02-21 11:40:05 -0500	workerpool0-0	  Epoch  63 | train=3.5471 [cls=0.6383 | reg=1.2226 | ctr=1.6861] | val=41.0698 | lr=6.92e-05 | img=224 | 
INFO	2026-02-21 11:40:19 -0500	workerpool0-0	  Epoch  64 | train=3.5405 [cls=0.6324 | reg=1.2220 | ctr=1.6861] | val=39.5189 | lr=6.73e-05 | img=224 | 
INFO	2026-02-21 11:40:33 -0500	workerpool0-0	  Epoch  65 | train=3.5106 [cls=0.6119 | reg=1.2147 | ctr=1.6840] | val=53.3766 | lr=6.55e-05 | img=224 | 
INFO	2026-02-21 11:40:46 -0500	workerpool0-0	  Epoch  66 | train=3.5260 [cls=0.6112 | reg=1.2270 | ctr=1.6878] | val=36.0798 | lr=6.36e-05 | img=224 | 
INFO	2026-02-21 11:41:00 -0500	workerpool0-0	  Epoch  67 | train=3.5144 [cls=0.6177 | reg=1.2131 | ctr=1.6837] | val=29.4713 | lr=6.17e-05 | img=224 | ★ best
INFO	2026-02-21 11:41:13 -0500	workerpool0-0	  Epoch  68 | train=3.4950 [cls=0.6139 | reg=1.1976 | ctr=1.6835] | val=41.8025 | lr=5.98e-05 | img=224 | 
INFO	2026-02-21 11:41:27 -0500	workerpool0-0	  Epoch  69 | train=3.4698 [cls=0.6000 | reg=1.1890 | ctr=1.6807] | val=37.8540 | lr=5.79e-05 | img=224 | 
INFO	2026-02-21 11:41:41 -0500	workerpool0-0	  Epoch  70 | train=3.5110 [cls=0.6146 | reg=1.2139 | ctr=1.6824] | val=51.0036 | lr=5.59e-05 | img=224 | 
INFO	2026-02-21 11:41:54 -0500	workerpool0-0	  Epoch  71 | train=3.4472 [cls=0.5921 | reg=1.1760 | ctr=1.6791] | val=50.6290 | lr=5.40e-05 | img=224 | 
INFO	2026-02-21 11:42:08 -0500	workerpool0-0	  Epoch  72 | train=3.4711 [cls=0.6033 | reg=1.1883 | ctr=1.6795] | val=37.9055 | lr=5.20e-05 | img=224 | 
INFO	2026-02-21 11:42:21 -0500	workerpool0-0	  Epoch  73 | train=3.4373 [cls=0.5866 | reg=1.1716 | ctr=1.6791] | val=41.4933 | lr=5.01e-05 | img=224 | 
INFO	2026-02-21 11:42:35 -0500	workerpool0-0	  Epoch  74 | train=3.4556 [cls=0.5975 | reg=1.1758 | ctr=1.6824] | val=33.5324 | lr=4.81e-05 | img=224 | 
INFO	2026-02-21 11:42:48 -0500	workerpool0-0	  Epoch  75 | train=3.4046 [cls=0.5681 | reg=1.1583 | ctr=1.6782] | val=44.8086 | lr=4.61e-05 | img=224 | 
INFO	2026-02-21 11:43:02 -0500	workerpool0-0	  Epoch  76 | train=3.4098 [cls=0.5734 | reg=1.1583 | ctr=1.6780] | val=47.1237 | lr=4.42e-05 | img=224 | 
INFO	2026-02-21 11:43:15 -0500	workerpool0-0	  Epoch  77 | train=3.4469 [cls=0.5939 | reg=1.1720 | ctr=1.6810] | val=40.4872 | lr=4.22e-05 | img=224 | 
INFO	2026-02-21 11:43:29 -0500	workerpool0-0	  Epoch  78 | train=3.3818 [cls=0.5571 | reg=1.1478 | ctr=1.6769] | val=39.7677 | lr=4.03e-05 | img=224 | 
INFO	2026-02-21 11:43:42 -0500	workerpool0-0	  Epoch  79 | train=3.4003 [cls=0.5663 | reg=1.1562 | ctr=1.6778] | val=31.3773 | lr=3.84e-05 | img=224 | 
INFO	2026-02-21 11:43:56 -0500	workerpool0-0	  Epoch  80 | train=3.3898 [cls=0.5587 | reg=1.1524 | ctr=1.6787] | val=28.1248 | lr=3.65e-05 | img=224 | ★ best
INFO	2026-02-21 11:44:09 -0500	workerpool0-0	  Epoch  81 | train=3.3679 [cls=0.5496 | reg=1.1409 | ctr=1.6774] | val=39.0885 | lr=3.46e-05 | img=224 | 
INFO	2026-02-21 11:44:23 -0500	workerpool0-0	  Epoch  82 | train=3.3883 [cls=0.5626 | reg=1.1486 | ctr=1.6771] | val=37.8488 | lr=3.28e-05 | img=224 | 
INFO	2026-02-21 11:44:36 -0500	workerpool0-0	  Epoch  83 | train=3.4199 [cls=0.5855 | reg=1.1551 | ctr=1.6794] | val=39.2990 | lr=3.09e-05 | img=224 | 
INFO	2026-02-21 11:44:50 -0500	workerpool0-0	  Epoch  84 | train=3.3909 [cls=0.5718 | reg=1.1399 | ctr=1.6791] | val=35.7942 | lr=2.91e-05 | img=224 | 
INFO	2026-02-21 11:45:03 -0500	workerpool0-0	  Epoch  85 | train=3.3474 [cls=0.5412 | reg=1.1294 | ctr=1.6767] | val=31.9420 | lr=2.74e-05 | img=224 | 
INFO	2026-02-21 11:45:17 -0500	workerpool0-0	  Epoch  86 | train=3.3598 [cls=0.5536 | reg=1.1282 | ctr=1.6780] | val=36.5503 | lr=2.56e-05 | img=224 | 
INFO	2026-02-21 11:45:30 -0500	workerpool0-0	  Epoch  87 | train=3.3631 [cls=0.5535 | reg=1.1355 | ctr=1.6740] | val=49.8956 | lr=2.40e-05 | img=224 | 
INFO	2026-02-21 11:45:44 -0500	workerpool0-0	  Epoch  88 | train=3.3327 [cls=0.5422 | reg=1.1159 | ctr=1.6746] | val=35.9951 | lr=2.23e-05 | img=224 | 
INFO	2026-02-21 11:45:57 -0500	workerpool0-0	  Epoch  89 | train=3.3220 [cls=0.5374 | reg=1.1087 | ctr=1.6758] | val=33.5970 | lr=2.07e-05 | img=224 | 
INFO	2026-02-21 11:46:11 -0500	workerpool0-0	  Epoch  90 | train=3.3551 [cls=0.5536 | reg=1.1238 | ctr=1.6777] | val=29.5986 | lr=1.91e-05 | img=224 | 
INFO	2026-02-21 11:46:25 -0500	workerpool0-0	  Epoch  91 | train=3.3299 [cls=0.5419 | reg=1.1118 | ctr=1.6761] | val=39.4894 | lr=1.76e-05 | img=224 | 
INFO	2026-02-21 11:46:38 -0500	workerpool0-0	  Epoch  92 | train=3.3308 [cls=0.5367 | reg=1.1189 | ctr=1.6752] | val=44.5682 | lr=1.61e-05 | img=224 | 
INFO	2026-02-21 11:46:52 -0500	workerpool0-0	  Epoch  93 | train=3.3305 [cls=0.5371 | reg=1.1185 | ctr=1.6750] | val=40.3271 | lr=1.47e-05 | img=224 | 
INFO	2026-02-21 11:47:05 -0500	workerpool0-0	  Epoch  94 | train=3.3383 [cls=0.5420 | reg=1.1219 | ctr=1.6745] | val=38.6960 | lr=1.34e-05 | img=224 | 
INFO	2026-02-21 11:47:19 -0500	workerpool0-0	  Epoch  95 | train=3.3148 [cls=0.5309 | reg=1.1094 | ctr=1.6745] | val=39.9908 | lr=1.21e-05 | img=224 | 
INFO	2026-02-21 11:47:32 -0500	workerpool0-0	  Epoch  96 | train=3.3198 [cls=0.5337 | reg=1.1135 | ctr=1.6726] | val=38.5357 | lr=1.08e-05 | img=224 | 
INFO	2026-02-21 11:47:45 -0500	workerpool0-0	  Epoch  97 | train=3.3178 [cls=0.5318 | reg=1.1112 | ctr=1.6748] | val=37.6560 | lr=9.64e-06 | img=224 | 
INFO	2026-02-21 11:47:59 -0500	workerpool0-0	  Epoch  98 | train=3.3258 [cls=0.5427 | reg=1.1061 | ctr=1.6770] | val=43.0391 | lr=8.52e-06 | img=224 | 
INFO	2026-02-21 11:48:12 -0500	workerpool0-0	  Epoch  99 | train=3.3205 [cls=0.5355 | reg=1.1100 | ctr=1.6750] | val=38.6487 | lr=7.46e-06 | img=224 | 
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	  Epoch 100 | train=3.2937 [cls=0.5243 | reg=1.0951 | ctr=1.6743] | val=36.9939 | lr=6.47e-06 | img=224 | 
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	⏹️  Early stopping at epoch 100 (patience=20)
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 16.0 min
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	📊 Entrenamiento completo: 101 epochs
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	   Mejor val_loss: 28.1248 (epoch 80)
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	   Tiempo total: 23.6 min
INFO	2026-02-21 11:48:26 -0500	workerpool0-0	⏱️  Entrenamiento completado en 23.6 min
INFO	2026-02-21 11:48:28 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 11:48:28 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:28 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 11:48:28 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	  Épocas: 101
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	  Mejor val_loss: 28.1248 (epoch 80)
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:29 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  mAP@50:    0.3761
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  mAP@50-95: 0.1791
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  Precision: 0.5910
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  Recall:    0.4267
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  F1-Score:  0.4956
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 555 | GT: 762
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  Avg inference: 4.8 ms
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	    dog                   0.2994
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	    door                  0.3847
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	    obstacle              0.3566
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	    person                0.3800
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	    stair                 0.4596
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	📊 Val mAP@50: 0.3761
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	   dog: 0.2994
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	   door: 0.3847
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	   obstacle: 0.3566
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	   person: 0.3800
INFO	2026-02-21 11:48:30 -0500	workerpool0-0	   stair: 0.4596
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 11:48:31 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  mAP@50:    0.5675
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  mAP@50-95: 0.2602
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  Precision: 0.6609
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  Recall:    0.6276
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  F1-Score:  0.6438
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 533 | GT: 576
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  Avg inference: 4.8 ms
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	    dog                   0.4957
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	    door                  0.5034
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	    obstacle              0.4575
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	    person                0.6359
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	    stair                 0.7451
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	📊 Test mAP@50: 0.5675
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	============================================================
INFO	2026-02-21 11:48:33 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.7s)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     Latencia mediana: 6.0ms
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/config.yaml
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/training_history.csv
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/training_curves.png
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/class_distribution.png
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/gt_samples.png
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/val_confusion_matrix.png
INFO	2026-02-21 11:48:34 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/val_per_class.png
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/val_evaluation.json
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/test_confusion_matrix.png
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/test_evaluation.json
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/experiment.json
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/checkpoints/best_fcos.pt
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx
INFO	2026-02-21 11:48:35 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 11:48:38 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 11:48:58 -0500	service	Tearing down training program.
INFO	2026-02-21 11:49:37 -0500	service	Finished tearing down training program.
INFO	2026-02-21 11:49:37 -0500	service	Job completed successfully.
```