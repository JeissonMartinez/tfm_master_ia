# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 1

```zsh
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 09:24:35 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 09:24:35 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 09:24:35 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 09:24:35 -0500	service	Waiting for training program to start.
INFO	2026-02-21 09:24:36 -0500	service	Job is preparing.
INFO	2026-02-21 09:28:40 -0500	service	Job is running.
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-1ik7oq50-37478d51-t6pp
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-99ab039ae0-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771683868.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771683868"]}
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:28:48 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 09:28:49 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 09:28:49 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 09:28:52 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 09:28:52 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 09:28:57 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:28:57 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:29:01 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:29:01 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 09:29:06 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 09:29:06 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 09:29:06 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 09:29:06 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=78810 sha256=cfda7e5a172b6878823ad0185fe502735c3331d35dc6698de20c07a01ca32134
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 09:29:07 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 09:29:07 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 09:29:08 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 09:29:08 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 09:29:08 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:29:09 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 09:29:09 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 09:29:13 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 09:29:13 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 09:29:13 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 09:29:13 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 09:29:14 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 09:29:14 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 09:29:14 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:14 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 09:29:14 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Collecting albumentations>=1.4 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 09:29:15 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=1.4->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 09:29:16 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:16 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 09:29:16 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:16 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 09:29:17 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:17 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 09:29:18 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:18 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 09:29:19 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 09:29:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 09:29:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:20 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 09:29:21 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 09:29:22 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 09:29:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 09:29:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 09:29:22 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 09:29:23 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 09:29:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 09:29:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 09:29:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 09:29:24 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 09:29:25 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 09:29:25 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:25 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=1.4->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 09:29:27 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 09:29:28 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 115.0 MB/s eta 0:00:00
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 47.6 MB/s eta 0:00:00
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 123.5 MB/s eta 0:00:00
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 151.4 MB/s eta 0:00:00
INFO	2026-02-21 09:29:29 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 144.0 MB/s eta 0:00:00
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 127.0 MB/s eta 0:00:00
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 28.8 MB/s eta 0:00:00
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 94.0 MB/s eta 0:00:00
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 09:29:30 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=78810 sha256=ca26ec83f79b745cfa6944d4c213534ec95c6924d0bf2a9b08b4124445d516df
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 09:29:31 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 09:29:33 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 09:29:33 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:33 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 09:29:36 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:36 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 09:29:36 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:36 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 09:29:41 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:41 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 09:29:41 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:41 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 09:29:43 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 09:29:43 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 09:29:49 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 09:29:49 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 09:29:49 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 09:29:49 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
INFO	2026-02-21 09:29:50 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 09:29:50 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 09:29:51 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771683868.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771683868
INFO	2026-02-21 09:29:54 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:29:54 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 09:29:54 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771683868.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Patience:     15
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  Conf Thresh:  0.25
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     Phase 1: 30 epochs @ LR=0.001
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     Phase 2: 60 epochs @ LR=0.0001
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 09:30:20 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 09:30:21 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 09:30:21 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 09:30:23 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	metadata {
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	}
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	metadata {
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	}
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	]
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	⚠️ Error escribiendo YAML: {'path': '/tmp/training/datasets/iodc_yolo', 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images', 'nc': 5, 'names': {0: 'dog', 1: 'door', 2: 'obstacle', 3: 'person', 4: 'stair'}} → expected str, bytes or os.PathLike object, not dict
INFO	2026-02-21 09:30:24 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 09:30:25 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 09:30:25 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 09:30:25 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 09:30:28 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 09:30:28 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:30:28 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 09:30:28 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 09:30:39 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 09:30:39 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 09:30:39 -0500	workerpool0-0	 69%|██████▊   | 6.75M/9.83M [00:00<00:00, 70.3MB/s]
ERROR	2026-02-21 09:30:39 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 84.1MB/s]
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 09:30:40 -0500	workerpool0-0	/root/.local/lib/python3.10/site-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
ERROR	2026-02-21 09:30:40 -0500	workerpool0-0	  original_init(self, **validated_kwargs)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	🔧 Two-Phase Training Config
ERROR	2026-02-21 09:30:40 -0500	workerpool0-0	/root/.local/lib/python3.10/site-packages/src_colab/utils_data.py:414: UserWarning: Argument(s) 'value' are not valid for transform ShiftScaleRotate
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Phase 1: 30 epochs | LR=0.001 | WD=0.0001
ERROR	2026-02-21 09:30:40 -0500	workerpool0-0	  A.ShiftScaleRotate(
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Phase 2: 60 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 15
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 30 epochs, LR=0.001
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 09:30:40 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=27)
INFO	2026-02-21 09:31:09 -0500	workerpool0-0	  Epoch   0 | train=687.4475 [cls=3.3112 | reg=682.2623 | ctr=1.8741] | val=2955.9308 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 09:31:29 -0500	workerpool0-0	  Epoch   1 | train=593.5757 [cls=2.8624 | reg=588.8953 | ctr=1.8180] | val=1425.2960 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 09:31:49 -0500	workerpool0-0	  Epoch   2 | train=496.2056 [cls=2.7807 | reg=491.6176 | ctr=1.8073] | val=1928.6243 | lr=1.00e-03 | img=640 | 
INFO	2026-02-21 09:32:09 -0500	workerpool0-0	  Epoch   3 | train=426.8949 [cls=2.7708 | reg=422.3225 | ctr=1.8017] | val=1768.3192 | lr=1.00e-03 | img=640 | 
INFO	2026-02-21 09:32:28 -0500	workerpool0-0	  Epoch   4 | train=399.5866 [cls=2.6212 | reg=395.1757 | ctr=1.7897] | val=2226.2774 | lr=9.97e-04 | img=640 | 
INFO	2026-02-21 09:32:47 -0500	workerpool0-0	  Epoch   5 | train=362.4417 [cls=2.4417 | reg=358.2212 | ctr=1.7787] | val=1532.2141 | lr=9.87e-04 | img=640 | 
INFO	2026-02-21 09:33:05 -0500	workerpool0-0	  Epoch   6 | train=333.8822 [cls=2.4286 | reg=329.6747 | ctr=1.7790] | val=1711.0466 | lr=9.70e-04 | img=640 | 
INFO	2026-02-21 09:33:21 -0500	workerpool0-0	  Epoch   7 | train=316.7436 [cls=2.3376 | reg=312.6329 | ctr=1.7732] | val=1391.6321 | lr=9.47e-04 | img=640 | ★ best
INFO	2026-02-21 09:33:38 -0500	workerpool0-0	  Epoch   8 | train=300.8077 [cls=2.3151 | reg=296.7221 | ctr=1.7704] | val=1884.3377 | lr=9.18e-04 | img=640 | 
INFO	2026-02-21 09:33:54 -0500	workerpool0-0	  Epoch   9 | train=292.8851 [cls=2.3204 | reg=288.7957 | ctr=1.7690] | val=2518.4142 | lr=8.83e-04 | img=640 | 
INFO	2026-02-21 09:34:07 -0500	workerpool0-0	  Epoch  10 | train=171.7258 [cls=2.2123 | reg=167.7471 | ctr=1.7664] | val=644.1192 | lr=8.43e-04 | img=416 | ★ best
INFO	2026-02-21 09:34:20 -0500	workerpool0-0	  Epoch  11 | train=151.5832 [cls=2.0953 | reg=147.7296 | ctr=1.7584] | val=602.2764 | lr=7.99e-04 | img=416 | ★ best
INFO	2026-02-21 09:34:32 -0500	workerpool0-0	  Epoch  12 | train=146.5218 [cls=2.0415 | reg=142.7251 | ctr=1.7552] | val=826.3722 | lr=7.50e-04 | img=416 | 
INFO	2026-02-21 09:34:45 -0500	workerpool0-0	  Epoch  13 | train=140.9470 [cls=2.0141 | reg=137.1802 | ctr=1.7527] | val=514.3845 | lr=6.98e-04 | img=416 | ★ best
INFO	2026-02-21 09:34:58 -0500	workerpool0-0	  Epoch  14 | train=138.5045 [cls=1.9548 | reg=134.8006 | ctr=1.7491] | val=681.2087 | lr=6.43e-04 | img=416 | 
INFO	2026-02-21 09:35:11 -0500	workerpool0-0	  Epoch  15 | train=137.6984 [cls=1.9060 | reg=134.0432 | ctr=1.7492] | val=643.1256 | lr=5.87e-04 | img=416 | 
INFO	2026-02-21 09:35:24 -0500	workerpool0-0	  Epoch  16 | train=132.8445 [cls=1.8340 | reg=129.2651 | ctr=1.7454] | val=739.6146 | lr=5.29e-04 | img=416 | 
INFO	2026-02-21 09:35:37 -0500	workerpool0-0	  Epoch  17 | train=132.1667 [cls=1.8181 | reg=128.6011 | ctr=1.7475] | val=646.5676 | lr=4.71e-04 | img=416 | 
INFO	2026-02-21 09:35:49 -0500	workerpool0-0	  Epoch  18 | train=134.0679 [cls=1.7958 | reg=130.5278 | ctr=1.7442] | val=670.5182 | lr=4.13e-04 | img=416 | 
INFO	2026-02-21 09:36:02 -0500	workerpool0-0	  Epoch  19 | train=128.7953 [cls=1.7683 | reg=125.2834 | ctr=1.7437] | val=691.3078 | lr=3.57e-04 | img=416 | 
INFO	2026-02-21 09:36:14 -0500	workerpool0-0	  Epoch  20 | train=100.7971 [cls=1.7853 | reg=97.2673 | ctr=1.7445] | val=344.2708 | lr=3.02e-04 | img=320 | ★ best
INFO	2026-02-21 09:36:27 -0500	workerpool0-0	  Epoch  21 | train=97.8512 [cls=1.7462 | reg=94.3644 | ctr=1.7406] | val=393.5571 | lr=2.50e-04 | img=320 | 
INFO	2026-02-21 09:36:39 -0500	workerpool0-0	  Epoch  22 | train=94.8109 [cls=1.6972 | reg=91.3732 | ctr=1.7405] | val=335.8109 | lr=2.02e-04 | img=320 | ★ best
INFO	2026-02-21 09:36:51 -0500	workerpool0-0	  Epoch  23 | train=92.9855 [cls=1.6604 | reg=89.5862 | ctr=1.7390] | val=358.0041 | lr=1.57e-04 | img=320 | 
INFO	2026-02-21 09:37:03 -0500	workerpool0-0	  Epoch  24 | train=93.2133 [cls=1.6803 | reg=89.7963 | ctr=1.7367] | val=352.5979 | lr=1.17e-04 | img=320 | 
INFO	2026-02-21 09:37:15 -0500	workerpool0-0	  Epoch  25 | train=93.0818 [cls=1.6599 | reg=89.6834 | ctr=1.7385] | val=358.9936 | lr=8.23e-05 | img=320 | 
INFO	2026-02-21 09:37:27 -0500	workerpool0-0	  Epoch  26 | train=92.3594 [cls=1.6453 | reg=88.9786 | ctr=1.7356] | val=353.8535 | lr=5.33e-05 | img=320 | 
INFO	2026-02-21 09:37:39 -0500	workerpool0-0	  Epoch  27 | train=91.6397 [cls=1.6240 | reg=88.2795 | ctr=1.7362] | val=351.6500 | lr=3.03e-05 | img=320 | 
INFO	2026-02-21 09:37:51 -0500	workerpool0-0	  Epoch  28 | train=92.3686 [cls=1.6212 | reg=89.0120 | ctr=1.7353] | val=355.3555 | lr=1.36e-05 | img=320 | 
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	  Epoch  29 | train=92.5946 [cls=1.6177 | reg=89.2415 | ctr=1.7353] | val=362.6344 | lr=3.48e-06 | img=320 | 
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 7.4 min
ERROR	2026-02-21 09:38:03 -0500	workerpool0-0	/root/.local/lib/python3.10/site-packages/src_colab/utils_train.py:614: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
ERROR	2026-02-21 09:38:03 -0500	workerpool0-0	  model.load_state_dict(torch.load(best_ckpt, map_location=config.device))
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 60 epochs, LR=0.0001
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 09:38:03 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=60)
INFO	2026-02-21 09:38:16 -0500	workerpool0-0	  Epoch  30 | train=75.5794 [cls=1.7976 | reg=72.0180 | ctr=1.7638] | val=226.4528 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 09:38:29 -0500	workerpool0-0	  Epoch  31 | train=64.9516 [cls=1.6833 | reg=61.5161 | ctr=1.7522] | val=178.3921 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 09:38:42 -0500	workerpool0-0	  Epoch  32 | train=61.9202 [cls=1.6059 | reg=58.5703 | ctr=1.7440] | val=185.1932 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 09:38:55 -0500	workerpool0-0	  Epoch  33 | train=59.1465 [cls=1.5847 | reg=55.8224 | ctr=1.7395] | val=160.5102 | lr=1.00e-04 | img=224 | ★ best
INFO	2026-02-21 09:39:08 -0500	workerpool0-0	  Epoch  34 | train=57.8728 [cls=1.5355 | reg=54.6038 | ctr=1.7335] | val=159.0207 | lr=9.99e-05 | img=224 | ★ best
INFO	2026-02-21 09:39:21 -0500	workerpool0-0	  Epoch  35 | train=56.3225 [cls=1.4827 | reg=53.1062 | ctr=1.7337] | val=179.3025 | lr=9.97e-05 | img=224 | 
INFO	2026-02-21 09:39:34 -0500	workerpool0-0	  Epoch  36 | train=55.0480 [cls=1.4568 | reg=51.8623 | ctr=1.7290] | val=135.4938 | lr=9.94e-05 | img=224 | ★ best
INFO	2026-02-21 09:39:46 -0500	workerpool0-0	  Epoch  37 | train=54.4032 [cls=1.4576 | reg=51.2183 | ctr=1.7273] | val=172.0443 | lr=9.89e-05 | img=224 | 
INFO	2026-02-21 09:39:59 -0500	workerpool0-0	  Epoch  38 | train=51.8887 [cls=1.4328 | reg=48.7311 | ctr=1.7248] | val=159.5751 | lr=9.83e-05 | img=224 | 
INFO	2026-02-21 09:40:12 -0500	workerpool0-0	  Epoch  39 | train=51.8195 [cls=1.4010 | reg=48.6976 | ctr=1.7209] | val=147.8221 | lr=9.76e-05 | img=224 | 
INFO	2026-02-21 09:40:25 -0500	workerpool0-0	  Epoch  40 | train=51.7655 [cls=1.3836 | reg=48.6613 | ctr=1.7207] | val=158.0602 | lr=9.67e-05 | img=224 | 
INFO	2026-02-21 09:40:37 -0500	workerpool0-0	  Epoch  41 | train=49.8658 [cls=1.3493 | reg=46.7993 | ctr=1.7173] | val=156.8879 | lr=9.57e-05 | img=224 | 
INFO	2026-02-21 09:40:50 -0500	workerpool0-0	  Epoch  42 | train=49.3308 [cls=1.2969 | reg=46.3204 | ctr=1.7135] | val=145.7431 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 09:41:03 -0500	workerpool0-0	  Epoch  43 | train=48.6953 [cls=1.3042 | reg=45.6763 | ctr=1.7147] | val=185.0785 | lr=9.33e-05 | img=224 | 
INFO	2026-02-21 09:41:16 -0500	workerpool0-0	  Epoch  44 | train=47.2784 [cls=1.2635 | reg=44.3048 | ctr=1.7101] | val=154.1804 | lr=9.19e-05 | img=224 | 
INFO	2026-02-21 09:41:29 -0500	workerpool0-0	  Epoch  45 | train=46.6472 [cls=1.2810 | reg=43.6527 | ctr=1.7135] | val=152.8424 | lr=9.05e-05 | img=224 | 
INFO	2026-02-21 09:41:41 -0500	workerpool0-0	  Epoch  46 | train=46.3835 [cls=1.2439 | reg=43.4296 | ctr=1.7101] | val=166.1492 | lr=8.89e-05 | img=224 | 
INFO	2026-02-21 09:41:54 -0500	workerpool0-0	  Epoch  47 | train=46.3914 [cls=1.2130 | reg=43.4722 | ctr=1.7061] | val=180.8808 | lr=8.72e-05 | img=224 | 
INFO	2026-02-21 09:42:07 -0500	workerpool0-0	  Epoch  48 | train=45.1991 [cls=1.1984 | reg=42.2933 | ctr=1.7075] | val=154.1041 | lr=8.54e-05 | img=224 | 
INFO	2026-02-21 09:42:20 -0500	workerpool0-0	  Epoch  49 | train=44.3994 [cls=1.2001 | reg=41.4938 | ctr=1.7054] | val=163.0445 | lr=8.35e-05 | img=224 | 
INFO	2026-02-21 09:42:33 -0500	workerpool0-0	  Epoch  50 | train=43.7484 [cls=1.1805 | reg=40.8659 | ctr=1.7020] | val=161.5034 | lr=8.15e-05 | img=224 | 
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	  Epoch  51 | train=44.1333 [cls=1.1880 | reg=41.2452 | ctr=1.7001] | val=153.6194 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	⏹️  Early stopping at epoch 51 (patience=15)
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 4.7 min
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	📊 Entrenamiento completo: 52 epochs
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	   Mejor val_loss: 135.4938 (epoch 36)
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	   Tiempo total: 12.1 min
INFO	2026-02-21 09:42:46 -0500	workerpool0-0	⏱️  Entrenamiento completado en 12.1 min
INFO	2026-02-21 09:42:47 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 09:42:47 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:47 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 09:42:47 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	  Épocas: 52
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	  Mejor val_loss: 135.4938 (epoch 36)
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 09:42:49 -0500	workerpool0-0	/root/.local/lib/python3.10/site-packages/trainer/task_fcos.py:440: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
ERROR	2026-02-21 09:42:49 -0500	workerpool0-0	  model.load_state_dict(torch.load(best_ckpt, map_location=device))
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:49 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  mAP@50:    0.2835
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  mAP@50-95: 0.0000
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  Precision: 0.5223
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  Recall:    0.3413
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  F1-Score:  0.4128
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 525 | GT: 762
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  Avg inference: 4.8 ms
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	    dog                   0.1956
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	    door                  0.2919
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	    obstacle              0.2360
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	    person                0.3530
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	    stair                 0.3409
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	📊 Val mAP@50: 0.2835
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	   dog: 0.1956
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	   door: 0.2919
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	   obstacle: 0.2360
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	   person: 0.3530
INFO	2026-02-21 09:42:50 -0500	workerpool0-0	   stair: 0.3409
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 09:42:51 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  mAP@50:    0.4304
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  mAP@50-95: 0.0000
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  Precision: 0.5427
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  Recall:    0.5291
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  F1-Score:  0.5358
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 548 | GT: 576
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  Avg inference: 5.0 ms
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	    dog                   0.4056
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	    door                  0.3391
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	    obstacle              0.3022
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	    person                0.5206
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	    stair                 0.5844
INFO	2026-02-21 09:42:52 -0500	workerpool0-0	📊 Test mAP@50: 0.4304
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	============================================================
INFO	2026-02-21 09:42:53 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.7s)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     Latencia mediana: 4.6ms
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/config.yaml
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/training_history.csv
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/training_curves.png
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/class_distribution.png
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/gt_samples.png
INFO	2026-02-21 09:42:54 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/val_confusion_matrix.png
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/val_per_class.png
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/val_evaluation.json
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/test_confusion_matrix.png
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/test_evaluation.json
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/experiment.json
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/checkpoints/best_fcos.pt
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/export/fcos_v3s.onnx
INFO	2026-02-21 09:42:55 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 09:42:58 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 09:43:14 -0500	service	Tearing down training program.
INFO	2026-02-21 09:44:04 -0500	service	Finished tearing down training program.
INFO	2026-02-21 09:44:04 -0500	service	Job completed successfully.

```