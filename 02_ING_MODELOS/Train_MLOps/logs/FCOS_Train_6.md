# Logs de Entrenamiento de Modelo FCOS (Task_Modelo_1_FCOS)
**Corrida Exitosa #:** 6

```zsh
% gcloud ai custom-jobs stream-logs 3443077695527714816 --region=us-central1
Using endpoint [https://us-central1-aiplatform.googleapis.com/]
INFO	2026-02-21 18:11:28 -0500	service	Waiting for job to be provisioned.
INFO	2026-02-21 18:11:28 -0500	service	Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.
INFO	2026-02-21 18:11:28 -0500	service	Vertex AI is setting up this job.
INFO	2026-02-21 18:11:28 -0500	service	Waiting for training program to start.
INFO	2026-02-21 18:11:29 -0500	service	Job is preparing.
INFO	2026-02-21 18:14:14 -0500	workerpool0-0	Updating master address to local address gk3-cml-0221-054317-2fe2-nap-6y014swo-8de8146d-zqbz
INFO	2026-02-21 18:14:14 -0500	workerpool0-0	Running run_module.py
INFO	2026-02-21 18:14:15 -0500	workerpool0-0	Running task with arguments: --cluster={"chief": ["cmle-training-workerpool0-89275d2add-0:2222"]} --task={"type": "chief", "index": 0} --job={"python_module":"trainer.task_fcos","package_uris":["gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz"],"job_args":["--config-uri\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771715459.yaml","--job-dir\u003dgs://project-18f58341-12cf-47bc-861-tfm-data/output","--project-id\u003dproject-18f58341-12cf-47bc-861","--region\u003dus-central1","--experiment-name\u003dtfm-deteccion-objetos","--run-name\u003dfcos_v3s_v1-1771715459"]}
INFO	2026-02-21 18:14:15 -0500	workerpool0-0	Running module trainer.task_fcos.
INFO	2026-02-21 18:14:15 -0500	workerpool0-0	Downloading the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:15 -0500	workerpool0-0	Running command: gsutil -q cp gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz tfm_trainer-2.0.0.tar.gz
ERROR	2026-02-21 18:14:15 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 18:14:15 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
ERROR	2026-02-21 18:14:19 -0500	workerpool0-0	Error in sitecustomize; set PYTHONVERBOSE for traceback:
ERROR	2026-02-21 18:14:19 -0500	workerpool0-0	ModuleNotFoundError: No module named 'pythonjsonlogger'
INFO	2026-02-21 18:14:23 -0500	workerpool0-0	Installing the package: gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:23 -0500	workerpool0-0	Running command: pip3 install --user --upgrade --force-reinstall --no-deps tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:24 -0500	service	Job is running.
INFO	2026-02-21 18:14:27 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:27 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 18:14:32 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 18:14:32 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 18:14:32 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 18:14:32 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=81841 sha256=b2e3d04546d8fd71946dd092c0db1c17f9d18f51961248bd0607639a48f63be9
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	Installing collected packages: tfm-trainer
INFO	2026-02-21 18:14:33 -0500	workerpool0-0	Successfully installed tfm-trainer-2.0.0
ERROR	2026-02-21 18:14:33 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 18:14:33 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 18:14:33 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 18:14:34 -0500	workerpool0-0	Running command: pip3 install --user tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:35 -0500	workerpool0-0	Processing /tfm_trainer-2.0.0.tar.gz
INFO	2026-02-21 18:14:35 -0500	workerpool0-0	  Installing build dependencies: started
INFO	2026-02-21 18:14:39 -0500	workerpool0-0	  Installing build dependencies: finished with status 'done'
INFO	2026-02-21 18:14:39 -0500	workerpool0-0	  Getting requirements to build wheel: started
INFO	2026-02-21 18:14:39 -0500	workerpool0-0	  Getting requirements to build wheel: finished with status 'done'
INFO	2026-02-21 18:14:39 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): started
INFO	2026-02-21 18:14:40 -0500	workerpool0-0	  Preparing metadata (pyproject.toml): finished with status 'done'
INFO	2026-02-21 18:14:40 -0500	workerpool0-0	Requirement already satisfied: pyyaml>=6.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (6.0.2)
INFO	2026-02-21 18:14:40 -0500	workerpool0-0	Collecting numpy<2.0,>=1.26 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:41 -0500	workerpool0-0	  Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
INFO	2026-02-21 18:14:41 -0500	workerpool0-0	Requirement already satisfied: pandas>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.2.3)
INFO	2026-02-21 18:14:41 -0500	workerpool0-0	Collecting matplotlib>=3.8 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:41 -0500	workerpool0-0	  Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
INFO	2026-02-21 18:14:41 -0500	workerpool0-0	Requirement already satisfied: scikit-learn>=1.4 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.6.1)
INFO	2026-02-21 18:14:42 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:42 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 18:14:42 -0500	workerpool0-0	Collecting albumentations>=2.0.0 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	  Downloading albumentations-2.0.8-py3-none-any.whl.metadata (43 kB)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	Requirement already satisfied: torchvision>=0.19 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (0.19.0+cu124)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	Requirement already satisfied: google-cloud-storage>=2.14 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (2.19.0)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	Requirement already satisfied: google-cloud-aiplatform>=1.40 in /opt/python/3.10/lib/python3.10/site-packages (from tfm-trainer==2.0.0) (1.91.0)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:43 -0500	workerpool0-0	  Downloading onnx-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 18:14:44 -0500	workerpool0-0	Collecting onnxruntime>=1.16 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:44 -0500	workerpool0-0	  Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
INFO	2026-02-21 18:14:44 -0500	workerpool0-0	Requirement already satisfied: scipy>=1.10.0 in /opt/python/3.10/lib/python3.10/site-packages (from albumentations>=2.0.0->tfm-trainer==2.0.0) (1.11.4)
INFO	2026-02-21 18:14:45 -0500	workerpool0-0	Collecting pydantic>=2.9.2 (from albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:45 -0500	workerpool0-0	  Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
INFO	2026-02-21 18:14:45 -0500	workerpool0-0	Collecting albucore==0.0.24 (from albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:45 -0500	workerpool0-0	  Downloading albucore-0.0.24-py3-none-any.whl.metadata (5.3 kB)
INFO	2026-02-21 18:14:47 -0500	workerpool0-0	Collecting stringzilla>=3.10.4 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:47 -0500	workerpool0-0	  Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (121 kB)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Collecting simsimd>=5.9.2 (from albucore==0.0.24->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	  Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (70 kB)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.34.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.40.3)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: proto-plus<2.0.0,>=1.22.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<7.0.0,>=3.20.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.20.3)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: packaging>=14.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (25.0)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.34.0)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: google-cloud-resource-manager<3.0.0,>=1.3.3 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.14.2)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: shapely<3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.1.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: typing-extensions in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.14.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: docstring-parser<1 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.16)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.70.0)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.32.4)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.73.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.48.2)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (5.5.2)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.4.2)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	Requirement already satisfied: rsa<5,>=3.1.4 in /opt/python/3.10/lib/python3.10/site-packages (from google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (4.9.1)
INFO	2026-02-21 18:14:48 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	Collecting google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0 (from google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.1-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.40.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.39.0-py3-none-any.whl.metadata (8.2 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.38.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.37.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.36.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.1-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	INFO: pip is still looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.35.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.29.0-py2.py3-none-any.whl.metadata (7.6 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.27.0-py2.py3-none-any.whl.metadata (8.6 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.26.0-py2.py3-none-any.whl.metadata (8.7 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	  Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl.metadata (8.9 kB)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.4.3)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.7.2)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	Requirement already satisfied: python-dateutil<3.0dev,>=2.7.2 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2.9.0.post0)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	Requirement already satisfied: grpc-google-iam-v1<1.0.0,>=0.14.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-resource-manager<3.0.0,>=1.3.3->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.14.2)
INFO	2026-02-21 18:14:49 -0500	workerpool0-0	INFO: pip is looking at multiple versions of google-cloud-storage to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	Collecting google-cloud-storage>=2.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.2-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.1-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.18.0-py2.py3-none-any.whl.metadata (9.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.17.0-py2.py3-none-any.whl.metadata (6.6 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.16.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.15.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl.metadata (6.1 kB)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in /opt/python/3.10/lib/python3.10/site-packages (from google-cloud-storage>=2.14->tfm-trainer==2.0.0) (1.7.1)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	Collecting annotated-types>=0.6.0 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:50 -0500	workerpool0-0	  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Collecting pydantic-core==2.41.5 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	  Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Collecting typing-inspection>=0.4.2 (from pydantic>=2.9.2->albumentations>=2.0.0->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: six>=1.5 in /opt/python/3.10/lib/python3.10/site-packages (from python-dateutil<3.0dev,>=2.7.2->google-cloud-bigquery!=3.20.0,<4.0.0,>=1.15.0->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.17.0)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: charset_normalizer<4,>=2 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: idna<4,>=2.5 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (3.10)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (1.26.20)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: certifi>=2017.4.17 in /opt/python/3.10/lib/python3.10/site-packages (from requests<3.0.0dev,>=2.18.0->google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3.0.0,>=1.34.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (2025.6.15)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: pyasn1>=0.1.3 in /opt/python/3.10/lib/python3.10/site-packages (from rsa<5,>=3.1.4->google-auth<3.0.0,>=2.14.1->google-cloud-aiplatform>=1.40->tfm-trainer==2.0.0) (0.6.1)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: contourpy>=1.0.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.3.2)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: cycler>=0.10 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (0.12.1)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: fonttools>=4.22.0 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (4.58.5)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: kiwisolver>=1.3.1 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (1.4.8)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: pillow>=8 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (11.3.0)
INFO	2026-02-21 18:14:51 -0500	workerpool0-0	Requirement already satisfied: pyparsing>=3 in /opt/python/3.10/lib/python3.10/site-packages (from matplotlib>=3.8->tfm-trainer==2.0.0) (3.2.3)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	INFO: pip is looking at multiple versions of onnx to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	Collecting onnx>=1.14 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading onnx-1.20.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading onnx-1.19.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading onnx-1.19.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.0 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading onnx-1.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	Collecting coloredlogs (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	Collecting flatbuffers (from onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	Requirement already satisfied: sympy in /opt/python/3.10/lib/python3.10/site-packages (from onnxruntime>=1.16->tfm-trainer==2.0.0) (1.14.0)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	INFO: pip is looking at multiple versions of opencv-python-headless to determine which version is compatible with other requirements. This could take a while.
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	Collecting opencv-python-headless>=4.9 (from tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading opencv_python_headless-4.13.0.90-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
INFO	2026-02-21 18:14:52 -0500	workerpool0-0	  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: pytz>=2020.1 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: tzdata>=2022.7 in /opt/python/3.10/lib/python3.10/site-packages (from pandas>=2.0->tfm-trainer==2.0.0) (2025.2)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: joblib>=1.2.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (1.5.1)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from scikit-learn>=1.4->tfm-trainer==2.0.0) (3.6.0)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: torch==2.4.0 in /opt/python/3.10/lib/python3.10/site-packages (from torchvision>=0.19->tfm-trainer==2.0.0) (2.4.0+cu124)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: filelock in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.18.0)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: networkx in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.4.2)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: jinja2 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.1.6)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: fsspec in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2025.5.1)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (9.1.0.70)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cublas-cu12==12.4.2.65 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.2.65)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cufft-cu12==11.2.0.44 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.2.0.44)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-curand-cu12==10.3.5.119 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (10.3.5.119)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusolver-cu12==11.6.0.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (11.6.0.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-cusparse-cu12==12.3.0.142 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.3.0.142)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-nccl-cu12==2.20.5 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (2.20.5)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvtx-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.99 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (12.4.99)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: triton==3.0.0 in /opt/python/3.10/lib/python3.10/site-packages (from torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.0)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.16->tfm-trainer==2.0.0)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: MarkupSafe>=2.0 in /opt/python/3.10/lib/python3.10/site-packages (from jinja2->torch==2.4.0->torchvision>=0.19->tfm-trainer==2.0.0) (3.0.2)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/python/3.10/lib/python3.10/site-packages (from sympy->onnxruntime>=1.16->tfm-trainer==2.0.0) (1.3.0)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 108.7 MB/s eta 0:00:00
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading albumentations-2.0.8-py3-none-any.whl (369 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading albucore-0.0.24-py3-none-any.whl (15 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading google_cloud_bigquery-3.25.0-py2.py3-none-any.whl (239 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading google_cloud_storage-2.14.0-py2.py3-none-any.whl (121 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 76.7 MB/s eta 0:00:00
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 134.5 MB/s eta 0:00:00
INFO	2026-02-21 18:14:53 -0500	workerpool0-0	Downloading onnx-1.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.0 MB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 109.3 MB/s eta 0:00:00
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.4 MB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 105.2 MB/s eta 0:00:00
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 136.2 MB/s eta 0:00:00
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading simsimd-6.5.13-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (582 kB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 582.3/582.3 kB 30.2 MB/s eta 0:00:00
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 96.4 MB/s eta 0:00:00
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
INFO	2026-02-21 18:14:54 -0500	workerpool0-0	Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	Building wheels for collected packages: tfm-trainer
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): started
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	  Building wheel for tfm-trainer (pyproject.toml): finished with status 'done'
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	  Created wheel for tfm-trainer: filename=tfm_trainer-2.0.0-py3-none-any.whl size=81841 sha256=51150b3eb141c8057ecf5c05f47ed8d7874e232eda93a3a5bfc999411b2b83df
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	  Stored in directory: /root/.cache/pip/wheels/7c/c7/b0/30a13f0c2cd9b1cdd73516fcf6defe763972b52dffedaab7be
INFO	2026-02-21 18:14:55 -0500	workerpool0-0	Successfully built tfm-trainer
INFO	2026-02-21 18:14:57 -0500	workerpool0-0	Installing collected packages: simsimd, flatbuffers, typing-inspection, stringzilla, pydantic-core, numpy, humanfriendly, annotated-types, pydantic, opencv-python-headless, onnx, coloredlogs, onnxruntime, matplotlib, albucore, albumentations, google-cloud-storage, google-cloud-bigquery, tfm-trainer
ERROR	2026-02-21 18:14:57 -0500	workerpool0-0	  WARNING: The scripts sz_split and sz_wc are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:14:57 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 18:15:00 -0500	workerpool0-0	  WARNING: The script f2py is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:15:00 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 18:15:00 -0500	workerpool0-0	  WARNING: The script humanfriendly is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:15:00 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 18:15:05 -0500	workerpool0-0	  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:15:05 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 18:15:05 -0500	workerpool0-0	  WARNING: The script coloredlogs is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:15:05 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR	2026-02-21 18:15:07 -0500	workerpool0-0	  WARNING: The script onnxruntime_test is installed in '/root/.local/bin' which is not on PATH.
ERROR	2026-02-21 18:15:07 -0500	workerpool0-0	  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
INFO	2026-02-21 18:15:12 -0500	workerpool0-0	  Attempting uninstall: tfm-trainer
INFO	2026-02-21 18:15:12 -0500	workerpool0-0	    Found existing installation: tfm-trainer 2.0.0
INFO	2026-02-21 18:15:12 -0500	workerpool0-0	    Uninstalling tfm-trainer-2.0.0:
INFO	2026-02-21 18:15:12 -0500	workerpool0-0	      Successfully uninstalled tfm-trainer-2.0.0
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	bigframes 0.22.0 requires pandas<2.1.4,>=1.5.0, but you have pandas 2.2.3 which is incompatible.
INFO	2026-02-21 18:15:12 -0500	workerpool0-0	Successfully installed albucore-0.0.24 albumentations-2.0.8 annotated-types-0.7.0 coloredlogs-15.0.1 flatbuffers-25.12.19 google-cloud-bigquery-3.25.0 google-cloud-storage-2.14.0 humanfriendly-10.0 matplotlib-3.10.8 numpy-1.26.4 onnx-1.17.0 onnxruntime-1.23.2 opencv-python-headless-4.11.0.86 pydantic-2.12.5 pydantic-core-2.41.5 simsimd-6.5.13 stringzilla-4.6.0 tfm-trainer-2.0.0 typing-inspection-0.4.2
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	dataproc-jupyter-plugin 0.1.80 requires pydantic~=1.10.0, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	pandas-gbq 0.29.1 requires google-api-core<3.0.0,>=2.10.2, but you have google-api-core 1.34.1 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	ydata-profiling 4.6.0 requires matplotlib<=3.7.3,>=3.2, but you have matplotlib 3.10.8 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	ydata-profiling 4.6.0 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pandas!=1.4.0,<2.1,>1.1, but you have pandas 2.2.3 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	ydata-profiling 4.6.0 requires pydantic<2,>=1.8.1, but you have pydantic 2.12.5 which is incompatible.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	[notice] A new release of pip is available: 25.1.1 -> 26.0.1
ERROR	2026-02-21 18:15:12 -0500	workerpool0-0	[notice] To update, run: pip install --upgrade pip
INFO	2026-02-21 18:15:13 -0500	workerpool0-0	Running command: python3 -m trainer.task_fcos --config-uri=gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771715459.yaml --job-dir=gs://project-18f58341-12cf-47bc-861-tfm-data/output --project-id=project-18f58341-12cf-47bc-861 --region=us-central1 --experiment-name=tfm-deteccion-objetos --run-name=fcos_v3s_v1-1771715459
INFO	2026-02-21 18:15:17 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:15:17 -0500	workerpool0-0	BLOQUE 1 — Setup y descarga de datos
INFO	2026-02-21 18:15:17 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1-1771715459.yaml → /tmp/training/config.yaml (0.0 MB)
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	🧪 CONFIGURACIÓN DEL EXPERIMENTO
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Nombre:       fcos_v3s_v1
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Familia:      FCOS
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Variante:     fcos_v3s
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Versión:      v1
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Descripción:  FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS head
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Dataset:      iodc_yolo
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Clases (5):  ['dog', 'door', 'obstacle', 'person', 'stair']
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Img Size:     224×224
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Batch Size:   16
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Patience:     20
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Seed:         42
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  Conf Thresh:  0.25
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  IoU Thresh:   0.45
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  📐 2-Phase Training:
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     Phase 1: 40 epochs @ LR=0.001
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     Phase 2: 80 epochs @ LR=0.0001
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     Resize Schedule: {0: 640, 10: 416, 20: 320, 30: 224}
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     Optimizer: AdamW | WD: 0.0005
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	  🔷 FCOS Config:
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     FPN Channels: 64
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	     Backbone: mobilenet_v3_small
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	✅ Configuración aplicada correctamente
INFO	2026-02-21 18:15:43 -0500	workerpool0-0	📥 Preparando dataset desde gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
INFO	2026-02-21 18:15:44 -0500	workerpool0-0	  ✅ Descargado: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo.zip (111.4 MB)
INFO	2026-02-21 18:15:44 -0500	workerpool0-0	  📦 Descomprimiendo /tmp/training/datasets/iodc_yolo.zip → /tmp/training/datasets/iodc_yolo ...
INFO	2026-02-21 18:15:45 -0500	workerpool0-0	  ✅ Descomprimido correctamente
INFO	2026-02-21 18:15:45 -0500	workerpool0-0	  🗑️  Eliminado zip temporal: /tmp/training/datasets/iodc_yolo.zip
INFO	2026-02-21 18:15:45 -0500	workerpool0-0	  📂 Dataset listo: /tmp/training/datasets/iodc_yolo
INFO	2026-02-21 18:15:45 -0500	workerpool0-0	🖥️  Device: cuda
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	⚠️  Vertex AI Experiments no disponible — el entrenamiento continuará sin registro de experimentos.
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	   Causa: 403 Request had insufficient authentication scopes. [reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	domain: "googleapis.com"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	metadata {
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  key: "method"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  value: "google.cloud.aiplatform.v1.MetadataService.GetMetadataStore"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	}
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	metadata {
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  key: "service"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  value: "aiplatform.googleapis.com"
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	}
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	]
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	BLOQUE 2 — Verificación del Dataset
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	📂 Dataset YOLO: iodc_yolo
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  ✅ train:  1470 imgs |  1470 labels | 0 sin label
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  ✅ valid:   188 imgs |   188 labels | 0 sin label
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	  ✅  test:   187 imgs |   187 labels | 0 sin label
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	📄 data.yaml generado: /tmp/training/datasets/iodc_yolo/data.yaml
INFO	2026-02-21 18:15:46 -0500	workerpool0-0	generated new fontManager
INFO	2026-02-21 18:15:47 -0500	workerpool0-0	  📊 Guardado: /tmp/training/class_distribution.png
INFO	2026-02-21 18:15:47 -0500	workerpool0-0	⚖️  Class weights: [1.0, 1.0, 0.9714795008912657, 1.0, 0.9945255474452555]
INFO	2026-02-21 18:15:49 -0500	workerpool0-0	  🖼️  Guardado: /tmp/training/gt_samples.png
INFO	2026-02-21 18:15:49 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:15:49 -0500	workerpool0-0	BLOQUE 3 — Construcción del Modelo FCOS
INFO	2026-02-21 18:15:49 -0500	workerpool0-0	============================================================
ERROR	2026-02-21 18:16:00 -0500	workerpool0-0	Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
ERROR	2026-02-21 18:16:00 -0500	workerpool0-0	  0%|          | 0.00/9.83M [00:00<?, ?B/s]
ERROR	2026-02-21 18:16:00 -0500	workerpool0-0	 61%|██████    | 6.00M/9.83M [00:00<00:00, 62.5MB/s]
ERROR	2026-02-21 18:16:00 -0500	workerpool0-0	100%|██████████| 9.83M/9.83M [00:00<00:00, 82.5MB/s]
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	✅ FCOS (MobileNetV3-S) construido: 1,233,450 params (1,233,450 trainable)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	   FPN channels: 64 | Classes: 5
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	📦 Modelo: FCOS
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Total params:        1,233,450
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Trainable:             306,442
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Non-trainable:         927,008
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Est. float32:           4.71 MB
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Est. INT8:              1.18 MB
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	📐 Tamaño estimado: 4.71 MB (FP32), 1.18 MB (INT8)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	BLOQUE 4 — Entrenamiento (2 fases)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	🔧 Two-Phase Training Config
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Phase 1: 40 epochs | LR=0.001 | WD=0.0001
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Phase 2: 80 epochs | LR=0.0001 | WD=1e-05
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Optimizer: adamw | Scheduler: cosine
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Batch: 16 | AMP: True | Patience: 20
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	  Resize schedule: [(0, 640), (10, 416), (20, 320), (30, 224)]
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	🔒 Backbone congelado: 927,008 params frozen
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	   Trainable: 306,442 / 1,233,450 (24.8%)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	🚀 Phase 1 (backbone frozen) — 40 epochs, LR=0.001
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.001, wd=0.0001)
INFO	2026-02-21 18:16:01 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=37)
INFO	2026-02-21 18:16:30 -0500	workerpool0-0	  Epoch   0 | train=9.3603 [cls=3.0204 | reg=4.5000 | ctr=1.8399] | val=2252.0779 | lr=3.33e-04 | img=640 | ★ best
INFO	2026-02-21 18:16:51 -0500	workerpool0-0	  Epoch   1 | train=8.8143 [cls=2.5171 | reg=4.5000 | ctr=1.7973] | val=1475.5932 | lr=6.67e-04 | img=640 | ★ best
INFO	2026-02-21 18:17:11 -0500	workerpool0-0	  Epoch   2 | train=8.6510 [cls=2.3581 | reg=4.5000 | ctr=1.7929] | val=1229.0344 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 18:17:31 -0500	workerpool0-0	  Epoch   3 | train=8.4291 [cls=2.1504 | reg=4.5000 | ctr=1.7787] | val=1032.5342 | lr=1.00e-03 | img=640 | ★ best
INFO	2026-02-21 18:17:51 -0500	workerpool0-0	  Epoch   4 | train=8.3379 [cls=2.0615 | reg=4.5000 | ctr=1.7763] | val=1090.1262 | lr=9.98e-04 | img=640 | 
INFO	2026-02-21 18:18:10 -0500	workerpool0-0	  Epoch   5 | train=8.2405 [cls=1.9682 | reg=4.5000 | ctr=1.7724] | val=959.1363 | lr=9.93e-04 | img=640 | ★ best
INFO	2026-02-21 18:18:27 -0500	workerpool0-0	  Epoch   6 | train=8.1778 [cls=1.9096 | reg=4.5000 | ctr=1.7681] | val=1632.8528 | lr=9.84e-04 | img=640 | 
INFO	2026-02-21 18:18:43 -0500	workerpool0-0	  Epoch   7 | train=8.1021 [cls=1.8348 | reg=4.5000 | ctr=1.7672] | val=1114.5944 | lr=9.71e-04 | img=640 | 
INFO	2026-02-21 18:18:59 -0500	workerpool0-0	  Epoch   8 | train=8.0526 [cls=1.7894 | reg=4.5000 | ctr=1.7631] | val=1334.3303 | lr=9.56e-04 | img=640 | 
INFO	2026-02-21 18:19:16 -0500	workerpool0-0	  Epoch   9 | train=8.0397 [cls=1.7778 | reg=4.5000 | ctr=1.7620] | val=951.5078 | lr=9.37e-04 | img=640 | ★ best
INFO	2026-02-21 18:19:29 -0500	workerpool0-0	  Epoch  10 | train=7.9370 [cls=1.6795 | reg=4.5000 | ctr=1.7575] | val=497.6752 | lr=9.14e-04 | img=416 | ★ best
INFO	2026-02-21 18:19:42 -0500	workerpool0-0	  Epoch  11 | train=7.7861 [cls=1.5387 | reg=4.5000 | ctr=1.7474] | val=431.4743 | lr=8.89e-04 | img=416 | ★ best
INFO	2026-02-21 18:19:55 -0500	workerpool0-0	  Epoch  12 | train=7.7989 [cls=1.5516 | reg=4.5000 | ctr=1.7473] | val=375.1511 | lr=8.61e-04 | img=416 | ★ best
INFO	2026-02-21 18:20:08 -0500	workerpool0-0	  Epoch  13 | train=7.7276 [cls=1.4846 | reg=4.5000 | ctr=1.7430] | val=308.4793 | lr=8.30e-04 | img=416 | ★ best
INFO	2026-02-21 18:20:21 -0500	workerpool0-0	  Epoch  14 | train=7.7146 [cls=1.4697 | reg=4.5000 | ctr=1.7449] | val=340.1859 | lr=7.97e-04 | img=416 | 
INFO	2026-02-21 18:20:34 -0500	workerpool0-0	  Epoch  15 | train=7.6663 [cls=1.4263 | reg=4.5000 | ctr=1.7399] | val=202.3285 | lr=7.62e-04 | img=416 | ★ best
INFO	2026-02-21 18:20:47 -0500	workerpool0-0	  Epoch  16 | train=7.7058 [cls=1.4664 | reg=4.5000 | ctr=1.7394] | val=381.1708 | lr=7.25e-04 | img=416 | 
INFO	2026-02-21 18:21:00 -0500	workerpool0-0	  Epoch  17 | train=7.6378 [cls=1.3975 | reg=4.5000 | ctr=1.7402] | val=391.7118 | lr=6.86e-04 | img=416 | 
INFO	2026-02-21 18:21:14 -0500	workerpool0-0	  Epoch  18 | train=7.5789 [cls=1.3448 | reg=4.5000 | ctr=1.7341] | val=293.0873 | lr=6.46e-04 | img=416 | 
INFO	2026-02-21 18:21:27 -0500	workerpool0-0	  Epoch  19 | train=7.5790 [cls=1.3442 | reg=4.5000 | ctr=1.7349] | val=230.0124 | lr=6.05e-04 | img=416 | 
INFO	2026-02-21 18:21:39 -0500	workerpool0-0	  Epoch  20 | train=7.6119 [cls=1.3755 | reg=4.5000 | ctr=1.7363] | val=265.9421 | lr=5.64e-04 | img=320 | 
INFO	2026-02-21 18:21:51 -0500	workerpool0-0	  Epoch  21 | train=7.5893 [cls=1.3547 | reg=4.5000 | ctr=1.7346] | val=264.9039 | lr=5.21e-04 | img=320 | 
INFO	2026-02-21 18:22:03 -0500	workerpool0-0	  Epoch  22 | train=7.5616 [cls=1.3288 | reg=4.5000 | ctr=1.7328] | val=220.0754 | lr=4.79e-04 | img=320 | 
INFO	2026-02-21 18:22:16 -0500	workerpool0-0	  Epoch  23 | train=7.5327 [cls=1.3036 | reg=4.5000 | ctr=1.7292] | val=146.4778 | lr=4.37e-04 | img=320 | ★ best
INFO	2026-02-21 18:22:28 -0500	workerpool0-0	  Epoch  24 | train=7.4776 [cls=1.2523 | reg=4.5000 | ctr=1.7253] | val=112.0594 | lr=3.95e-04 | img=320 | ★ best
INFO	2026-02-21 18:22:40 -0500	workerpool0-0	  Epoch  25 | train=7.4570 [cls=1.2313 | reg=4.5000 | ctr=1.7257] | val=151.4857 | lr=3.54e-04 | img=320 | 
INFO	2026-02-21 18:22:52 -0500	workerpool0-0	  Epoch  26 | train=7.4658 [cls=1.2395 | reg=4.5000 | ctr=1.7263] | val=131.4550 | lr=3.14e-04 | img=320 | 
INFO	2026-02-21 18:23:05 -0500	workerpool0-0	  Epoch  27 | train=7.4560 [cls=1.2305 | reg=4.5000 | ctr=1.7255] | val=141.3758 | lr=2.75e-04 | img=320 | 
INFO	2026-02-21 18:23:17 -0500	workerpool0-0	  Epoch  28 | train=7.4317 [cls=1.2068 | reg=4.5000 | ctr=1.7249] | val=210.4034 | lr=2.38e-04 | img=320 | 
INFO	2026-02-21 18:23:29 -0500	workerpool0-0	  Epoch  29 | train=7.4111 [cls=1.1884 | reg=4.5000 | ctr=1.7227] | val=137.6530 | lr=2.03e-04 | img=320 | 
INFO	2026-02-21 18:23:41 -0500	workerpool0-0	  Epoch  30 | train=7.6489 [cls=1.3942 | reg=4.5000 | ctr=1.7547] | val=112.8412 | lr=1.70e-04 | img=224 | 
INFO	2026-02-21 18:23:52 -0500	workerpool0-0	  Epoch  31 | train=7.5975 [cls=1.3493 | reg=4.5000 | ctr=1.7482] | val=71.6401 | lr=1.39e-04 | img=224 | ★ best
INFO	2026-02-21 18:24:04 -0500	workerpool0-0	  Epoch  32 | train=7.5447 [cls=1.2971 | reg=4.5000 | ctr=1.7476] | val=85.2934 | lr=1.11e-04 | img=224 | 
INFO	2026-02-21 18:24:15 -0500	workerpool0-0	  Epoch  33 | train=7.5440 [cls=1.2967 | reg=4.5000 | ctr=1.7474] | val=64.3115 | lr=8.58e-05 | img=224 | ★ best
INFO	2026-02-21 18:24:27 -0500	workerpool0-0	  Epoch  34 | train=7.5381 [cls=1.2917 | reg=4.5000 | ctr=1.7463] | val=59.2822 | lr=6.36e-05 | img=224 | ★ best
INFO	2026-02-21 18:24:39 -0500	workerpool0-0	  Epoch  35 | train=7.4831 [cls=1.2397 | reg=4.5000 | ctr=1.7433] | val=67.7065 | lr=4.45e-05 | img=224 | 
INFO	2026-02-21 18:24:50 -0500	workerpool0-0	  Epoch  36 | train=7.5010 [cls=1.2566 | reg=4.5000 | ctr=1.7444] | val=63.2392 | lr=2.87e-05 | img=224 | 
INFO	2026-02-21 18:25:02 -0500	workerpool0-0	  Epoch  37 | train=7.4906 [cls=1.2474 | reg=4.5000 | ctr=1.7433] | val=68.2850 | lr=1.62e-05 | img=224 | 
INFO	2026-02-21 18:25:14 -0500	workerpool0-0	  Epoch  38 | train=7.4832 [cls=1.2388 | reg=4.5000 | ctr=1.7444] | val=66.7409 | lr=7.29e-06 | img=224 | 
INFO	2026-02-21 18:25:25 -0500	workerpool0-0	  Epoch  39 | train=7.4914 [cls=1.2460 | reg=4.5000 | ctr=1.7454] | val=66.0402 | lr=1.90e-06 | img=224 | 
INFO	2026-02-21 18:25:25 -0500	workerpool0-0	✅ Phase 1 (backbone frozen) completada en 9.4 min
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	🔄 Mejor checkpoint de Phase 1 recargado
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	🔓 Todas las capas desbloqueadas: 927,008 params unfrozen
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	   Total trainable: 1,233,450
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	🚀 Phase 2 (full fine-tuning) — 80 epochs, LR=0.0001
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	⚙️  Optimizer: ADAMW (lr=0.0001, wd=1e-05)
INFO	2026-02-21 18:25:26 -0500	workerpool0-0	📈 Scheduler: CosineAnnealing (T_max=80)
INFO	2026-02-21 18:25:39 -0500	workerpool0-0	  Epoch  40 | train=7.4704 [cls=1.2291 | reg=4.5000 | ctr=1.7413] | val=117.5170 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 18:25:52 -0500	workerpool0-0	  Epoch  41 | train=7.3725 [cls=1.1375 | reg=4.5000 | ctr=1.7351] | val=59.2831 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 18:26:05 -0500	workerpool0-0	  Epoch  42 | train=6.3160 [cls=1.0859 | reg=3.4977 | ctr=1.7325] | val=97.5668 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 18:26:19 -0500	workerpool0-0	  Epoch  43 | train=5.1917 [cls=1.0307 | reg=2.4335 | ctr=1.7275] | val=77.2554 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 18:26:32 -0500	workerpool0-0	  Epoch  44 | train=4.8885 [cls=0.9969 | reg=2.1675 | ctr=1.7241] | val=75.1473 | lr=1.00e-04 | img=224 | 
INFO	2026-02-21 18:26:45 -0500	workerpool0-0	  Epoch  45 | train=4.7057 [cls=0.9720 | reg=2.0141 | ctr=1.7195] | val=65.5019 | lr=9.98e-05 | img=224 | 
INFO	2026-02-21 18:26:58 -0500	workerpool0-0	  Epoch  46 | train=4.5886 [cls=0.9563 | reg=1.9145 | ctr=1.7177] | val=52.5386 | lr=9.97e-05 | img=224 | ★ best
INFO	2026-02-21 18:27:11 -0500	workerpool0-0	  Epoch  47 | train=4.4223 [cls=0.9024 | reg=1.8051 | ctr=1.7148] | val=69.4074 | lr=9.94e-05 | img=224 | 
INFO	2026-02-21 18:27:24 -0500	workerpool0-0	  Epoch  48 | train=4.3344 [cls=0.8787 | reg=1.7430 | ctr=1.7127] | val=52.5078 | lr=9.90e-05 | img=224 | ★ best
INFO	2026-02-21 18:27:37 -0500	workerpool0-0	  Epoch  49 | train=4.2592 [cls=0.8616 | reg=1.6867 | ctr=1.7108] | val=59.2220 | lr=9.86e-05 | img=224 | 
INFO	2026-02-21 18:27:51 -0500	workerpool0-0	  Epoch  50 | train=4.2025 [cls=0.8473 | reg=1.6434 | ctr=1.7118] | val=64.3254 | lr=9.81e-05 | img=224 | 
INFO	2026-02-21 18:28:04 -0500	workerpool0-0	  Epoch  51 | train=4.1291 [cls=0.8192 | reg=1.6033 | ctr=1.7065] | val=38.5634 | lr=9.76e-05 | img=224 | ★ best
INFO	2026-02-21 18:28:17 -0500	workerpool0-0	  Epoch  52 | train=4.1049 [cls=0.8191 | reg=1.5784 | ctr=1.7074] | val=46.3070 | lr=9.69e-05 | img=224 | 
INFO	2026-02-21 18:28:30 -0500	workerpool0-0	  Epoch  53 | train=4.0171 [cls=0.7854 | reg=1.5285 | ctr=1.7032] | val=48.6276 | lr=9.62e-05 | img=224 | 
INFO	2026-02-21 18:28:44 -0500	workerpool0-0	  Epoch  54 | train=4.0250 [cls=0.7852 | reg=1.5365 | ctr=1.7033] | val=78.5550 | lr=9.54e-05 | img=224 | 
INFO	2026-02-21 18:28:57 -0500	workerpool0-0	  Epoch  55 | train=3.9633 [cls=0.7810 | reg=1.4808 | ctr=1.7015] | val=50.2891 | lr=9.46e-05 | img=224 | 
INFO	2026-02-21 18:29:10 -0500	workerpool0-0	  Epoch  56 | train=3.9009 [cls=0.7474 | reg=1.4516 | ctr=1.7019] | val=64.6499 | lr=9.36e-05 | img=224 | 
INFO	2026-02-21 18:29:23 -0500	workerpool0-0	  Epoch  57 | train=3.8692 [cls=0.7330 | reg=1.4386 | ctr=1.6976] | val=47.7405 | lr=9.26e-05 | img=224 | 
INFO	2026-02-21 18:29:36 -0500	workerpool0-0	  Epoch  58 | train=3.8671 [cls=0.7461 | reg=1.4224 | ctr=1.6985] | val=82.2821 | lr=9.16e-05 | img=224 | 
INFO	2026-02-21 18:29:49 -0500	workerpool0-0	  Epoch  59 | train=3.8011 [cls=0.7127 | reg=1.3926 | ctr=1.6958] | val=68.3829 | lr=9.05e-05 | img=224 | 
INFO	2026-02-21 18:30:02 -0500	workerpool0-0	  Epoch  60 | train=3.7421 [cls=0.6932 | reg=1.3533 | ctr=1.6955] | val=40.2314 | lr=8.93e-05 | img=224 | 
INFO	2026-02-21 18:30:16 -0500	workerpool0-0	  Epoch  61 | train=3.7738 [cls=0.7017 | reg=1.3781 | ctr=1.6941] | val=36.9275 | lr=8.80e-05 | img=224 | ★ best
INFO	2026-02-21 18:30:29 -0500	workerpool0-0	  Epoch  62 | train=3.7280 [cls=0.6982 | reg=1.3383 | ctr=1.6915] | val=65.8181 | lr=8.67e-05 | img=224 | 
INFO	2026-02-21 18:30:42 -0500	workerpool0-0	  Epoch  63 | train=3.7175 [cls=0.6821 | reg=1.3399 | ctr=1.6955] | val=75.7193 | lr=8.54e-05 | img=224 | 
INFO	2026-02-21 18:30:55 -0500	workerpool0-0	  Epoch  64 | train=3.6595 [cls=0.6645 | reg=1.3045 | ctr=1.6905] | val=52.2446 | lr=8.40e-05 | img=224 | 
INFO	2026-02-21 18:31:08 -0500	workerpool0-0	  Epoch  65 | train=3.6529 [cls=0.6552 | reg=1.3076 | ctr=1.6901] | val=33.3135 | lr=8.25e-05 | img=224 | ★ best
INFO	2026-02-21 18:31:21 -0500	workerpool0-0	  Epoch  66 | train=3.6319 [cls=0.6473 | reg=1.2973 | ctr=1.6873] | val=58.4150 | lr=8.10e-05 | img=224 | 
INFO	2026-02-21 18:31:35 -0500	workerpool0-0	  Epoch  67 | train=3.6080 [cls=0.6453 | reg=1.2759 | ctr=1.6868] | val=41.5704 | lr=7.94e-05 | img=224 | 
INFO	2026-02-21 18:31:48 -0500	workerpool0-0	  Epoch  68 | train=3.5826 [cls=0.6304 | reg=1.2644 | ctr=1.6877] | val=48.2946 | lr=7.78e-05 | img=224 | 
INFO	2026-02-21 18:32:01 -0500	workerpool0-0	  Epoch  69 | train=3.5681 [cls=0.6353 | reg=1.2489 | ctr=1.6840] | val=41.7264 | lr=7.61e-05 | img=224 | 
INFO	2026-02-21 18:32:14 -0500	workerpool0-0	  Epoch  70 | train=3.5396 [cls=0.6133 | reg=1.2410 | ctr=1.6852] | val=60.2868 | lr=7.45e-05 | img=224 | 
INFO	2026-02-21 18:32:27 -0500	workerpool0-0	  Epoch  71 | train=3.5231 [cls=0.6093 | reg=1.2277 | ctr=1.6861] | val=57.1623 | lr=7.27e-05 | img=224 | 
INFO	2026-02-21 18:32:40 -0500	workerpool0-0	  Epoch  72 | train=3.5297 [cls=0.6192 | reg=1.2243 | ctr=1.6862] | val=49.9242 | lr=7.10e-05 | img=224 | 
INFO	2026-02-21 18:32:54 -0500	workerpool0-0	  Epoch  73 | train=3.5077 [cls=0.6145 | reg=1.2085 | ctr=1.6847] | val=49.3598 | lr=6.92e-05 | img=224 | 
INFO	2026-02-21 18:33:07 -0500	workerpool0-0	  Epoch  74 | train=3.4660 [cls=0.5923 | reg=1.1907 | ctr=1.6830] | val=47.1740 | lr=6.73e-05 | img=224 | 
INFO	2026-02-21 18:33:20 -0500	workerpool0-0	  Epoch  75 | train=3.4621 [cls=0.5866 | reg=1.1952 | ctr=1.6803] | val=48.7388 | lr=6.55e-05 | img=224 | 
INFO	2026-02-21 18:33:33 -0500	workerpool0-0	  Epoch  76 | train=3.4702 [cls=0.5967 | reg=1.1932 | ctr=1.6802] | val=48.8524 | lr=6.36e-05 | img=224 | 
INFO	2026-02-21 18:33:47 -0500	workerpool0-0	  Epoch  77 | train=3.4096 [cls=0.5680 | reg=1.1612 | ctr=1.6804] | val=48.2970 | lr=6.17e-05 | img=224 | 
INFO	2026-02-21 18:34:00 -0500	workerpool0-0	  Epoch  78 | train=3.4543 [cls=0.5805 | reg=1.1938 | ctr=1.6800] | val=57.2978 | lr=5.98e-05 | img=224 | 
INFO	2026-02-21 18:34:13 -0500	workerpool0-0	  Epoch  79 | train=3.4133 [cls=0.5754 | reg=1.1595 | ctr=1.6784] | val=36.2762 | lr=5.79e-05 | img=224 | 
INFO	2026-02-21 18:34:26 -0500	workerpool0-0	  Epoch  80 | train=3.4362 [cls=0.5873 | reg=1.1689 | ctr=1.6800] | val=39.9388 | lr=5.59e-05 | img=224 | 
INFO	2026-02-21 18:34:40 -0500	workerpool0-0	  Epoch  81 | train=3.4042 [cls=0.5676 | reg=1.1568 | ctr=1.6798] | val=53.7481 | lr=5.40e-05 | img=224 | 
INFO	2026-02-21 18:34:53 -0500	workerpool0-0	  Epoch  82 | train=3.3980 [cls=0.5601 | reg=1.1582 | ctr=1.6797] | val=50.7712 | lr=5.20e-05 | img=224 | 
INFO	2026-02-21 18:35:06 -0500	workerpool0-0	  Epoch  83 | train=3.3448 [cls=0.5381 | reg=1.1287 | ctr=1.6781] | val=43.7994 | lr=5.01e-05 | img=224 | 
INFO	2026-02-21 18:35:19 -0500	workerpool0-0	  Epoch  84 | train=3.3735 [cls=0.5558 | reg=1.1388 | ctr=1.6788] | val=55.5816 | lr=4.81e-05 | img=224 | 
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	  Epoch  85 | train=3.3472 [cls=0.5485 | reg=1.1197 | ctr=1.6791] | val=46.4658 | lr=4.61e-05 | img=224 | 
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	⏹️  Early stopping at epoch 85 (patience=20)
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	✅ Phase 2 (full fine-tuning) completada en 10.1 min
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	📊 Entrenamiento completo: 86 epochs
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	   Mejor val_loss: 33.3135 (epoch 65)
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	   Tiempo total: 19.5 min
INFO	2026-02-21 18:35:32 -0500	workerpool0-0	⏱️  Entrenamiento completado en 19.5 min
INFO	2026-02-21 18:35:34 -0500	workerpool0-0	💾 Historial guardado: /tmp/training/training_history.csv
INFO	2026-02-21 18:35:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:34 -0500	workerpool0-0	BLOQUE 5 — Curvas de Entrenamiento
INFO	2026-02-21 18:35:34 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	📊 Curvas guardadas: /tmp/training/training_curves.png
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	📈 Resumen – PYTORCH 
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	  Épocas: 86
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	  Mejor val_loss: 33.3135 (epoch 65)
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	  Resoluciones: [640, 416, 320, 224]
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	BLOQUE 6 — Evaluación en Validación
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:35 -0500	workerpool0-0	✅ Cargado mejor checkpoint: /tmp/training/checkpoints/best_fcos.pt
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=val
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  mAP@50:    0.3799
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  mAP@50-95: 0.1632
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  Precision: 0.2904
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  Recall:    0.4778
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  F1-Score:  0.3613
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  Imágenes: 188 | Detecciones: 1289 | GT: 762
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  Avg inference: 5.5 ms
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	    dog                   0.2845
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	    door                  0.3847
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	    obstacle              0.4081
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	    person                0.4096
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	    stair                 0.4128
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	📊 Val mAP@50: 0.3799
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	   dog: 0.2845
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	   door: 0.3847
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	   obstacle: 0.4081
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	   person: 0.4096
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	   stair: 0.4128
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/val_confusion_matrix.png
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	📊 Per-class metrics guardados: /tmp/training/val_per_class.png
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/val_evaluation.json
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	BLOQUE 7 — Evaluación en Test
INFO	2026-02-21 18:35:37 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	📊 Evaluación: fcos_v3s (FCOS) – split=test
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  mAP@50:    0.5572
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  mAP@50-95: 0.2511
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  Precision: 0.3290
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  Recall:    0.6558
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  F1-Score:  0.4382
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  Imágenes: 187 | Detecciones: 1124 | GT: 576
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  Avg inference: 5.1 ms
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	  Per-class AP@50:
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	    dog                   0.4424
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	    door                  0.4642
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	    obstacle              0.4600
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	    person                0.6834
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	    stair                 0.7359
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	📊 Test mAP@50: 0.5572
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	📊 Confusion matrix guardada: /tmp/training/test_confusion_matrix.png
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	💾 Evaluación guardada: /tmp/training/test_evaluation.json
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	BLOQUE 8 — Guardado y subida a GCS
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	============================================================
INFO	2026-02-21 18:35:39 -0500	workerpool0-0	📦 Exportando FCOS → ONNX (opset=13, shape=[1, 3, 224, 224], outputs=9)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	  ℹ️ onnxsim not installed, skipping simplification
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	  ✅ Exportado: /tmp/training/export/fcos_v3s.onnx (4.74 MB, 0.8s)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	  ✅ ONNX verificado: /tmp/training/export/fcos_v3s.onnx
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     Latencia mediana: 6.1ms
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     cls_lvl0: (1, 5, 28, 28)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     cls_lvl1: (1, 5, 14, 14)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     cls_lvl2: (1, 5, 7, 7)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     reg_lvl0: (1, 4, 28, 28)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     reg_lvl1: (1, 4, 14, 14)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     reg_lvl2: (1, 4, 7, 7)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     centerness_lvl0: (1, 1, 28, 28)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     centerness_lvl1: (1, 1, 14, 14)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	     centerness_lvl2: (1, 1, 7, 7)
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	💾 Experimento guardado: /tmp/training/experiment.json
INFO	2026-02-21 18:35:40 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/config.yaml → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/config.yaml
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_history.csv → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/training_history.csv
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/training_curves.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/training_curves.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/class_distribution.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/class_distribution.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/gt_samples.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/gt_samples.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/val_confusion_matrix.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_per_class.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/val_per_class.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/val_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/val_evaluation.json
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_confusion_matrix.png → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/test_confusion_matrix.png
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/test_evaluation.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/test_evaluation.json
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/experiment.json → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/experiment.json
INFO	2026-02-21 18:35:41 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/checkpoints/best_fcos.pt → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/checkpoints/best_fcos.pt
INFO	2026-02-21 18:35:42 -0500	workerpool0-0	  ☁️  Subido: /tmp/training/export/fcos_v3s.onnx → gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/export/fcos_v3s.onnx
INFO	2026-02-21 18:35:42 -0500	workerpool0-0	✅ Pipeline FCOS completado exitosamente.
INFO	2026-02-21 18:35:44 -0500	workerpool0-0	Task completed. Exit code (0). Exit reason (SUCCEEDED)
INFO	2026-02-21 18:36:02 -0500	service	Tearing down training program.
INFO	2026-02-21 18:36:49 -0500	service	Finished tearing down training program.
INFO	2026-02-21 18:36:49 -0500	service	Job completed successfully.
```