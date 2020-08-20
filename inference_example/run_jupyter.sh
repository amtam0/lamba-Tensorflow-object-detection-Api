apt update
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
cd inference_example
pip install -r requirements.txt
jupyter notebook --ip=0.0.0.0 --allow-root
