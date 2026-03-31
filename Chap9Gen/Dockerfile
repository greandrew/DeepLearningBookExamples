FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
COPY requirements.txt requirements.txt
RUN pip install --break-system-package -r requirements.txt
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
