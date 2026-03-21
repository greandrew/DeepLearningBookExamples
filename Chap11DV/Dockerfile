FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --break-system-packages notebook pandas scipy scikit-learn torch-metrics numpy torchsummary IProgress ipywidgets matplotlib quantlib yfinance --upgrade --no-cache-dir fdc tqdm

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
