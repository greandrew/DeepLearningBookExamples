FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel && \
    pip install "setuptools<82"

RUN pip install \
    torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu130

RUN pip install \
    notebook scipy scikit-learn torchmetrics pandas==1.5.3 numpy torchsummary \
    ipywidgets matplotlib seaborn quantlib tqdm optuna flask \
    gpy gpyopt parameter-sherpa

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

