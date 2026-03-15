FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
        torchtext==0.18.0 \
        transformers==4.41.2 \
        datasets==2.20.0 \
        scikit-learn==1.4.2 \
        numpy==1.26.4 \
        scipy==1.11.4 \
        pandas==2.2.2 \
        notebook \
        torchmetrics \
        torchsummary \
        IProgress \
        ipywidgets \
        matplotlib \
        quantlib \
        yfinance \
        genomic-benchmarks \
        nltk \
        gensim \
        spacy \
        seaborn && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm && \
    python -m spacy download de_core_news_sm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
