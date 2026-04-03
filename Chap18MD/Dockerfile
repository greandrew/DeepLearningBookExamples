FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages --no-cache-dir \
    yfinance \
    pandas \
    scipy \
    scikit-learn \
    torchmetrics \
    numpy \
    torchsummary \
    IProgress \
    ipywidgets \
    matplotlib \
    statsmodels \
    hurst \
    svolfit \
    arch \
    torchcde \
    torchsde \
    fbm \
    openpyxl \
    adjustText \
    notebook \
    tqdm

RUN pip install --break-system-packages --no-cache-dir --no-build-isolation signatory

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
