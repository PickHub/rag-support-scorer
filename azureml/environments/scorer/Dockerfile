FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04@sha256:2f88524bcaaaeecd457236fe6ab35965a780640449402c57aab14de91f5569b9

ARG INSTALL_BITSANDBYTES=1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements*.txt /opt/rag-support-scorer/
RUN python -m pip install --no-cache-dir \
      -r /opt/rag-support-scorer/requirements.txt && \
    if [ "$INSTALL_BITSANDBYTES" = "1" ]; then \
      python -m pip install --no-cache-dir \
        -r /opt/rag-support-scorer/requirements-bitsandbytes.txt; \
    fi

WORKDIR /mnt/azureml/cr/j
