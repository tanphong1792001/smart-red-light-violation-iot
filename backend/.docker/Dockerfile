FROM ultralytics/ultralytics:latest as dev

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
     build-essential \
    git \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean -y


RUN apt-get update \
    && pip install --upgrade pip \
    && pip install fastapi python-multipart easyocr uvicorn


