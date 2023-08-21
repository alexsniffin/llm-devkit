FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /devkit

COPY requirements.txt .
RUN pip3 install -r requirements.txt