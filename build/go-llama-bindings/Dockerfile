FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y git wget cmake && \
    git clone --recursive https://github.com/go-skynet/go-llama.cpp.git

RUN cd go-llama.cpp && make libbinding.a

RUN wget https://golang.org/dl/go1.21.1.linux-amd64.tar.gz && \
    tar -xvf go1.21.1.linux-amd64.tar.gz && \
    mv go /usr/local && \
    rm go1.21.1.linux-amd64.tar.gz

ENV GOPATH=/go
ENV PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

WORKDIR /dev