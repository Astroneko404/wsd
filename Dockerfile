# To build docker:
# 1) Go to /home/luoz3/wsd_clone/
# 2) sudo docker build --tag=wsd .
#
# To run docker (with mounted folder)
# sudo docker run -it --name=wsd_test --mount type=bind,src=/home/luoz3/wsd_data_test,dst=/wsd_data_test wsd

FROM ubuntu:18.04

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y gcc git g++ python3.6 python3-pip python3-setuptools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# gensim and sklearn requires NumPy and Scipy
RUN pip3 install numpy scipy tqdm \
 && pip3 install gensim scikit-learn

# Environment for fastText
RUN pip3 install pybind11 \
 && pip3 install Cython --install-option="--no-cython-compile"

# fastText installation
RUN git clone https://github.com/facebookresearch/fastText.git \
 && cd fastText \
 && pip3 install .

WORKDIR "/"

# Copy dependency files
COPY /baseline/ /wsd_clone/baseline/
COPY /model/ /wsd_clone/model/
COPY /pipeline/ /wsd_clone/pipeline/
COPY /preprocess/ /wsd_clone/preprocess/
COPY /util/ /wsd_clone/util/

ENTRYPOINT ["/bin/bash"]