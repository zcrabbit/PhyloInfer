FROM continuumio/anaconda2:2019.07

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl-dev \
    libgsl0-dev

COPY environment.yml .
RUN /opt/conda/bin/conda env create -f environment.yml

RUN git clone https://github.com/pgfoster/p4-phylogenetics.git
RUN cd p4-phylogenetics && git checkout 388099345b3a37b3156e52e87a65a7c231eb53c8 && /opt/conda/bin/conda run -n phyloinfer python setup.py install

RUN /opt/conda/bin/conda run -n phyloinfer pip install phyloinfer
