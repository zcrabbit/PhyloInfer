FROM continuumio/anaconda2:2019.07

COPY environment.yml .

RUN /opt/conda/bin/conda env create -f environment.yml

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgsl0-dev

#RUN conda install -c anaconda gxx_linux-64

RUN git clone https://github.com/pgfoster/p4-phylogenetics.git
RUN cd p4-phylogenetics && git checkout 46e91e5e511d70c3f4eb76db0097dd9e4f3fc96a && /opt/conda/bin/conda run -n phyloinfer python setup.py install
RUN /opt/conda/bin/conda run -n phyloinfer pip install phyloinfer
