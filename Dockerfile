FROM continuumio/miniconda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgsl0-dev

RUN conda install -c anaconda gxx_linux-64
RUN conda install -y biopython git future
RUN conda install -c etetoolkit ete3

RUN git clone https://github.com/pgfoster/p4-phylogenetics.git
RUN cd p4-phylogenetics && python setup.py install
RUN pip install phyloinfer
