FROM continuumio/anaconda2:2019.07

COPY environment.yml .

RUN /opt/conda/bin/conda env create -f environment.yml

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgsl0-dev

#RUN conda install -c anaconda gxx_linux-64

RUN git clone https://github.com/pgfoster/p4-phylogenetics.git
COPY p4-phylogenetics .
RUN cd p4-phylogenetics
RUN /opt/conda/bin/conda run -n libsbn python setup.py install
RUN /opt/conda/bin/conda run -n libsbn pip install phyloinfer
