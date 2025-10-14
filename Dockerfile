FROM jupyter/base-notebook:latest

USER root
RUN apt-get update && apt-get install -y g++ && apt-get clean

RUN pip install mlflow \
    && pip install scikit-learn \
    && pip install matplotlib \
    && pip install IProgress \
    && pip install ipywidgets \
    && pip install cmdstanpy

RUN python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"

USER jovyan

# disable security
RUN echo "c.NotebookApp.token = ''" >> /etc/jupyter/jupyter_notebook_config.py
WORKDIR /notebook/

EXPOSE 80
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=80"]
