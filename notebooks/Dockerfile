FROM nvcr.io/nvidia/pytorch:22.05-py3

WORKDIR /sdg

COPY checkpoints /sdg/checkpoints
COPY data /sdg/data
COPY megatron /sdg/megatron
COPY examples /sdg/examples
COPY tools /sdg/tools
COPY tasks /sdg/tasks

RUN conda install -c conda-forge nodejs && \
    pip3 install flask-restful iso18245 seaborn aquirdturtle_collapsible_headings && \
    jupyter lab build

COPY README.md /sdg/
COPY LICENSE /sdg/
COPY coder /sdg/coder
COPY images /sdg/images
COPY Dockerfile /sdg/Dockerfile
COPY *.sh /sdg/
COPY *.py /sdg/
COPY *.ipynb /sdg/

EXPOSE 6006
ENTRYPOINT jupyter lab --NotebookApp.token '' --allow-root --ip 0.0.0.0 --port 8888


