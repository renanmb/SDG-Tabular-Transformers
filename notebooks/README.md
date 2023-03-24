Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for ongoing research on training large transformer language models at scale. We developed efficient, model-parallel (tensor and pipeline), and multi-node pre-training oftransformer based models such as [GPT](https://arxiv.org/abs/2005.14165), [BERT](https://arxiv.org/pdf/1810.04805.pdf), and [T5](https://arxiv.org/abs/1910.10683) using mixed precision.


# Download Example [TabFormer Credit Card Data](https://github.com/IBM/TabFormer)

Download it from IBM Box: [https://ibm.box.com/v/tabformer-data](https://github.com/IBM/TabFormer)

# Setup
We have tested Megatron with [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) version 20.12, which uses python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3.

To use this repository, please install the latest supported versions of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We strongly recommend using one of [NGC's recent PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) (the latest compatible version at time of publication can be pulled with `docker pull nvcr.io/nvidia/pytorch:20.12-py3`). Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

