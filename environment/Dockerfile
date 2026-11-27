# hash:sha256:571b2f0bc0254303f6e266f10bfd519841597a05d639593e00ab39517123b9ad
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.10.9 \
        numpy==2.5.3 \
        pandas==3.0.5 \
        pytorch-lightning==2.6.5 \
        scikit-learn==1.9.0 \
        scipy==1.18.0 \
        seaborn==0.13.2 \
        tensorboard==2.20.0 \
        torchvision==0.19.0 \
        tqdm==4.70.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    datasets==5.0.1 \
    gpytorch==1.15.2 \
    h5py==3.16.0 \
    jsonargparse[signatures]==4.52.0 \
    ucimlrepo==0.0.7
