#!/usr/bin/env bash
conda create -n py36 python=3.6 -y
conda install pytorch torchvision cuda80 -c soumith -y
conda install visdom -c conda-forge -y
pip install git+https://github.com/pytorch/pytorch
pip install git+https://github.com/pytorch/vision

conda install scipy