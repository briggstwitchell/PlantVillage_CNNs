#!/bin/bash
conda create --name TF_env python=3.9 -y
source activate TF_env
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install --upgrade pip
pip install tensorflow==2.13.*
pip install tensorflow-gpu
pip install chardet
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=TF_env