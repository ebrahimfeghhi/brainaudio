#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n brainaudio python=3.12.9 -y 

conda activate brainaudio 

pip install --upgrade pip 

pip install -r requirements.txt

echo
echo "Setup complete, verify it worked by activating the conda environment with the command 'conda activate brainaudio'."
echo