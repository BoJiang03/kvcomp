#! /bin/bash

cd ~
mkdir micromamba
cd micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ./micromamba_install

source ~/.bashrc
