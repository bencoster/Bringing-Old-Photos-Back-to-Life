#!/bin/bash
sudo install python3

cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../

cd Face_Enhancement/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip
cd ../
cd Global/
wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip
cd ../

pip3 install -r requirements.txt

python3 get_models.py