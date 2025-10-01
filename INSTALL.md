# Training and evaluation environment setup

```bash
conda create -n djepa python=3.10 -y

conda activate djepa

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -U diffusers transformers accelerate scipy ftfy safetensors

pip3 install -U opencv-python omegaconf matplotlib torchdiffeq timm tensorboard webdataset einops tfrecord torchdata
```
