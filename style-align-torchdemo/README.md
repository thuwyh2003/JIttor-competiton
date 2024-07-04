# Style Aligned PyTorch Demo Usege

## Set Environment

```text
conda create -n style-aligned-torchdemo python=3.10
conda activate style-aligned-torchdemo
```

### Install PyTorch

```text
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Install Other Dependencies

```text
pip install -r requirement.txt
```

## Test Style Aligned Text to Image

```text
python test_sa_sdxl.py
```

## Test Style Aligned + Multidiffusion

```text
python test_sa_md.py
```
