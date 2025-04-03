# Brain Foundation Model
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)


## Steps to reproduce
1. Data preparation.
If you work on Misha, you can find the preprocessed data at `/gpfs/radev/home/cl2482/project/BrainFM/data/Dynamic_Natural_Vision`.

1.1 Download Dynamic Natural Vision data.
```
cd src/data_download
python download_DNV.py
```

1.2 Unzip (sorry this step is messy)
The desired structure is:
data/Dynamic_Natural_Vision/{subject1,subject2,subject3,video}
Under {subject1,subject2,subject3} there are {fmri/smri} folders.
Under {video} there are {seg1.mp4,seg2.mp4,...,test5.mp4}.

1.3 Preprocess the video. This will create a {video_frames} folder.
```
cd src/preprocessing
python preprocess_DNV_videos.py
```

1.4 Preprocess the fMRI. This will create a {fMRI_scans} folder.
```
cd src/preprocessing
python preprocess_DNV_fmri.py
```

2. Train.
```
cd src/
python main.py --batch-size 4 --max-training-iters 64 --max-validation-iters 64
```


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name brainfm pytorch==2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c anaconda -c conda-forge -y
conda activate brainfm
conda install scikit-learn scikit-image pillow matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge -y
conda install accelerate -c conda-forge -y
conda install nibabel -y
python -m pip install opencv-python
python -m pip install torch_geometric einops
python -m pip install git+https://github.com/openai/CLIP.git


# For NeuroClips
python -m pip install webdataset pytorch-lightning einops kornia open-clip-torch omegaconf transformers
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install diffusers["torch"]==0.21.4 transformers huggingface_hub==0.25.2
python -m pip install xformers==0.0.22.post7
python -m pip install dalle2-pytorch==1.15.6
python -m pip install huggingface_hub
python -m pip install natsort
```


<!-- conda install read-roi -c conda-forge
python -m pip install -U albumentations
python -m pip install timm
python -m pip install opencv-python
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
python -m pip install monai
python -m pip install torchdiffeq
python -m pip install torch-ema
python -m pip install torchcde
python -m pip install torchsde
python -m pip install phate
python -m pip install psutil
python -m pip install ninja -->