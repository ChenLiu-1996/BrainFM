# Brain Foundation Model
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name brainfm pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge
conda activate brainfm
conda install scikit-learn scikit-image pillow matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
conda install -c conda-forge accelerate

python -m pip install webdataset pytorch-lightning
```
<!-- python -m pip install -r requirements_neuroclips.txt -->


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