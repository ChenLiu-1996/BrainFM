import os
import re
from glob import glob
from multiprocessing import Pool

data_zip_folder = '/gpfs/radev/scratch/krishnaswamy_smita/cl2482/BrainFM_2024_HBN/zip/'
data_unzip_folder = '/gpfs/radev/scratch/krishnaswamy_smita/cl2482/BrainFM_2024_HBN/'

def unzip(zipped_file: str) -> None:
    os.system(f'tar -xvzf {zipped_file} -C {unzip_dir}')


if __name__ == '__main__':
    for subfolder in ['MRI', 'EEG']:

        # Find the zipped files.
        zipped_file_list = glob(f'{data_zip_folder}/{subfolder}/*.tar.gz')

        # Unzip.
        unzip_dir = os.path.join(data_unzip_folder, subfolder)
        os.makedirs(unzip_dir, exist_ok=True)
        Pool(8).map(unzip, zipped_file_list)