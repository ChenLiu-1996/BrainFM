'''
Loader of the Dynamic Natural Vision dataset.
The dataset contains acquisitions of fMRI.
'''
from typing import Tuple

import os
import numpy as np
from glob import glob
import nibabel as nib
from torch.utils.data import Dataset
from natsort import natsorted


class DynamicNaturalVisionDataset(Dataset):
    def __init__(self,
                 base_path: str = '../../data/Dynamic_Natural_Vision/',
                 subject_idx: int = None,
                 mode: str = 'train'):

        assert mode in ['train', 'test'], f'`DynamicNaturalVisionDataset` mode must be `train` or `test`, but got {mode}.'
        self.mode = mode
        string_for_mode = {
            'train': 'seg',
            'test': 'test',
        }

        if subject_idx is None:
            self.fmri_paths = np.array(natsorted(glob(os.path.join(base_path, 'subject*', 'fmri', f'{string_for_mode[self.mode]}*', 'mni', '*.nii.gz'))))
        else:
            self.fmri_paths = np.array(natsorted(glob(os.path.join(base_path, f'subject{subject_idx}', 'fmri', f'{string_for_mode[self.mode]}*', 'mni', '*.nii.gz'))))

        self.video_frame_paths = np.array(natsorted(glob(os.path.join(base_path, 'video_frames', f'{string_for_mode[self.mode]}*', '*.png'))))

    def __len__(self) -> int:
        return len(self.video_frame_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image_nii = nib.load(self.fmri_paths[idx])
        image = image_nii.get_fdata()
        import pdb; pdb.set_trace()

        # assert os.path.isfile(file_EEG)

        # return image, label


if __name__ == '__main__':
    dataset = DynamicNaturalVisionDataset()
    print('Loading Dynamic Natural Vision dataset.')
    print('Length of dataset:', len(dataset))
    item = dataset[0]