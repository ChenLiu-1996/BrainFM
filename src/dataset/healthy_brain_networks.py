'''
Loader of the Healthy Brain Network Dataset.
The dataset contains acquisitions of EEG and fMRI.
'''
from typing import Tuple

import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset


class HealthyBrainNetworks(Dataset):
    def __init__(self, base_path: str = '../../data/BrainFM_2024_HBN/'):
        self.folders_fMRI = np.array(sorted(glob(os.path.join(base_path, 'MRI', 'sub-*'))))
        self.folders_EEG = np.array(sorted(glob(os.path.join(base_path, 'EEG', '*'))))

        self.__filter_unmatched()


    def __filter_unmatched(self) -> None:
        '''
        Remove subjects that do not have a counterpart in EEG/fMRI.
        '''
        subject_set_fMRI = set()
        subject_set_EEG = set()

        for folder in self.folders_fMRI:
            subject_id = os.path.basename(folder).replace('sub-', '')
            subject_set_fMRI.add(subject_id)

        for folder in self.folders_EEG:
            subject_id = os.path.basename(folder)
            subject_set_EEG.add(subject_id)

        common_subject_set = subject_set_fMRI.intersection(subject_set_EEG)

        indices_to_keep = []
        for i, folder in enumerate(self.folders_fMRI):
            subject_id = os.path.basename(folder).replace('sub-', '')
            if subject_id in common_subject_set:
                indices_to_keep.append(i)
        self.folders_fMRI = self.folders_fMRI[indices_to_keep]

        indices_to_keep = []
        for i, folder in enumerate(self.folders_EEG):
            subject_id = os.path.basename(folder)
            if subject_id in common_subject_set:
                indices_to_keep.append(i)
        self.folders_EEG = self.folders_EEG[indices_to_keep]

        assert len(self.folders_fMRI) == len(self.folders_EEG)


    def __len__(self) -> int:
        return len(self.folders_fMRI)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        file_EEG = os.path.join(self.folders_EEG[idx], 'raw', 'mat_format', 'Video-DM.mat')
        file_fMRI = os.path.join(self.folders_fMRI[idx], 'fmap', 'mat_format', 'Video-DM.mat')
        assert os.path.isfile(file_EEG)


        return image, label


if __name__ == '__main__':
    dataset = HealthyBrainNetworks()
    print('Loading HealthyBrainNetworks dataset.')
    print('Length of dataset:', len(dataset))
