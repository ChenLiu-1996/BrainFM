'''
Loader of the Dynamic Natural Vision dataset.
The dataset contains acquisitions of fMRI.
'''
from typing import Tuple

import os
import cv2
import numpy as np
from glob import glob
import nibabel as nib
import nibabel.processing as processing
from torch.utils.data import Dataset
from natsort import natsorted


class DynamicNaturalVisionDataset(Dataset):
    '''
    We are using the MNI files. These are the 4D volumes registered to the MNI152 template space.
    Scans are given by (height, width, depth, time),
        where (height, width, depth) are fixed for 3 mm voxel spacing, and (time) is either 245 or 246.
    Following the official preprocessing, we will drop 1 at beginning and 4 at end if 245, and 2/4 if 246, to
        make sure all of them share the same time axis of 240.

    CIFTI files provides information on voxel values and brain region for each voxel.
    But unfortunately, the granularity of brain regions is super bad. Essentially most voxels of interest are assigned
    'CIFTI_STRUCTURE_CORTEX_LEFT' or 'CIFTI_STRUCTURE_CORTEX_RIGHT'.
    '''
    def __init__(self,
                 subject_idx: int = None,
                 fMRI_window_frames: int = 3,
                 fMRI_frames_per_session: int = 240,
                 video_frames_per_session: int = 1440,
                 voxel_spacing: int = 3,
                 target_image_dim: Tuple[int] = (256, 256),
                 base_path: str = '../../data/Dynamic_Natural_Vision/',
                 brain_atlas_path: str = '../../data/brain_parcellation/AAL3/AAL3v1_1mm.nii.gz',
                 mode: str = 'train'):
        '''
        subject_idx: subject index. A total of 3 subjects in this dataset.
        fMRI_window_frames: Number of fMRI frames in the time window.
        fMRI_frames_per_session: [intrinsic to this dataset] Number of fMRI frames per session.
        video_frames_per_session: [intrinsic to this dataset] Number of video frames per session.
        voxel_spacing: [intrinsic to this dataset] Voxel spacing in mm.
        target_image_dim: Desired image dimension (height, width) of video frame.
        '''

        self.subject_idx = subject_idx
        self.fMRI_window_frames = fMRI_window_frames
        self.fMRI_frames_per_session = fMRI_frames_per_session
        self.fMRI_frames_per_session_indexable = self.fMRI_frames_per_session - self.fMRI_window_frames + 1
        self.video_frames_per_session = video_frames_per_session
        assert self.video_frames_per_session % self.fMRI_frames_per_session == 0
        self.voxel_spacing = voxel_spacing
        self.target_image_dim = target_image_dim
        self.brain_atlas_path = brain_atlas_path
        assert mode in ['train', 'test'], f'`DynamicNaturalVisionDataset` mode must be `train` or `test`, but got {mode}.'
        self.mode = mode
        string_for_mode = {
            'train': 'seg',
            'test': 'test',
        }

        if self.subject_idx is None:
            self.fMRI_folders = np.array(natsorted(glob(
                os.path.join(base_path, 'subject*', 'fmri', f'{string_for_mode[self.mode]}*', 'mni'))))
        else:
            self.fMRI_folders = np.array(natsorted(glob(
                os.path.join(base_path, f'subject{self.subject_idx}', 'fmri', f'{string_for_mode[self.mode]}*', 'mni'))))

        self.video_frame_folder = os.path.join(base_path, 'video_frames')

    def __len__(self) -> int:
        return len(self.fMRI_folders) * self.fMRI_frames_per_session_indexable

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        '''
        There are several repetitions per subject per session.
        Following the official preprocessing,
        we will perform frame dropping, removal of 4-th order trends, standardization, and
        take the average across all repetitions per subject per session.
        '''
        # Load the fMRI scan.
        fMRI_scan_idx = idx // self.fMRI_frames_per_session_indexable
        fMRI_frame_idx_start = idx % self.fMRI_frames_per_session_indexable

        fMRI_folder = self.fMRI_folders[fMRI_scan_idx]
        fMRI_path_list = natsorted(glob(os.path.join(fMRI_folder, '*nii*')))

        repetitions = None
        for fMRI_path in fMRI_path_list:
            fMRI_nii = nib.load(fMRI_path)
            voxel_spacing = fMRI_nii.header['pixdim']
            assert (voxel_spacing[1:4] == self.voxel_spacing).all(), f'Voxel spacing is not {self.voxel_spacing} mm.'

            # fMRI scan dimension is: (height, width, depth, time).
            fMRI = fMRI_nii.get_fdata()
            assert len(fMRI.shape) == 4

            # Drop the first few and last few fMRI frames.
            drop_last = 4
            assert fMRI.shape[-1] > self.fMRI_frames_per_session + drop_last
            drop_first = fMRI.shape[-1] - self.fMRI_frames_per_session - drop_last
            fMRI = fMRI[..., drop_first:-drop_last]
            assert fMRI.shape[-1] == self.fMRI_frames_per_session

            # Data standardization.
            # 1. Remove 4-th order trends (separately for each voxel, along time axis).
            # 2. Zero mean and unit variance (separately for each voxel, along time axis).
            fMRI = detrend_poly(fMRI, polyorder=4, axis=-1)
            # After removal of 4-th order trends, the mean along time axis should be close to zero already.
            assert np.isclose(fMRI.mean(axis=-1), np.zeros_like(fMRI.mean(axis=-1)), atol=1e-4).all(), \
                'Removal of 4-th order trends still gives nonzero mean along time axis.'
            mean_along_time = fMRI.mean(axis=-1, keepdims=True)
            min_std = 1e-6
            std_along_time = np.maximum(min_std, fMRI.std(axis=-1, keepdims=True))
            fMRI = (fMRI - mean_along_time) / std_along_time

            # Take the queried fMRI frames.
            fMRI_frames = fMRI[..., fMRI_frame_idx_start : fMRI_frame_idx_start + self.fMRI_window_frames]
            if self.fMRI_window_frames == 1:
                assert len(fMRI_frames.shape) == 3
                fMRI_frames = fMRI_frames[..., None]

            if repetitions is None:
                repetitions = fMRI_frames[..., None]
            else:
                repetitions = np.concatenate((repetitions, fMRI_frames[..., None]), axis=-1)
            del fMRI_frames

        assert repetitions.shape[-1] == len(fMRI_path_list)
        fMRI_frames = repetitions.mean(axis=-1)

        # Take the queried video frames.
        video_id = self.fMRI_folders[fMRI_scan_idx].split('fmri/')[1].split('/mni')[0]
        video_frame_paths = natsorted(glob(os.path.join(self.video_frame_folder, video_id, '*.png')))
        video_frame_per_fMRI_frame = int(self.video_frames_per_session / self.fMRI_frames_per_session)
        video_frame_idx_start = video_frame_per_fMRI_frame * fMRI_frame_idx_start
        video_frame_paths_selected = video_frame_paths[
            video_frame_idx_start : video_frame_idx_start + video_frame_per_fMRI_frame * self.fMRI_window_frames]

        video_frames = None
        for video_frame_path in video_frame_paths_selected:
            image = np.array(
                cv2.resize(
                    cv2.cvtColor(cv2.imread(video_frame_path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB),
                    self.target_image_dim))
            image = normalize_natural_image(image)
            if video_frames is None:
                video_frames = image[..., None]
            else:
                video_frames = np.concatenate((video_frames, image[..., None]), axis=-1)
            del image

        # Load and resample the brain atlas.
        atlas_nii = nib.load(self.brain_atlas_path)
        atlas_nii = processing.resample_to_output(atlas_nii, voxel_sizes=(self.voxel_spacing, self.voxel_spacing, self.voxel_spacing))
        atlas = atlas_nii.get_fdata()

        assert fMRI.shape[:3] == atlas.shape, f'fMRI ({fMRI.shape[:3]}) and brain atlas {atlas.shape} dimension mismatch.'

        return fMRI_frames, video_frames, atlas


def normalize_natural_image(image):
    image = image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

def detrend_poly(signal, polyorder=1, axis=-1):
    '''
    Remove polynomial trend from a signal.

    Parameters:
    ----------
    signal : np.ndarray
        Input array. Detrending is performed along the specified axis.
    polyorder : int
        Order of the polynomial to remove (default is 1, linear detrend).
    axis : int
        Axis along which to detrend (default is -1).

    Returns:
    -------
    detrended : np.ndarray
        Signal after polynomial detrending, same shape as input.
    '''
    signal = np.asarray(signal)
    detrended = np.empty_like(signal)

    # Move the target axis to the end
    signal_moved = np.moveaxis(signal, axis, -1)
    orig_shape = signal_moved.shape

    # Reshape to 2D for easier computation
    reshaped = signal_moved.reshape(-1, orig_shape[-1])

    time = np.arange(orig_shape[-1])
    X = np.vander(time, polyorder + 1)

    for i in range(reshaped.shape[0]):
        coefs = np.linalg.lstsq(X, reshaped[i], rcond=None)[0]
        trend = X @ coefs
        reshaped[i] -= trend

    detrended = reshaped.reshape(orig_shape)
    return np.moveaxis(detrended, -1, axis)


if __name__ == '__main__':
    dataset = DynamicNaturalVisionDataset()
    print('Loading Dynamic Natural Vision dataset.')
    print('Length of dataset:', len(dataset))
    item = dataset[0]
