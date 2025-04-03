'''
Preprocess the fMRI data in Dynamic Natural Vision dataset.
'''

import os
import numpy as np
import nibabel as nib
from glob import glob
from natsort import natsorted
from tqdm import tqdm

BASE_FOLDER = '../../data/Dynamic_Natural_Vision/'


def detrend_and_standardize(fMRI_path_list: str,
                            correct_voxel_spacing: int = 3) -> np.ndarray:
    '''
    Following the official preprocessing,
    we will remove 4-th order trends and perform standardization along time (zero mean, unit variance).

    Args:
        fMRI_path_list: List of fMRI paths.
        correct_voxel_spacing: Correct voxel spacing in mm for sanity checking.
    '''

    fMRI_repetitions = None
    for fMRI_path in fMRI_path_list:
        fMRI_nii = nib.load(fMRI_path)
        voxel_spacing = fMRI_nii.header['pixdim']
        assert (voxel_spacing[1:4] == correct_voxel_spacing).all(), f'Voxel spacing is not {correct_voxel_spacing} mm.'

        # fMRI scan dimension is: (height, width, depth, time).
        fMRI = fMRI_nii.get_fdata()
        assert len(fMRI.shape) == 4

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

        if fMRI_repetitions is None:
            fMRI_repetitions = fMRI[..., None]
        else:
            fMRI_repetitions = np.concatenate((fMRI_repetitions, fMRI[..., None]), axis=-1)

    assert fMRI_repetitions.shape[-1] == len(fMRI_path_list)

    # Now shape of repetitions: [H, W, D, T, num_repetitions].
    return fMRI_repetitions


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
    fMRI_folders = np.array(natsorted(glob(os.path.join(BASE_FOLDER, f'subject*', 'fmri', '*', 'mni'))))

    for fMRI_folder in tqdm(fMRI_folders):
        fMRI_path_list = natsorted(glob(os.path.join(fMRI_folder, '*nii*')))

        path_items = fMRI_folder.split('/')
        assert path_items[-1] == 'mni'

        subject_identifier = path_items[-4]
        acquisition_identifier = path_items[-2]

        if 'seg' in acquisition_identifier:
            mode = 'train'
        elif 'test' in acquisition_identifier:
            mode = 'test'
        else:
            raise ValueError(f'`acquisition_identifier` should contain `seg` or `test`, but found {acquisition_identifier}.')

        acquisition_id = acquisition_identifier.lstrip('seg').lstrip('test')
        npz_save_path = os.path.join(BASE_FOLDER, 'fMRI_scans', subject_identifier, mode + acquisition_id + '_mni_detrended.npz')

        os.makedirs(os.path.dirname(npz_save_path), exist_ok=True)
        fMRI_5D = detrend_and_standardize(fMRI_path_list)

        # Save the fMRI in npz format.
        with open(npz_save_path, 'wb+') as f:
            np.savez(f, fMRI=fMRI_5D)
