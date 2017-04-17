import joblib
import os
import h5py as h5
import nibabel as nib
import numpy as np


def l2(x, axis=None):
    """
    Computes the squared L2-norm of an array.

    Parameters
    ----------
    x: numpy array.
        Array from which the norm will be computed
    axis: int.
        Axis along which to compute. If None, over all the values.

    Returns
    -------
    squared L2-norm: numpy array or float.
        Padded image.
    """
    if axis == None:
        return np.sum(x ** 2.0)
    else:
        return np.sum(x ** 2.0, axis=axis)


def get_mri_sequences(path, sequences):
    """
    Reads a set of MRI sequences from a directory.

    Parameters
    ----------
    path: string.
        Path to the directory containing the MRI volumes.
    sequences: list.
        List containing the name of the MRI sequences

    Returns
    -------
    Dict.
        Dict with sequences keys containing the MRI volumes.
    """
    files = os.listdir(path)
    mri = {}

    for sequence in sequences:
        for f in files:
            if sequence + '.' in f:
                vol, _ = load_mri(os.path.join(path, f))
                mri[sequence] = vol

    return mri


def array_tf_90(arr):
    """
    Rotates a 2D image by 90 degrees.

    Parameters
    ----------
    arr: numpy array.
        Image to be rotated

    Returns
    -------
    Numpy array
        The input array rotated by 90 degrees.
    """
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
        [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


def get_slice(vol, slice_n, axis, rotate):
    """
    Given an axis, extracts a slice from a MRI volume.

    Parameters
    ----------
    vol: numpy array.
        MRI 3D volume
    slice_n: int
        Slice to be extracted.
    axis: int
        Axis along which must extract the MRI sequence.
    rotate: boolean
        If the slice should be rotated by 90 degrees

    Returns
    -------
    Numpy array
        The extracted slice.
    """
    if axis == 0:
        s = vol[slice_n, :, :]
    elif axis == 1:
        s = vol[:, slice_n, :]
    else:
        s = vol[:, :, slice_n]

    if rotate:
        return array_tf_90(s)
    else:
        return s


def get_slice_from_volumes(volumes, slice_n, axis, rotate):
    """
    Given a dict with the MRI volumes, extracts a slice from each MRI volume.

    Parameters
    ----------
    volumes: Dict.
        Dict with MRI sequences keys containing the MRI volumes
    slice_n: int
        Slice to be extracted.
    axis: int
        Axis along which must extract the MRI sequence.
    rotate: boolean
        If the slice should be rotated by 90 degrees

    Returns
    -------
    Dict
        Dict with the desired slice from each MRI sequence.
    """
    slices = {}
    for seq in volumes.keys():
        slices[seq] = get_slice(volumes[seq], slice_n, axis, rotate)

    return slices


def read_data(path):
    """
    Reads pickled data.

    Parameters
    ----------
    path: string
        Path to the file to be read.

    Returns
    -------
    Some data
        The data that was pickled.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise ValueError('Path to weight matrix is invalid.')


def load_mri(path):
    """
    Reads an MRI.

    Parameters
    ----------
    path: string
        Path to the file to be read.

    Returns
    -------
    Numpy array
        The MRI volume.
    Numpy array
        The affine matrix of the MRI volume.
    """
    mri = nib.load(path)

    return mri.get_data(), mri.get_affine()


def padding(image, padding_size, padding_value=0):
    """
    Pads a given image with some given value.

    Parameters
    ----------
    image: numpy array
        Image to be padded.
    padding size: int
        Amount of data to be added to each side of the image.
    padding_value: float
        Value to be put in the padding.

    Returns
    -------
    Numpy array
        The padded image
    """
    if padding_size > 0:
        nimg = np.pad(image, (padding_size, padding_size), 'constant',
                      constant_values=(padding_value, padding_value))
        return np.array(nimg, dtype=np.float32)
    return image


def unpadding(image, padding_size):
    """
    Unpads a given image.

    Parameters
    ----------
    image: numpy array
        Image to be padded.
    padding size: int
        Amount of data to be removed from each side of the image.

    Returns
    -------
    Numpy array
        The unpadded image
    """
    if padding_size > 0:
        if len(image.shape) == 2:
            return image[padding_size:-padding_size, padding_size:-padding_size]
        elif len(image.shape) == 3:
            return image[padding_size:-padding_size, padding_size:-padding_size,
                    padding_size:-padding_size]
    return image


def scale_to(image, amax=255, amin=0):
    """
    Scales values to a given interval.

    Parameters
    ----------
    image: numpy array
        Image to be scaled.
    amax: int
        Max of the scaled image.
    amin: int
        Minimum of the scaled image.

    Returns
    -------
    Numpy array
        The scaled image
    """
    vmin, vmax = image.min(), image.max()
    return (amax - amin) * (image - vmin) / (vmax - vmin) + amin


def create_dir(path):
    """
    Creates a directory.

    Parameters
    ----------
    path: string
        Directory to be created
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def load_test_subject(test_features_path):
    """
    Loads the data of a test subject.

    Parameters
    ----------
    test_features_path: string
        Path to the hdf5 file containing the data

    Returns
    -------
    Numpy array
        The test subject features
    Numpy array
        The test subject image coordinates
    """
    test_data = h5.File(test_features_path, 'r')
    X_test = np.array(test_data['features'])
    coordinates = np.array(test_data['coordinates'])
    test_data.close()

    return X_test, coordinates


def dump_data(data, save_dir, save_name):
    """
    Saves data.

    Parameters
    ----------
    data: some data
        Data that will be saved.
    save_dir: string
        Directory where the data will be saved.
    save_name: string
        Name of the file
    """
    joblib.dump(data, os.path.join(save_dir, save_name))


def rebuild(prediction, coordinates, mask):
    """
    Rebuilds the prediction into the MRI volume space.

    Parameters
    ----------
    prediction: numpy array
        1-D array containing the predictions.
    coordinates: numpy array
        The 3D coordinates of each prediction in the MRI volume space.
    mask: numpy array
        Mask volume used to define the brain region.

    Returns
    -------
    Numpy array
        The segmentation
    """
    segmentation = np.zeros(mask.shape)
    segmentation[coordinates[:, 0],
                 coordinates[:, 1], coordinates[:, 2]] = prediction

    return segmentation


def save_image(volume, affine, save_dir, save_name, unpad):
    """
    Saves a MRI image.

    Parameters
    ----------
    volume: numpy array
        A MRI volume.
    affine: numpy array
        The affine matrix of the MRI volume
    save_dir: string
        Directory where the image will be saved.
    save_name: string
        Name of the MRI volume
    unpad: int
        Value to be unpad from the image.
    """
    nii_image = nib.Nifti1Image(unpadding(image=volume,
                                          padding_size=unpad),
                                affine)
    nib.save(nii_image, os.path.join(save_dir, save_name))