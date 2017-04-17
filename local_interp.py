import numpy as np
import nibabel as nib
import joblib
import os
import sys
from utils import read_data, load_mri, padding, unpadding, create_dir, \
    load_test_subject, dump_data, rebuild, scale_to, save_image, \
    get_mri_sequences, array_tf_90, get_slice, get_slice_from_volumes
from lime import lime_tabular
import argparse
import h5py as h5
import time
import multiprocessing
import progressbar
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


_AXIS = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}


def dump_results_as_image(results, affine, save_folder, base_name):
    """
    Saves the interpretability results into 3D MRI images.

    Parameters
    ----------
    results: Dict.
        Dict containing the interpretability results for each class and 
        sequence.
    affine: numpy array.
        Affine matrix of the MRI volumes.
    save_folder: string
        Path where the images will be saved.
    base_name: string
        Prefix name of the saved images.
    """
    for c in results.keys():
        for s in results[c].keys():
            nii_image = nib.Nifti1Image(results[c][s], affine)
            nib.save(nii_image, os.path.join(save_folder, base_name +
                                             '_class_' + str(c) + '_' + s +
                                             '.nii.gz'))


def prepare_lime(training_path, class_names, discretize_continuous=True,
                 sel_feat_file=None):
    """
    Prepared LIME.

    Parameters
    ----------
    training_path: string
        Path to the hdf5 file of the training data.
    class_names: list
        List containing the name of the classes.
    discretize_continuous: boolean
        To discretize continuous data using LIME.
    sel_feat_file: string
        Path to a file with a list of the indexes of the selected features.

    Returns
    -------
    Lime explainer
    """
    train_data = h5.File(training_path, 'r')
    X_train = \
        np.array(train_data['features'][
                 :, 0:train_data['features'].shape[1] - 1])
    train_data.close()

    if sel_feat_file is not None:
        selected_feat = read_data(sel_feat_file)
        X_train = X_train[:, selected_feat]

    explainer = \
        lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=list(
                np.arange(0, X_train.shape[1]).astype(str)),
            class_names=class_names,
            discretize_continuous=discretize_continuous)

    return explainer


def get_rf_predictions(X_test, rf):
    """
    Computes the RF predictions.

    Parameters
    ----------
    X_test: numpy array
        Features of the test subject.
    rf: trained sklearn Random Forest
        Instance of a Random Forest classifier from Scikit-Learn.

    Returns
    -------
    Numpy array
        Predictions of the Random Forest.
    """
    return rf.predict(X_test)


def preprocess_W(W, scale=True):
    """
    Preprocesses the weights matrix of the RBM.

    Parameters
    ----------
    W: numpy array
        Weights matrix of the RBM.
    scale: boolean
        To scale or not to the [0, 1] range.

    Returns
    -------
    Numpy array
        Pre-processed RBM weights matrix.
    """
    W = np.absolute(W)
    if scale:
        for i in range(0, W.shape[1]):
            W[:, i] = scale_to(W[:, i], 1.0, 0.0)

    return W


def apply_mask(results, mask):
    """
    Applies the mask to the interpretation results.

    Parameters
    ----------
    results: dict
        Dictionary containing the interpretation results from each class and 
        MRI sequence.
    mask: numpy array
        Brain mask.

    Returns
    -------
    Dict
        Dictionary with the interpretability results.
    """
    for c in results.keys():
        for s in results[c].keys():
            results[c][s] *= mask
            results[c][s][np.round(results[c][s], 4) ==
                          np.round(1.0 / float(len(results[c].keys())), 4)] = 0.0

    return results


def unpad_results(results, pad):
    """
    Unpads the interpretation results.

    Parameters
    ----------
    results: dict
        Dictionary containing the interpretation results from each class and 
        MRI sequence.
    pad: int
        Pad size for unpadding the interpretation results volumes.

    Returns
    -------
    Dict
        Dictionary with the unpadded interpretability results.
    """
    for c in results.keys():
        for s in results[c].keys():
            results[c][s] = unpadding(image=results[c][s], padding_size=pad)
    return results


def reshape_W(W_matrix, patch_size, sequences):
    """
    Reshapes the RBM weights matrix in order to give the visible units 
    connecting to each hidden unit the correct patch size.

    Parameters
    ----------
    W_matrix: numpy array
        RBM weights matrix.
    patch_size: list
        List containing the patch size.
    sequences: list
        The list of MRI sequences

    Returns
    -------
    numpy array
        The reshaped RBM weights matrix
    """
    new_W = np.zeros((W_matrix.shape[1], len(sequences), patch_size[
                     0], patch_size[1], patch_size[2]))

    step = int(np.product(patch_size))

    for f in range(0, W_matrix.shape[1]):
        for s in range(0, len(sequences)):
            new_W[f, s, :, :, :] = W_matrix[
                s * step: s * step + step, f].reshape(patch_size)

    return new_W


def init_seq_vols(mask, sequences_name):
    """
    Initializes the volumes where the interpretation will be stored.

    Parameters
    ----------
    mask: numpy array
        3D volume of the brain mask.
    sequences_name: list
        The list of MRI sequences

    Returns
    -------
    Dict
        Dictionary with the initialized volumes. One for each sequence.
    """
    vols = {}
    for seq in sequences_name:
        vols[seq] = np.zeros(mask.shape)
    return vols


def init_store_vols(mask, segmentation, sequences_name, only_foreground):
    """
    Initializes the volumes where the interpretation will be stored.

    Parameters
    ----------
    mask: numpy array
        3D volume of the brain mask.
    segmentation: numpy array
        3D volume with the segmentation
    sequences_name: list
        The list of MRI sequences
    only_foreground: boolean
        If only the foreground ground classes are wanted.

    Returns
    -------
    Dict
        Dictionary with the initialized volumes. One for each class.
    """
    start = 0 if only_foreground is False else 1
    store_vols = {}
    for i in range(start, int(segmentation.max()) + 1):
        store_vols[i] = init_seq_vols(mask, sequences_name)
    return store_vols


def fill_interpretation(W, selected_features, result_vols, coors,
                        sequences_name, patch_size):
    """
    Provides the interpretation by placing the weights values in the 
    corresponding positions in the interpretation volumes.

    Parameters
    ----------
    W: numpy array
        RBM weights matrix, after reshaping.
    selected_features: list
        indexes of the selected features (=hidden units)
    result_vols: dict
        Dict of numpy arrays where the interpretation will be stored.
    coors: numpy array
        3D coordinates of the voxels under analysis.
    sequences_name: list
        The MRI sequences names.
    patch_size: list
        The list with the patch sizes.

    Returns
    -------
    Dict
        Dictionary with the the interpretations.
    """
    x, y, z = coors
    x_half = patch_size[0] // 2
    y_half = patch_size[1] // 2
    z_half = patch_size[2] // 2

    for selected_feature in selected_features:
        for i, seq in enumerate(sequences_name):
            result_vols[seq][x - x_half: x + x_half + 1,
                             y - y_half: y + y_half + 1,
                             z - z_half: z + z_half + 1] += W[selected_feature, i, :, :, :]

    return result_vols


def get_interpretation(W, patch_size, coordinates, sequences_name, mask,
                       features, segmentation, only_foreground):
    """
    Computes the interpretation.

    Parameters
    ----------
    W: numpy array
        RBM weights matrix, after reshaping.
    patch_size: list
        The list with the patch sizes.
    coordinates: numpy array
        3D coordinates of the voxels under analysis.
    sequences_name: list
        The MRI sequences names.
    mask: numpy array
        3D volume with the brain mask.
    features: numpy array
        Features selected by LIME
    segmentation: numpy array
        3D image with the segmentation.
    only_foreground: boolean
        If just the foreground object is to be interpreted.

    Returns
    -------
    Dict
        Dictionary with the the interpretations.
    """
    store_vols = init_store_vols(mask, segmentation, sequences_name,
                                 only_foreground)
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    line = 0

    for i, j, k in zip(x, y, z):
        selected_features = features[line, :]

        segmented_class = int(segmentation[i, j, k])

        store_vols[segmented_class] = \
            fill_interpretation(W, selected_features,
                                store_vols[segmented_class], [i, j, k],
                                sequences_name, patch_size)

        line += 1

    return store_vols


def explain_sample(sample, classifier, explainer, num_features=10,
                   n_neighbor_samples=5000, top_labels=1):
    """
    Computes the interpretation.

    Parameters
    ----------
    sample: numpy array
        Sample of the test subject to be explained.
    classifier: sklearn trained classifier
        Trained classifier.
    explainer: LIME object
        LIME explainer for interpretation.
    num_features: int
        Number of features to be selected by LIME.
    n_neighbor_samples: int
        Number of neighbooring samples to be synthesized by LIME.

    Returns
    -------
    list
        Features selected by LIME.
    """
    exp = explainer.explain_instance(sample, classifier.predict_proba,
                                     num_features=num_features,
                                     top_labels=top_labels,
                                     num_samples=n_neighbor_samples)

    return dict(exp.local_exp[exp.local_exp.keys()[0]]).keys()


def explain_subset(chosen_data, classifier, explainer, num_features=10,
                   n_neighbor_samples=5000, top_labels=1):
    """
    Explaines a group of voxels from the testing subjects.

    Parameters
    ----------
    chosen_data: numpy array
        Subset of voxels to be interpreted. The features.
    classifier: sklearn trained classifier
        Trained classifier.
    explainer: LIME object
        LIME explainer for interpretation.
    num_features: int
        Number of features to be selected by LIME.
    n_neighbor_samples: int
        Number of neighbooring samples to be synthesized by LIME.

    Returns
    -------
    list
        Features selected by LIME.
    """
    widgets = ['Explaining: ', progressbar.Percentage(), ' ',
               progressbar.Bar(marker='#', left='[', right=']'), ' ',
               progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,
                                   maxval=chosen_data.shape[0])
    pbar.start()

    lime_interpreted_feat = np.zeros((chosen_data.shape[0], num_features))

    for i in range(0, chosen_data.shape[0]):
        pbar.update(i)
        sample_interpreted_feat = explain_sample(
            chosen_data[i], classifier, explainer, num_features=num_features,
            n_neighbor_samples=n_neighbor_samples, top_labels=top_labels)

        lime_interpreted_feat[i, :] = np.array(sample_interpreted_feat[:])

    pbar.finish()
    return lime_interpreted_feat.astype(np.int)


def _parallel_explain_subset(procnum, chosen_data, classifier,
                             explainer, save_dir, num_features=10,
                             n_neighbor_samples=5000, top_labels=1):
    """
    Runs the interpretation algorithm in parallel.

    Parameters
    ----------
    procnum: int
        Identifier of the parallel.
    chosen_data: numpy array
        Subset of voxels to be interpreted. The features.
    classifier: sklearn trained classifier
        Trained classifier.
    explainer: LIME object
        LIME explainer for interpretation.
    num_features: int
        Number of features to be selected by LIME.
    n_neighbor_samples: int
        Number of neighbooring samples to be synthesized by LIME.

    Returns
    -------
    list
        Features selected by LIME.
    """
    explanation = explain_subset(
        chosen_data, classifier, explainer, num_features=num_features,
        n_neighbor_samples=n_neighbor_samples, top_labels=top_labels)

    dump_data(explanation, save_dir=save_dir,
              save_name='explainer_proc_id_' + str(procnum))


def parallel_explain_subset(test_features, classifier, explainer, save_dir,
                            num_features=10, n_neighbor_samples=5000,
                            top_labels=1, n_jobs=8):
    """
    Runs the interpretation algorithm in parallel. Launches the processes, 
    basically.

    Parameters
    ----------
    test_features: numpy array
        The features of the samples to be interpreted.
    classifier: sklearn trained classifier
        Trained classifier.
    explainer: LIME object
        LIME explainer for interpretation.
    save_dir: string
        Directory to save temporary data.
    num_features: int
        Number of features to be selected by LIME.
    n_neighbor_samples: int
        Number of neighbooring samples to be synthesized by LIME.
    n_jobs: int
        The number of parallel jobs.

    Returns
    -------
    numpy array
        The features selected by LIME.
    """
    step = int(np.ceil(float(test_features.shape[0]) / float(n_jobs)))

    jobs = [None] * n_jobs

    for i in range(0, n_jobs):

        if i * step + step < test_features.shape[0]:
            job_data = test_features[i * step: i * step + step, :]
        else:
            job_data = test_features[i * step: test_features.shape[0], :]

        jobs[i] = multiprocessing.Process(
            target=_parallel_explain_subset,
            args=(i, job_data, classifier, explainer, save_dir,
                  num_features, n_neighbor_samples, top_labels,))

        jobs[i].start()

    for proc_id in range(0, n_jobs):
        jobs[proc_id].join(None)

    selected_features = read_data(os.path.join(save_dir,
                                               'explainer_proc_id_0'))
    os.remove(os.path.join(save_dir, 'explainer_proc_id_0'))
    for i in range(1, n_jobs):
        explainer_features = read_data(
            os.path.join(save_dir, 'explainer_proc_id_' + str(i)))
        selected_features = np.append(selected_features,
                                      explainer_features, axis=0)
        os.remove(os.path.join(save_dir, 'explainer_proc_id_' + str(i)))

    return selected_features


def serial_explain_subset(test_features, classifier, explainer, save_dir,
                          num_features=10, n_neighbor_samples=5000,
                          top_labels=1):
    """
    Runs the interpretation algorithm in parallel. Launches the processes, 
    basically.

    Parameters
    ----------
    test_features: numpy array
        The features of the samples to be interpreted.
    classifier: sklearn trained classifier
        Trained classifier.
    explainer: LIME object
        LIME explainer for interpretation.
    save_dir: string
        Directory to save temporary data.
    num_features: int
        Number of features to be selected by LIME.
    n_neighbor_samples: int
        Number of neighbooring samples to be synthesized by LIME.

    Returns
    -------
    numpy array
        The features selected by LIME.
    """
    selected_features = explain_subset(test_features, classifier, explainer,
                                       num_features=num_features,
                                       n_neighbor_samples=n_neighbor_samples,
                                       top_labels=top_labels)

    return selected_features


def normalize_max_all_volumes(data):
    """
    Normalizes the interpretations with the max of the class. It helps the 
    visualization, afterwards.

    Parameters
    ----------
    data: dict
        The dict with the volumes of the interpretations

    Returns
    -------
    dict
        The normalized interpretations.
    """
    new_data = deepcopy(data)

    for label in data.keys():
        label_max = 0.0
        for sequence in data[label].keys():
            vol = data[label][sequence]
            if vol.max() > label_max:
                label_max = vol.max()

        for sequence in data[label].keys():
            seq_max = data[label][sequence].max()
            vol = scale_to(data[label][sequence], 1.0, 0.0)
            vol *= float(seq_max) / float(label_max)

            new_data[label][sequence] = vol

    return new_data


def smooth_interpretation(results, sigma=1.0):
    """
    Smooths the interpretability results with a gaussian filter.

    Parameters
    ----------
    results: Dict.
        Dict containing the interpretability results for each class and 
        sequence.
    sigma: float.
        Standard deviation of the gaussian filter.

    Returns
    -------
    Dict
        Dict with the interpretability results after being filtered.
    """
    filtered = deepcopy(results)

    for c in results.keys():
        for s in results[c].keys():
            filtered[c][s] = gaussian_filter(results[c][s], sigma=sigma)

    return filtered


def get_strided_sample(test_features, test_coordinates, prediction, mask,
                       only_foreground, stride, return_stride_image=False):
    """
    Downsamples the images to speed up the interpretation process.

    Parameters
    ----------
    test_features: numpy array
        The features of the samples to be interpreted.
    test_coordinates: numpy array
        The coordinates of the voxels under analysis.
    prediction: numpy array
        The volume image with the segmentation.
    mask: numpy array
        The volume image with the brain mask.
    only_foreground: boolean
        True, if only the foreground object must be studied
    stride: list
        List with the downsample factor in each axis.
    Return_stride_image: boolean
        True, if the image with the strides must be return. Only useful for 
        debugging.

    Returns
    -------
    numpy array
        The features of the selected voxels.
    numpy array
        The coordinates of the selected voxels
    numpy array
        The image with the strides
    """
    x, y, z = test_coordinates[:, 0], test_coordinates[:, 1], \
        test_coordinates[:, 2]

    if only_foreground:
        prediction = prediction[prediction > 0]
        rebuild_seg = np.zeros(mask.shape)
        rebuild_seg[x, y, z] = prediction
    else:
        rebuild_seg = mask

    tmp = np.zeros(mask.shape)
    tmp[::stride[0], ::stride[1], ::stride[2]] = 1
    tmp *= rebuild_seg

    tmp_labels = tmp[x, y, z]
    indexes = np.nonzero(tmp_labels)[0]

    if not return_stride_image:
        return test_features[indexes], test_coordinates[indexes]
    else:
        stride_image = np.zeros(mask.shape)
        stride_image[x[indexes], y[indexes], z[indexes]] = 1

        return test_features[indexes], test_coordinates[indexes], stride_image


def get_slices_samples(test_features, test_coordinates, mask, slices, axis,
                       patch_size, return_slices_image=False):
    """
    Gets just some slices of the image to be interpreted. It also keeps 
    slices in the nearby.

    Parameters
    ----------
    test_features: numpy array
        The features of the samples to be interpreted.
    test_coordinates: numpy array
        The coordinates of the voxels under analysis.
    mask: numpy array
        The volume image with the brain mask.
    slices: list
        list of slices to be kept.
    axis: int
        Axis along which the slices must be extracted.
    patch_size: list
        Size of the patch in each axi
    return_slices_image: boolean
        True, if the image with the slices must be return. Only useful for 
        debugging.

    Returns
    -------
    numpy array
        The features of the selected voxels.
    numpy array
        The coordinates of the selected voxels
    numpy array
        The image with the slices
    """
    def get_half_patch(patch_size, axis):
        if patch_size[axis] % 2.0 == 0:
            return int(float(patch_size[axis]) // 2.0), \
                int(float(patch_size[axis]) // 2.0)
        else:
            return int(float(patch_size[axis]) // 2.0), \
                int(float(patch_size[axis]) // 2.0) + 1

    slices = [s + np.sum(patch_size) for s in slices]

    x, y, z = test_coordinates[:, 0], test_coordinates[:, 1], \
        test_coordinates[:, 2]

    tmp = np.zeros(mask.shape)
    h1, h2 = get_half_patch(patch_size, axis)

    for s in slices:
        if axis == 0:
            tmp[s - h1: s + h2, :, :] = 1
        elif axis == 0:
            tmp[:, s - h1: s + h2, :] = 1
        else:
            tmp[:, :, s - h1: s + h2] = 1

    tmp *= mask

    tmp_labels = tmp[x, y, z]
    indexes = np.nonzero(tmp_labels)[0]

    if not return_slices_image:
        return test_features[indexes], test_coordinates[indexes]
    else:
        slices_image = np.zeros(mask.shape)
        slices_image[x[indexes], y[indexes], z[indexes]] = 1

        return test_features[indexes], test_coordinates[indexes], slices_image


def convert_selected_lime_to_selected_mi(lime_selected_features,
                                         mi_selected_features):
    """
    Converts the index of the features selected by lime to the indexes of 
    the previously selected features.

    Parameters
    ----------
    lime_selected_features: numpy array
        The features selected by LIME.
    mi_selected_features: list
        The indexes of the previously selected features.

    Returns
    -------
    numpy array
        The features selected by LIME, converted to the right indexes.
    """
    converted_lime = np.zeros_like(lime_selected_features)

    for i in range(0, lime_selected_features.shape[0]):
        for j in range(0, lime_selected_features.shape[1]):
            converted_lime[i, j] = \
                mi_selected_features[lime_selected_features[i, j]]

    return converted_lime


def plot_local_interp(mri_sequences, save_dir, interpretation, prediction,
                      classes, slices, axis, dpi, rotate=True, exclude=[],
                      vmax=None):
    """
    Creates the plots with the local interpretation.

    Parameters
    ----------
    mri_sequences: dict
        The MRI images for each sequence.
    save_dir: string
        Path to the directory where results will be saved.
    interpretation: dict
        Dictionary with the volumes of the interpretation
    prediction: numpy array
        3D image with the segmentation
    classes: list
        The classes to be plotted.
    slices: list
        The chosen slices.
    axis: int
        Axis along which we extract the slices.
    dpi: int
        The dpi of the images that will be plotted.
    rotate: boolean
        If the image should be rotated for plot.
    exclude: list
        If some class should be excluded from the plot.
    vmax: int
        The value to take as maximum while plotting.

    Returns
    -------
    numpy array
        The features selected by LIME, converted to the right indexes.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_lines = len(classes) + 1 - len(exclude)
    n_cols = len(mri_sequences.keys()) + 1

    with PdfPages(os.path.join(save_dir, 'local_interpretation.pdf')) as pdf:

        for slice_n in slices:

            slices = get_slice_from_volumes(volumes=mri_sequences,
                                            slice_n=slice_n, axis=axis,
                                            rotate=rotate)
            seg_slice = get_slice(vol=prediction, slice_n=slice_n, axis=axis,
                                  rotate=rotate)

            plt.figure(figsize=(20, 9))
            plt.suptitle('Slice ' + str(slice_n) + ' of the ' +
                         _AXIS[axis] + ' plane')
            sub_id = 1

            for seq in sorted(slices.keys()):
                plt.subplot(n_lines, n_cols, sub_id)
                plt.imshow(slices[seq], cmap='gray')
                plt.title(seq)
                plt.gca().set_axis_off()
                sub_id += 1

            plt.subplot(n_lines, n_cols, sub_id)
            sub_id += 1

            plt.imshow(slices['T1c'], alpha=1.0, cmap='gray')
            plt.gca().set_axis_off()
            plt.hold(True)
            over_cmap = plt.cm.get_cmap('jet')
            over_cmap.set_bad(alpha=0)
            seg_slice[seg_slice <= 0] = np.nan
            plt.imshow(seg_slice, cmap=over_cmap)
            plt.gca().set_axis_off()
            plt.title('Segmentation')

            for c in sorted(interpretation.keys()):
                if c not in exclude:
                    t = 0
                    for seq in sorted(interpretation[c].keys()):
                        plt.subplot(n_lines, n_cols, sub_id)
                        sub_id += 1

                        plt.imshow(slices['T1c'], alpha=1.0, cmap='gray')
                        plt.gca().set_axis_off()
                        plt.hold(True)
                        over_cmap = plt.cm.get_cmap('jet')
                        over_cmap.set_bad(alpha=0)
                        tmp_interp = interpretation[c][seq]
                        tmp_interp = get_slice(vol=tmp_interp,
                                               slice_n=slice_n, axis=axis,
                                               rotate=rotate)
                        tmp_interp[tmp_interp <= 0] = np.nan
                        plt.imshow(tmp_interp, cmap=over_cmap, vmin=0, vmax=1)
                        cbar = plt.colorbar(shrink=.9, format='%0.2f')
                        cbar.ax.tick_params(labelsize=8)

                        t += 1
                        if t == 3:
                            plt.title(classes[c - 1])

                    tmp_seg = deepcopy(seg_slice)
                    tmp_seg[tmp_seg != c] = 0

                    plt.subplot(n_lines, n_cols, sub_id)
                    sub_id += 1
                    plt.imshow(slices['T1c'], alpha=1.0, cmap='gray')
                    plt.gca().set_axis_off()
                    plt.hold(True)
                    over_cmap = plt.cm.get_cmap('jet')
                    over_cmap.set_bad(alpha=0)
                    tmp_seg[tmp_seg <= 0] = np.nan
                    plt.imshow(tmp_seg, cmap=over_cmap, vmin=0,
                               vmax=prediction.max())
                    plt.gca().set_axis_off()

            pdf.savefig()
            plt.close()


def main():
    start = time.time()

    ###### PARAMS ######
    save_dir = 'results'
    mri_path = 'data/BRATS_subj_0310'
    rf_path = 'data/rf/RF.npy'
    test_subj_path = 'data/test_subj.hdf5'
    training_feat_path = 'data/train_BRATS.hdf5'
    selected_features_file = 'data/selected_feat.npy'
    slices = [104]  # None if we want all the segmentation
    axis = 0
    patch_size = [9, 9, 9]
    stride = [2, 2, 2]  # if we want to decimate the image, otherwise None
    # Gaussian filter to remove the high frequency transitions made by the
    # decimation
    apply_gaussian_filter = True
    class_names = ['Normal', 'Necrosis', 'Edema', 'Non-enhanced', 'Enhanced']
    sequences = ['T1', 'T1c', 'T2', 'Flair']
    lime_features = 10
    lime_neighbor_samples = 2400
    n_jobs = 8
    ####################

    create_dir(save_dir)

    print 'Loading W matrix...'
    start_subtask = time.time()

    W = read_data('data/W.npy')
    W = preprocess_W(W=W, scale=True)
    W = reshape_W(W_matrix=W, patch_size=patch_size, sequences=sequences)
    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    # Load mask
    print '\nGetting mask...'
    start_subtask = time.time()
    mask, _ = load_mri('data/BRATS_subj_0310/VSD.Mask_HG_310.17608.nii.gz')
    mask = padding(image=mask, padding_size=np.sum(patch_size))

    _, affine = \
        load_mri('data/BRATS_subj_0310/VSD.Brain.XX.O.MR_Flair.17608.nii.gz')
    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    # Segment the test subject
    print '\nSegmenting test subject...'
    start_subtask = time.time()

    rf = read_data(rf_path)

    X_test, test_coors = load_test_subject(test_features_path=test_subj_path)

    mi_selected_features = read_data(selected_features_file)
    X_test = X_test[:, mi_selected_features]

    print '\tStarting RF prediction'

    prediction = None
    if slices is not None:
        X_test, test_coors, slices_image = \
            get_slices_samples(test_features=X_test,
                               test_coordinates=test_coors, mask=mask,
                               slices=slices, axis=axis,
                               patch_size=patch_size,
                               return_slices_image=True)

    prediction = get_rf_predictions(X_test=X_test, rf=rf)

    seg = rebuild(prediction=prediction, coordinates=test_coors, mask=mask)

    save_image(volume=seg, affine=affine, save_dir=save_dir,
               save_name='prediction.nii.gz', unpad=np.sum(patch_size))

    X_test = X_test[prediction > 0, :]
    test_coors = test_coors[prediction > 0, :]

    if stride is not None:
        X_test, test_coors, stride_image = \
            get_strided_sample(test_features=X_test,
                               test_coordinates=test_coors,
                               prediction=prediction, mask=mask,
                               only_foreground=True,
                               stride=stride, return_stride_image=True)

    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    # Prepare LIME explainer
    print '\nPreparing LIME...'
    start_subtask = time.time()
    explainer = prepare_lime(training_path=training_feat_path,
                             class_names=class_names,
                             discretize_continuous=True,
                             sel_feat_file=selected_features_file)

    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    # Getting LIME selected features
    print '\nGetting explaining features from LIME...'
    start_subtask = time.time()
    if n_jobs > 1:
        print '\tExecuting parallel algorithm.'
        lime_selected_features = parallel_explain_subset(
            test_features=X_test, classifier=rf, explainer=explainer,
            save_dir=save_dir, num_features=lime_features,
            n_neighbor_samples=lime_neighbor_samples, top_labels=1,
            n_jobs=n_jobs)
    elif n_jobs == 1:
        print '\tExecuting serial algorithm.'
        lime_selected_features = serial_explain_subset(
            X_test, rf, explainer, save_dir,
            num_features=lime_features,
            n_neighbor_samples=lime_neighbor_samples,
            top_labels=1)
        print '\t', lime_selected_features.shape
    else:
        raise ValueError('Invalid number of jobs')

    if selected_features_file is not None:
        lime_selected_features = \
            convert_selected_lime_to_selected_mi(
                lime_selected_features=lime_selected_features,
                mi_selected_features=mi_selected_features)

    del X_test

    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    # Getting interpretation images
    print '\nGetting interpretation over sequences...'
    start_subtask = time.time()

    results = get_interpretation(W=W, patch_size=patch_size,
                                 coordinates=test_coors,
                                 sequences_name=sequences, mask=mask,
                                 features=lime_selected_features, 
                                 segmentation=seg,
                                 only_foreground=True)

    results = apply_mask(results, mask)
    results = unpad_results(results=results, pad=np.sum(patch_size))

    if apply_gaussian_filter:
        results = smooth_interpretation(results=results, sigma=1.0)

    normalized_results = normalize_max_all_volumes(results)

    mri_sequences = get_mri_sequences(path=mri_path, sequences=sequences)
    seg = unpadding(seg, np.sum(patch_size))

    plot_local_interp(mri_sequences=mri_sequences, save_dir=save_dir,
                      interpretation=normalized_results, 
                      prediction=seg, 
                      classes=class_names[1: len(class_names)],
                      slices=slices, axis=axis, dpi=250, exclude=[3], 
                      vmax=None)

    print '\tDone! Took', float(time.time() - start_subtask) / 60.0,
    'minutes.'

    print 'Elapsed time:', float(time.time() - start) / 60.0, 'minutes.'
    print 'Have a nice day!'


if __name__ == '__main__':
    main()
