import joblib
import numpy as np
import os
import pylab as pl

from utils import read_data, l2


def get_selected_weights(w_matrix, selected_indexes):
    """
    Given the index of selected features, it gets the weights connecting to 
    the corresponding weights matrix of the RBM.

    Parameters
    ----------
    w_matrix: numpy array.
        Weights matrix of the RBM.
    selected_indexes: list.
        List containing the indexes of the selected features.

    Returns
    -------
    numpy array.
        RBM weights matrix cntaining just the hidden units corresponding to 
        the selected features.
    """
    index_features = np.array(selected_indexes)

    return w_matrix[:, index_features]


def get_l2_measures(w_matrix, sequences):
    """
    Computes the squared L2-norm between the visible units representing a MRI 
    sequence and each hidden unit.

    Parameters
    ----------
    w_matrix: numpy array.
        Weights matrix of the RBM.
    sequences: list.
        List containing the name of the MRI sequences

    Returns
    -------
    numpy array.
        Array [n_sequences X n_hidden_units] containing the squared L2-norm 
        between each hidden unit and the MRI sequences
    """
    l2_measures = np.zeros((len(sequences), w_matrix.shape[1]))
    n_visibles_voxels = int(float(w_matrix.shape[0]) / float(len(sequences)))

    for i in xrange(0, len(sequences)):
        l2_measures[i, :] = \
            l2(w_matrix[i * n_visibles_voxels: (i + 1) * n_visibles_voxels, :],
               axis=0)

    return l2_measures


def plot_global_interpretability(l2_measures, sequences, selected_indexes,
                                 title='', figsize=(25, 3), x_font=13,
                                 y_font=13, title_font=14, x_tick_font=10,
                                 y_tick_font=10, save_dir=None):
    """
    Creates the global interpretability plots.

    Parameters
    ----------
    l2_measures: numpy array.
        Array [n_sequences X n_hidden_units] containing the squared L2-norm 
        between each hidden unit and the MRI sequences.
    sequences: list.
        List containing the name of the MRI sequences.
    selected_indexes: list.
        List containing the indexes of the selected features.
    title: string
        Title of the plot.
    figsize: tuple
        Tuple with 2 elements containing the figure size
    x_font: float
        Font size of the x axis
    y_font: float
        Font size of the y axis
    title_font: float
        Font size of the title
    x_tick_font: float
        Font size of the ticks in the x axis
    y_tick_font: float
        Font size of the ticks in the y axis
    save_dir: string
        Path to the saving directory. If None, results will be displayed 
        instead of saved in a PDF.
    """

    f, axarr = pl.subplots(1, 1, figsize=figsize)
    im = axarr.imshow(l2_measures, interpolation='nearest', cmap='gray',
                      aspect='auto', vmin=l2_measures.min(),
                      vmax=l2_measures.max())
    axarr.set_xticks(np.arange(0, np.size(selected_indexes)))
    axarr.set_xticklabels(selected_indexes, {
                          'rotation': 'vertical', 'fontsize': x_tick_font})
    axarr.set_yticks(np.arange(0, len(sequences)))
    axarr.set_yticklabels(sequences, {'rotation': 'horizontal',
                                      'fontsize': y_tick_font})
    axarr.set_xlabel('Feature Index (Selected ' +
                     str(len(selected_indexes)) + ' features)',
                     fontsize=x_font, fontweight='bold')
    axarr.set_ylabel('MRI sequence', fontsize=y_font, fontweight='bold')
    axarr.set_title(title, fontsize=title_font, fontweight='bold')

    if save_dir == None:
        pl.show()
    else:
        pl.savefig(os.path.join(save_dir, 'global_interpretability.pdf'),
                   dpi=250, pad_inches=0.0, bbox_inches='tight')


def main():
    mri_sequences = ['T1', 'T1c', 'T2', 'Flair']
    save_dir = 'results'

    print 'Loading RBM weight matrix...'
    w = read_data('data/W.npy')
    print '\tVisible units:', w.shape[0], ', Hidden units:', w.shape[1]

    print 'Getting selected hidden nodes'
    selected_features = read_data('data/selected_feat.npy')

    # get the weights connecting to the selected hidden units
    w_selected = get_selected_weights(w_matrix=w,
                                      selected_indexes=selected_features)
    print '\tVisible units:', w_selected.shape[0], ', Hidden units:', \
        w_selected.shape[1]

    l2_measures = get_l2_measures(w_matrix=w_selected,
                                  sequences=mri_sequences)

    print 'Plotting the image for global interpratability...'
    plot_global_interpretability(l2_measures=l2_measures,
                                 sequences=mri_sequences,
                                 selected_indexes=selected_features,
                                 title='Normal tissues vs. Necrosis vs. '
                                 'Edema vs. Non-enhancing vs. Enhancing',
                                 figsize=(100, 10), x_font=30,
                                 y_font=40, title_font=60, x_tick_font=14,
                                 y_tick_font=30, save_dir=save_dir)

    print 'Have a nice day!'

if __name__ == "__main__":
    main()
