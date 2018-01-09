# EIML

This repository contains code corresponding to the algorithms of the paper: Pereira, Sérgio, et al., "Enhancing interpretability of automatically extracted machine learning features: application to a RBM-Random Forest system on brain lesion segmentation", Medical Image Analysis, Volume 44, February 2018. [link](https://www.sciencedirect.com/science/article/pii/S1361841517301901)

## Overview

The scrips global_interp.py and local_interp.py contain the implementation of the algorithms for the global and local interpretability, respectively. utils.py includes auxiliary code.

To run the code, please download the data from this [link](https://uminho365-my.sharepoint.com/personal/id5692_uminho_pt/_layouts/15/guestaccess.aspx?docid=071160f2e810645838c4d7b6bcf810616&authkey=AYIC4NVo_eISktwwTR-axKY) and unzip it in the same directory as the scripts. Then, just run each script in the command line.

## Requirements
The code was tested (in Linux Mint 18) with the following packages and corresponding versions:

- Python 2.7
- numpy 1.11.2
- nibabel 2.1.0
- joblib 0.10.3
- [LIME](https://github.com/marcotcr/lime) 0.1.1.16
- h5py 2.6.0
- progressbar 3.10.1
- scipy 0.18.1
- matplotlib 1.5.3
