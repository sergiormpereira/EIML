# EIML

This repository contains code corresponding to the algorithms of the paper: Pereira, Sérgio, et al., "Enhancing interpretability of automatically extracted machine learning features: application to a RBM-Random Forest system on brain lesion segmentation", Medical Image Analysis, Volume 44, February 2018. [link](https://www.sciencedirect.com/science/article/pii/S1361841517301901)

## Abstract

Machine learning systems are achieving better performances at the cost of becoming increasingly complex. However, because of that, they become less interpretable, which may cause some distrust by the end-user of the system. This is especially important as these systems are pervasively being introduced to critical domains, such as the medical field. Representation Learning techniques are general methods for automatic feature computation. Nevertheless, these techniques are regarded as uninterpretable “black boxes”. In this paper, we propose a methodology to enhance the interpretability of automatically extracted machine learning features. The proposed system is composed of a Restricted Boltzmann Machine for unsupervised feature learning, and a Random Forest classifier, which are combined to jointly consider existing correlations between imaging data, features, and target variables. We define two levels of interpretation: global and local. The former is devoted to understanding if the system learned the relevant relations in the data correctly, while the later is focused on predictions performed on a voxel- and patient-level. In addition, we propose a novel feature importance strategy that considers both imaging data and target variables, and we demonstrate the ability of the approach to leverage the interpretability of the obtained representation for the task at hand. We evaluated the proposed methodology in brain tumor segmentation and penumbra estimation in ischemic stroke lesions. We show the ability of the proposed methodology to unveil information regarding relationships between imaging modalities and extracted features and their usefulness for the task at hand. In both clinical scenarios, we demonstrate that the proposed methodology enhances the interpretability of automatically learned features, highlighting specific learning patterns that resemble how an expert extracts relevant data from medical images.

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
