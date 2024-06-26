# DL-SOC
Deep learning code for digital soil organic carbon (SOC) mapping and covariate importance analysis

![Abstract Image](abstract_image/abstract_image_landscape.PNG)
## Overview
- `convert_multiscale_features/` constains a script to generate multiscale covariate layers using Gaussian Pyramids
- `dl_models/` contains scripts to build deep learning models for digital soil mapping as well as the weights of the trained models
- `permutation_analysis/` contains scripts to perform local and continental scales feature permutation analysis
- `shap/` contains a script to compute the contribution of covariates to SOC using SHapley Additive exPlanation (SHAP)
- To make the scripts work correctly, please create a `data/` folder in the `main/` directory and download the processed datasets from [DL-SOC-data](https://zenodo.org/records/11409768) and paste all the subfolders into `data/`
## Package Requirements
- python 3.8.2
- pandas 1.4.1
- shap 0.41.0
- scikit-learn 1.2.2
- numpy 1.20.1
- tensorflow 2.6.0
- evidential-deep-learning 0.4.0
- opencv-python 4.7.0.72
## Citation
TBD
