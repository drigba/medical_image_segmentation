#  Model Ensemble for Medical Image Segmentation

## Team name
Another bomb training - Újabb bomba tanítás

## Team members

- Benyák Bence József - AQTBDV
- Klenk Botond - FTNYN1
- Bánfi Zsombor - L1N5IV

# Project Description: Enhancing Deep-Learning Solutions through Model Ensembles for Semantic Segmentation on ACDC Dataset

## Overview
This project focuses on the implementation and analysis of model ensembles in deep learning to improve the accuracy of semantic segmentation. Model ensembles, a proven technique used by leading AI competition winners, involve integrating multiple models to enhance the overall precision of deep learning solutions. Although this approach incurs additional computational costs, it offers a promising avenue for significantly refining the performance of our deep-learning models.

## Objective
Our main objective is to explore and apply various strategies for constructing effective model ensembles tailored specifically for semantic segmentation tasks. We will leverage the ACDC (Automated Cardiac Diagnosis Challenge) dataset, a rich repository of complex cardiac MRI scans. Our ultimate goal is to accurately identify distinct components of the heart, including the RV cavity, the myocardium, and the LV cavity.

## Methodology
The project methodology entails comprehensive training of multiple deep-learning models, followed by the construction of an ensemble from these models. The focus will be on scrutinizing the iterative enhancements in accuracy facilitated by the ensemble approach. Simultaneously, we will conduct a thorough assessment of the associated benefits and computational overheads incurred while leveraging model ensembles.

## Anticipated Outcomes
We anticipate an improvement in the accuracy of semantic segmentation for the ACDC dataset through the implementation of our model ensemble approach. Additionally, we aim to derive valuable insights into the practical advantages and potential trade-offs associated with deploying model ensembles. 


# Project Status Report

## Data Preparation
- [x] DataModule code has been completed.
- [x] Dataset code is ready.
- [x] Data Util files have been coded.

## Data Analysis
- [x] Data analysis on the dataset has been concluded.
- [x] Key insights from the analysis:
  - the dataset contains different number of frames for each patient
  - correlation between distinct groups and the proportion of specific cardiac components
  - correlation between the record's phase (ED or ES) and the proportion of specific cardiac components

## Docker
- [x] Initial version of the Dockerfile is complete.

## Next Steps
- [ ] Define evaluation metrics
- [ ] Define and implement baseline model.
- [ ] Train the model with the prepared data.
- [ ] Document the model training process.
- [ ] Construct model ensemble
- [ ] Train ensemble
- [ ] Document the ensemble training process
- [ ] Compare ensemble and baseline model

## Milestones
- [x] I. Milestone  - Containerization, data acquisition, preparation and analysis
- [ ] II. Milestone - Baseline evaluation, baseline model
- [ ] III. Milestone - Final submission


# Files descriptions
 - acdc_datamodule.py: contains the Datamodule for the pytorch lightning framework. Handles Dataset and Dataloader creation.
 - acdc_dataset.py: contains the ACDCDataset pytorch Dataset class, which handles the loading, transformation, and indexing of data samples
 - acdc_utils.py: contains utility functions for the ACDCDataset and ACDCDataModule classes
   - DualTransform: class, that applies the same augmentation transformation for both the image and the ground-truth
   - get_acdc: loads and splits the MRI scan files from the given path
   - convert_masks: splits multi-class masks into multiple single-class masks
   - convert_mask_single: splits multi-class mask into multiple single-class masks
   - visualize: visualize Nib images
   - get_images_with_info: like the get_acdc but it appends additional infos to the images for easier analysing
   - get_label_percentages: returns the the percentages of pixels for the different labels in a labeled image
 - analysis.ipynb: dataset analysis

# How to

   ```bash
   [chmod +x start.sh]
   ./start.sh
   ```

This script automates dataset download and Docker container setup for a medical image machine learning project. It simplifies the initial project setup by downloading the dataset, preparing the data structure, and launching a Docker container with GPU support.

# Related works
https://www.creatis.insa-lyon.fr/Challenge/acdc/

https://ieeexplore.ieee.org/document/8360453

https://github.com/kingo233/FCT-Pytorch/tree/main
