#  Enhancing Deep-Learning Solutions through Model Ensembles for Semantic Segmentation on ACDC Dataset

## Team name
Another bomb training - Újabb bomba tanítás

## Team members

- Benyák Bence József - AQTBDV
- Klenk Botond - FTNYN1
- Bánfi Zsombor - L1N5IV

## Overview
This project focuses on the implementation and analysis of model ensembles in deep learning to improve the accuracy of semantic segmentation. Model ensembles, a proven technique used by leading AI competition winners, involve integrating multiple models to enhance the overall precision of deep learning solutions. Although this approach incurs additional computational costs, it offers a promising avenue for significantly refining the performance of our deep-learning models.

## Data Acquisition
The ACDC (Automated Cardiac Diagnosis Challenge) dataset used in this project was acquired from [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb). We also created a shell script that automatically downloads and unzips the dataset from a Google Drive folder. The dataset comprises complex cardiac MRI scans, providing a rich and diverse set of data for training and evaluating our model ensemble.


## Objective
Our main objective is to explore and apply various strategies for constructing effective model ensembles tailored specifically for semantic segmentation tasks. We will leverage the ACDC dataset for training and evaluation purposes.


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
- [x] Define evaluation metrics
- [x] Define and implement baseline model.
- [x] Train the model with the prepared data.
- [x] Document the model training process.
- [x] Construct model ensemble
- [x] Train ensemble
- [x] Document the ensemble training process
- [x] Compare ensemble and baseline model

## Milestones
- [x] I. Milestone  - Containerization, data acquisition, preparation and analysis
- [x] II. Milestone - Baseline evaluation, baseline model
- [x] III. Milestone - Final submission


# File descriptions
 - acdc_datamodule.py: contains the Datamodule for the pytorch lightning framework. Handles Dataset and Dataloader creation.
 - acdc_dataset.py: contains the ACDCDataset pytorch Dataset class, which handles the loading, transformation, and indexing of data samples
 - acdc_utils.py: contains utility functions for the ACDCDataset and ACDCDataModule classes
   - DualTransform: class, that applies the same augmentation transformation for both the image and the ground-truth
   - get_acdc: loads and splits the MRI scan files from the given path
   - convert_masks: splits multi-class masks into multiple single-class masks
   - convert_mask_single: splits multi-class mask into multiple single-class masks
   - visualize: visualize Nib images
   - get_images_with_info: like the get_acdc but it appends additional info to the images for easier analysis
   - get_label_percentages: returns the percentages of pixels for the different labels in a labeled image
 - data_analysis.ipynb: dataset analysis
 - model_analysis.ipynb: analysis of model runs
 - start.sh: downloads the dataset and runs the container

# How to

## Setup environment

   ```bash
   [chmod +x start.sh]
   ./start.sh
   ```

This script automates dataset download and Docker container setup for a medical image machine-learning project. It simplifies the initial project setup by downloading the dataset, preparing the data structure, and launching a Docker container with GPU support.

## Train models

The training pipeline is implemented in the `modeling.ipynb` file. Open this Jupyter Notebook to access the training code and follow the steps below.

### Model Selection

There are four different models available for training. Choose the desired model by uncommenting the corresponding section in the notebook. The available models are:
1. Unet from the segmentation_models_pytorch library
2. Unet from the [original Unet paper] (https://arxiv.org/abs/1505.04597)
3. [fcn_resnet50](https://arxiv.org/abs/1411.4038) from the pytorch pretrained models
4. Simple segmentation model

### Loss Functions

Two different loss functions have been implemented for experimentation. You can choose between Crossentropy Loss and Dice Loss by uncommenting the appropriate section in the notebook.


### Dataset Modification

Depending on the selected loss function, the dataset may need to be modified. If using the Crossentropy Loss, the masks should be converted to one-hot format. Set the `convert_to_single` flag accordingly in the dataset constructor.

### Training Process

After setting the model, loss function, and dataset modification parameters, run the notebook to initiate the training process. Follow the instructions and monitor the training progress. Adjust hyperparameters as needed based on the training results.

Feel free to experiment with different models, loss functions, and dataset configurations to find the best combination for your specific use case.

Happy training!




 

# Related works
https://www.creatis.insa-lyon.fr/Challenge/acdc/

https://ieeexplore.ieee.org/document/8360453

https://github.com/kingo233/FCT-Pytorch/tree/main
