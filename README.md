# Workflow Blood Vessel Graft Segmentation
This repository contains the code and results for the quantification of histological staining in blood vessel grafts segmentation workflow. 
Three different segmentation models were trained and evaluated using 512px by 512px image patches. The repository contains the data organization, training code, evaluation scripts, transformation analysis and experimental results.

# Repository Structure
## Code
The `code/` folder contains the complete project pipeline:
- `Cytomine.py`: Conversion of full Cytomine slides to usable slides.
- `Data_processing`: Processing of data from slides to training, testing and validation patches.
- `Data_distribution`: Utility script to find the data distribution of the random sampled patches.
- `Model.py`: Implementation of the three segmentation models.
- `Losses_metrics.py`: Loss functions and evaluation metrics.
- `Evaluate.py`: Extraction of (sigmoid) prediction mask of patches using the three trained models.
- `Full_slide_evaluation.py`: Extraction of prediction mask of full slide using the three trained models.
- `Transformation.py`: scripts for the three spatial analysis implementations.
The three models each have their own version of Model.py, Evaluate.py and Full_slide_evaluation.py
## Data
The `data/` folder contains the csv files detailing the patches and slides used during training, validating and testing the models.
- `Slides`: Whole-slide image information
- `Training`: Training patches information
- `Validation`: Validation patches information
- `Testing`: Testing patches information
## Results
The `results/` folder contains the results obtained during model training and evaluation
- `Metrics`: Folder with csv files of the training history metrics as well as model summaries and slide metrics
- `Visualizations`: Folder containing the graphs of the training performance over the training epochs
## Requirements
Install the required libraries specified in `Requirements.txt`.
# Models
Three different model architectures were made, trained and analysed:
- Model 1 (standard): U-Net with ResNet34 encoder using pretrained weights from ImageNet
- Model 2 (DPT): Custom ViT-DPT architecture using DINOv2 pretrained weights
- Model 3 (DINO): U-Net with ResNet50 encoder using DINOv1 pretrained weights
# Spatial Analysis Transforms
Three different types of spatial analysis methods where applied to the slides:
- Distance transformation: Implementation of tissue intensity as a function of distance from lumen
- Polar transformation: Implementation of tissue intensity as a function of normalized tissue thickness and angle using a polar representation
- Finite Element Method (FEM) transformation: Implementation of tissue intensity as a function of distance to center of tissue and angle using a FEM transformation
# Author
L.M. van Dam

Eindhoven University of Technology - Biomedical Engineering

2026
# Acknowledgements
I would like to thank Prof. M. Veta and Dr. P.J. Besseling for their guidance and support throughout this project

