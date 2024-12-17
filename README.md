# Detecting AI-Generated Images Using Machine Learning and Deep Learning Techniques

In this repo, we present a study that uses machine learning techniques
to classify between Real and AI-generated images. This is created as part
of a graduate Applied ML class at Stevens Institute of Technology.

We implement and compare four models, by training them on two different datasets (outlined below): 
 1. Random Forest classifier,
 2. Support Vector Machine (SVM), 
 3. Vanilla Convolutional Neural Network (CNN)
 4. MobileNet CNN

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- CUDA
- cuml
- numpy
- numba
- joblib
- opencv-python
- imbalanced-learn
- pillow 

## Authors
- [Amudhan Manivasagam](https://github.com/smamudhan/)
- [Sameer Rajendra](https://github.com/SameerRajendra/)
- [Nikhil Krishna Bramhandam](https://github.com/Nikhil-wannabe/)

## Dataset
- [Full Dataset (Dataset 1)](https://www.kaggle.com/datasets/nikhil2k3/artifact-customcurated-extras)
- [Reduced Dataset (Dataset 2)](https://www.kaggle.com/datasets/nikhil2k3/project-data)

## Results
The /output folder contains the final model weights of the python notebooks, trained models, any relevant graphs, and sometimes - intermediate files of extracted features from the dataset.

The python files in the root folder are mostly identical to the notebooks; but were used to simplify training in a cloud environment (easier to run a script than setup jupyterlab)
