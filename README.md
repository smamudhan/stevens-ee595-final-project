# Real vs AI Image Classifier

This project is a deep learning-based binary classifier built with PyTorch. It aims to classify images into two categories (e.g., real and AI-generated) using a Convolutional Neural Network (CNN) model. The code is designed to train on a dataset organized in a specific folder structure and save the trained model for later use.

## Requiements
- Python 3.6+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- CUDA (optional for GPU support)

## Key Functions
load_data: Loads and preprocesses the dataset.
CNNClassifier: Defines the CNN model architecture.
train_classifier: Trains the model and evaluates performance on the validation set.
plot_training_history: Plots the training and validation loss curves.
