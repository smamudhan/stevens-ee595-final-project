import cv2
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Train SVM

# Loading Saved Data
print("Loading Saved Data")
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')
val_features = np.load('val_features.npy')
val_labels = np.load('val_labels.npy')

print("Scaling Loaded Data")
scaler = joblib.load('scaler.pkl')
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print("Starting training")
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(train_features, train_labels)
print("End Training")

# Save SVM model and scaler
joblib.dump(svm, 'svm_model.pkl')
print("Saved SVM model.")

print("Evaluating on validation data")
# Evaluate SVM on validation data
val_preds = svm.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Evaluating on test data")
# Evaluate SVM on test data
test_preds = svm.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix
print("Classification Report on Test Set:")
print(classification_report(test_labels, test_preds))

conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap='coolwarm', fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('confusion_matrix.png')
plt.show()
