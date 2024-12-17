import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class to handle image preprocessing and loading
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = os.listdir(data_dir)
        
        # Collect all image paths and labels
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read and preprocess image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (150, 150))  # Resize to 150x150
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur
        
        # Convert to tensor and move to GPU
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add channel dimension and move to GPU
        image = image / 255.0  # Normalize the image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image.squeeze(0).flatten(), label  # Flatten image for the model

# Data transforms (if needed)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
data_dir = "path_to_dataset"  # Replace with the actual path
dataset = CustomDataset(data_dir=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Feature extraction (using PyTorch with GPU)
def extract_features(data_loader):
    features = []
    labels = []
    for inputs, target in data_loader:
        inputs = inputs.to(device)  # Move inputs to GPU
        features.append(inputs.cpu().detach().numpy())  # Move back to CPU for later use
        labels.append(target)
    return np.vstack(features), np.array(labels)

# Extract features from the dataset
features, labels = extract_features(data_loader)

# Split dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(train_features, train_labels)

# Predict on training and testing sets
train_preds = rf_model.predict(train_features)
test_preds = rf_model.predict(test_features)

# Calculate accuracy
train_accuracy = accuracy_score(train_labels, train_preds)
test_accuracy = accuracy_score(test_labels, test_preds)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds))

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dataset.class_names)
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_rf_cuda.png")
print("Confusion matrix saved as 'confusion_matrix_rf_cuda.png'")
plt.show()

# Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 important features

plt.figure(figsize=(10, 5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
plt.xlabel("Relative Importance")
plt.title("Top 10 Feature Importances")
plt.savefig("feature_importances_rf_cuda.png")
print("Feature importance plot saved as 'feature_importances_rf_cuda.png'")
plt.show()
