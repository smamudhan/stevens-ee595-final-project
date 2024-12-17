import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_images(data_dir, image_size=(256, 256)):
    data = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize
                    data.append(img)
                    labels.append(label_dir)
    return np.array(data), np.array(labels)

def prepare_data(data_dir):
    # Load data
    X, y = preprocess_images(data_dir)
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, label_encoder

if __name__ == "__main__":
    data_dir = "path/to/your/dataset"
    X_train, X_val, y_train, y_val, label_encoder = prepare_data(data_dir)
    np.savez("preprocessed_data.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, classes=label_encoder.classes_)
