import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# Preprocess dataset and load data
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

data_dir = "./dataset"  # Update with the actual path
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Convert dataset to numpy arrays
data = []
labels = []
for img, label in dataset:
    data.append(img.numpy().flatten())  # Flatten the image
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Assuming features and labels are already loaded as NumPy arrays
# data = np.load('features.npy')  # Replace with actual feature array
# labels = np.load('labels.npy')  # Replace with actual label array

# Split the dataset into training, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

# Scale the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
train_data, train_labels = smote.fit_resample(train_data, train_labels)

# Compute class weights for SVM
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(enumerate(class_weights))

# Define and tune the SVM model using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(class_weight=class_weight_dict, probability=True), param_grid, refit=True, cv=3, verbose=2, return_train_score=True)
grid.fit(train_data, train_labels)

print(f"Best Parameters: {grid.best_params_}")

# Train the best model
svm_model = grid.best_estimator_

# Evaluate on training and validation sets
train_preds = svm_model.predict(train_data)
val_preds = svm_model.predict(val_data)

train_accuracy = accuracy_score(train_labels, train_preds)
val_accuracy = accuracy_score(val_labels, val_preds)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on the test set
test_preds = svm_model.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_preds)
conf_matrix = confusion_matrix(test_labels, test_preds)
class_report = classification_report(test_labels, test_preds)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap='coolwarm', fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('confusion_matrix.png')
plt.show()

# Save training and validation accuracy plots
epochs = list(range(1, len(grid.cv_results_['mean_test_score']) + 1))
train_accuracies = grid.cv_results_['mean_train_score']  # Now this works
val_accuracies = grid.cv_results_['mean_test_score']

plt.figure()
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# Cross-validation scores
cross_val_scores = cross_val_score(svm_model, data, labels, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Accuracy: {cross_val_scores.mean():.4f}")

# Save the model
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
print("Model saved as 'svm_model.pkl'")

