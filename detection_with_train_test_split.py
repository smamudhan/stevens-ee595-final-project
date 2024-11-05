import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess dataset
def load_data(data_directory, img_size=(128, 128), batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load images from both 'real' and 'ai' folders
    full_dataset = datasets.ImageFolder(root="dataset", transform=transform)
    # ai_dataset = datasets.ImageFolder("dataset\AiArtData", transform=transform)
    
    # Concatenate datasets
    # full_dataset = real_dataset + ai_dataset
    
    # Split dataset into training and testing
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# CNN model for binary classification
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Training function
def train_classifier(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        
        # Validation step
        model.eval()
        test_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        test_losses.append(test_loss / len(test_loader))
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {test_loss / len(test_loader)}")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    return train_losses, test_losses

# Plot training results
def plot_training_history(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# Main script
if __name__ == "__main__":
    # Define data directory
    data_directory = r"stevens-ee595-final-project\dataset"  # Update with your dataset path
    print("path",os.path.join(data_directory,"RealArt"))

    # Load data
    train_loader, test_loader = load_data(data_directory)

    # Build model
    classifier = CNNClassifier()

    # Train model
    train_losses, test_losses = train_classifier(classifier, train_loader, test_loader)

    # Plot training history
    plot_training_history(train_losses, test_losses)
    
    
    

    # Save the model
    torch.save(classifier.state_dict(), "real_vs_ai_classifier.pth")
    print("Model saved as real_vs_ai_classifier.pth")
