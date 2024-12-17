from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import joblib

def predict_with_pth(image_path, pth_file_path, model_class):
    """Predicts the class of an image using a PyTorch model.

    Args:
        image_path: Path to the image file.
        pth_file_path: Path to the .pth file containing the state_dict.
        model_class: The model class to instantiate and load the state_dict.

    Returns:
        The predicted class (0 or 1).
    """

    # Instantiate the model and load the state_dict
    model = model_class()
    state_dict = torch.load(pth_file_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((200, 200)),  # Match the input size used in training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(image).squeeze()  # Get the raw output
        predicted = (output > 0.5).float().item()  # Binary classification threshold

    return int(predicted)

def predict_with_pkl_model(image_path, model_path, target_size=(200, 200)):
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Shape: (H, W, 3)

    # Flatten the image into a 1D feature vector
    feature_vector = image_array.flatten()  # Shape: (H*W*3,)

    # Load the model
    model = joblib.load(model_path)

    # Reshape input for the model
    feature_vector = feature_vector.reshape(1, -1)  # Model expects 2D input: (1, num_features)

    # Make the prediction
    prediction = model.predict(feature_vector)

    return int(prediction[0])

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Calculate the flattened size after the convolutional layers
        test_input = torch.zeros(1, 3, 200, 200)  # Example input
        conv_output_size = self.conv_layers(test_input).view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class MobileNetClassifier(nn.Module):
    def __init__(self):
        super(MobileNetClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False
        # Replace the classifier with a custom classifier
        self.mobilenet.classifier[1] = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mobilenet(x)

# Example usage
# image_file = '/workspace/145675.jpg'
# pth_file = '/workspace/cnn_model.pth'
# predicted_class = predict_with_pth(image_file, pth_file, model_class=CNNModel)
# print(f"Predicted class: {predicted_class}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # prediction/app/weights/cnn_full.pth
        cnn_full = predict_with_pth(file, '../app/weights/cnn_full.pth', model_class=CNNModel)
        mobilenet_full = predict_with_pth(file, '../app/weights/mobilenet_full.pth', model_class=MobileNetClassifier)
        svm_full = predict_with_pkl_model(file, '../app/weights/svm_full.pkl', target_size=(150, 150))
        rf_full = predict_with_pkl_model(file, '../app/weights/rf_full.pkl')
        cnn_reduced = predict_with_pth(file, '../app/weights/cnn_reduced.pth', model_class=CNNModel)
        mobilenet_reduced = predict_with_pth(file, '../app/weights/mobilenet_reduced.pth', model_class=MobileNetClassifier)
        svm_reduced = predict_with_pkl_model(file, '../app/weights/svm_reduced.pkl', target_size=(150, 150))
        rf_reduced = predict_with_pkl_model(file, '../app/weights/rf_reduced.pkl')
        
        return jsonify({
            "cnn_full": cnn_full,
            "mobilenet_full": mobilenet_full,
            "svm_full": svm_full,
            "rf_full": rf_full,
            "cnn_reduced": cnn_reduced,
            "mobilenet_reduced": mobilenet_reduced,
            "svm_reduced": svm_reduced,
            "rf_reduced": rf_reduced
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

