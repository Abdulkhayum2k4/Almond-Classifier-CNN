import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === Paths ===
model_path = "C:/Users/sigal/OneDrive/Desktop/almondclassifier/models/almond_model.pth"
dataset_path = "C:/Users/sigal/OneDrive/Desktop/almondclassifier/dataset"

# === Classes (get from dataset subfolder names) ===
classes = sorted(os.listdir(dataset_path))

# === Define the same transform used in train.py ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Load the model architecture ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))  # same output layer
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === Function to predict the almond class ===
def predict(image_path):
    if not os.path.isfile(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
        print(f"✅ Predicted almond class: {predicted_class}")

# === Main Execution ===
if __name__ == "__main__":
    image_path = input("Enter the full path to the almond image: ").strip()
    predict(image_path)

#  C:\Users\sigal\OneDrive\Desktop\almondclassifier\dataset\SIRA\test1.jpg 
# C:\Users\sigal\OneDrive\Desktop\almondclassifier\dataset\NURLU\nurlu (1).jpg
#  C:\Users\sigal\OneDrive\Desktop\almondclassifier\dataset\SIRA\SAM_2683 kopyası 25.jpg 
   









