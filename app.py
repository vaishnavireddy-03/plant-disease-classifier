import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------------------
# CNN Model
# -----------------------------

class PlantCNN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)

        self.conv3 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)

        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# -----------------------------
# Classes from training
# -----------------------------

classes = [
    "PlantVillage",
    "plantvillage"
]

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = PlantCNN(num_classes=2)

    state_dict = torch.load(
        "plantcnn.pth",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image and the model will make a prediction."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(img_tensor)

        _, prediction = torch.max(output, 1)

    predicted_class = classes[prediction.item()]

    st.success(
        f"Prediction: {predicted_class}"
    )
