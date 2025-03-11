import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import CNN  # Import CNN model structure, but NOT train it

# Function to load the trained model (ensures it only loads ONCE)
@st.cache_resource
def load_trained_model():
    model = CNN()  # Initializes the CNN model (BUT doesn't train it)
    model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=torch.device("cpu")))  # Load trained weights
    model.eval()  # Set to evaluation mode
    return model

# Load the model once and store in cache
model = load_trained_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("MNIST Digit Recognizer 🧠✏️")
st.write("Draw a digit below and get the model's prediction!")

# Create a drawing canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Process the drawn image and make a prediction
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.write(f"**Prediction: {predicted.item()}** 🎯")
