import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from trainCNN import CNN  # Import the CNN model

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("cnn_mnist_model.pth"))
model.eval()  # Set the model to evaluation mode (no training)

# Define the transform (same as in training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)  # Load one image at a time

# Function to visualize predictions
def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(npimg.squeeze(), cmap="gray")
    plt.axis("off")

# Get a few test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Run the model on the images
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)  # Get the highest probability class

# Display the image and prediction
plt.figure(figsize=(5, 5))
imshow(images[0])  # Show the image
plt.title(f"Predicted: {predicted.item()}, Actual: {labels.item()}")
plt.show()
