import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import CNN  # Import model

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        images = images.to(torch.float32)  # Ensure input is float32
        labels = labels.to(torch.long)  # Ensure labels are long type
        outputs = model(images)

        # Debugging: Check shapes
        #print(f"Batch size (images): {images.shape}, Model output shape: {outputs.shape}, Labels shape: {labels.shape}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), "cnn_mnist_model.pth")
print("Model saved as cnn_mnist_model.pth")
