import torch
import torch.nn as nn
import torch.optim as optim

# Set device to MPS if available, otherwise fall back to CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Print the device being used
if torch.backends.mps.is_available():
    print("Using Metal Performance Shaders (MPS) for computations.")
else:
    print("MPS not available. Using CPU for computations.")

# Simple model example: A simple feedforward neural network
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = YourModel()
model.to(device)  # Move the model to the selected device (MPS or CPU)

# Define a simple loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example tensor moved to Metal device
input_tensor = torch.randn(2, 2).to(device)  # A random input tensor
target_tensor = torch.randn(2, 1).to(device)  # A random target tensor

# Training loop (simple example)
for epoch in range(100):
    optimizer.zero_grad()  # Clear previous gradients
    output = model(input_tensor)  # Forward pass
    loss = criterion(output, target_tensor)  # Calculate loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item()}")

print("Training completed.")