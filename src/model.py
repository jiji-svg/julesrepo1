import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Adjust the input features to the linear layer based on the output of conv and pool layers
        # Assuming input image size is 28x28, after one pool layer, it becomes 14x14
        # So, 64 channels * 14 * 14 = 12544. If another pool, then 7*7 = 3136
        # For this example, let's assume one pooling layer.
        # The input size to the first linear layer depends on the output size of the last pooling layer.
        # If input is (B, 1, 28, 28):
        # Conv1 -> (B, 32, 28, 28)
        # Pool1 -> (B, 32, 14, 14)
        # Conv2 -> (B, 64, 14, 14)
        # Pool2 (if we add another one) -> (B, 64, 7, 7)
        # For this example, let's use one pool layer, so after conv2, size is (B, 64, 14, 14)
        # And after the second pool, it would be (B, 64, 7, 7)
        # So, fc1 input features should be 64 * 7 * 7 = 3136 if we have two pool layers
        # Let's stick to one pool layer for simplicity in this example, so after conv2 it's 64*14*14
        # self.fc1 = nn.Linear(64 * 14 * 14, 128) # If one pool layer
        # Let's add a second pooling layer for a more typical CNN structure
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # After two pool layers (28x28 -> 14x14 -> 7x7)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # Add second pool layer
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example usage:
    model = SimpleCNN(num_classes=10)
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(64, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (64, 10)
