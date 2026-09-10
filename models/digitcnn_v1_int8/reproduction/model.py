"""Candidate v1: floating-point reference architecture, not trained yet."""

from torch import nn


class DigitCNN(nn.Module):
    """Input NCHW [N, 1, 28, 28]; output ten raw class scores."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(4 * 13 * 13, 10, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)
