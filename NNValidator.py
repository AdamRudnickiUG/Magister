import torch
import torch.nn as nn
import torch.optim as optim
import random

# Dummy dimensions - adjust to match your input
INPUT_SIZE = 20
HIDDEN_SIZE = 64

class NNValidator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)  # Regression output
        )

    def forward(self, x):
        return self.model(x)