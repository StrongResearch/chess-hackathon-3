import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_filters=32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(19, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 1858),
            nn.Tanh()
        )

    def forward(self, input_planes: torch.Tensor):
        x = input_planes.reshape(-1, 19, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        policy_out = self.policy_head(x)
        return policy_out