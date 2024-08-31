import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return policy_out, None
    



class BiggerNet(nn.Module):
    def __init__(self, num_filters=32, num_blocks=3):
        super(BiggerNet, self).__init__()
        self.conv_input = nn.Conv2d(19, num_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1) for _ in range(num_blocks)])
        self.relu = nn.ReLU()

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 128, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 1858),
            nn.Tanh()
        )

    def forward(self, s):
        s = s.view(-1, 19, 8, 8)
        s = self.relu(self.conv_input(s))
        
        for block in self.blocks:
            s = self.relu(block(s))
        
        policy_out = self.policy_head(s)
        return policy_out, None
    






# Initialization function
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
