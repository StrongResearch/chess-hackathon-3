from hackathon_train.train import VisionTransformerTwoHeads
from torch import nn

class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = VisionTransformerTwoHeads(
            in_channels=model['in_channels'],
            patch_size=model['patch_size'],
            embed_dim=model['embed_dim'],
            num_heads=model['num_heads'],
            num_layers=model['num_layers'],
            num_classes1=model['num_classes1'],
            num_classes2=model['num_classes2']
        )
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
