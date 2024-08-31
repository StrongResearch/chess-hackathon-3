from hackathon_train.train import VisionTransformerTwoHeads
from torch import nn

class VisionTransformerInference(nn.module):
    def __init__(self, config):
        super().__init__()
        self.model = VisionTransformerTwoHeads(
            in_channels=config['model']['in_channels'],
            patch_size=config['model']['patch_size'],
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            num_classes1=config['model']['num_classes1'],
            num_classes2=config['model']['num_classes2']
        )
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

    def forward(self, x):
        return self.model(x)