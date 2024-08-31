import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse


from torch.utils.data import Dataset


from models.VIT_multihead.vit_multi_head import VisionTransformerTwoHeads


class DummyDatasetVisionTransformer(Dataset):
    def __init__(self, num_samples=1000, input_channels=11, input_size=8, num_classes1=3, num_classes2=1858):
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        
        # Generate random data
        self.data = torch.randn(num_samples, input_channels, input_size, input_size)
        self.labels1 = torch.randint(0, num_classes1, (num_samples,))
        self.labels2 = torch.randint(0, num_classes2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels1[idx], self.labels2[idx]


class VisionTransformerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = VisionTransformerTwoHeads(
            in_channels=config['model']['in_channels'],
            patch_size=config['model']['patch_size'],
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            num_classes1=config['model']['num_classes1'],
            num_classes2=config['model']['num_classes2']
        )
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y1, y2 = batch
        out1, out2 = self(x)
        loss1 = self.criterion1(out1, y1)
        loss2 = self.criterion2(out2, y2)
        loss = loss1 + loss2
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y1, y2 = batch
        out1, out2 = self(x)
        loss1 = self.criterion1(out1, y1)
        loss2 = self.criterion2(out2, y2)
        loss = loss1 + loss2
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.training['learning_rate'], 
            weight_decay=self.hparams.training['weight_decay']
        )
        return optimizer

    def train_dataloader(self):
        dataset = DummyDatasetVisionTransformer(
            num_samples=self.hparams.data['num_samples'],
            input_channels=self.hparams.model['in_channels'],
            input_size=self.hparams.data['input_size'],
            num_classes1=self.hparams.model['num_classes1'],
            num_classes2=self.hparams.model['num_classes2']
        )
        return DataLoader(dataset, batch_size=self.hparams.training['batch_size'], shuffle=True)

    def val_dataloader(self):
        dataset = DummyDatasetVisionTransformer(
            num_samples=self.hparams.data['num_samples'] // 5,  # Smaller validation set
            input_channels=self.hparams.model['in_channels'],
            input_size=self.hparams.data['input_size'],
            num_classes1=self.hparams.model['num_classes1'],
            num_classes2=self.hparams.model['num_classes2']
        )
        return DataLoader(dataset, batch_size=self.hparams.training['batch_size'])

def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set up model
    model = VisionTransformerLightning(config)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='/shared/checkpoints',
        filename='vit-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Set up logger
    logger = TensorBoardLogger("lightning_logs", name="vision_transformer")

    is_cuda_available = torch.cuda.is_available()
    strategy = "auto"
    accelerator = "cpu"
    if is_cuda_available:
        strategy = "ddp"
        accelerator="gpu"

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy=strategy,
        accelerator=accelerator,
    )

    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    main(args.config)