import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from torchvision import models
import segmentation_models_pytorch as smp
from config import *
from utils import *
from sklearn.metrics import classification_report

class SegmentationModule(pl.LightningModule):

    def __init__(self, model, loss_type, encoder, encoder_weights, step_lr):
        super().__init__()
        self.save_hyperparameters()
        if loss_type == 'focal':
            self.loss = smp.losses.FocalLoss(LOSS_MODE)
        elif loss_type == 'dice':
            self.loss = smp.losses.DiceLoss(LOSS_MODE)
        elif loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        if model == 'unet':
            self.model = smp.Unet(
                            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=4,                      # model output channels (number of classes in your dataset)
                        )
        elif model == 'fpn':
            self.model = smp.FPN(
                            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=4,                      # model output channels (number of classes in your dataset)
                        )
        elif model == 'linknet':
            self.model = smp.Linknet(
                            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=4,                      # model output channels (number of classes in your dataset)
                        )

    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
        return pred.argmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return torch.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx): 
        x, y = batch # N * C * H * W, N * H * W
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'val_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'test_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def validation_epoch_end(self, val_batch_outputs):
        loss = torch.stack([x['val_loss'] for x in val_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in val_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in val_batch_outputs], dim=0).flatten()
        report = classification_report(y.cpu().numpy(), y_hat.cpu().numpy(), output_dict=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_precision', report['macro avg']['precision'], sync_dist=True)
        self.log('val_recall', report['macro avg']['recall'], sync_dist=True)
        self.log('val_f1', report['macro avg']['f1-score'], sync_dist=True)


    def test_epoch_end(self, test_batch_outputs):
        loss = torch.stack([x['test_loss'] for x in test_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in test_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in test_batch_outputs], dim=0).flatten()
        print(classification_report(y.cpu().numpy(), y_hat.cpu().numpy()))
        # print(report)
        # self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        # self.log('test_precision', report['macro avg']['precision'], sync_dist=True)
        # self.log('test_recall', report['macro avg']['recall'], sync_dist=True)
        # self.log('test_f1', report['macro avg']['f1-score'], sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-2)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
        # lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        return optimizer
