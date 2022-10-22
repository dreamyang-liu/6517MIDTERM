import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score
import segmentation_models_pytorch as smp

class SegmentationModule(pl.LightningModule):

    def __init__(self, input_channel=3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.model.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
        # self.model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))

    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            pred = torch.softmax(pred, dim=1)
        return pred.argmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return torch.softmax(x['out'], dim=1)
    
    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self.forward(x)
        flat_y_hat = y_hat.view(x.shape[0], 4, -1)
        flat_y = y.view(x.shape[0], -1)
        loss = torch.nn.functional.cross_entropy(flat_y_hat, flat_y, weight = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(self.device))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        flat_y_hat = y_hat.view(x.shape[0], 4, -1)
        flat_y = y.view(x.shape[0], -1)
        loss = torch.nn.functional.cross_entropy(flat_y_hat, flat_y, weight = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(self.device))
        return {"val_loss": loss, "flat_y_hat": flat_y_hat.argmax(dim=1).cpu().reshape(-1, 1), "flat_y": flat_y.cpu().reshape(-1, 1)}
    
    def validation_epoch_end(self, val_batch_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in val_batch_outputs]).mean()
        flat_y = torch.cat([x['flat_y'] for x in val_batch_outputs])
        flat_y_hat = torch.cat([x['flat_y_hat'] for x in val_batch_outputs])
        accuracy = accuracy_score(flat_y, flat_y_hat)
        f1_score = f1_score(flat_y, flat_y_hat, average='macro')
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_accuracy", accuracy, sync_dist=True)
        self.log("val_f1_score", f1_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.forward(x)
        flat_y_hat = y_hat.view(x.shape[0], 4, -1)
        flat_y = y.view(x.shape[0], -1)
        loss = torch.nn.functional.cross_entropy(flat_y_hat, flat_y, weight = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(self.device))
        return {"test_loss": loss, "flat_y_hat": flat_y_hat.argmax(dim=1).cpu().reshape(-1, 1), "flat_y": flat_y.cpu().reshape(-1, 1)}

    
    def test_epoch_end(self, test_batch_outputs):
        avg_loss = torch.stack([x['test_loss'] for x in test_batch_outputs]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in test_batch_outputs]).mean()
        avg_f1_score = torch.tensor([x['f1_score'] for x in test_batch_outputs]).mean()
        self.log("final_test_loss", avg_loss, sync_dist=True)
        self.log("final_test_accuracy", avg_accuracy, sync_dist=True)
        self.log("final_test_f1_score", avg_f1_score, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
