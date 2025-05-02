from .rnn import RNN
import torch
import lightning as pl
from torchmetrics import Accuracy

class RNNTrainer(pl.LightningModule):
    def __init__(self, model: RNN, loss_fn: str = 'crossentropy', optimizer: str = 'adam'):
        super().__init__()
        self.model = model
        self.loss_fn = self._init_loss_fn(loss_fn)
        self.optimizer = optimizer
        self.accuracy = Accuracy("multiclass", num_classes=model.n_classes)

    def _init_loss_fn(self, loss_fn: str):
        if loss_fn == 'crossentropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_fn == 'mse':
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(f"Training step {batch_idx}")

        x, y = batch
        print(f"Batch x shape: {x.shape}")
        print(f"Batch y shape: {y.shape}")

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def validation_step(self, batch, batch_idx):
        print(f"Training step {batch_idx}")

        x, y = batch
        print(f"Batch x shape: {x.shape}")
        print(f"Batch y shape: {y.shape}")

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y))

        return loss