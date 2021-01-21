import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset import CIFAR10DataModule
from differentiable_randaugment import RandAugmentModule
from model import wide_resnet_28x10


class MyLightningModule(pl.LightningModule):
    def __init__(self, base_lr: float, total_epochs: int):
        super().__init__()
        self.base_lr = base_lr
        self.total_epochs = total_epochs

        self.augmentor = RandAugmentModule(num_ops=2, normalized=True)
        self.model = wide_resnet_28x10(10)

    def configure_optimizers(self):
        param_groups = [
            {"params": self.augmentor.parameters(), "lr": 10 * self.base_lr},
            {"params": self.model.parameters(), "lr": self.base_lr},
        ]

        optimizer = optim.Adam(param_groups, lr=self.base_lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.total_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, labels = batch

        logits = self.model(self.augmentor(x))
        loss = F.cross_entropy(logits, labels)

        self.log("train/loss", loss, prog_bar=False)
        self.log("randaug/magnitude", self.augmentor.get_magnitude(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, labels = batch

        logits = self.model(x)
        loss = F.cross_entropy(logits, labels)

        self.log("val/loss", loss)
        self.log("val/acc1", self._accuracy(logits, labels, topk=1), prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, labels = batch
        logits = self.model(x)
        self.log("test/acc1", self._accuracy(logits, labels, topk=1))

    def _accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor, topk: int = 1
    ) -> torch.Tensor:
        _, indices = torch.topk(logits, k=topk)
        correct = indices == labels.unsqueeze(-1)
        return correct.any(-1).float().mean()


if __name__ == "__main__":
    datamodule = CIFAR10DataModule(
        train_batch_size=128, val_batch_size=128, num_workers=4
    )
    model = MyLightningModule(base_lr=1e-3, total_epochs=200)

    trainer = pl.Trainer(gpus=1, max_epochs=200)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
