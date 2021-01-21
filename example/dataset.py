from typing import Optional

import albumentations as A
import albumentations.pytorch as AP
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int = 4,
        data_dir: str = "./",
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                AP.ToTensorV2(),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            total_data = CIFAR10(
                self.data_dir,
                train=True,
                transform=lambda x: self.transform(image=x)["image"],
            )
            self.train_set, self.val_set = random_split(total_data, [45000, 5000])

        if stage == "test" or stage is None:
            self.test_set = CIFAR10(
                self.data_dir,
                train=False,
                transform=lambda x: self.transform(image=x)["image"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
