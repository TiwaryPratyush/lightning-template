import os
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from src.utils.transforms import get_transforms

class DogBreedDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/dog_breeds", batch_size: int = 32, img_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = datasets.ImageFolder(root=self.data_dir, transform=get_transforms(self.img_size, train=True))
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
            self.val_dataset.dataset.transform = get_transforms(self.img_size, train=False)

        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset  # Using validation dataset for testing

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    @property
    def num_classes(self):
        return len(self.train_dataset.dataset.classes)
