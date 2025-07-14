from typing import Optional, Dict

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNet256DataModule(pl.LightningDataModule):
    """
    LightningDataModule for ImageNet-256 folder structure.
    """

    def __init__(self, data_dir, batch_size, num_workers, image_size, val_split, seed):
        super().__init__()
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.image_size: int = image_size
        self.val_split: float = val_split
        self.seed: int = seed

        # Normalization for ImageNet (mean/std)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        # Define train/val transforms
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.15)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        # Placeholders; will be set in setup()
        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None

    def prepare_data(self):
        """Download or prepare the dataset if necessary."""
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Create train/val splits from the ImageFolder.
        """
        # Only load once
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        # Load the entire ImageFolder
        full_dataset = ImageFolder(
            root=self.data_dir,
            transform=self.train_transforms,
            target_transform=add_one # 1…N are real classes, 0 is null
        )

        # Compute lengths for train/val
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        # Use `random_split` to partition (ensure reproducibility via a fixed seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Override the transforms for val subset
        self.val_dataset.dataset.transform = self.val_transforms

        # Class-index mapping:
        self.class_to_idx: Dict[str, int] = full_dataset.class_to_idx
        self.num_classes: int = len(self.class_to_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        return None


def add_one(label: int) -> int:
    return label + 1  