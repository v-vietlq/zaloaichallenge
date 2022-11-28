import pytorch_lightning as pl
from video_dataset import VideoFrameDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.dataset import Dataset, Subset


class ZaloLivenessDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root,
                 val_root,
                 train_list,
                 val_list,
                 batch_size,
                 num_threads,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.train_root = train_root
        self.train_list = train_list
        self.val_root = val_root
        self.val_list = val_list
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_threads = num_threads

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.zalo_train = VideoFrameDataset(
                root_path=self.train_root,
                annotationfile_path=self.train_list,
                num_segments=4,
                frames_per_segment=1,
                imagefile_template='img_{:05d}.jpg',
                transform=self.train_transforms,
                test_mode=False
            )
            self.zalo_val = VideoFrameDataset(
                root_path=self.val_root,
                annotationfile_path=self.val_list,
                num_segments=4,
                frames_per_segment=1,
                imagefile_template='img_{:05d}.jpg',
                transform=self.val_transforms,
                test_mode=True
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.zalo_train, batch_size=self.batch_size, num_workers=self.num_threads, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.zalo_val, batch_size=self.batch_size, num_workers=self.num_threads, shuffle=False)
        return val_loader


class ZaloLivenessKfoldDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root,
                 val_root,
                 train_list,
                 val_list,
                 batch_size,
                 num_threads,
                 k=1,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.train_root = train_root
        self.train_list = train_list
        self.val_root = val_root
        self.val_list = val_list
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_threads = num_threads

        self.k = k

    def setup(self, stage=None) -> None:
        self.full_data = VideoFrameDataset(
            root_path=self.train_root,
            annotationfile_path=self.train_list,
            num_segments=4,
            frames_per_segment=1,
            imagefile_template='img_{:05d}.jpg',
            transform=self.train_transforms,
            test_mode=False
        )
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_splits = [k for k in kf.split(self.full_data)]

        train_indexes, val_indexes = all_splits[self.k]

        self.zalo_train, self.zalo_val = Subset(
            self.full_data, train_indexes), Subset(self.full_data, val_indexes)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.zalo_train, batch_size=self.batch_size, num_workers=self.num_threads, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.zalo_val, batch_size=self.batch_size, num_workers=self.num_threads, shuffle=False)
        return val_loader
