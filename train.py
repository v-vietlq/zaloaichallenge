import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from video_dataset import VideoFrameDataset
from options.train_options import TrainOptions
import random
import os
from fasmodule import FasModule
from torchvision import transforms as T
import torch.utils.data as data
from randaugment import RandAugment
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == '__main__':
    train_opt = TrainOptions().parse()
    np.random.seed(train_opt.seed)
    random.seed(train_opt.seed)
    torch.manual_seed(train_opt.seed)
    train_opt.phase = 'train'

    val_opt = TrainOptions().parse()
    val_opt.phase = 'val'
    val_opt.batch_size = 1

    fas_module = FasModule(train_opt, val_opt)

    if train_opt.pretrained is not None:
        fas_module.print('Loading pretrained model from ',
                         train_opt.pretrained)
        state_dict = torch.load(train_opt.pretrained, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict['state_dcit']
            state_dict = {k.replace('net', ''): v for k,
                          v in state_dict.items()}
        fas_module.net.load_state_dict(state_dict)

    # save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.2f}',
        save_last=True,
        verbose=True,
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=train_opt.patience,
        verbose=False,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logging = TensorBoardLogger(
        save_dir=train_opt.save_dir, name=train_opt.name)

    trainer = pl.Trainer(gpus=train_opt.gpus,
                         resume_from_checkpoint=train_opt.resume,
                         accelerator=train_opt.accelerator,
                         logger=logging,
                         max_epochs=train_opt.max_epoch,
                         callbacks=[early_stopping_callback,
                                    checkpoint_callback, lr_monitor]
                         )

    # # Folder in which all videos lie in a specific structure
    # train_root = os.path.join(os.getcwd(), 'zaloai/train/videos')
    # # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
    # train_annotation_file = os.path.join(
    #     train_root.replace('videos', ''), 'train_annotations.txt')

    # train_transform = T.Compose([
    #     # T.RandomResizedCrop((224, 224)),
    #     # T.RandomRotation(degrees=30.),
    #     # T.RandomPerspective(distortion_scale=0.4),
    #     T.Resize((224, 224)),
    #     # T.RandomHorizontalFlip(p=0.5),
    #     # T.RandomRotation((-180, 180)),
    #     # T.RandomVerticalFlip(p=0.5),
    #     RandAugment(),
    #     T.ToTensor(),

    # ])
    train_transform = A.Compose([
        # RandomSizedCrop not working for some reason. I'll post a thread for this issue soon.
        # Any help or suggestions are appreciated.
        # A.RandomSizedCrop(min_max_height=(224, 224), height=224, width=224, p=0.5),
        #         A.RandomSizedCrop(min_max_height=(300, 1000), height=1000, width=1000, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Resize(height=224, width=224, p=1),
        A.Cutout(num_holes=8, max_h_size=32,
                 max_w_size=32, fill_value=0, p=0.5),
        ToTensorV2(p=1.0)

    ],
    )

    train_dataset = VideoFrameDataset(
        root_path=train_opt.train_root,
        annotationfile_path=train_opt.train_list,
        num_segments=4,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=train_transform,
        test_mode=False
    )

    # val_root = os.path.join(os.getcwd(), 'zaloai/train/videos')
    # # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
    # val_annotation_file = os.path.join(
    #     train_root.replace('videos', ''), 'val_annotations.txt')

    val_transform = T.Compose([
        # T.RandomResizedCrop((224, 224)),
        # T.RandomRotation(degrees=30.),
        # T.RandomPerspective(distortion_scale=0.4),
        T.Resize((224, 224)),
        T.ToTensor(),


    ])

    val_dataset = VideoFrameDataset(
        root_path=train_opt.val_root,
        annotationfile_path=train_opt.val_list,
        num_segments=4,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=val_transform,
        test_mode=True
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_opt.batch_size, num_workers=train_opt.num_threads, shuffle=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=train_opt.batch_size, num_workers=train_opt.num_threads, shuffle=False)

    trainer.fit(fas_module, train_loader, val_loader)
