import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_gallery_transform(config):
    return A.Compose(
        [
            A.RandomCropFromBorders(
                always_apply=False,
                p=1.0,
                crop_left=0.11,
                crop_right=0.11,
                crop_top=0.11,
                crop_bottom=0.11,
            ),
            A.Resize(
                always_apply=False,
                p=1,
                height=config.dataset.input_size,
                width=config.dataset.input_size,
                interpolation=cv2.INTER_AREA,
            ),
            A.OneOf(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(always_apply=False, p=0.5),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                always_apply=False,
                p=0.4,
                brightness_limit=(-0.05, 0.05),
                contrast_limit=(-0.05, 0.05),
                brightness_by_max=True,
            ),
            A.CoarseDropout(
                always_apply=False,
                p=0.4,
                max_holes=20,
                max_height=8,
                max_width=8,
                min_holes=8,
                min_height=3,
                min_width=3,
            ),
            A.OneOf(
                [
                    A.ChannelShuffle(always_apply=False, p=0.5),
                    A.HueSaturationValue(always_apply=False, p=0.5),
                ],
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )


def get_query_transform(config):
    return A.Compose(
        [
            A.RandomCropFromBorders(
                always_apply=False,
                p=1.0,
                crop_left=0.05,
                crop_right=0.05,
                crop_top=0.05,
                crop_bottom=0.05,
            ),
            A.Resize(
                always_apply=False,
                p=1.0,
                height=config.dataset.input_size,
                width=config.dataset.input_size,
                interpolation=cv2.INTER_AREA,
            ),
            A.OneOf(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(always_apply=False, p=0.5),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                always_apply=False,
                p=0.4,
                brightness_limit=(-0.05, 0.05),
                contrast_limit=(-0.05, 0.05),
                brightness_by_max=True,
            ),
            A.CoarseDropout(
                always_apply=False,
                p=0.4,
                max_holes=20,
                max_height=8,
                max_width=8,
                min_holes=8,
                min_height=4,
                min_width=4,
            ),
            A.OneOf(
                [
                    A.ChannelShuffle(always_apply=False),
                    A.HueSaturationValue(always_apply=False),
                ],
                p=0.4,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )


def get_train_aug(config):
    if config.dataset.augmentations == "default":

        train_augs = A.Compose(
            [
                A.RandomCropFromBorders(
                    always_apply=False,
                    p=1.0,
                    crop_left=0.11,
                    crop_right=0.11,
                    crop_top=0.11,
                    crop_bottom=0.11,
                ),
                A.Resize(
                    always_apply=False,
                    p=1.0,
                    height=config.dataset.input_size,
                    width=config.dataset.input_size,
                    interpolation=cv2.INTER_AREA,
                ),
                A.ImageCompression(
                    always_apply=False, p=0.4, quality_lower=90, quality_upper=100
                ),
                A.OneOf(
                    [
                        A.Flip(p=0.5),
                        A.RandomRotate90(always_apply=False, p=0.5),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    always_apply=False,
                    p=0.4,
                    brightness_limit=(-0.05, 0.05),
                    contrast_limit=(-0.05, 0.05),
                    brightness_by_max=True,
                ),
                A.CoarseDropout(
                    always_apply=False,
                    p=0.4,
                    max_holes=20,
                    max_height=8,
                    max_width=8,
                    min_holes=8,
                    min_height=4,
                    min_width=4,
                ),
                A.OneOf(
                    [
                        A.ChannelShuffle(always_apply=False),
                        A.HueSaturationValue(always_apply=False),
                    ],
                    p=0.4,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ToTensorV2(),

            ]
        )
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == "default":
        val_augs = A.Compose(
            [
                A.Resize(
                    always_apply=False,
                    p=1.0,
                    height=config.dataset.input_size,
                    width=config.dataset.input_size,

                    interpolation=cv2.INTER_AREA,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
                ToTensorV2(),
            ]
        )

    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs


def get_val_query_aug(config):

    val_augs = A.Compose(
        [
            A.Resize(
                always_apply=False,
                p=1.0,
                height=config.dataset.input_size,
                width=config.dataset.input_size,
                interpolation=cv2.INTER_AREA,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.CenterCrop(height=config.dataset.input_size,width=config.dataset.input_size),
            # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )
    return val_augs


def get_val_gallery_aug(config):
    val_augs = A.Compose(
        [
            A.Resize(
                always_apply=False,
                p=1.0,
                height=config.dataset.input_size,
                width=config.dataset.input_size,
                interpolation=cv2.INTER_AREA,
            ),
            # A.CenterCrop(height=config.dataset.input_size,
            #              width=config.dataset.input_size,
            #              ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # malize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )

    return val_augs


def get_eval_query_aug():

    val_augs = A.Compose(
        [
            A.Resize(
                always_apply=False,
                p=1.0,
                height=256,
                width=256,
                interpolation=cv2.INTER_AREA,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.CenterCrop(height=config.dataset.input_size,width=config.dataset.input_size),
            # A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )
    return val_augs


def get_eval_gallery_aug():
    val_augs = A.Compose(
        [
            A.Resize(
                always_apply=False,
                p=1.0,
                height=256 + 12,
                width=256 + 12,
                interpolation=cv2.INTER_AREA,
            ),
            A.CenterCrop(
                height=256,
                width=256,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # malize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2(),
        ]
    )

    return val_augs
