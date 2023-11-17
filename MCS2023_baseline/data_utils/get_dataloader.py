import torch

from . import dataset, augmentations
from pytorch_metric_learning import samplers
from .sampler import MPerClassBalanceSampler


def get_dataloaders(config, feature_extractor):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")


    if config.dataset.name == "wb":
        train_dataset = dataset.WBFlexDataset(root=config.dataset.train_prefix,
            annotation_file=config.dataset.train_list,config=config)
        sampler = MPerClassBalanceSampler(
            labels=train_dataset.labels[:, 1],
            is_galleries=train_dataset.is_galleries,
            m=config.train.mperclass,
            batch_size=config.dataset.batch_size,
            length_before_new_iter=len(train_dataset),
        )
    else:
        train_dataset = dataset.Product10KDataset(
            root=config.dataset.train_prefix,
            annotation_file=config.dataset.train_list,
            transforms=augmentations.get_train_aug(config),
            feature_extractor=feature_extractor,
        )
        sampler = samplers.MPerClassSampler(
            labels=train_dataset.labels[:, 1],
            m=config.train.mperclass,
            batch_size=config.dataset.batch_size,
            length_before_new_iter=len(train_dataset),
        )


    
    classes_count = train_dataset.classes_count

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        sampler=sampler,
        # shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        # persistent_workers=True,
        prefetch_factor=2,
    )
    print("Done.")

    return train_loader, train_loader, classes_count  # fix it


def get_public_dataloaders(config):

    gallery_dataset = dataset.SubmissionDataset(
        root=config.dataset.public_dir,
        annotation_file=config.dataset.public_gallery_annotation,
        transforms=augmentations.get_val_aug(config),
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
    )

    query_dataset = dataset.SubmissionDataset(
        root=config.dataset.public_dir,
        annotation_file=config.dataset.public_query_annotation,
        transforms=augmentations.get_val_aug(config),
        with_bbox=True,
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
    )
    # print(len(gallery_dataset), len(query_dataset))
    # exit(1)
    return gallery_loader, query_loader
