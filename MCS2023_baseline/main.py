import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import utils

from omegaconf import OmegaConf
from accelerate import Accelerator
from data_utils import get_dataloader


from models.model import RetrivealNet, Trunk
from pytorch_metric_learning import (
    losses,
    regularizers,
    miners,
    distances,
)

from custom_miner import BatchEasyHardMinerCustom
from tqdm import tqdm
from train import train, validation_public
from loguru import logger
from accelerate import DistributedDataParallelKwargs



def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """

    config = OmegaConf.load(args.cfg)
    checkpoint_path = args.chkp

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    feature_extractor = None

    print("Preparing train and val dataloaders...")
    train_loader, val_loader, classes_count = get_dataloader.get_dataloaders(
        config, feature_extractor
    )
    config.dataset.num_of_classes = classes_count
    gallery_loader, query_loader = get_dataloader.get_public_dataloaders(config)

    trunk = Trunk(
        backbone=config.model.backbone,
        embedding_dim=config.model.embedding_dim,
        dropout=config.train.dropout,
        pretrained=config.model.pretrained,
    )

    model = RetrivealNet(trunk=trunk)
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        model.load_state_dict(checkpoint)

    print("Load model...")

    model.to(device, memory_format=torch.channels_last)

    print("Prepare training params...")

    distance = distances.CosineSimilarity()

    if config.train.classes.loss.name == "arcface":
        R = regularizers.RegularFaceRegularizer()
        class_loss = losses.SubCenterArcFaceLoss(
            num_classes=config.dataset.num_of_classes,
            embedding_size=config.model.embedding_dim,
            margin=config.train.classes.loss.arcface.m,
            scale=config.train.classes.loss.arcface.s,
            sub_centers=config.train.classes.loss.arcface.sub_centers,
            weight_regularizer=R,
        ).to(device)

    elif config.train.classes.loss.name == "triplet_margin":
        class_loss = losses.TripletMarginLoss(
            margin=config.train.classes.loss.triplet_margin.margin,
            swap=config.train.classes.loss.triplet_margin.swap,
            smooth_loss=config.train.classes.loss.triplet_margin.smooth_loss,
            triplets_per_anchor="all",
            distance=distance,
        )

    elif config.train.classes.loss.name == "soft_triplet":

        class_loss = losses.SoftTripleLoss(
            config.dataset.num_of_classes,
            config.model.embedding_dim,
            centers_per_class=config.train.classes.loss.soft_triplet.centers_per_class,
            la=config.train.classes.loss.soft_triplet.la,
            gamma=config.train.classes.loss.soft_triplet.gamma,
            margin=config.train.classes.loss.soft_triplet.margin,
            distance=distance,
            # weight_regularizer=R
        ).to(device)

    elif config.train.classes.loss.name == "proxy_anchor":
        class_loss = losses.ProxyAnchorLoss(
            config.dataset.num_of_classes,
            config.model.embedding_dim,
            margin=config.train.classes.loss.proxy_anchor.margin,
            alpha=config.train.classes.loss.proxy_anchor.alpha,
            distance=distance,
        ).to(accelerator.device)
    else:
        raise ValueError("неизвестный лосс!!!22!1!!")

    if config.train.classes.second_loss.name == "tuplet":
        second_class_loss = losses.TupletMarginLoss(
            margin=config.train.classes.second_loss.tuplet.margin,
            scale=config.train.classes.second_loss.tuplet.scale,
            distance=distance,
        )
    elif config.train.classes.second_loss.name == "contrastive":

        second_class_loss = losses.ContrastiveLoss(
            pos_margin=config.train.classes.second_loss.contrastive.pos_margin,
            neg_margin=config.train.classes.second_loss.contrastive.neg_margin,
            distance=distance,
        )

    if config.train.classes.loss.name in ["soft_triplet", "arcface", "proxy_anchor"]:

        if config.train.loss_optimizer == "AdamW":
            loss_optimizer = torch.optim.AdamW(
                class_loss.parameters(),
                betas=(config.train.adamw_beta1, config.train.adamw_beta2),
                lr=config.train.classes.loss.lr,
                weight_decay=config.train.classes.loss.weight_decay,
            )
        else:
            loss_optimizer = torch.optim.SGD(
                class_loss.parameters(),
                momentum=config.train.momentum,
                lr=config.train.classes.loss.lr,
                weight_decay=config.train.classes.loss.weight_decay,
            )

    if config.train.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            trunk.parameters(),
            betas=(config.train.adamw_beta1, config.train.adamw_beta2),
            lr=config.train.trunk.lr,
            weight_decay=config.train.trunk.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            trunk.parameters(),
            momentum=config.train.momentum,
            lr=config.train.trunk.lr,
            weight_decay=config.train.trunk.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.2, last_epoch=-1, verbose=False
    )

    loss_scheduler = torch.optim.lr_scheduler.StepLR(
        loss_optimizer, step_size=1, gamma=0.5, last_epoch=-1, verbose=False
    )

    (
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        loss_optimizer,
        loss_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        loss_optimizer,
        loss_scheduler,
    )

    print("Done.")

    train_epoch = tqdm(
        range(config.train.n_epoch), dynamic_ncols=True, desc="Epochs", position=0
    )

    if config.train.classes.miner.name == "multi_simularity":
        class_miner = miners.MultiSimilarityMiner(
            epsilon=config.train.classes.miner.multi_simularity.epsilon,
            distance=distance,
        )

    elif config.train.classes.miner.name == "pair_margin":
        class_miner = miners.PairMarginMiner(
            pos_margin=config.train.classes.miner.pair_margin.pos_margin,
            neg_margin=config.train.classes.miner.pair_margin.neg_margin,
            distance=distance,
        )

    elif config.train.classes.miner.name == "triplet_margin":
        class_miner = miners.TripletMarginMiner(
            margin=config.train.classes.miner.triplet_margin.margin,
            type_of_triplets=config.train.classes.miner.triplet_margin.type_of_triplets,
            distance=distance,
        )

    elif config.train.classes.miner.name == "batch_easy_hard":

        class_miner = miners.BatchEasyHardMiner(
            pos_strategy=config.train.classes.miner.batch_easy_hard.pos_strategy,
            neg_strategy=config.train.classes.miner.batch_easy_hard.neg_strategy,
            distance=distance,
        )
    elif config.train.classes.miner.name == "batch_easy_hard_custom":

        class_miner = BatchEasyHardMinerCustom(
            pos_strategy=config.train.classes.miner.batch_easy_hard.pos_strategy,
            neg_strategy=config.train.classes.miner.batch_easy_hard.neg_strategy,
            mode="q",
            distance=distance,
        )
    else:
        raise ValueError(f"неизвестный майнер {config.train.classes.miner}")

    # main process
    best_acc = 0.61

    print(class_loss, class_miner)
    for epoch in train_epoch:
        train_loss = train(
            model,
            accelerator,
            train_loader,
            class_loss,
            second_class_loss, 
            class_miner,
            optimizer,
            config,
            epoch,
            gallery_loader,
            query_loader,
            scheduler,
            loss_optimizer,
            loss_scheduler,
        )

        if accelerator.is_main_process:
            epoch_avg_acc = validation_public(
                model, config, gallery_loader, query_loader, epoch
            )

            logger.info(
                f"""Epoch {epoch}
                public mAP: {epoch_avg_acc}
                Train loss: {train_loss}
                """
            )

            saved_model = accelerator.unwrap_model(model)
            if epoch_avg_acc >= best_acc:
                best_acc = epoch_avg_acc
                epoch_avg_acc = f"{epoch_avg_acc:.4f}"

                utils.save_checkpoint(
                    saved_model,
                    class_loss,
                    optimizer,
                    scheduler,
                    epoch,
                    outdir,
                    epoch_avg_acc,
                )

        scheduler.step()
        loss_scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to config file.")
    parser.add_argument("--chkp", type=str, default=None, help="Path to checkpoint file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
