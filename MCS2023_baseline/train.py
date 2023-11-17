import numpy as np
import torch
import os
from collections import OrderedDict

from tqdm import tqdm
from accelerate import Accelerator
from utils import AverageMeter
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import pandas as pd
from mean_average_precision import calculate_map


def train(
    model: torch.nn.Module,
    accelerator: Accelerator,
    train_loader: torch.utils.data.DataLoader,
    class_loss_metric_fn: torch.nn.Module,
    class_second_loss_metric_fn: torch.nn.Module,
    class_miner,
    optimizer: torch.optim.Optimizer,
    config,
    epoch,
    gallery_loader,
    query_loader,
    scheduler,
    loss_optimizer,
    loss_scheduler,
) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    model.train()

    class_metric_loss_stat = AverageMeter("class_metric_loss")
    class_metric_second_loss_stat = AverageMeter("class_metric_second_loss")
    class_ce_loss_stat = AverageMeter("Class CrossEntropy")

    mean_loss_stat = AverageMeter("Mean Loss")

    train_iter = train_loader
    if accelerator.is_main_process:
        train_iter = tqdm(train_loader, desc="Train", dynamic_ncols=True, position=1)

    best_val = 0
    best_metric = 0

    for step, (x, y1, y2, is_gallery) in enumerate(train_iter, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss_optimizer.zero_grad(set_to_none=True)

        embeddings = model(x.to(memory_format=torch.channels_last))
        if config.dataset.name == "wb":
            class_miner_out = class_miner(embeddings, y2, embeddings, is_gallery)
        else:
            class_miner_out = class_miner(embeddings, y2)

        class_metric_loss = class_loss_metric_fn(embeddings, y2, class_miner_out)
        class_second_loss = class_second_loss_metric_fn(embeddings, y2, class_miner_out)

        num_of_samples = x.shape[0]
        loss = (
            +config.train.losses_k.k_class_metric_l * class_metric_loss
            + config.train.losses_k.k_class_second_l * class_second_loss
        )
        accelerator.backward(loss)

        class_metric_loss_stat.update(
            class_metric_loss.detach().cpu().item(), num_of_samples
        )
        class_metric_second_loss_stat.update(
            class_second_loss.detach().cpu().item(), num_of_samples
        )

        mean_loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        if accelerator.sync_gradients:
            accelerator.clip_grad_value_(model.parameters(), config.train.grad_clipping)
        optimizer.step()
        loss_optimizer.step()

        if step % config.train.freq_vis == 0 and not step == 0:

            _, class_metric_loss_avg = class_metric_loss_stat()
            _, class_metric_second_loss_avg = class_metric_second_loss_stat()
            _, avg_mean_loss = mean_loss_stat()

            if accelerator.is_main_process:
                print(
                    f"""Epoch {epoch}, step: {step}:
                        class_metric_loss: {class_metric_loss_avg},
                        class_second_loss: {class_metric_second_loss_avg},
                        mean_loss: {avg_mean_loss}
                    """
                )
                if config.train.save_weights:
                    with torch.no_grad():
                        val_map = validation_public(
                            model, config, gallery_loader, query_loader, epoch
                        )

                        if val_map >= 0.61:
                            saved_model = accelerator.unwrap_model(model)
                            weights = saved_model.state_dict()
                            loss_weights = class_loss_metric_fn.state_dict()

                            state = OrderedDict(
                                [
                                    ("state_dict", weights),
                                    ("state_dict_loss", loss_weights),
                                ]
                            )
                            torch.save(state, f"part_epoch_{epoch}_{step}_{val_map}.pt")
                            os.chmod(f"try_part_epoch_{epoch}_{step}_{val_map}.pt", 0o777)

                model.train()

    _, class_metric_loss_avg = class_metric_loss_stat()
    _, class_metric_second_loss_avg = class_metric_second_loss_stat()
    _, avg_mean_loss = mean_loss_stat()
    if accelerator.is_main_process:
        print(
            f"""Train process of epoch {epoch} is done:
                        class_metric_loss: {class_metric_loss_avg},
                        class_second_loss: {class_metric_second_loss_avg},
                        mean_loss: {avg_mean_loss}
                    """
        )
    return avg_mean_loss


def validation_public(
    model: torch.nn.Module,
    config,
    gallery_loader: torch.utils.data.DataLoader,
    query_loader: torch.utils.data.DataLoader,
    epoch,
) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
    """

        
    print("Calculating embeddings")

    gallery_embeddings = torch.zeros((1067, config.model.embedding_dim), device="cuda", requires_grad=False, dtype=torch.half)
    query_embeddings = torch.zeros((1935, config.model.embedding_dim), device="cuda", requires_grad=False, dtype=torch.half)


    with torch.no_grad():
        for i, images in tqdm(enumerate(gallery_loader), total=len(gallery_loader)):
            images = images.contiguous(memory_format=torch.channels_last)
            outputs = model(images.half().cuda())

            gallery_embeddings[
                i
                * config.dataset.batch_size : (
                    i * config.dataset.batch_size + config.dataset.batch_size
                ),
                :,
            ] = outputs


        for i, images in tqdm(enumerate(query_loader), total=len(query_loader)):
            images = images.contiguous(memory_format=torch.channels_last)
            outputs = model(images.half().cuda())


            query_embeddings[
                i
                * config.dataset.batch_size : (
                    i * config.dataset.batch_size + config.dataset.batch_size
                ),
                :,
            ] = outputs
            
        
        query_embeddings, gallery_embeddings = query_embeddings.float(), gallery_embeddings.float()
    
        distances = torch.cdist(query_embeddings, gallery_embeddings)

        sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances = sorted_distances.cpu().numpy()[:, :1000]


        seller_gt = pd.read_csv(config.dataset.public_gallery_annotation)
        gallery_labels = seller_gt["product_id"].values
        user_gt = pd.read_csv(config.dataset.public_query_annotation)
        query_labels = user_gt["product_id"].values

        # Evalaute metrics

        public_map = calculate_map(sorted_distances, query_labels, gallery_labels)
        print(f"Validation on epoch {epoch}: mAP: {public_map}")
        return public_map
