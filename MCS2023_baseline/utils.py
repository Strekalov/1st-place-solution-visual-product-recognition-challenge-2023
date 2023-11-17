import os

import numpy as np
import torch

from collections import namedtuple, OrderedDict

from torch import nn
from torch.nn import functional as func
import torch
import torch.nn as nn
import torch.nn.functional as F
import imutils


class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            size_average: bool = True,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def save_checkpoint(model, class_loss, optimizer, scheduler, epoch, outdir, epoch_avg_acc):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}_{}.pth".format(epoch, epoch_avg_acc)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict(
        [
            ("state_dict", weights),
            ("state_dict_loss", class_loss.state_dict()),
            ("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("epoch", epoch),
        ]
    )

    torch.save(state, filename)
    os.chmod(filename, 0o777)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = func.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, net):
    lr = config.train.learning_rate

    print("Opt: ", config.train.optimizer)

    if config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )
    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.n_epoch
        )
    elif config.train.lr_schedule.name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_schedule.step_size,
            gamma=config.train.lr_schedule.gamma,
        )
    else:
        raise Exception(
            "Unknown type of lr schedule: {}".format(config.train.lr_schedule)
        )
    return scheduler


def get_training_parameters(config, net):
    criterion = torch.nn.CrossEntropyLoss().to("cuda")
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return criterion, optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]


def freeze_batch_norm(model):
    for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
        module.eval()
        module.train = lambda _: None
    return model
