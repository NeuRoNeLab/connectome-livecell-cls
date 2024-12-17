import functools
from typing import Dict, Any, Optional, Final, Callable
from enum import Enum
import torch
import torchmetrics
from utils.utils import SerializableConfig


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    RMS_PROP = "rms_prop"
    RMSPROP = "rmsprop"
    RPROP = "rprop"
    ADAGRAD = "adagrad"
    NADAM = "nadam"
    RADAM = "radam"
    ASGD = "asgd"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"
    SGD = "sgd"


class LRSchedulerType(str, Enum):
    NONE = "none"
    COS_ANNEALING = "cos_anneal"
    COS_ANNEALING_RESTARTS = "cos_anneal_restarts"
    REDUCE_LR_PLATEAU = "reduce_lr_plateau"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CYCLIC = "cyclic"


_LR_SCHEDULER_DEFAULTS: Final[Dict[str, Any]] = {
    LRSchedulerType.NONE: {},
    LRSchedulerType.COS_ANNEALING: {
        "T_max": 100,
        "eta_min": 0,
        "last_epoch": -1
    },
    LRSchedulerType.COS_ANNEALING_RESTARTS: {
        "T_0": 20,
        "T_mult": 1,
        "eta_min": 0,
        "last_epoch": -1
    },
    LRSchedulerType.REDUCE_LR_PLATEAU: {
        "mode": "min",
        "factor": 0.1,
        "patience": 10,
        "verbose": False,
        "threshold": 1e-4,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0,
        "eps": 1e-08
    },
    LRSchedulerType.LINEAR: {
        "start_factor": 0.3333333333333333,
        "end_factor": 1.0,
        "total_iters": 5,
        "last_epoch": -1
    },
    LRSchedulerType.EXPONENTIAL: {
        "gamma": 0.01,
        "last_epoch": -1
    },
    LRSchedulerType.CYCLIC: {
        "step_size_up": 2000,
        "step_size_down": None,
        "mode": "triangular",
        "gamma": 1.0,
        "scale_fn": None,
        "scale_mode": "cycle",
        "cycle_momentum": True,
        "base_momentum": 0.8,
        "max_momentum": 0.9,
        "last_epoch": -1,
        "verbose": False
    }
}


class LRSchedulerConfig(SerializableConfig):

    def __init__(self,
                 name: LRSchedulerType,
                 params: Optional[Dict[str, Any]] = None,
                 interval: str = 'step',
                 frequency: int = 1,
                 monitor: Optional[str] = 'val_loss'):
        """
        Configuration class for Learning Rate (LR) schedulers. It specifies parameters such as the scheduler name,
        update interval, update frequency, and the monitor for LR adjustments.

        Args:
            name (LRSchedulerType): The type of LR scheduler to use.
            params (Optional[Dict[str, Any]]): Additional parameters for the LR scheduler. Default is None.
            interval (str): Update interval for LR adjustments, either 'step' or 'epoch'. Default is 'step'.
            frequency (int): Update frequency for LR adjustments, indicating the number of epochs or steps between
                updates. Default is 1.
            monitor (Optional[str]): The monitor for LR adjustments, used in strategies like reducing LR on a plateau.
                Default is 'val_loss'.

        Attributes:
            name (LRSchedulerType): The type of LR scheduler to use.
            params (Dict[str, Any]): Additional parameters for the LR scheduler.
            interval (str): Update interval for LR adjustments, either 'step' or 'epoch'.
            frequency (int): Update frequency for LR adjustments, indicating the number of epochs or steps between
                updates.
            monitor (Optional[str]): The monitor for LR adjustments, used in strategies like reducing LR on a plateau.
        """
        super().__init__()

        self.name: LRSchedulerType = name
        self.params: Dict[str, Any] = params if params is not None else _LR_SCHEDULER_DEFAULTS[name]
        self.interval: str = interval
        self.frequency: int = frequency
        self.monitor: str = monitor if name == LRSchedulerType.REDUCE_LR_PLATEAU else None


def resolve_optimizer(optimizer_name: OptimizerType, optimizer_kwargs: Dict[str, Any]) -> torch.optim.Optimizer:
    if optimizer_name == OptimizerType.ADAM or optimizer_name == OptimizerType.ADAMW:
        if "weight_decay" not in optimizer_kwargs and optimizer_name == OptimizerType.ADAM:
            optimizer_kwargs["weight_decay"] = 0.0  # weight decay is 0.0 by default in basic Adam
        return torch.optim.AdamW(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.RMS_PROP or optimizer_name == OptimizerType.RMSPROP:
        return torch.optim.RMSprop(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.RPROP:
        return torch.optim.Rprop(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.ADAGRAD:
        return torch.optim.Adagrad(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.NADAM:
        return torch.optim.NAdam(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.RADAM:
        return torch.optim.RAdam(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.ASGD:
        return torch.optim.ASGD(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.ADADELTA:
        return torch.optim.Adadelta(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.ADAMAX:
        return torch.optim.Adamax(**optimizer_kwargs)
    elif optimizer_name == OptimizerType.SGD:
        return torch.optim.SGD(**optimizer_kwargs)
    else:
        raise NotImplementedError(f"Unsupported optimizer type {optimizer_name}.")


def resolve_lr_scheduler(optimizer: torch.optim.Optimizer,
                         lr_scheduler_name: LRSchedulerType,
                         lr_scheduler_config: LRSchedulerConfig) -> Optional[Any]:
    if lr_scheduler_config.params is None:
        lr_scheduler_config.params = _LR_SCHEDULER_DEFAULTS[lr_scheduler_name]

    if lr_scheduler_name == LRSchedulerType.NONE:
        return None
    if lr_scheduler_name == LRSchedulerType.COS_ANNEALING:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **lr_scheduler_config.params)
    elif lr_scheduler_name == LRSchedulerType.COS_ANNEALING_RESTARTS:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **lr_scheduler_config.params)
    elif lr_scheduler_name == LRSchedulerType.REDUCE_LR_PLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_config.params)
    elif lr_scheduler_name == LRSchedulerType.LINEAR:
        return torch.optim.lr_scheduler.LinearLR(optimizer, **lr_scheduler_config.params)
    elif lr_scheduler_name == LRSchedulerType.EXPONENTIAL:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **lr_scheduler_config.params)
    elif lr_scheduler_name == LRSchedulerType.CYCLIC:
        max_lr = None
        base_lr = None

        if "max_lr" in lr_scheduler_config.params:
            max_lr = lr_scheduler_config.params["max_lr"]
            base_lr = max_lr / 10

        if "base_lr" in lr_scheduler_config.params:
            base_lr = lr_scheduler_config.params["base_lr"]
            max_lr = base_lr * 10 if max_lr is None else max_lr

        if "lr" in optimizer.defaults:
            max_lr = optimizer.defaults["lr"]
            base_lr = max_lr / 10

        if max_lr is None or base_lr is None:
            max_lr = 0.01
            base_lr = 0.001

        return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                 base_lr=base_lr,
                                                 max_lr=max_lr,
                                                 **lr_scheduler_config.params)
    else:
        raise NotImplementedError(f"Unsupported LR scheduler type {lr_scheduler_name}.")


def binary_dice_score(preds: torch.Tensor,
                      targets: torch.Tensor,
                      smooth: float = 1.0,
                      p: float = 1.0,
                      from_logits: bool = True,
                      reduction: str = "mean") -> torch.Tensor:
    assert preds.shape[0] == targets.shape[0], "predict & target batch size don't match"

    if from_logits:
        preds = preds.sigmoid()

    preds = preds.contiguous().view(preds.shape[0], -1)
    target = targets.contiguous().view(targets.shape[0], -1)

    num = torch.sum(torch.mul(preds, target), dim=1) + smooth
    den = torch.sum(preds.pow(p) + target.pow(p), dim=1) + smooth

    score = (2 * num) / den

    if reduction == 'mean':
        return score.mean()
    elif reduction == 'sum':
        return score.sum()
    elif reduction == 'none':
        return score
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


def dice_score(preds: torch.Tensor,
               targets: torch.Tensor,
               n_classes: int = 2,
               smooth: float = 1.0,
               p: float = 1.0,
               reduction: str = "mean",
               ignore_index: int = -1,
               weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if n_classes > 2:
        assert preds.shape == targets.shape, 'predict & target shape do not match'
        dice = functools.partial(binary_dice_score, smooth=smooth, p=p, reduction=reduction, from_logits=False)
        total_score = 0

        class_dim = 1 if len(preds.shape) < 4 else -3
        predict = preds.softmax(dim=-class_dim)

        for i in range(targets.shape[class_dim]):
            if i != ignore_index:
                dice_loss = dice(predict[:, i], targets[:, i])
                if weight is not None:
                    assert weight.shape[0] == targets.shape[class_dim], \
                        'Expect weight shape [{}], get[{}]'.format(targets.shape[class_dim], weight.shape[0])
                    dice_loss *= weight[i]
                total_score += dice_loss

        return total_score / targets.shape[class_dim]
    return binary_dice_score(
        preds=preds,
        targets=targets,
        smooth=smooth,
        p=p,
        from_logits=True,
        reduction=reduction
    )


_METRICS: Final[Dict[str, torchmetrics.Metric]] = {
    "accuracy": torchmetrics.functional.accuracy,
    "top_k_accuracy": torchmetrics.functional.accuracy,
    "auroc": torchmetrics.functional.auroc,
    "precision": torchmetrics.functional.precision,
    "recall": torchmetrics.functional.recall,
    "f1": torchmetrics.functional.f1_score,
    "fbeta": torchmetrics.functional.fbeta_score,
    "dice": dice_score,
    "jaccard": torchmetrics.functional.jaccard_index,
    "iou": torchmetrics.functional.jaccard_index,
    "mcc": torchmetrics.functional.matthews_corrcoef,
    "specificity": torchmetrics.functional.specificity,
    "ssim": torchmetrics.functional.structural_similarity_index_measure,
    "msssim": torchmetrics.functional.multiscale_structural_similarity_index_measure,
    "mse": torchmetrics.functional.mean_squared_error,
    "mae": torchmetrics.functional.mean_absolute_error
}


def resolve_metric(metric: str, **kwargs) -> Callable[[torch.Tensor, ...], torch.Tensor]:  # torchmetrics.Metric:
    # return _METRICS[metric](**kwargs)
    return functools.partial(_METRICS[metric], **kwargs)
