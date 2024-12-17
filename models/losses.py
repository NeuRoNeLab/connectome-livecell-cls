from typing import Final, Union, Tuple, List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as nnf
from torchvision.ops import sigmoid_focal_loss
from pytorch_msssim import SSIM, MS_SSIM

EPSILON: Final[float] = 1e-7  # this should be precision-dependant


def unnormalized_cross_entropy(prediction: torch.Tensor,
                               target: torch.Tensor,
                               reduction: str = "mean",
                               from_logits: bool = False,
                               keras_like_behavior: bool = True,
                               label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Calculate the cross-entropy loss between predicted values and target values, allowing un-normalized prediction
    values to be interpreted as probabilities or logits. This loss function is useful for training classification models
    where the goal is to predict class labels for input samples.

    Parameters:
        prediction (torch.Tensor): The predicted values of the model, either logits or probabilities.
        target (torch.Tensor): The true target values.
        reduction (str, optional): Specifies the reduction to apply to the computed loss. Can be "mean", "sum", or
            "none".
        from_logits (bool, optional): Indicates whether the prediction values are logits (raw scores) or should be
            interpreted as probabilities (even if they are un-normalized and do not sum to 1).
        keras_like_behavior (bool, optional): Indicates whether the function must compute the cross-entropy mimicking
            the behavior of the Keras cross-entropy. It clips the values between eps (a very small quantity) and
            1 - eps, therefore applies log function to obtain final logits that are fed into standard cross-entropy.
        label_smoothing (float, optional): A value between 0 and 1 representing the amount of label smoothing to apply.

    Returns:
        torch.Tensor: The computed UnNormalizedCrossEntropy loss.
    """
    # Sparse labels
    if len(target.shape) < len(prediction.shape):
        one_hot_target = nnf.one_hot(target, num_classes=prediction.shape[-1])
    else:
        one_hot_target = target

    if label_smoothing > 0:
        one_hot_target = one_hot_target.float()
        n_classes = one_hot_target.shape[1]
        perturbation = - one_hot_target * label_smoothing + (1 - one_hot_target) * (label_smoothing / (n_classes - 1))
        one_hot_target += perturbation

    if from_logits:
        loss = torch.sum(one_hot_target * torch.log_softmax(prediction, dim=-1), dim=-1)
    elif keras_like_behavior:
        prediction = torch.clip(prediction, EPSILON, 1 - EPSILON)
        prediction = torch.log(prediction)
        # loss = torch.sum(one_hot_target * torch.log_softmax(prediction, dim=-1), dim=-1)
        return torch.nn.functional.cross_entropy(
            input=prediction,
            target=target,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    else:
        loss = torch.sum(one_hot_target * torch.log(prediction + EPSILON), dim=-1)
    # if weights is None:
    #    loss = torch.sum(one_hot_target * torch.log(predictions), dim=-1)
    # else:
    #    loss = torch.sum(weights * one_hot_target * torch.log(predictions), dim=-1)

    if reduction == "mean":
        loss = -torch.mean(loss)
    elif reduction == "sum":
        loss = -torch.sum(loss)
    else:
        loss = -loss

    return loss


class UnNormalizedCrossEntropy(torch.nn.Module):

    def __init__(self,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 from_logits: bool = False,
                 keras_like_behavior: bool = True, ):
        """
        A PyTorch module implementing the un-normalized cross-entropy loss function, allowing un-normalized prediction
        values to be interpreted as probabilities or logits. This loss function is useful for training classification
        models where the goal is to predict class labels for input samples.

        Parameters:
            reduction (str, optional): Specifies the reduction to apply to the computed loss. Can be "mean", "sum", or
                "none".
            label_smoothing (float, optional): A value between 0 and 1 representing the amount of label smoothing to
                apply.
            from_logits (bool, optional): Indicates whether the prediction values are logits (raw scores) or should be
                interpreted as probabilities (even if they are un-normalized and do not sum to 1).
            keras_like_behavior (bool, optional): Indicates whether the function must compute the cross-entropy
                mimicking the behavior of the Keras cross-entropy.
        """

        super(UnNormalizedCrossEntropy, self).__init__()

        if not 0 <= label_smoothing < 1:
            raise ValueError(f'`label_smoothing` must be between 0 (inclusive) and 1 (exclusive). '
                             f'{label_smoothing} given.')
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f'`reduction` must be either `mean` `sum` or `none`. {reduction} given.')

        self._label_smoothing: float = label_smoothing
        self._reduction: str = reduction
        self._from_logits: bool = from_logits
        self._keras_like_behavior: bool = keras_like_behavior

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing

    @label_smoothing.setter
    def label_smoothing(self, label_smoothing: float):
        if not 0 <= label_smoothing < 1:
            raise ValueError(f'`label_smoothing must be` between 0 (inclusive) and 1 (exclusive). '
                             f'{label_smoothing} given.')
        self._label_smoothing = label_smoothing

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @from_logits.setter
    def from_logits(self, from_logits: bool):
        self._from_logits = from_logits

    @property
    def keras_like_behavior(self) -> bool:
        return self._keras_like_behavior

    @keras_like_behavior.setter
    def keras_like_behavior(self, keras_like_behavior: bool):
        self._keras_like_behavior = keras_like_behavior

    @reduction.setter
    def reduction(self, reduction: str):
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f'`reduction` must be either `mean` `sum` or `none`. {reduction} given.')
        self._reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the UnNormalizedCrossEntropy loss using provided prediction and target tensors.

        Parameters:
            prediction (torch.Tensor): The predicted values of the model.
            target (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The computed UnNormalizedCrossEntropy loss.
        """
        return unnormalized_cross_entropy(prediction=prediction,
                                          target=target,
                                          reduction=self.reduction,
                                          from_logits=self.from_logits,
                                          keras_like_behavior=self.keras_like_behavior,
                                          label_smoothing=self.label_smoothing)


class SSIMLoss(torch.nn.Module):
    r""" Structural Similarity Index Measure loss class.
    Args:
        data_range (float or int, optional): Value range of input images, usually 1.0 or 255. Default: 1.0.
        size_average (bool, optional): If True, SSIM of all images will be averaged as a scalar. Default: True.
        win_size: (int, optional): The size of Gaussian kernel. Default: 11.
        win_sigma: (float, optional): Sigma of normal distribution. Default: 1.5.
        channels (int, optional): Number of input channels. Default: 3.
        k (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a
            negative or NaN results. Default: (0.01, 0.03).
        nonnegative (bool, optional): Forces the SSIM response to be non-negative with relu. Default: False.
    """

    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = True,
                 win_size: int = 11,
                 win_sigma: float = 1.5,
                 channels: int = 3,
                 spatial_dims: int = 2,
                 k: Union[tuple[float, float], list[float]] = (0.01, 0.03),
                 nonnegative: bool = False):
        super().__init__()

        self._win_sigma: float = win_sigma
        self._channels: int = channels
        self._spacial_dims: int = spatial_dims
        self._k: Union[tuple[float, float], list[float]] = k
        self._ssmim = SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=channels,
            spatial_dims=spatial_dims,
            K=k,
            nonnegative_ssim=nonnegative
        )

    @property
    def ssim(self) -> SSIM:
        return self._ssmim

    @property
    def data_range(self) -> float:
        return self.ssmim.data_range

    @property
    def size_average(self) -> bool:
        return self.ssmim.size_average

    @property
    def win_size(self) -> int:
        return self.ssmim.win_size

    @property
    def win_sigma(self) -> float:
        return self._win_sigma

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def spatial_dims(self) -> int:
        return self._spacial_dims

    @property
    def k(self) -> Union[tuple[float, float], list[float]]:
        return self._k

    @property
    def nonnegative(self) -> bool:
        return self.ssmim.nonnegative_ssim

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 1 - self.ssmim(prediction, target)
        return loss


class MSSSIMLoss(SSIMLoss):

    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = True,
                 win_size: int = 11,
                 win_sigma: float = 1.5,
                 channels: int = 3,
                 spatial_dims: int = 2,
                 weights: Optional[List[float]] = None,
                 k: Union[Tuple[float, float], List[float]] = (0.01, 0.03)):
        super().__init__(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channels=channels,
            spatial_dims=spatial_dims,
            k=k,
            nonnegative=True
        )
        self._mssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=channels,
            weights=weights,
            spatial_dims=spatial_dims,
            K=k
        )

    @property
    def ssim(self) -> MS_SSIM:
        return self._mssim

    @property
    def mssim(self) -> MS_SSIM:
        return self._mssim

    @property
    def nonnegative(self) -> bool:
        return True


class FocalLoss(torch.nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, logits scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = inputs.shape[1]
            inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            targets = targets.view(-1)

        unignored_mask = targets != self.ignore_index
        targets = targets[unignored_mask]
        if len(targets) == 0:
            return torch.tensor(0.)
        inputs = inputs[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = nnf.log_softmax(inputs, dim=-1)
        ce = self.nll_loss(log_p, targets)

        # get true class column from each row
        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def make_one_hot(inputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert class index tensor to one hot encoding tensor.

    Args:
         inputs: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A one-hot encoded tensor of shape [N, num_classes, *]
    """
    shape = np.array(inputs.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, inputs.cpu(), 1)

    return result


class BinaryDiceLoss(torch.nn.Module):
    """
    Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: `math: \sum{x^p} + \sum{y^p}`, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean', return sum if 'sum', return a tensor
            of shape [N,] if 'none'
        from_logits: True if logits are given as predictions, False otherwise (default: True).
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth: float = 1.0, p: float = 2.0, reduction: str = 'mean', from_logits: bool = True):
        super(BinaryDiceLoss, self).__init__()
        self._smooth: float = smooth
        self._p: float = p
        self._reduction: str = reduction
        self._from_logits: bool = from_logits

    @property
    def smooth(self) -> float:
        return self._smooth

    @property
    def p(self) -> float:
        return self._p

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss between the predicted and target binary tensors.

        Args:
            predict (torch.Tensor): A tensor of shape [N, C, *], where N is the batch size, C is the number of classes,
                and * is the spatial dimensions.
            target (torch.Tensor): A tensor of the same shape as the predicted tensor.

        Returns:
            Loss tensor according to arg reduction.

        Raises:
            AssertionError: If the batch sizes of the predicted and target tensors do not match.
            Exception: If unexpected reduction.
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        if self.from_logits:
            predict = predict.sigmoid()

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BinaryFocalLoss(torch.nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()

        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )


class DiceLoss(torch.nn.Module):
    """
    Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: Optional[int] = None, **kwargs):
        super(DiceLoss, self).__init__()
        self._kwargs: Dict[str, Any] = kwargs
        self._weight: Optional[torch.Tensor] = weight
        self._ignore_index: Optional[int] = ignore_index

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs

    @property
    def weight(self) -> Optional[torch.Tensor]:
        return self._weight

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignore_index

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss between the predicted and target multi-class segmentation maps.

        Args:
            predict (torch.Tensor): A tensor of shape [N, C, H, W] or [N, C], where N is the batch size, C is the number
                of classes, H is the height, and W is the width.
            target (torch.Tensor): A tensor of the same shape as the predicted tensor.

        Returns:
            The average Dice loss for all classes except the ignored class.

        Raises:
            AssertionError: If the shapes of the predicted and target tensors do not match.
            AssertionError: If the weight tensor has the wrong shape.
        """
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(from_logits=False, **self.kwargs)
        total_loss = 0

        if len(predict.shape) < 4:
            predict = predict.softmax(dim=1)
        else:
            predict = predict.softmax(dim=-3)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


class DiceCrossLoss(torch.nn.Module):

    def __init__(self,
                 num_classes: int,
                 weight_dice: float = 1.0,
                 weight_ce: float = 0.001,
                 use_focal_loss: bool = False,
                 dice_kwargs: Optional[Dict[str, Any]] = None,
                 cross_entropy_kwargs: Dict[str, Any] = None):

        super().__init__()

        if cross_entropy_kwargs is None:
            cross_entropy_kwargs = {}

        self._ce_ignore_index: Optional[int] = None
        if "ignore_index" in cross_entropy_kwargs:
            self._ce_ignore_index = cross_entropy_kwargs["ignore_index"]
            if num_classes <= 2:
                del cross_entropy_kwargs["ignore_index"]
                cross_entropy_kwargs["reduction"] = "none"  # we must multiply element-wise

        if num_classes > 2:
            if dice_kwargs is None:
                dice_kwargs = {
                    "weight": None,
                    "ignore_index": None,
                    "smooth": 1.0,
                    "p": 2.0,
                    "reduction": "mean"
                }
            if use_focal_loss:
                self._cross_entropy = FocalLoss(**cross_entropy_kwargs)
            else:
                self._cross_entropy = torch.nn.CrossEntropyLoss(**cross_entropy_kwargs)
            self._dice = DiceLoss(**dice_kwargs)
        else:
            if dice_kwargs is None:
                dice_kwargs = {
                    "smooth": 1.0,
                    "p": 2.0,
                    "reduction": "mean"
                }

            if use_focal_loss:
                self._cross_entropy = BinaryFocalLoss(**cross_entropy_kwargs)
            else:
                self._cross_entropy = torch.nn.BCEWithLogitsLoss(**cross_entropy_kwargs)
            self._dice = BinaryDiceLoss(**dice_kwargs)

        self._num_classes: int = num_classes
        self._weight_ce: float = weight_ce
        self._weight_dice: float = weight_dice
        self._use_focal_loss: bool = use_focal_loss

    @property
    def cross_entropy(self) -> Union[torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss, FocalLoss, BinaryFocalLoss]:
        return self._cross_entropy

    @property
    def dice(self) -> Union[DiceLoss, BinaryDiceLoss]:
        return self._dice

    @property
    def num_classes(self) -> int:
        return self._num_classes if self._num_classes > 2 else 2

    @property
    def weight_ce(self) -> float:
        return self._weight_ce

    @property
    def weight_dice(self) -> float:
        return self._weight_dice

    @property
    def use_focal_loss(self) -> bool:
        return self._use_focal_loss

    def forward(self,
                inputs: torch.Tensor,
                target: torch.Tensor,
                return_all: bool = True,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Computes the combined Dice and Cross-Entropy loss between the predicted and target segmentation maps.

        Args:
            inputs (torch.Tensor): The predicted segmentation maps. The shape of the tensor depends on the number of
                classes. If `num_classes` is greater than 2, the shape is `(N, C, H, W)`, where `N` is the number of
                images, `C` is the number of classes, `H` is the height of the image, and `W` is the width of the image.
                If `num_classes` is 2, the shape is `(N, H, W)`.
            target (torch.Tensor): The target segmentation maps. The shape of the tensor is the same as the shape of the
                `inputs` tensor.
            return_all (bool, optional): Whether to return the combined dice-cross loss, or the all three losses

        Returns:
            The weighted sum of the Dice loss and the Cross-Entropy loss, plus the cross-entropy and the dice loss
            tensors (in this order), if `return_all` is True.
        """
        ce = self.cross_entropy(inputs, target)
        dice = self.dice(inputs, target)

        if self.num_classes <= 2 and self._ce_ignore_index is not None:
            # noinspection PyUnresolvedReferences
            ce = ce * ((target >= 0) & (target != self._ce_ignore_index)).float()  # mask all positions with the ignored

            reduction = self.dice.reduction
            if reduction == 'none':
                pass
            elif reduction == 'mean':
                ce = ce.mean()
            elif reduction == 'sum':
                ce = ce.sum()
            else:
                raise ValueError(f'Unsupported reduction type {reduction}.')

        if return_all:
            return ce * self.weight_ce + dice * self.weight_dice, ce, dice
        return ce * self.weight_ce + dice * self.weight_dice
        # return self.cross_entropy(inputs, target) * self.weight_ce + self.dice(inputs, target) * self.weight_dice
