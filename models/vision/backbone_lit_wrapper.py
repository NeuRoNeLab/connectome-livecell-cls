from typing import Optional, Dict, Any, Callable, Union, List
from enum import Enum
import torch
import lightning as lit
from lightning.pytorch.utilities.types import STEP_OUTPUT
from models.configs import OptimizerType, LRSchedulerConfig, LRSchedulerType, resolve_optimizer, resolve_lr_scheduler, \
    resolve_metric
from models.layers.misc import resolve_activation
from models.vision.backbone import TorchVisionBackbone, BackboneOutput
from models.vision.backbone_utils import TorchVisionBackboneConfig, BackboneConfig
from utils.utils import SerializableConfig


class BackboneLossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    CUSTOM = "custom"


class BackboneHeadConfig(SerializableConfig):

    def __init__(self, in_dim: Optional[int] = None, dims: Optional[List[int]] = None, activation: str = "tanh"):
        super().__init__()
        self.in_dim: Optional[int] = in_dim
        self.dims: Optional[List[int]] = dims
        self.activation: str = activation


class BackboneWrapper(lit.LightningModule):

    def __init__(self,
                 config: TorchVisionBackboneConfig,
                 head_config: Optional[BackboneHeadConfig] = None,
                 loss_type: BackboneLossType = BackboneLossType.CROSS_ENTROPY,
                 loss_config: Optional[Dict] = None,
                 custom_loss_fn: Optional[torch.nn.Module] = None,
                 optimizer_type: OptimizerType = OptimizerType.ADAMW,
                 optimizer_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 lr_scheduler_config: LRSchedulerConfig = None,
                 metric_params: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None):
        super().__init__()

        # Store hyper-parameters
        self.save_hyperparameters()

        if optimizer_config is None:
            optimizer_config = {"": {}}

        # Store given parameters
        self._config: TorchVisionBackboneConfig = config
        self._loss_type: BackboneLossType = loss_type
        self._loss_config: Optional[Dict] = loss_config
        self._optimizer_type: OptimizerType = optimizer_type
        self._optimizer_config: Optional[Dict[str, Dict[str, Any]]] = optimizer_config
        self._lr_scheduler_config: LRSchedulerConfig = lr_scheduler_config
        self._head_config: Optional[BackboneHeadConfig] = head_config

        # Check for the number of labels from BackboneViTConfig
        if hasattr(config, 'output_dim'):
            out_channels = config.output_dim
        else:
            out_channels = config.backbone_kwargs['output_dim']
        self._out_channels: int = out_channels

        # Initialize loss function
        if loss_config is None:
            loss_config = {}
        if loss_type == BackboneLossType.CROSS_ENTROPY:
            if out_channels > 1:
                self._loss_fn = torch.nn.CrossEntropyLoss(**loss_config)
            else:
                self._loss_fn = torch.nn.BCEWithLogitsLoss(**loss_config)
        elif loss_type == BackboneLossType.CUSTOM:
            if custom_loss_fn is None:
                raise ValueError(f"Custom loss function must be provided if `loss_type` is {loss_type}.")
            self._loss_fn = custom_loss_fn
        else:
            raise ValueError(f"Unsupported loss type {loss_type}.")

        # Setup metrics
        task_type = "multiclass" if out_channels > 1 else "binary"

        if metric_params is not None and "average" in metric_params:
            average = metric_params["average"]
            del metric_params["average"]
            metric_params = metric_params if len(metric_params) > 0 else None
        else:
            average = "macro"
        default_metric_kwargs = {
            "task": task_type,
            "num_classes": out_channels if out_channels > 1 else 2,  # TODO: check if this works
            "average": average
        }
        if metric_params is None:
            metric_params = {
                "accuracy": default_metric_kwargs,
                "top_k_accuracy": default_metric_kwargs,
                "precision": default_metric_kwargs,
                "recall": default_metric_kwargs,
                "f1": default_metric_kwargs,
                "fbeta": {
                    **default_metric_kwargs,
                    "beta": 2.0
                },
                "auroc": default_metric_kwargs,
            }
        if self.out_channels < 2 and "top_k_accuracy" in metric_params:
            del metric_params["top_k_accuracy"]
        self._metrics = {}
        self._metric_params = metric_params
        for metric_name in metric_params:
            self._metrics[metric_name] = resolve_metric(metric_name, **metric_params[metric_name])

        # Initialize BackboneViT model from dictionary config
        self._backbone = TorchVisionBackbone(config)

        # Initialize MLP head if needed
        self._head = torch.nn.Identity()
        if head_config is not None and head_config.dims is not None and head_config.in_dim is not None:
            self._head = torch.nn.Sequential()
            in_dim = head_config.in_dim
            for i, dim in enumerate(head_config.dims):
                act_fn = resolve_activation(head_config.activation) if i < len(head_config.dims) - 1 else None
                lin = torch.nn.Linear(in_dim, dim)
                self._head.append(lin)
                if act_fn is not None:
                    self._head.append(act_fn)
                in_dim = dim
            self._head.append(torch.nn.Linear(in_dim, out_channels))

    @property
    def config(self) -> BackboneConfig:
        return self._config

    @property
    def head_config(self) -> Optional[BackboneHeadConfig]:
        return self._head_config

    @property
    def loss_type(self) -> BackboneLossType:
        return self._loss_type

    @property
    def loss_config(self) -> Optional[Dict]:
        return self._loss_config

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def optimizer_type(self) -> OptimizerType:
        return self._optimizer_type

    @property
    def optimizer_config(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self._optimizer_config

    @property
    def lr_scheduler_type(self) -> LRSchedulerType:
        return LRSchedulerType.NONE if self._lr_scheduler_config is None else self._lr_scheduler_config.name

    @property
    def lr_scheduler_config(self) -> LRSchedulerConfig:
        return self._lr_scheduler_config

    @property
    def backbone(self) -> TorchVisionBackbone:
        return self._backbone

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    @property
    def loss_fn(self) -> torch.nn.Module:
        return self._loss_fn

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor, ...], torch.Tensor]]:  # torch.nn.ModuleDict:
        return self._metrics

    @property
    def metric_params(self) -> Dict[str, Dict[str, Any]]:
        return self._metric_params

    def forward(self,
                x: torch.Tensor,
                apply_head: bool = False,
                attention_kwargs: Optional[Dict] = None,
                *args, **kwargs) -> BackboneOutput:
        out = self.backbone(
            x=x,
            *args,
            **kwargs
        )
        if apply_head:
            head_out = self.head(out.output)
            # noinspection PyProtectedMember
            # fields = out._asdict()
            # del fields["output"]
            out = BackboneOutput(output=head_out)

        return out

    def configure_optimizers(self):

        # Get the parameters
        model_params = self.backbone.named_parameters()

        # Define the parameter group names
        parameter_groups = {
            param_group_name: {
                "params": [],  # parameters of the group
                **self.optimizer_config[param_group_name]  # optimizer config for that parameter group
            } for param_group_name in self.optimizer_config
        }

        # Split the parameters into groups
        for param_name, param in model_params:
            for param_group_name in parameter_groups:
                if param_group_name in param_name or param_group_name == "all":  # maybe replace with regex
                    parameter_groups[param_group_name]["params"].append(param)
                    break

        optimizer_kwargs = {"params": list(parameter_groups.values())}
        optimizer = resolve_optimizer(
            optimizer_name=self.optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )

        lr_scheduler = resolve_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_name=self.lr_scheduler_type,
            lr_scheduler_config=self.lr_scheduler_config
        )

        if lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                'interval': self.lr_scheduler_config.interval,
                'frequency': self.lr_scheduler_config.frequency,
                "monitor": self.lr_scheduler_config.monitor
            }
        }

    def _step(self, batch, batch_idx, step_type: str, *args, **kwargs) -> STEP_OUTPUT:
        # Get relevant kwargs
        # tensornet_kwargs = None
        # if "tensornet_kwargs" in kwargs:
        #     tensornet_kwargs = kwargs["tensornet_kwargs"]
        #     del kwargs["tensornet_kwargs"]
        attention_kwargs = None
        if "attention_kwargs" in kwargs:
            attention_kwargs = kwargs["attention_kwargs"]
            del kwargs["attention_kwargs"]

        # Get the batch
        inputs, target = batch

        # Call the model
        output: BackboneOutput = self(
            x=inputs,
            apply_head=True,
            *args,
            **kwargs
        )

        # Compute the loss
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            target = target.float()
        loss = self.loss_fn(output.output, target)

        # Logging
        log_dict = {
            f'{step_type}/final_loss': loss.item()
        }

        # If loss is BCE, convert targets back to long
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            target = target.long()
        preds: torch.Tensor = output.output.detach()

        # If target is integer, convert logits to distribution
        if target.dtype == torch.long or self.loss_type == BackboneLossType.CROSS_ENTROPY:
            if self.out_channels == 1:
                preds = preds.sigmoid()
            else:
                preds = preds.softmax(dim=-1)

        # Compute the metrics
        for metric in self.metrics:

            # Convert target to one-hot if needed
            target_tmp = target
            preds_tmp = preds
            # if target.dtype == torch.long and (metric == "mse" or metric == "mae" or "ssim" in metric):
            #     if self.out_channels > 1:
            #         target_tmp = torch.nn.functional.one_hot(target, num_classes=self.out_channels).float()
            #     else:
            #         target_tmp = target_tmp.float()
            # if metric == "dice":
            #     preds_tmp = output.decoder_out.detach()

            # self.metrics[metric](preds_tmp, target_tmp)
            # log_dict[f"{step_type}_{metric}"] = self.metrics[metric]
            log_dict[f"{step_type}/{metric}"] = self.metrics[metric](preds_tmp, target_tmp).item()

        # Log the metrics
        on_step = step_type == "train"
        self.log_dict(dictionary=log_dict, on_step=on_step, sync_dist=True, prog_bar=True)

        return {"loss": loss,
                "output": preds if step_type == 'test' else None}

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._step(batch=batch, batch_idx=batch_idx, step_type="train", *args, **kwargs)

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._step(batch=batch, batch_idx=batch_idx, step_type="val", *args, **kwargs)

    def test_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        return self._step(batch=batch, batch_idx=batch_idx, step_type="test", *args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: implement this
        pass
