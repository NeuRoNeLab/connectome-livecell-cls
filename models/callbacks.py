import os
from typing import Any, Optional, Final, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchmetrics
import torchvision
import wandb
import lightning as lit
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_only


PREDICTED_MASK: Final[str] = "pred_mask"


class MakeBinaryMaskGrid(lit.Callback):
    def __init__(self,
                 interval: int = 5,
                 n_images: int = 4,
                 max_batches_per_epoch: int = 5,
                 pred_mask_key: str = PREDICTED_MASK):
        self._interval: int = interval
        self._n_images: int = n_images
        self._max_batches_per_epoch: int = max_batches_per_epoch
        self._pred_mask_key: str = pred_mask_key
        self._logged_batches_epoch: Dict[str, int] = {
            "train": 0,
            "val": 0,
            "test": 0
        }

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def n_images(self) -> int:
        return self._n_images

    @property
    def max_batches_per_epoch(self) -> int:
        return self._max_batches_per_epoch

    @property
    def pred_mask_key(self) -> str:
        return self._pred_mask_key

    @rank_zero_only
    def on_train_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        print(f"Image prediction callback with interval {self.interval}")

    def _store_predictions(self,
                           trainer: lit.Trainer,
                           _pl_module: lit.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           step_type: str):
        if ((step_type == "train" and trainer.global_step % self.interval == 0) or
            (step_type in ["val", "test"] and batch_idx % self.interval == 0)) and \
                self._logged_batches_epoch[step_type] < self.max_batches_per_epoch:
            self._logged_batches_epoch[step_type] += 1
            x, y = batch
            n = min(self.n_images, y.shape[0])
            data = []
            '''# for x_i, y_i, y_pred in zip(x[:n], y[:n], outputs['predicted_mask'][:n]):
            #     # print(x_i.shape, y_i.shape, y_pred.shape)
            #     x_i.clamp_(min=x_i.min(), max=x_i.max())
            #     x_i.sub_(x_i.min()).div_(max(x_i.max() - x_i.min(), 1e-5))
            #     y_i = (y_i * 255).to(torch.uint8)
            #     y_pred = (y_pred * 255).to(torch.uint8)
            #     data.append(torch.cat([x_i, y_i, y_pred], dim=-1))'''

            # Perform a random shuffling from the batch to be sure everytime we look at different images
            idx = torch.randperm(y.shape[0])
            y_perm = y[idx, ...]
            outputs_perm = outputs[self.pred_mask_key][idx, ...]
            # for y_true, y_pred in zip(y[:n], outputs[self.pred_mask_key][:n]):
            for y_true, y_pred in zip(y_perm[:n], outputs_perm[:n]):
                y_true = (y_true * 255).to(torch.uint8).unsqueeze(-3)
                # y_pred = (torch.sigmoid(y_pred) * 255).to(torch.uint8)
                y_pred = (y_pred * 255).to(torch.uint8).unsqueeze(-3)
                data.append(torch.cat([y_true, y_pred], dim=-1))
            if data:
                grid = torchvision.utils.make_grid(data, nrow=len(data), normalize=False, scale_each=False)
                grid = wandb.Image(grid, mode='L')

                if isinstance(trainer.logger, WandbLogger):
                    # log to wandb
                    # noinspection PyUnresolvedReferences
                    trainer.logger.experiment.log(
                        #{f"{step_type}/images/epoch_{trainer.current_epoch}_gs_{trainer.global_step}": grid},
                        {f"{step_type}/images/epoch_{trainer.current_epoch}": grid},
                        # step=trainer.global_step,
                        # batch_idx=
                        #if step_type == 'train' else batch_idx
                    )
                else:
                    path = trainer.log_dir
                    grid.image.save(
                        os.path.join(path, f"{step_type}_epoch_{trainer.current_epoch}_batch_{batch_idx}_preds.png"),
                        format="png"
                    )

    def on_train_batch_end(self,
                           trainer: lit.Trainer,
                           pl_module: lit.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int):
        self._store_predictions(
            trainer=trainer,
            _pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            step_type="train"
        )

    def on_validation_batch_end(self,
                                trainer: lit.Trainer,
                                pl_module: lit.LightningModule,
                                outputs: Optional[STEP_OUTPUT],
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        self._store_predictions(
            trainer=trainer,
            _pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            step_type="val"
        )

    def on_test_batch_end(self,
                          trainer: lit.Trainer,
                          pl_module: lit.LightningModule,
                          outputs: Optional[STEP_OUTPUT],
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int = 0, ):
        self._store_predictions(
            trainer=trainer,
            _pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            step_type="test"
        )

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        self._logged_batches_epoch["train"] = 0

    def on_validation_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        self._logged_batches_epoch["val"] = 0

    def on_sanity_check_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        self._logged_batches_epoch["val"] = 0

    def on_test_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        self._logged_batches_epoch["test"] = 0


class ConfusionMatrixCallback(lit.Callback):
    def __init__(self, out_name: str, out_path: str, normalize: str):
        self.out_name = out_name
        self.out_path = out_path
        self.normalize = normalize
        self.cf_preds = []
        self.cf_targets = []

    def on_test_batch_end(
            self,
            trainer: lit.Trainer,
            pl_module: lit.LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.cf_targets.append(batch[1])
        preds = outputs['output']
        self.cf_preds.append(preds.argmax(dim=-1))

    def on_test_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        confusion_matrix = torchmetrics.ConfusionMatrix(task='binary' if trainer.model.config.output_dim <= 2
                                                             else 'multiclass',
                                                        num_classes=trainer.model.config.output_dim,
                                                        normalize='true')
        self.cf_preds = torch.IntTensor([item for row in self.cf_preds for item in row])
        self.cf_targets = torch.IntTensor([item for row in self.cf_targets for item in row])
        confusion_matrix(self.cf_preds, self.cf_targets)

        confusion_matrix_computed = confusion_matrix.compute().detach().cpu().numpy().astype(float)
        confusion_matrix_computed = np.around(confusion_matrix_computed, decimals=2)
        df_cm = pd.DataFrame(confusion_matrix_computed)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, linewidths=2, cmap='Blues', annot_kws={"size": 17})
        outpath = os.path.join(self.out_path, self.out_name)
        plt.savefig(outpath + '.png')
        df_cm.to_csv(outpath + '.csv')
