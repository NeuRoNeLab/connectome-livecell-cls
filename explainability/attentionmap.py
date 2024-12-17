from typing import Sequence, Union, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch


def attention_rollout(attn_weights: Sequence[torch.Tensor],
                      img: Image,
                      label: Optional[Union[int, Tuple[str, int]]] = None,
                      logits: Optional[torch.Tensor] = None,
                      return_masked_img: str = "all",
                      plot_img: bool = True,
                      figsize: Tuple[int, int] = (16, 16),
                      out_path: Optional[str] = "attn_map.png",) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                                          List[Tuple[torch.Tensor, torch.Tensor]]]:

    # Average the attention weights across all heads
    attn_weights = torch.stack(list(attn_weights), dim=0).squeeze(dim=1)
    if len(attn_weights[0].shape) >= 4:
        attn_weights = attn_weights.mean(dim=1)

    # To account for residual connections, we add an identity matrix to the attn matrix and re-normalize the weights
    residual_att = torch.eye(attn_weights.size(1))
    aug_attn_weights = attn_weights + residual_att
    aug_attn_weights = aug_attn_weights / aug_attn_weights.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    # joint_attentions = torch.zeros(aug_attn_weights.size())
    # joint_attentions[0] = aug_attn_weights[0]
    joint_attentions = [aug_attn_weights[0]]

    for n in range(1, aug_attn_weights.size(0)):
        # joint_attentions[n] = torch.matmul(aug_attn_weights[n], joint_attentions[n - 1])
        joint_attentions.append(torch.matmul(aug_attn_weights[n], joint_attentions[n - 1]))
    joint_attentions = torch.stack(joint_attentions, dim=0)

    # Attention from the output token to the input space
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_attn_weights.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")

    suffix = "" if label is None else f"Label {label}"
    if logits is not None and label is not None:
        label_idx = label if isinstance(label, int) else label[0]
        prob = logits.softmax(dim=-1)[..., label_idx].item()
        suffix = f"{suffix}, Prob {prob:.4f}"

    if out_path is not None:
        out_path = f"{'.'.join(out_path.split('.')[:-1])}_original.png"
        img.save(out_path)

    if return_masked_img == "last":
        if plot_img or out_path is not None:
            result_pil = Image.fromarray(result)

            if out_path is not None:
                result_pil.save(out_path)
            if plot_img:
                result_pil.show()

        return result, v

    elif return_masked_img == "all":
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(joint_attentions.size(0)):
            v = joint_attentions[i]
            # Attention from the output token to the input space.
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            result = (mask * img).astype("uint8")
            results.append((result, v))

            if plot_img or out_path is not None:
                result_pil = Image.fromarray(result)
                if out_path is not None:
                    extension = out_path.split(".")[-1]
                    out_path = f"{'.'.join(out_path.split('.')[:-1])}_layer{i}.{extension}"
                    result_pil.save(out_path)

                if plot_img:
                    result_pil.show()

        return results

    elif return_masked_img == "mean":
        # Attention from the output token to the input space
        v = joint_attentions.mean(dim=0)
        grid_size = int(np.sqrt(aug_attn_weights.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

        if plot_img or out_path is not None:
            result_pil = Image.fromarray(result)

            if out_path is not None:
                result_pil.save(out_path)
            if plot_img:
                result_pil.show()

        return result, v
    else:
        raise ValueError(f"Unrecognized attention mask return type: {return_masked_img}")
