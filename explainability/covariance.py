from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import einops


def batched_cov(points: torch.Tensor, correction: int = 1) -> torch.Tensor:
    b, s, e = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(b * s, e)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(b, s, e, e)
    bcov = prods.sum(dim=1) / (s - correction)  # Unbiased estimate
    return bcov  # (B, D, D)


def compute_cov(tensor: torch.Tensor, correction: int = 1, batched: bool = False) -> torch.Tensor:
    if len(tensor.shape) > 2:
        if batched:
            tensor = einops.rearrange(tensor, "b s ... e -> b (s ...) e")
        else:
            tensor = einops.rearrange(tensor, "b ... e -> (b ...) e")

    if batched:
        cov = batched_cov(tensor, correction=correction)
    else:
        cov = torch.cov(tensor, correction=correction)

    return cov


def get_spectral_values(tensor: torch.Tensor, method: str = "svd", absolute: bool = True, **kwargs) -> torch.Tensor:
    if method == "svd":
        vals = torch.linalg.svdvals(tensor, **kwargs)
    elif method == "eig" or method == "eigvals":
        vals = torch.linalg.eigvals(tensor, **kwargs)
    elif method == "eigh" or method == "eigvalsh":
        vals = torch.linalg.eigvalsh(tensor, **kwargs)
    else:
        raise ValueError(f"Method {method} not supported")
    return vals if not absolute else vals.abs()


def plot_spectral_values(tensor: torch.Tensor,
                         method: str = "svd",
                         absolute: bool = True,
                         plot_type: str = "bar",
                         path: Optional[str] = None,
                         show_plot: bool = True,
                         **kwargs) -> torch.Tensor:
    vals = get_spectral_values(tensor, method=method, absolute=absolute, **kwargs)

    if absolute:
        x = np.arange(vals.shape[-1])
        y = vals.cpu().detach().numpy()
        x_label = f"{'eigenvalue#' if method in ['eigvalsh', 'eigh', 'eig', 'eigvals'] else 'singular_value#'}"
        y_label = "absolute value"

        if plot_type in ["bar", "barplot"]:
            df = pd.DataFrame({x_label: x, y_label: y})
            plot = sns.barplot(data=df, x=x_label, y=y_label)
        elif plot_type in ["hist", "histogram"]:
            x_label = f"abs({x_label[:-1]})"
            df = pd.DataFrame({x_label: y})
            plot = sns.histplot(data=df, x=x_label, bins="auto", kde=True)
        elif plot_type in ["box", "boxplot"]:
            x_label = f"abs({x_label[:-1]})"
            df = pd.DataFrame({x_label: y})
            plot = sns.boxplot(data=df, x=x_label)
        else:
            raise ValueError(f"Plot type {plot_type} not supported")

        fig = plot.get_figure()
    else:
        raise ValueError(f"Complex plots not supported yet.")

    if path is not None:
        fig.savefig(path)
    if show_plot:
        plt.show()

    return vals
