import numpy as np
from typing import Callable, List, Optional
import tqdm
import torch
import ttach as tta
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils import get_2d_projection
from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels, scale_cam_image
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive, find_layer_predicate_recursive
from pytorch_grad_cam.ablation_layer import AblationLayer


class ActivationsAndGradientsV2(ActivationsAndGradients):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform, layer_keys=None):
        super().__init__(model, target_layers, reshape_transform)
        self.layer_keys = layer_keys

        if layer_keys is not None:
            self.layer_keys_in_storage_order = []
            self.layers_ids = {
                hash(layer): layer_key for layer, layer_key in zip(target_layers, layer_keys)
            }

    def save_activation(self, module, input, output):

        if isinstance(output, (tuple, list)):
            output = output[0]

        super().save_activation(module, input, output)

    def save_gradient(self, module, input, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        if self.layer_keys is not None:
            layer_key = self.layers_ids[hash(module)]
            self.layer_keys_in_storage_order.append(layer_key)

        # super().save_gradient(module, input, output)

        # Code from super-scall, need this to fix bug with fullgradcam
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)


class BaseCAMV2(BaseCAM):
    def __init__(
            self,
            model: torch.nn.Module,
            target_layers: List[torch.nn.Module],
            reshape_transform: Callable = None,
            compute_input_gradient: bool = False,
            uses_gradients: bool = True,
            tta_transforms: Optional[tta.Compose] = None,
    ) -> None:
        super().__init__(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            compute_input_gradient=compute_input_gradient,
            uses_gradients=uses_gradients,
            tta_transforms=tta_transforms
        )
        self.activations_and_grads = ActivationsAndGradientsV2(self.model, target_layers, reshape_transform)


class GradCAMV2(BaseCAMV2):
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAMV2, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            print(f'Grad shape inval: {grads.shape}')
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


class GradCAMPlusPlusV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradCAMPlusPlusV2, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


class ScoreCAMV2(BaseCAMV2):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(ScoreCAMV2, self).__init__(model,
                                         target_layers,
                                         reshape_transform=reshape_transform,
                                         uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                            :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights


""" Implementation of AblationCAM
https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf

Ablate individual activations, and then measure the drop in the target scores.

In the current implementation, the target layer activations is cached, so it won't be re-computed.
However layers before it, if any, will not be cached.
This means that if the target layer is a large block, for example model.featuers (in vgg), there will
be a large save in run time.

Since we have to go over many channels and ablate them, and every channel ablation requires a forward pass,
it would be nice if we could avoid doing that for channels that won't contribute anwyay, making it much faster.
The parameter ratio_channels_to_ablate controls how many channels should be ablated, using an experimental method
(to be improved). The default 1.0 value means that all channels will be ablated.
"""


class AblationCAMV2(BaseCAMV2):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 reshape_transform: Callable = None,
                 ablation_layer: torch.nn.Module = AblationLayer(),
                 batch_size: int = 32,
                 ratio_channels_to_ablate: float = 1.0) -> None:

        super(AblationCAMV2, self).__init__(model,
                                            target_layers,
                                            reshape_transform,
                                            uses_gradients=False)
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer
        self.ratio_channels_to_ablate = ratio_channels_to_ablate

    def save_activation(self, module, input, output) -> None:
        """ Helper function to save the raw activations from the target layer """
        self.activations = output

    def assemble_ablation_scores(self,
                                 new_scores: list,
                                 original_score: float,
                                 ablated_channels: np.ndarray,
                                 number_of_channels: int) -> np.ndarray:
        """ Take the value from the channels that were ablated,
            and just set the original score for the channels that were skipped """

        index = 0
        result = []
        sorted_indices = np.argsort(ablated_channels)
        ablated_channels = ablated_channels[sorted_indices]
        new_scores = np.float32(new_scores)[sorted_indices]

        for i in range(number_of_channels):
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]
                index = index + 1
            else:
                weight = original_score
            result.append(weight)

        return result

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: torch.nn.Module,
                        targets: List[Callable],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:

        # Do a forward pass, compute the target scores, and cache the
        # activations
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32(
                [target(output).cpu().item() for target, output in zip(targets, outputs)])

        # Replace the layer with the ablation layer.
        # When we finish, we will replace it back, so the
        # original model is unchanged.
        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []
        # This is a "gradient free" method, so we don't need gradients here.
        with torch.no_grad():
            # Loop over each of the batch images and ablate activations for it.
            for batch_index, (target, tensor) in enumerate(
                    zip(targets, input_tensor)):
                new_scores = []
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                # Check which channels should be ablated. Normally this will be all channels,
                # But we can also try to speed this up by using a low
                # ratio_channels_to_ablate.
                channels_to_ablate = ablation_layer.activations_to_be_ablated(
                    activations[batch_index, :], self.ratio_channels_to_ablate)
                number_channels_to_ablate = len(channels_to_ablate)

                for i in tqdm.tqdm(
                        range(
                            0,
                            number_channels_to_ablate,
                            self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[:(
                                number_channels_to_ablate - i)]

                    # Change the state of the ablation layer so it ablates the next channels.
                    # TBD: Move this into the ablation layer forward pass.
                    ablation_layer.set_next_batch(
                        input_batch_index=batch_index,
                        activations=self.activations,
                        num_channels_to_ablate=batch_tensor.size(0))
                    score = [target(o).cpu().item()
                             for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(
                        0):]

                new_scores = self.assemble_ablation_scores(
                    new_scores,
                    original_scores[batch_index],
                    channels_to_ablate,
                    number_of_channels)
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        # Returning the weights from new_scores
        return weights


class XGradCAMV2(BaseCAMV2):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(XGradCAMV2, self).__init__(model,
                                         target_layers,
                                         reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
                  (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights


# https://arxiv.org/abs/2008.00299
class EigenCAMV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(EigenCAMV2, self).__init__(model,
                                         target_layers,
                                         reshape_transform,
                                         uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)


# Like Eigen CAM: https://arxiv.org/abs/2008.00299
# But multiply the activations x gradients
class EigenGradCAMV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(EigenGradCAMV2, self).__init__(model, target_layers, reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(grads * activations)


# https://ieeexplore.ieee.org/document/9462463
class LayerCAMV2(BaseCAMV2):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(LayerCAMV2, self).__init__(
            model,
            target_layers,
            reshape_transform
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        spatial_weighted_activations = np.maximum(grads, 0) * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam


# https://arxiv.org/abs/1905.00780
class FullGradV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        if len(target_layers) > 0:
            print(
                "Warning: target_layers is ignored in FullGrad. All bias layers will be used instead")

        def layer_with_2D_bias(layer):
            bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
            if type(layer) in bias_target_layers and layer.bias is not None:
                return True
            return False

        target_layers = find_layer_predicate_recursive(
            model, layer_with_2D_bias)

        layer_keys = []
        i = 0
        for key, layer in model.named_modules():
            if i < len(target_layers) and layer == target_layers[i]:
                layer_keys.append(key)
                i += 1
        self.layer_keys = layer_keys

        super(FullGradV2, self).__init__(
            model,
            target_layers,
            reshape_transform,
            compute_input_gradient=True)
        self.activations_and_grads = ActivationsAndGradientsV2(self.model, target_layers, reshape_transform, layer_keys)
        self.bias_data = [self.get_bias_data(
            layer).cpu().numpy() for layer in target_layers]
        self.reordered_bias_list = False

    def get_bias_data(self, layer):
        # Borrowed from official paper impl:
        # https://github.com/idiap/fullgrad-saliency/blob/master/saliency/tensor_extractor.py#L47
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight
                      / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias.data
        else:
            return layer.bias.data

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            eigen_smooth):

        if not self.reordered_bias_list:
            # Need this to fix bug with activations and gradients being saved in the execution order and the biases
            # being saved in the module instantiation order
            self.bias_data = [
                self.get_bias_data(self.model.get_submodule(layer_key)).cpu().numpy()
                for layer_key in self.activations_and_grads.layer_keys_in_storage_order
            ]
            self.reordered_bias_list = True

        input_grad = input_tensor.grad.data.cpu().numpy()
        grads_list = [g.cpu().data.numpy() for g in
                      self.activations_and_grads.gradients]
        cam_per_target_layer = []
        target_size = self.get_target_width_height(input_tensor)

        gradient_multiplied_input = input_grad * input_tensor.data.cpu().numpy()
        gradient_multiplied_input = np.abs(gradient_multiplied_input)
        gradient_multiplied_input = scale_accross_batch_and_channels(
            gradient_multiplied_input,
            target_size)
        cam_per_target_layer.append(gradient_multiplied_input)

        # Loop over the saliency image from every layer
        assert (len(self.bias_data) == len(grads_list))
        for bias, grads in zip(self.bias_data, grads_list):
            bias = bias[None, :, None, None]
            # In the paper they take the absolute value,
            # but possibily taking only the positive gradients will work
            # better.
            bias_grad = np.abs(bias * grads)
            result = scale_accross_batch_and_channels(
                bias_grad, target_size)
            result = np.sum(result, axis=1)
            cam_per_target_layer.append(result[:, None, :])
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        if eigen_smooth:
            # Resize to a smaller image, since this method typically has a very large number of channels,
            # and then consumes a lot of memory
            cam_per_target_layer = scale_accross_batch_and_channels(
                cam_per_target_layer, (target_size[0] // 8, target_size[1] // 8))
            cam_per_target_layer = get_2d_projection(cam_per_target_layer)
            cam_per_target_layer = cam_per_target_layer[:, None, :, :]
            cam_per_target_layer = scale_accross_batch_and_channels(
                cam_per_target_layer,
                target_size)
        else:
            cam_per_target_layer = np.sum(
                cam_per_target_layer, axis=1)[:, None, :]

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        result = np.sum(cam_per_target_layer, axis=1)
        return scale_cam_image(result)


class HiResCAMV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(HiResCAMV2, self).__init__(
            model,
            target_layers,
            reshape_transform
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = grads * activations

        if eigen_smooth:
            print(
                "Warning: HiResCAM's faithfulness guarantees do not hold if smoothing is applied")
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam


class RandomCAMV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(RandomCAMV2, self).__init__(
            model,
            target_layers,
            reshape_transform
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.random.uniform(-1, 1, size=(grads.shape[0], grads.shape[1]))


class GradCAMElementWiseV2(BaseCAMV2):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradCAMElementWiseV2, self).__init__(
            model,
            target_layers,
            reshape_transform
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = np.maximum(grads * activations, 0)

        if eigen_smooth:
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam
