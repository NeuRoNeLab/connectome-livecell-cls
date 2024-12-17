from typing import Union
import numpy as np
import torchvision
from skimage.segmentation import mark_boundaries
from PIL import Image
import torch

def lime_explain(model,
                 img,
                 explainer,
                 out_path,
                 key: Union[int, str] = 0,
                 device: str = 'cpu'):

    def preprocess_transform(input_image: np.ndarray):
        t = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224))
        ])
        input = t(input_image).unsqueeze(0).to(device)
        return input

    def predict4lime(img2):
        # print(img2.shape)
        img2 = preprocess_transform(img2[0, ..., 0])
        # If error, comment this. Some model need 1 channel, others 3
        if img2.shape != (1, 3, 224, 224):
            img2 = torch.tile(img2, (1, 3, 1, 1))
        output = model(img2)
        output = output.output
        if isinstance(output, (tuple, list, dict)):
            output = output[key]
        return output.softmax(-1).detach().cpu().numpy()

    explanation = explainer.explain_instance(np.asarray(img),
                                             predict4lime,  # classification function
                                             labels=(0, 1, 2, 3, 4, 5, 6, 7),
                                             top_labels=8,
                                             num_samples=100,
                                             batch_size=1)  # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                                hide_rest=False)  # , min_weight=0.1) TODO:parametrizzare
    img_boundry = mark_boundaries(temp / 255.0, mask)
    img_boundry = np.asarray(img_boundry * 255, dtype=np.uint8)
    Image.fromarray(img_boundry).save(out_path)

