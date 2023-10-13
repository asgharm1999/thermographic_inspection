"""
utils.py

Utility functions for Segformer training
"""


# Import Libraries
import albumentations as A
import numpy as np

from torch import no_grad, from_numpy
from torch.nn import functional as F

# Transforms
noTransform = A.Compose([A.Resize(512, 512)], is_check_shapes=False)

lightTransform = A.Compose(
    [A.Resize(512, 512), A.HorizontalFlip(p=0.5)],
    is_check_shapes=False,
)


class Transform:
    """
    Creates a transform object that can be used to transform images and masks

    Parameters
    ----------
    preprocessor : obj
        Preprocessor object
    transform : str
        Type of transform to use. ['none', 'light'] (default: 'none')
    isTrain : bool
        Whether the transform is for training or not (default: True)

    Returns
    -------
    obj
        Transform object
    """

    def __init__(self, preprocessor, transform: str = "none", isTrain: bool = True):
        self.preprocessor = preprocessor
        self.isTrain = isTrain

        if transform == "none":
            self.transform = noTransform
        elif transform == "light":
            self.transform = lightTransform
        else:
            raise ValueError("Invalid transform")

    def __call__(self, batch):
        if self.isTrain:
            images, labels = [], []

            for image, mask in zip(batch["image"], batch["mask"]):
                transformed = self.transform(image=np.array(image.convert('RGB')), mask=np.array(mask))
                images.append(transformed["image"])
                labels.append(transformed["mask"])

            return self.preprocessor(images, labels)

        else:
            images = [x for x in batch["image"]]
            labels = [x for x in batch["mask"]]
            return self.preprocessor(images, labels)


# Metrics
class ComputeMetrics:
    def __init__(self, metric, id2label, numLabels):
        self.metric = metric
        self.id2label = id2label
        self.numLabels = numLabels

    def computeMetrics(self, pred):
        with no_grad():
            logits, labels = pred
            logits = from_numpy(logits)
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            ).argmax(dim=1)

            predLabels = logits.detach().cpu().numpy()
            metrics = self.metric.compute(
                predictions=predLabels,
                references=labels,
                num_labels=self.numLabels,
                ignore_index=-1,
            )

            perCatAcc = metrics.pop("per_category_accuracy").tolist()
            perCatIOU = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{self.id2label[i]}": v for i, v in enumerate(perCatAcc)})
            metrics.update({f"iou_{self.id2label[i]}": v for i, v in enumerate(perCatIOU)})

            return {"val_" + k: v for k, v in metrics.items()}
