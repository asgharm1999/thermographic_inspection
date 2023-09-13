'''
utils.py

Utility functions for Segformer training
'''


# Import Libraries
import albumentations as A 
import numpy as np

# Transforms
noTransform = A.Compose([A.Resize(512, 512)], is_check_shapes=False)

lightTransform = A.Compose(
    [A.Resize(512, 512), A.HorizontalFlip(p=0.5)],
    is_check_shapes=False,
)


class Transform:
    ''' 
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
    '''

    def __init__(self, preprocessor, transform='none', isTrain=True):
        self.preprocessor = preprocessor
        self.isTrain = isTrain

        if transform == 'none':
            self.transform = noTransform
        elif transform == 'light':
            self.transform = lightTransform
        else:
            raise ValueError('Invalid transform')

    def __call__(self, batch):
        if self.isTrain:
            images, labels = [], []

            for image, mask in zip(batch['image'], batch['mask']):
                transformed = self.transform(image=np.array(image), mask=np.array(mask))
                images.append(transformed['image'])
                labels.append(transformed['mask'])
            
            return self.preprocessor(images, labels)
        
        else:
            images = [x for x in batch['image']]
            labels = [x for x in batch['mask']]
            return self.preprocessor(images, labels)
