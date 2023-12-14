"""
DeepLabV3Plus.py
DeepLabV3+ model implementation in Tensorflow
"""

# Import Libraries
import os
from keras import Sequential
from keras.layers import (
    Conv2D,
    BatchNormalization,
    AveragePooling2D,
    Activation,
    UpSampling2D,
    Input,
    Concatenate,
    Dropout,
)
from keras.models import Model
from keras.applications import Xception

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Create Blocks
def ConvBlock(filters: int, dilation=1, kernel_size=3):
    """
    Convolutional Block

    Parameters
    ----------
    filters : int
        Number of filters
    dilation : int, optional
        Dilation rate, by default 1
    kernel_size : int, optional
        Kernel size, by default 3

    Returns
    -------
    keras.Sequential object
    """

    return Sequential(
        [
            Conv2D(
                filters,
                kernel_size,
                dilation_rate=dilation,
                padding="same",
                use_bias=False,
            ),
            BatchNormalization(),
            Activation("relu"),
        ]
    )


def ASPP(inputLayer):
    ''' 
    Atrous Spatial Pyramid Pooling

    Parameters
    ----------
    inputLayer : Input
        Input layer

    Returns
    -------
    keras.Sequential object
    '''
    shape = inputLayer.shape

    # Global Average Pooling
    pool = Sequential(
        [
            AveragePooling2D(pool_size=(shape[1], shape[2])),
            Conv2D(256, 1, padding="same", use_bias=False),
            UpSampling2D((shape[1], shape[2]), interpolation="bilinear"),
        ]
    )(inputLayer)

    # 1x1 Convolution
    conv1 = ConvBlock(256, kernel_size=1)(inputLayer)

    # 3x3 Convolution with dilation rate 6
    conv6 = ConvBlock(256, dilation=6)(inputLayer)

    # 3x3 Convolution with dilation rate 12
    conv12 = ConvBlock(256, dilation=12)(inputLayer)

    # 3x3 Convolution with dilation rate 18
    conv18 = ConvBlock(256, dilation=18)(inputLayer)

    # Concatenate
    concat = Concatenate()([pool, conv1, conv6, conv12, conv18])

    # 1x1 Convolution
    return ConvBlock(256, kernel_size=1)(concat)


# Create DeepLabV3+ Model
def DeepLabV3Plus(inputShape, numClasses):
    '''
    DeepLabV3+ Model, using Xception backbone

    Parameters
    ----------
    inputShape : tuple
        Input shape
    numClasses : int
        Number of classes

    Returns
    -------
    keras.model.Model object
    '''
    inputs = Input(inputShape)

    # Encoder
    encoder = Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        # input_shape=inputShape,
        # pooling=None,
        # classes=numClasses,
    )
    encoder.trainable = False
    encoderOutput = encoder.get_layer("block13_pool").output

    # ASPP
    aspp = ASPP(encoderOutput)
    upsampled = UpSampling2D((4, 4), interpolation="bilinear")(aspp)

    # Decoder

    # 1x1 Convolution on low-level features
    low = ConvBlock(48, kernel_size=1)(encoder.get_layer("block4_sepconv2_bn").output)
    
    # Concatenate
    concat = Concatenate()([upsampled, low])

    # 3x3 Convolutions
    conv1 = ConvBlock(256, kernel_size=3)(concat)
    conv2 = ConvBlock(256, kernel_size=3)(conv1)

    # Upsampling
    upsampled = UpSampling2D((8, 8), interpolation="bilinear")(conv2)

    # 1x1 Convolution
    output = Conv2D(numClasses, 1, padding="same", activation="softmax")(upsampled)

    return Model(inputs=inputs, outputs=output)


# Test
if __name__ == "__main__":
    model = DeepLabV3Plus((480, 640, 3), 4)
    model.summary()

    import numpy as np

    test = model.predict(np.zeros((1, 480, 640, 3)))
