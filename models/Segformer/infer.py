# Import Libraries
from transformers import pipeline
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

# Load model
segmenter = pipeline("image-segmentation", model="modelName")

# Read images
testImage = Image.open("data/images/name.png")
testMask = Image.open("data/masks/name.png")

def maskToRGB(mask, type: str):
    arr = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)

    if type == "background":
        pass
    elif type == "class1":
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (mask[i][j] == [255, 255, 255]).all():
                    arr[i][j] = [255, 0, 0]  # Set RGB values
    elif type == "class2":
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (mask[i][j] == [255, 255, 255]).all():
                    arr[i][j] = [0, 255, 0]  # Set RGB values
    
    return arr

def label2num(label: str):
    if label == "background":
        return 0
    elif label == "class1":
        return 1
    elif label == "class2":
        return 2
    

# Predict
res = segmenter(testImage)
    
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

ax[0].set_title("Prediction")
ax[1].set_title("Ground Truth")

for label in range(len(res)):
    mask = maskToRGB(np.array(res[label]["mask"]), res[label]["label"])
    ax[0].imshow(mask, alpha=0.5)

ax[0].imshow(testImage, alpha=0.5)
ax[1].imshow(testImage)
