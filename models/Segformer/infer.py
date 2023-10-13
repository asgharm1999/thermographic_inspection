# Import Libraries
from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import json

# Load model
segmenter = pipeline(
    "image-segmentation", model="ChristopherS27/testModel"
)  # TODO: Set model name

# Read images
testImage = Image.open("data/TIR/TIR-SS/Img8bit/val/64_1332.bmp")
testMask = Image.open("data/TIR/TIR-SS/gtFine/val/64_1332.png")

# Get mappings
id2label = hf_hub_download(
    repo_id="ChristopherS27/TIR", filename="id2label.json", repo_type="dataset"
)
id2label = json.load(open(id2label, "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

label2color = {
    label: list(np.random.choice(range(256), size=3))
    if label != "background"
    else [0, 0, 0]
    for label in label2id.keys()
}


# Convert mask to RGB
def maskToRGB(mask, type: str | None = None):
    arr = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if type != None:
                if (mask[i][j] == [255, 255, 255]).all():
                    arr[i][j] = label2color[type]
            else:
                arr[i][j] = label2color[id2label[mask[i][j]]]

    return arr


# Predict
res = segmenter(testImage)

fig, ax = plt.subplots(1, 2, figsize=(15, 15))

ax[0].set_title("Prediction")
ax[1].set_title("Ground Truth")

predMask = np.zeros(np.array(testMask).shape[:2] + (3,), dtype=np.uint8)
for index in range(len(res)):
    label = res[index]["label"]
    id = label2id[label]

    temp = np.array(res[index]["mask"])

    for i in range(predMask.shape[0]):
        for j in range(predMask.shape[1]):
            if (temp[i][j] == [255, 255, 255]).all():
                predMask[i][j] = label2color[label]


ax[0].imshow(testImage)
ax[0].imshow(predMask, alpha=0.5)

ax[1].imshow(testImage)
ax[1].imshow(maskToRGB(np.array(testMask)), alpha=0.5)

plt.show()
