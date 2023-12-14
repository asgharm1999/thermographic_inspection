"""
Reads images and masks from a directory and create a HuggingFace Dataset
"""

from datasets import Dataset, DatasetDict, Image
from pandas import DataFrame
import glob

imageTrain = glob.glob("data/TIR/TIR-SS/Img8bit/test/*.bmp")
maskTrain = glob.glob("data/TIR/TIR-SS/gtFine/test/*.png")

imageTest = glob.glob("data/TIR/TIR-SS/Img8bit/train/*.bmp")
maskTest = glob.glob("data/TIR/TIR-SS/gtFine/train/*.png")

imageVal = glob.glob("data/TIR/TIR-SS/Img8bit/val/*.bmp")
maskVal = glob.glob("data/TIR/TIR-SS/gtFine/val/*.png")

def createDataset(imagePaths, maskPaths):
    pairs = zip(imagePaths, maskPaths)

    dataset = Dataset.from_pandas(DataFrame(pairs, columns=["image", "mask"]))
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("mask", Image(decode=True, id=None))

    return dataset

train = createDataset(imageTrain, maskTrain)
test = createDataset(imageTest, maskTest)
val = createDataset(imageVal, maskVal)

# Upload dataset to HuggingFace
dataset = DatasetDict({"train": train, "test": test, "val": val})
dataset.push_to_hub("ChristopherS27/TIR")
