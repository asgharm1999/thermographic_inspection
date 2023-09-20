"""
Reads images and masks from a directory and create a HuggingFace Dataset
"""

from datasets import Dataset, DatasetDict, Image
from pandas import DataFrame
import glob

imageTrain = glob.glob("path/to/train/images/*.png")
maskTrain = glob.glob("path/to/train/masks/*.png")

imageTest = glob.glob("path/to/test/images/*.png")
maskTest = glob.glob("path/to/test/masks/*.png")

def createDataset(imagePaths, maskPaths):
    pairs = zip(imagePaths, maskPaths)

    dataset = Dataset.from_pandas(DataFrame(pairs, columns=["image", "label"]))
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image(decode=True, id=None))

    return dataset

train = createDataset(imageTrain, maskTrain)
test = createDataset(imageTest, maskTest)

# Upload dataset to HuggingFace
dataset = DatasetDict({"train": train, "test": test})
dataset.push_to_hub("nameOfDataset")
