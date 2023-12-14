# Loads images and labels from folders, and saves it

# Import libraries
import tensorflow as tf
import glob
import os
import argparse
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load dataset")
parser.add_argument(
    "--savePath",
    type=str,
    default="data/datasets/default/",
    help="Path to save dataset",
)
parser.add_argument(
    "--imagePath", type=str, default="data/images/", help="Path to images folder"
)
parser.add_argument(
    "--maskPath", type=str, default="data/masks/", help="Path to masks folder"
)
parser.add_argument("--imageType", type=str, default="png", help="Image type")
parser.add_argument("--maskType", type=str, default="png", help="Mask type")

args = parser.parse_args()

# Get list of file names
fileNames = glob.glob(args.imagePath + "*." + args.imageType)
fileDS = tf.data.Dataset.list_files(fileNames)


# Load images
class ImageLoader:
    def __init__(self, imageType, maskType) -> None:
        self.imageType = imageType
        self.maskType = maskType

    def processPath(self, filePath: str):
        image = tf.io.read_file(filePath)
        mask = tf.io.read_file(
            tf.strings.regex_replace(filePath, args.imagePath, args.maskPath)
        )

        if self.imageType == "jpg":
            image = tf.image.decode_jpeg(image)
        elif self.imageType == "png":
            image = tf.image.decode_png(image, channels=3)

        if self.maskType == "jpg":
            mask = tf.image.decode_jpeg(mask)
        elif self.maskType == "png":
            mask = tf.image.decode_png(mask, channels=1)

        return image, mask

    def tfFunc(self, filePath: str):
        return tf.py_function(self.processPath, [filePath], [tf.uint8, tf.uint8])


dataset = fileDS.map(ImageLoader(args.imageType, args.maskType).tfFunc)  # type: ignore
dataset.save(args.savePath)

if __name__ == "__main__":
    # Load dataset to see if it works
    ds = tf.data.Dataset.load(args.savePath)

    for image, mask in ds.take(2):
        print(image.shape, mask.shape)
