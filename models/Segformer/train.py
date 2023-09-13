# Import Libraries
import wandb
import argparse
from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
)
import evaluate

from utils import Transform

# Parse Arguments 
parser = argparse.ArgumentParser(description='Train Segformer')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batchSize', type=int, default=6, help='Batch size')
parser.add_argument('--earlyStop', type=int, default=10, help='Early stopping patience')
parser.add_argument('--name', type=str, default='Test', help='Name of the run')
parser.add_argument('--scheduler', type=str, default='polynomial', help='Learning rate scheduler')
parser.add_argument('--modelName', type=str, default='nvidia/mit-b0', help='Model name')

args = parser.parse_args()

# Initialize WandB
wandb.init(
    project='projectName',  # TODO: Set project name
    config={
        'modelName': args.modelName,
        'batchSize': args.batchSize,
        'epochs': args.epochs,
        'lr': args.lr,
        'scheduler': args.scheduler,
        'earlyStop': args.earlyStop,
        'evalMetric': 'mean_iou'
    },
    name=args.name,
)
config = wandb.config

# Load dataset
train, test = load_dataset('nameOfDataset', split=['train', 'test'])  # TODO: Set dataset name

id2label = {0: 'background', 1: 'object'}  # TODO: Set id2label
label2id = {v: k for k, v in id2label.items()}
numLabels = len(id2label)

# Preprocess dataset
preprocessor = SegformerImageProcessor()
train.set_transform(Transform(preprocessor, transform='None', isTrain=True))
test.set_transform(Transform(preprocessor, transform='None', isTrain=False))

# Define evaluation metrics
metric = evaluate.load(config.evalMetric)
