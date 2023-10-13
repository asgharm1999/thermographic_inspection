# Import Libraries
import wandb
import argparse
from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import evaluate

from utils import Transform, ComputeMetrics

# Parse Arguments
parser = argparse.ArgumentParser(description="Train Segformer")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--batchSize", type=int, default=6, help="Batch size")
parser.add_argument("--earlyStop", type=int, default=10, help="Early stopping patience")
parser.add_argument("--name", type=str, default="Test", help="Name of the run")
parser.add_argument("--scheduler", type=str, default="polynomial", help="Learning rate scheduler")
parser.add_argument("--modelName", type=str, default="nvidia/mit-b0", help="Model name")
parser.add_argument("--evalMetric", type=str, default="mean_iou", help="Evaluation metric")

args = parser.parse_args()

# Initialize WandB
wandb.init(
    project="test",  # TODO: Set project name
    config={
        "modelName": args.modelName,
        "batchSize": args.batchSize,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "earlyStop": args.earlyStop,
        "evalMetric": args.evalMetric,
    },
    name=args.name,
)
config = wandb.config

# Load dataset
train, test = load_dataset("ChristopherS27/bridgeSeg", split=["originalTrain", "originalTest"])  # TODO: Set dataset name

id2label = {0: "good", 1: "fair", 2: "poor", 3: "severe"}
label2id = {v: k for k, v in id2label.items()}
numLabels = len(id2label)

# Preprocess dataset
preprocessor = SegformerImageProcessor(do_reduce_labels=True)
train.set_transform(Transform(preprocessor, transform="none", isTrain=True))
test.set_transform(Transform(preprocessor, transform="none", isTrain=False))

# Define evaluation metrics
metric = evaluate.load(config.evalMetric)
metricComputer = ComputeMetrics(metric, id2label, numLabels)

# Define model
model = SegformerForSemanticSegmentation.from_pretrained(
    config.modelName, num_labels=numLabels, id2label=id2label, label2id=label2id
)
wandb.watch(model)

# Set up training arguments
trainArgs = TrainingArguments(
    output_dir="runs/" + args.name,
    learning_rate=config.lr,
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batchSize,
    per_device_eval_batch_size=config.batchSize,
    save_total_limit=1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    lr_scheduler_type=config.scheduler,
    optim="adamw_torch",
)

# Set up callbacks
earlyStop = EarlyStoppingCallback(early_stopping_patience=config.earlyStop)

# Train model
trainer = Trainer(
    model=model,
    args=trainArgs,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=metricComputer.computeMetrics,
    callbacks=[earlyStop],
)
trainer.train()

# End
wandb.finish()
