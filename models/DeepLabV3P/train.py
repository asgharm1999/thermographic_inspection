# Import Libraries
import tensorflow as tf
import wandb
import argparse

from deeplabv3plus import DeepLabV3Plus
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# Set up argument parsing
parser = argparse.ArgumentParser(description='Train DeepLabV3P')
parser.add_argument('--batchSize', type=int, default=6, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--scheduler', type=str, default='polynomial', help='Learning rate scheduler')
parser.add_arugment('--earlyStop', type=int, default=10, help='Early stopping patience')
parser.add_argument('--name', type=str, default='Test', help='Name of the run')

args = parser.parse_args()

# Enable multiple GPUs
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Initialize WandB
    wandb.init(
        project="projectName",
        config={
            "batchSize": args.batchSize,
            "epochs": args.epochs,
            "lr": args.lr,
            "scheduler": args.scheduler,
            "earlyStop": args.earlyStop,
        },
        name=args.name,
    )
    config = wandb.config

    # Load dataset
    # TODO: Load dataset

    # Load model
    model = DeepLabV3Plus((512, 512, 3), 1)  # TODO: Change input shape and number of classes

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.earlyStop, restore_best_weights=True
        ),
        WandbMetricsLogger(),
        WandbModelCheckpoint(
            monitor="val_loss", save_weights_only=True, save_best_only=True
        ),
    ]

    # Train model
    # TODO: Define once dataset is done

    wandb.finish()
