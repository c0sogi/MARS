import os
import torch
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed


def train_model(
    epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    debug=Config.DEBUG,
):
    """
    Trains the GHG-CRCN model using the Trainer class.

    Args:
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Initializing Trainer (Device: {'cuda' if torch.cuda.is_available() else 'cpu'})..."
    )
    trainer = Trainer()

    print(f"Starting Training: Epochs={epochs}, Patience={patience}, Debug={debug}")
    trainer.fit(epochs=epochs, patience=patience, debug=debug)

    # Verify checkpoint exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(
            f"Best model found. Saved at epoch {checkpoint['epoch']} with validation score: {checkpoint['best_score']}"
        )
    else:
        print("Warning: No checkpoint found after training.")


def generate_submission(output_file="submission.csv", debug=Config.DEBUG):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        output_file (str): Name of the output CSV file.
        debug (bool): If True, predicts on a debug subset.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Initializing Trainer for Inference...")
    trainer = Trainer()

    print("Generating predictions for test set...")
    # Trainer.predict loads the best model automatically
    predictions = trainer.predict(split="test")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(Config.SUBMISSION_DIR, output_file)

    print(f"Writing submission to {output_path}...")
    try:
        with open(output_path, "w") as f:
            # Iterate over predictions.
            # predictions is a dict: {sample_id: [label1, label2, ...]}
            for sample_id, gesture_list in predictions.items():
                # Format: SessionID,Label1,Label2,Label3
                # Join labels with commas
                labels_str = ",".join(map(str, gesture_list))

                # Construct line
                if labels_str:
                    line = f"{sample_id},{labels_str}"
                else:
                    # Handle empty predictions (though unlikely with valid input)
                    line = f"{sample_id},"

                f.write(line + "\n")

        print("Submission generation complete.")

    except Exception as e:
        print(f"Error writing submission file: {e}")


def run_pipeline(
    epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    debug=Config.DEBUG,
):
    """
    Executes the full pipeline: Training followed by Inference.

    Args:
        epochs (int): Training epochs.
        patience (int): Early stopping patience.
        debug (bool): Debug mode flag.
    """
    print("=== Starting GHG-CRCN Pipeline ===")

    # Step 1: Train
    train_model(epochs=epochs, patience=patience, debug=debug)

    # Step 2: Inference
    generate_submission(debug=debug)

    print("=== Pipeline Complete ===")
