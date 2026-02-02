import os
from library.utils import seed_everything
from library.model import run_training, generate_submission


def train(
    epochs=15,
    batch_size=32,
    lr=1e-4,
    patience=5,
    save_path="./working/best_model.pth",
    debug=False,
):
    """
    Orchestrates the training of the VAMGNet model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training and validation.
        lr (float): Learning rate for the optimizer.
        patience (int): Patience for early stopping.
        save_path (str): File path to save the best model checkpoint.
        debug (bool): If True, reduces epochs to facilitate quick debugging.

    Returns:
        float: The best validation AUC score achieved.
    """
    seed_everything(42)

    # Handle debugging constraints
    if debug:
        print("Debug mode enabled: Limiting training to 2 epochs.")
        epochs = 2

    # Ensure the working directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Delegate to the library's training implementation
    best_auc = run_training(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        save_path=save_path,
    )

    return best_auc


def inference(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=32,
):
    """
    Generates predictions for the test set and creates a submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
    """
    seed_everything(42)

    # Delegate to the library's submission generation implementation
    generate_submission(
        model_path=model_path, output_path=output_path, batch_size=batch_size
    )
