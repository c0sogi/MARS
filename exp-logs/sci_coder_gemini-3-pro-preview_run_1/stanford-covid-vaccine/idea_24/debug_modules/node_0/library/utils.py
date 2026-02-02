import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The MCRMSE score.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate MSE per column
    # y_true and y_pred are expected to be of shape (N, 3) where N is total masked positions
    mse = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE per column
    rmse = np.sqrt(mse)

    # Calculate Mean of RMSEs
    score = np.mean(rmse)

    return float(score)


def format_submission(preds, test_ids, output_path=Config.SUBMISSION_FILE):
    """
    Formats the predictions into the required submission CSV format.

    Args:
        preds: Numpy array of shape (num_samples, seq_scored, 3) containing predictions.
               Channels are expected to be [reactivity, deg_Mg_pH10, deg_Mg_50C].
        test_ids: List or array of sample IDs corresponding to the predictions.
        output_path: Path to save the submission CSV.
    """
    # Submission requirements
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Total length per sample: 107
    # Scored length: 68 (provided in preds)

    num_samples = len(test_ids)
    full_seq_len = Config.SEQ_LEN  # 107
    scored_len = preds.shape[1]  # Should be 68

    # Initialize full prediction matrix for all columns (5 targets)
    # Columns order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Indices:     0           1            2         3           4
    final_preds = np.zeros((num_samples, full_seq_len, 5), dtype=np.float32)

    # Map model predictions to the correct columns
    # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # Map to:        [0,          1,           3         ]

    # Fill reactivity
    final_preds[:, :scored_len, 0] = preds[:, :, 0]
    # Fill deg_Mg_pH10
    final_preds[:, :scored_len, 1] = preds[:, :, 1]
    # Fill deg_Mg_50C
    final_preds[:, :scored_len, 3] = preds[:, :, 2]

    # Reshape for dataframe construction: (num_samples * seq_len, 5)
    flat_preds = final_preds.reshape(-1, 5)

    # Generate id_seqpos column
    id_seqpos_list = []
    for sample_id in test_ids:
        for i in range(full_seq_len):
            id_seqpos_list.append(f"{sample_id}_{i}")

    # Create DataFrame
    submission_df = pd.DataFrame(
        flat_preds,
        columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    )
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
