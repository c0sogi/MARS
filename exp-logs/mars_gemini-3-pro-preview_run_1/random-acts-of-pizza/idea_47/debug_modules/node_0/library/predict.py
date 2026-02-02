import os
import pandas as pd
import numpy as np
import torch

from library.config import (
    OUTPUT_SUBMISSION_PATH,
    TEST_PATH,
    DEVICE,
    ENSEMBLE_WEIGHT_RF,
    ENSEMBLE_WEIGHT_MLP,
    MLP_BATCH_SIZE,
)
from library.utils import seed_everything
from library.train import train_rf, train_mlp
from library.feature_engineering import FeaturePipeline
from library.dataset import create_dataloaders


def generate_rf_predictions(model, X_test):
    """
    Generates predictions using the Random Forest model.

    Args:
        model: Trained InteractionRandomForest model.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    print("Generating Random Forest predictions...")
    probs = model.predict_proba(X_test)
    # Return probability of class 1
    return probs[:, 1]


def generate_mlp_predictions(model, test_loader):
    """
    Generates predictions using the MLP model.

    Args:
        model: Trained PizzaFiLMMLP model.
        test_loader: DataLoader for the test set.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    print("Generating MLP predictions...")
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            title = batch["title_emb"].to(DEVICE)
            body = batch["body_emb"].to(DEVICE)
            hist = batch["history_emb"].to(DEVICE)
            mask = batch["history_mask"].to(DEVICE)
            cent = batch["centroid_emb"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)

            logits = model(title, body, hist, mask, cent, meta)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)

    return np.array(preds).flatten()


def ensemble_predictions(rf_probs, mlp_probs):
    """
    Combines predictions from RF and MLP using weighted averaging.

    Args:
        rf_probs (np.ndarray): RF probabilities.
        mlp_probs (np.ndarray): MLP probabilities.

    Returns:
        np.ndarray: Ensembled probabilities.
    """
    print(
        f"Ensembling predictions (RF: {ENSEMBLE_WEIGHT_RF}, MLP: {ENSEMBLE_WEIGHT_MLP})..."
    )
    ensemble_probs = (rf_probs * ENSEMBLE_WEIGHT_RF) + (mlp_probs * ENSEMBLE_WEIGHT_MLP)
    return ensemble_probs


def run_prediction(load_cached_data=True):
    """
    Main function to run the prediction pipeline.

    Args:
        load_cached_data (bool): Whether to use cached features and models.
    """
    seed_everything()

    # 1. Load Features and Data
    print("Loading features...")
    pipeline = FeaturePipeline()
    rf_out, _ = pipeline.run(load_cached_data=load_cached_data)
    X_test_rf = rf_out["test_X"]

    # 2. Random Forest Stream
    # train_rf handles loading the model from cache if it exists, or training it
    rf_model = train_rf(load_cached_data=load_cached_data)
    rf_probs = generate_rf_predictions(rf_model, X_test_rf)

    # 3. MLP Stream
    # train_mlp handles loading the model from cache if it exists, or training it
    mlp_model = train_mlp(load_cached_data=load_cached_data, batch_size=MLP_BATCH_SIZE)

    # Create test loader for MLP inference
    _, _, test_loader = create_dataloaders(
        load_cached_data=load_cached_data, batch_size=MLP_BATCH_SIZE
    )
    mlp_probs = generate_mlp_predictions(mlp_model, test_loader)

    # 4. Ensemble
    final_probs = ensemble_predictions(rf_probs, mlp_probs)

    # 5. Generate Submission File
    print("Generating submission file...")
    # Load test metadata to get request_ids in the correct order
    test_df = pd.read_csv(TEST_PATH)

    submission_df = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": final_probs}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"Submission saved to {OUTPUT_SUBMISSION_PATH}")

    # Print head of submission for verification
    print("\nSubmission Head:")
    print(submission_df.head())
