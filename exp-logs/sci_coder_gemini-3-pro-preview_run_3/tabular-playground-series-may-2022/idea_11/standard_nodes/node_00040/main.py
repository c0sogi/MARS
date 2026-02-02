import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.data_processing import get_dataloaders
from library.model import LayerNormFunnelMLP, train_model, predict


def main():
    # 1. Setup Configuration and Seeds
    Config.setup()

    # Override Config for Fast Baseline
    # We use a sufficient number of epochs to allow the One-Cycle schedule to fully anneal
    # and reach the "Super-Convergence" optimum (Cite solution_lesson_node_00002).
    FAST_EPOCHS = 30

    print(f"Initializing Fast Baseline Run. Device: {Config.DEVICE}")

    # 2. Data Loading
    # Load cached data if available to save processing time
    train_loader, val_loader, test_loader, vocab_sizes = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    cont_dim = len(Config.CONTINUOUS_COLS)

    model = LayerNormFunnelMLP(
        vocab_sizes=vocab_sizes,
        cont_dim=cont_dim,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        token_dropout_rate=Config.TOKEN_DROPOUT_RATE,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Training
    # Train the model using the provided library function
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=FAST_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=Config.DEVICE,
    )

    # 5. Validation & Metric Assessment
    print("Performing final validation assessment...")
    model.eval()
    model.to(Config.DEVICE)

    all_preds = []
    all_targets = []

    # Lists to store features for failure analysis
    all_cats = []
    all_conts = []

    with torch.no_grad():
        for batch_cat, batch_cont, batch_y in val_loader:
            batch_cat = batch_cat.to(Config.DEVICE)
            batch_cont = batch_cont.to(Config.DEVICE)

            # Forward pass
            logits = model(batch_cat, batch_cont)
            probs = torch.sigmoid(logits)

            # Store data for metric and analysis
            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

            # Store features (move to CPU numpy for analysis)
            all_cats.append(batch_cat.cpu().numpy())
            all_conts.append(batch_cont.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets).ravel()
    y_pred = np.concatenate(all_preds).ravel()
    X_cat = np.concatenate(all_cats, axis=0)
    X_cont = np.concatenate(all_conts, axis=0)

    # Compute Metric
    val_auc = roc_auc_score(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Construct a DataFrame of features to correlate with error
    # Categorical columns
    cat_cols = Config.CATEGORICAL_COLS + [
        f"f_27_{i}" for i in range(Config.F_27_LENGTH)
    ]
    # Continuous columns
    cont_cols = Config.CONTINUOUS_COLS

    # Create feature dataframe
    # Note: X_cat shape is (N, num_cat), X_cont is (N, num_cont)
    df_analysis = pd.DataFrame(X_cat, columns=cat_cols)
    df_cont = pd.DataFrame(X_cont, columns=cont_cols)
    df_analysis = pd.concat([df_analysis, df_cont], axis=1)

    # Add error column
    df_analysis["error_magnitude"] = errors

    # Calculate correlation
    correlations = (
        df_analysis.corrwith(df_analysis["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features Correlated with Error Magnitude:")
    print(correlations.head(6).iloc[1:])  # Skip self-correlation with error_magnitude

    # 7. Conditional Submission
    THRESHOLD = 0.9971550270448856

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions
        submission_df = predict(model, test_loader, device=Config.DEVICE)

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {val_auc} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
