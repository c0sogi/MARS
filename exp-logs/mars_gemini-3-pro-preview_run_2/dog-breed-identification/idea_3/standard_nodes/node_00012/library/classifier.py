import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from library.config import CACHE_DIR, SUBMISSION_PATH, MODELS, TRAIN_CSV
from library.utils import seed_everything


def get_class_names():
    """
    Retrieves the sorted list of class names from the training metadata.
    This ensures consistency with the DataLoader class mapping.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    classes = sorted(df["breed"].unique().tolist())
    return classes


def save_model(model, model_name, score, best_c):
    """
    Saves the LogisticRegression model weights and metadata using numpy.
    Avoids using pickle as per constraints.
    """
    save_path = os.path.join(CACHE_DIR, f"{model_name}_logreg.npz")
    np.savez(
        save_path,
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        score=np.array(score),
        best_c=np.array(best_c),
    )
    print(f"Model weights for {model_name} saved to {save_path}")


def load_model(model_name):
    """
    Loads a LogisticRegression model from numpy weights.
    Returns None if the file does not exist.
    """
    load_path = os.path.join(CACHE_DIR, f"{model_name}_logreg.npz")
    if not os.path.exists(load_path):
        return None

    data = np.load(load_path, allow_pickle=True)

    # Reconstruct the model
    # We initialize with the best C found, though for prediction only weights matter
    best_c = float(data["best_c"])
    model = LogisticRegression(
        C=best_c, multi_class="multinomial", solver="lbfgs", max_iter=1000
    )

    # Manually set attributes to bypass fitting
    model.coef_ = data["coef"]
    model.intercept_ = data["intercept"]
    model.classes_ = data["classes"]

    score = float(data["score"])
    print(f"Loaded {model_name} from cache (Val LogLoss: {score})")

    return model


def train_single_model(model_name, train_x, train_y, val_x, val_y, random_state=42):
    """
    Trains a Logistic Regression model with grid search for C.
    """
    print(f"\nTraining Classifier for: {model_name}")
    print(f"Train shape: {train_x.shape}, Val shape: {val_x.shape}")

    # Grid search for regularization parameter C
    # Smaller C = stronger regularization
    c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_loss = float("inf")
    best_model = None
    best_c = 1.0

    for c in c_candidates:
        clf = LogisticRegression(
            C=c,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,  # Increased max_iter for convergence
            random_state=random_state,
            n_jobs=-1,  # Use all cores
        )

        clf.fit(train_x, train_y)

        # Validate
        val_probs = clf.predict_proba(val_x)
        loss = log_loss(val_y, val_probs)

        print(f"  C={c}: Validation Log Loss = {loss}")

        if loss < best_loss:
            best_loss = loss
            best_model = clf
            best_c = c

    print(f"Best C for {model_name}: {best_c} with Log Loss: {best_loss}")

    return best_model, best_loss, best_c


def predict_model(model, test_x):
    """
    Generates probabilities for the test set.
    """
    return model.predict_proba(test_x)


def generate_submission(test_ids, ensemble_probs, class_names, output_path):
    """
    Creates and saves the submission CSV file.
    """
    # Create DataFrame
    df = pd.DataFrame(ensemble_probs, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", test_ids)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {df.shape}")

    # Display first few rows
    print("Head of submission:")
    print(df.head())


def run_classification_pipeline(features_dict, load_cached_models=True):
    """
    Orchestrates the training, prediction, and ensembling process.

    Args:
        features_dict (dict): Dictionary containing embeddings for 'train', 'val', 'test'
                              for each model key.
        load_cached_models (bool): Whether to try loading trained classifiers from disk.
    """
    seed_everything()

    class_names = get_class_names()
    num_classes = len(class_names)

    # We need to ensure test IDs are consistent across models (they should be)
    # We'll grab them from the first model processed
    test_ids = None

    # Accumulator for ensemble probabilities
    ensemble_probs_sum = None
    models_processed = 0

    for model_name in MODELS.keys():
        if model_name not in features_dict:
            print(f"Skipping {model_name} (embeddings not found in input dict)")
            continue

        print(f"Processing classification for {model_name}...")

        # Unpack data
        train_emb, train_lbl, _ = features_dict[model_name]["train"]
        val_emb, val_lbl, _ = features_dict[model_name]["val"]
        test_emb, _, current_test_ids = features_dict[model_name]["test"]

        if test_ids is None:
            test_ids = current_test_ids
        else:
            # Sanity check
            if not np.array_equal(test_ids, current_test_ids):
                raise ValueError(f"Test ID mismatch for model {model_name}")

        # Try load or train
        model = None
        if load_cached_models:
            model = load_model(model_name)

        if model is None:
            model, score, best_c = train_single_model(
                model_name, train_emb, train_lbl, val_emb, val_lbl
            )
            save_model(model, model_name, score, best_c)

        # Predict on test
        probs = predict_model(model, test_emb)

        # Initialize or accumulate
        if ensemble_probs_sum is None:
            ensemble_probs_sum = probs
        else:
            ensemble_probs_sum += probs

        models_processed += 1

    if models_processed == 0:
        raise RuntimeError("No models were processed. Check feature extraction.")

    # Average probabilities (Late Fusion)
    final_probs = ensemble_probs_sum / models_processed

    # Generate Submission
    generate_submission(test_ids, final_probs, class_names, SUBMISSION_PATH)
