import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from library.config import *
from library.data_loader import get_regression_data, get_inference_data
from library.metrics import score_dataset


class RankPredictor:
    """
    A regression model that predicts the normalized rank of a markdown cell
    based on its text content using TF-IDF features.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=VOCAB_SIZE,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)

    def fit(self, df_train):
        """
        Fits the vectorizer and regressor on the training data.

        Args:
            df_train (pd.DataFrame): DataFrame containing 'text' and 'rank' columns.
        """
        # Ensure text is string and handle NaNs
        texts = df_train["text"].fillna("").astype(str).tolist()
        ranks = df_train["rank"].values

        # Fit vectorizer and transform text
        X = self.vectorizer.fit_transform(texts)

        # Train regressor
        self.model.fit(X, ranks)
        return self

    def predict(self, texts):
        """
        Predicts normalized ranks for a list of text strings.

        Args:
            texts (list): List of markdown text strings.

        Returns:
            np.array: Predicted ranks.
        """
        # Ensure input is list of strings
        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        X = self.vectorizer.transform(cleaned_texts)
        return self.model.predict(X)


def predict_notebook_order(model, notebook_data):
    """
    Generates the cell order for a single notebook using the Interleaving Sort strategy.

    Args:
        model (RankPredictor): Trained model instance.
        notebook_data (dict): Dictionary containing 'code_cells' (list of ids)
                              and 'markdown_cells' (list of (id, text) tuples).

    Returns:
        str: Space-delimited string of ordered cell IDs.
    """
    code_cells = notebook_data["code_cells"]
    md_cells = notebook_data["markdown_cells"]

    cells_with_ranks = []

    # 1. Assign ranks to code cells based on their fixed relative order
    # Formula: rank = i / N_code
    n_code = len(code_cells)
    for i, cell_id in enumerate(code_cells):
        rank = i / n_code if n_code > 0 else 0.0
        cells_with_ranks.append((cell_id, rank))

    # 2. Predict ranks for markdown cells
    if md_cells:
        md_ids = [c[0] for c in md_cells]
        md_texts = [c[1] for c in md_cells]

        pred_ranks = model.predict(md_texts)

        for cell_id, rank in zip(md_ids, pred_ranks):
            cells_with_ranks.append((cell_id, rank))

    # 3. Global Sort
    cells_with_ranks.sort(key=lambda x: x[1])

    # 4. Extract IDs
    sorted_ids = [x[0] for x in cells_with_ranks]
    return " ".join(sorted_ids)


def run_pipeline(
    max_train_samples=MAX_TRAIN_SAMPLES, max_test_samples=MAX_TEST_SAMPLES
):
    """
    Executes the full training, validation, and submission pipeline.
    """
    set_seed(RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    print("Loading training data...")
    df_train = get_regression_data(data_type="train", max_samples=max_train_samples)

    print(f"Training model on {len(df_train)} markdown cells...")
    model = RankPredictor()
    model.fit(df_train)

    # -------------------------------------------------------------------------
    # 2. Validation
    # -------------------------------------------------------------------------
    print("Loading validation data...")
    # Load validation notebooks in inference format
    val_notebooks = get_inference_data(data_type="val", max_samples=max_test_samples)

    print(f"Predicting on {len(val_notebooks)} validation notebooks...")
    val_preds = []
    for nb in val_notebooks:
        order_str = predict_notebook_order(model, nb)
        val_preds.append({"id": nb["id"], "cell_order": order_str})

    df_val_pred = pd.DataFrame(val_preds)

    # Load Ground Truth for validation
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Ensure we only score the notebooks we predicted (in case of max_samples)
    val_ids = set(df_val_pred["id"])
    df_val_meta = df_val_meta[df_val_meta["id"].isin(val_ids)]

    # Compute Metric
    score = score_dataset(df_val_meta, df_val_pred)
    print(f"Validation Kendall Tau: {score}")

    # -------------------------------------------------------------------------
    # 3. Submission
    # -------------------------------------------------------------------------
    print("Loading test data...")
    test_notebooks = get_inference_data(data_type="test", max_samples=max_test_samples)

    print(f"Generating submission for {len(test_notebooks)} notebooks...")
    test_preds = []
    for nb in test_notebooks:
        order_str = predict_notebook_order(model, nb)
        test_preds.append({"id": nb["id"], "cell_order": order_str})

    df_submission = pd.DataFrame(test_preds)

    # Save submission
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
