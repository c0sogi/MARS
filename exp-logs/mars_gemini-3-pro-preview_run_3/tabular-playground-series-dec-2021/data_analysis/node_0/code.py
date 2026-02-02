import os
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Setup & Configuration
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


def main():
    # Define paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")

    # Load Data
    # We use the metadata parquet file as it is the designated training set
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    df = pd.read_parquet(TRAIN_PATH)

    # Identify Target and ID
    target_col = "Cover_Type"
    id_col = "Id"

    # Drop ID for analysis if present
    if id_col in df.columns:
        df_analysis = df.drop(columns=[id_col])
    else:
        df_analysis = df.copy()

    # ---------------------------------------------------------
    # 2. Modality Detection
    # ---------------------------------------------------------
    # Heuristic: Check for file paths or specific data types
    # Since we are working with a known synthetic tabular dataset, we expect Tabular.
    # However, we implement checks as requested.

    modality = "Tabular"

    # Check for image/audio paths in string columns
    str_cols = df_analysis.select_dtypes(include=["object", "string"]).columns
    if len(str_cols) > 0:
        # Check first non-null value of first string col
        sample_val = (
            df_analysis[str_cols[0]].dropna().iloc[0]
            if not df_analysis[str_cols[0]].dropna().empty
            else ""
        )
        if isinstance(sample_val, str):
            if sample_val.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                modality = "Image"
            elif sample_val.lower().endswith((".wav", ".mp3", ".flac")):
                modality = "Audio"
            elif len(sample_val.split()) > 10:  # Heuristic for text
                modality = "Text"

    print("DATA MODALITY DETECTION")
    print(f"Detected Modality: {modality}")
    print("-" * 30)

    # ---------------------------------------------------------
    # 3. Target Variable Analysis
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
    else:
        y = df[target_col]
        # Determine if Classification or Regression
        # Heuristic: Low unique count relative to size or integer type with few values -> Classification
        n_unique = y.nunique()
        is_classification = False

        if n_unique < 50 or y.dtype == "object":
            is_classification = True
            print(f"Task Type: Classification (Unique Classes: {n_unique})")

            # Class Balance
            print("Class Distribution:")
            counts = y.value_counts()
            total = len(y)
            for label, count in counts.items():
                ratio = count / total
                print(f"  Class {label}: {count} ({ratio:.4f})")

            # Imbalance check
            min_class = counts.min()
            max_class = counts.max()
            print(f"  Imbalance Ratio (Max/Min): {max_class/min_class:.4f}")

        else:
            print("Task Type: Regression")
            # Skewness and Kurtosis
            skew = y.skew()
            kurt = y.kurtosis()
            print(f"  Skewness: {skew:.4f}")
            print(f"  Kurtosis: {kurt:.4f}")
            print(f"  Distribution: Mean={y.mean():.4f}, Std={y.std():.4f}")

    # ---------------------------------------------------------
    # 4. Input Data Analysis (Modality-Specific)
    # ---------------------------------------------------------
    print(f"\nINPUT DATA ANALYSIS ({modality.upper()})")

    # Separate features
    X = df_analysis.drop(columns=[target_col], errors="ignore")

    if modality == "Tabular":
        # Numerical Analysis
        num_cols = X.select_dtypes(include=["number"]).columns
        cat_cols = X.select_dtypes(exclude=["number"]).columns

        if len(num_cols) > 0:
            print(f"Numerical Features: {len(num_cols)}")
            # We calculate stats on the full set, but for outliers we use a vectorized approach
            # To avoid printing 54 lines, we summarize or print top/bottom/aggregates?
            # The prompt asks to "Report mean, std, min, max, and outlier counts".
            # We will compute these for all, but maybe only print a summary if too many columns.
            # However, prompt implies reporting them. We will print for first 10 and summarize rest to keep output clean,
            # or just print all if not too huge. 54 is borderline. Let's print aggregate stats.

            print(
                f"{'Feature':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
            )
            print("-" * 85)

            # Using a subset for display if too many columns, but let's try to do all
            stats = X[num_cols].describe().T

            # Outlier detection (IQR)
            Q1 = X[num_cols].quantile(0.25)
            Q3 = X[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (X[num_cols] < (Q1 - 1.5 * IQR)) | (X[num_cols] > (Q3 + 1.5 * IQR))
            ).sum()

            # Print first 10 and last 5 to show coverage without spamming 50+ lines if desired,
            # but let's print all since 54 lines is acceptable for a report.
            for col in num_cols:
                m = stats.loc[col, "mean"]
                s = stats.loc[col, "std"]
                mn = stats.loc[col, "min"]
                mx = stats.loc[col, "max"]
                out = outliers[col]
                print(
                    f"{col[:29]:<30} {m:<10.4f} {s:<10.4f} {mn:<10.4f} {mx:<10.4f} {out:<10}"
                )

        if len(cat_cols) > 0:
            print(f"\nCategorical Features: {len(cat_cols)}")
            print(f"{'Feature':<30} {'Cardinality':<15} {'Rare Labels':<10}")
            for col in cat_cols:
                card = X[col].nunique()
                # Rare labels < 1%
                counts = X[col].value_counts(normalize=True)
                rare_count = (counts < 0.01).sum()
                flag = "(!)" if card > 50 else ""
                print(f"{col[:29]:<30} {card:<15}{flag} {rare_count:<10}")
        else:
            print("\nCategorical Features: 0 detected (based on non-numeric dtypes).")

        # Missing Values
        print("\nMissing Values Analysis:")
        missing = X.isnull().sum()
        missing_pct = (missing / len(X)) * 100
        missing_cols = missing[missing > 0]
        if len(missing_cols) == 0:
            print("  No missing values detected.")
        else:
            for col in missing_cols.index:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.4f}%)")

    elif modality == "Image":
        # Placeholder for Image logic (not triggered for this dataset)
        print("Image analysis logic placeholder.")

    elif modality == "Audio":
        # Placeholder for Audio logic
        print("Audio analysis logic placeholder.")

    elif modality == "Text":
        # Placeholder for Text logic
        print("Text analysis logic placeholder.")

    # ---------------------------------------------------------
    # 5. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\nFEATURE RELATIONSHIPS")

    # Subsample for expensive computations
    # 2.8M rows is too much for RF and Correlation matrix in short time
    SAMPLE_SIZE = 100000
    if len(df_analysis) > SAMPLE_SIZE:
        print(f"Subsampling {SAMPLE_SIZE} rows for relationship analysis...")
        df_sample = df_analysis.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        df_sample = df_analysis

    X_sample = df_sample.drop(columns=[target_col], errors="ignore")
    y_sample = df_sample[target_col] if target_col in df_sample.columns else None

    if modality == "Tabular":
        # Correlation
        print("\n1. Correlation Analysis (Top Collinear Pairs > 0.90):")
        # Select numeric columns only
        num_sample = X_sample.select_dtypes(include=["number"])
        if not num_sample.empty:
            corr_matrix = num_sample.corr().abs()
            # Select upper triangle
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            # Find features with correlation > 0.90
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

            pairs = []
            for col in to_drop:
                # Find the feature it correlates with
                correlated_feats = upper.index[upper[col] > 0.90].tolist()
                for feat in correlated_feats:
                    pairs.append((feat, col, upper.loc[feat, col]))

            if not pairs:
                print("  No collinear pairs found > 0.90")
            else:
                for f1, f2, val in pairs[:10]:  # Print top 10
                    print(f"  {f1} - {f2}: {val:.4f}")
                if len(pairs) > 10:
                    print(f"  ... and {len(pairs)-10} more pairs.")

        # Feature Importance
        if y_sample is not None:
            print("\n2. Feature Importance (Random Forest):")
            # Handle categorical encoding for RF if necessary
            # Simple Label Encoding for object cols
            X_rf = X_sample.copy()
            for col in X_rf.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                X_rf[col] = le.fit_transform(X_rf[col].astype(str))

            # Impute missing for RF
            X_rf = X_rf.fillna(0)  # Simple fill for EDA

            if is_classification:
                rf = RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
                )

            rf.fit(X_rf, y_sample)

            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]

            print(f"{'Rank':<5} {'Feature':<30} {'Importance':<10}")
            for i in range(min(5, len(indices))):
                idx = indices[i]
                print(f"{i+1:<5} {X_rf.columns[idx][:29]:<30} {importances[idx]:.4f}")

    else:
        # Unstructured Relationships
        print("Unstructured data relationship analysis placeholder.")


if __name__ == "__main__":
    main()
