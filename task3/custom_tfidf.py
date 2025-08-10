import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def clean_text(text):
    """Basic text cleaning function (you may need to adjust based on your data)"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and digits (keeping spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_custom_tfidf_features():
    """Extract TF-IDF features using your friend's optimized configuration"""

    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Identify text column (assuming it's the column that's not 'id' or 'label')
    text_col = [c for c in train_df.columns if c not in ("id", "label")][0]
    print(f"Using text column: {text_col}")

    # Prepare data with text cleaning
    print("Cleaning text...")
    X_train_raw = train_df[text_col].apply(clean_text)
    X_test_raw = test_df[text_col].apply(clean_text)
    y_train = train_df["label"].values

    print("Building TF-IDF vectorizer with optimized configuration...")
    # Extract the exact TF-IDF configuration from your friend's code
    tfidf_vectorizer = TfidfVectorizer(
        max_features=15000,  # More features for better coverage
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=2,  # Ignore very rare terms
        max_df=0.95,  # Ignore very common terms
        stop_words="english",  # Remove English stop words
        sublinear_tf=True,  # Apply sublinear TF scaling
        strip_accents="unicode",  # Handle unicode characters
    )

    # Fit on training data and transform both train and test
    print("Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)

    print("Transforming test data...")
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)

    print(f"TF-IDF shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")

    # Apply feature selection using chi2 (same as in your friend's pipeline)
    print("Applying feature selection (SelectKBest with chi2)...")
    feature_selector = SelectKBest(score_func=chi2, k=5000)  # Select top 5000 features

    X_train_selected = feature_selector.fit_transform(
        X_train_tfidf, y_train.astype(int)
    )
    X_test_selected = feature_selector.transform(X_test_tfidf)

    print(
        f"After feature selection - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}"
    )

    # Convert sparse matrices to dense for saving
    print("Converting to dense matrices...")
    # SelectKBest typically returns sparse matrices, so convert them
    try:
        X_train_dense = X_train_selected.toarray()
    except AttributeError:
        X_train_dense = np.asarray(X_train_selected)

    try:
        X_test_dense = X_test_selected.toarray()
    except AttributeError:
        X_test_dense = np.asarray(X_test_selected)

    # Create feature column names
    feature_names = [f"tfidf_feature_{i}" for i in range(X_train_dense.shape[1])]

    # Create DataFrames with proper indexing
    print("Creating train DataFrame...")
    train_tfidf_df = pd.DataFrame(X_train_dense, columns=feature_names)
    train_tfidf_df["id"] = train_df["id"].values
    train_tfidf_df["label"] = train_df["label"].values
    train_tfidf_df = train_tfidf_df.set_index("id")

    print("Creating test DataFrame...")
    test_tfidf_df = pd.DataFrame(X_test_dense, columns=feature_names)
    test_tfidf_df["id"] = test_df["id"].values
    test_tfidf_df = test_tfidf_df.set_index("id")

    # Save to CSV files
    print("Saving train TF-IDF features...")
    train_tfidf_df.to_csv("train_tfidf_features_custom.csv")

    print("Saving test TF-IDF features...")
    test_tfidf_df.to_csv("test_tfidf_features_custom.csv")

    # Print some statistics
    print("\nTF-IDF Feature Extraction Complete!")
    print("Training features saved to: train_tfidf_features_custom.csv")
    print("Test features saved to: test_tfidf_features_custom.csv")
    print(f"Number of features: {len(feature_names)}")
    print(f"Training samples: {len(train_tfidf_df)}")
    print(f"Test samples: {len(test_tfidf_df)}")

    # Print vocabulary size and some sample features
    print(f"\nOriginal TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print("Sample feature names from original vocabulary:")
    vocab_items = list(tfidf_vectorizer.vocabulary_.items())[:10]
    for word, idx in vocab_items:
        print(f"  '{word}': index {idx}")

    return train_tfidf_df, test_tfidf_df, tfidf_vectorizer, feature_selector


if __name__ == "__main__":
    # Run the feature extraction
    train_features, test_features, vectorizer, selector = (
        extract_custom_tfidf_features()
    )
