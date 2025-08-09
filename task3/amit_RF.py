# task3_random_forest_cv.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time


def load_data():
    """
    Load the TF-IDF feature files
    """
    print("Loading TF-IDF feature files...")

    # Load training data
    train_features = pd.read_csv('train_tfidf_features.csv')
    print(f"Training data shape: {train_features.shape}")

    # Load test data
    test_features = pd.read_csv('test_tfidf_features.csv')
    print(f"Test data shape: {test_features.shape}")

    # Extract feature columns (0 to 4999)
    feature_columns = [str(i) for i in range(0, 5000)]

    # Training data
    X_train = train_features[feature_columns].values
    y_train = train_features['label'].values
    train_ids = train_features['id'].values

    # Test data
    X_test = test_features[feature_columns].values
    test_ids = test_features['id'].values

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, test_ids


def apply_pca_500_components(X_train, X_test):
    """
    Apply PCA with 500 components
    """
    print("Applying PCA with 500 components...")

    pca = PCA(n_components=500, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance ratio: {explained_variance:.4f}")
    print(f"X_train_pca shape: {X_train_pca.shape}")
    print(f"X_test_pca shape: {X_test_pca.shape}")

    return X_train_pca, X_test_pca


def perform_cross_validation(X_train, y_train):
    """
    Perform cross-validation to estimate local accuracy
    """
    print("\n" + "=" * 50)
    print("PERFORMING CROSS-VALIDATION")
    print("=" * 50)

    # Create the same model configuration used for final training
    rf_model_cv = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Use StratifiedKFold to ensure each fold has a representative sample of each class
    # 5-fold CV is a good balance between computation time and reliable estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running 5-fold Cross-Validation...")
    start_time = time.time()

    # Perform cross-validation
    cv_scores = cross_val_score(
        rf_model_cv,
        X_train,
        y_train,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    cv_time = time.time() - start_time

    # Print results
    print(f"Cross-Validation completed in {cv_time:.2f} seconds")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Min CV Accuracy: {cv_scores.min():.4f}")
    print(f"Max CV Accuracy: {cv_scores.max():.4f}")

    return cv_scores.mean(), cv_scores.std()


def train_random_forest_model(X_train, y_train):
    """
    Train Random Forest model
    """
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 50)

    # Random Forest parameters (same as used in CV)
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    print("Training Random Forest model...")
    start_time = time.time()

    # Create and train Random Forest classifier
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    return rf_model


def make_final_predictions(model, X_test, test_ids, model_name, cv_mean=None, cv_std=None):
    """
    Make final predictions and save submission file
    """
    print(f"\nMaking predictions with {model_name}...")
    y_pred = model.predict(X_test)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': y_pred
    })

    # Ensure proper column order and data types
    submission_df = submission_df[['id', 'label']]
    submission_df['id'] = submission_df['id'].astype(int)
    submission_df['label'] = submission_df['label'].astype(int)

    # Save submission file
    filename = f"{model_name.replace(' ', '_')}_predictions.csv"
    submission_df.to_csv(filename, index=False)
    print(f"Saved {filename}")
    print(f"File shape: {submission_df.shape}")
    print("First 10 predictions:")
    print(submission_df.head(10))

    # If CV results are available, print them in the summary
    if cv_mean is not None and cv_std is not None:
        print(f"\nEstimated Local Accuracy (from CV): {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    return filename


def run_task3_random_forest():
    """
    Main function to run Task 3 with Random Forest
    """
    # Load data
    X_train, X_test, y_train, test_ids = load_data()

    # Apply PCA with 500 components
    X_train_pca, X_test_pca = apply_pca_500_components(X_train, X_test)

    # Perform Cross-Validation on the PCA-transformed training data
    cv_mean, cv_std = perform_cross_validation(X_train_pca, y_train)

    # Train Random Forest model
    rf_model = train_random_forest_model(X_train_pca, y_train)

    # Make final predictions
    final_filename = make_final_predictions(
        rf_model, X_test_pca, test_ids, "RandomForest_PCA_500_again", cv_mean, cv_std
    )

    print("\n" + "=" * 50)
    print("TASK 3 RANDOM FOREST COMPLETE!")
    print("=" * 50)
    print(f"Estimated Local Accuracy (from CV): {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print(f"Generated submission file: {final_filename}")
    print("\nNext steps:")
    print("1. Upload the CSV file to Kaggle")
    print("2. Compare performance with SVM and XGBoost")
    print("3. Try other models if needed")


# Run the analysis
if __name__ == "__main__":
    run_task3_random_forest()