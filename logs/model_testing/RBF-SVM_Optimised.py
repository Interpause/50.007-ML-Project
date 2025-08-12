import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Import the required library
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore') # Suppress potential warnings for cleaner output

# --- 1. Load Data ---
print("Loading data...")
# Load training data
train_df = pd.read_csv('train_tfidf_features.csv')
X_train_full = train_df.drop(['id', 'label'], axis=1) # Features (columns 0-4999)
y_train_full = train_df['label']                       # Labels

# Load test data
test_df = pd.read_csv('test_tfidf_features.csv')
X_test = test_df.drop(['id'], axis=1) # Features (columns 0-4999)
test_ids = test_df['id']              # Test IDs for submission

print(f"Training set shape: {X_train_full.shape}, Labels shape: {y_train_full.shape}")
print(f"Test set shape: {X_test.shape}")

# --- 2. Define SVM Model and Hyperparameter Grid ---
print("\nDefining SVM model and hyperparameter grid...")
# Base SVM model (parameters that won't be searched are set here)
# probability=True is crucial for getting probability outputs needed for threshold tuning and ensembling
base_svm_model = SVC(
    kernel='rbf',         # Use RBF kernel as requested
    probability=True,     # Enable probability estimates
    random_state=42       # For reproducibility
    # Note: SVM does not natively support n_jobs for probability=True in many versions.
    # If parallel processing is needed and supported, it might be via external libraries or joblib.
)

# Define hyperparameter grid for tuning
# Note: SVM can be sensitive to hyperparameters. Start with a coarse grid.
# Consider using RandomizedSearchCV for larger grids or continuous parameters like gamma.
param_grid = {
    'C': [0.1, 1.0, 10.0],      # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01] # Kernel coefficient for 'rbf'
    # You can add more values like 'C': [0.01, 0.1, 1, 10, 100] and 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    # Be aware that SVM training can be slow, especially with large datasets and many hyperparameter combinations.
}

# --- 3. Perform Hyperparameter Tuning with Cross-Validation ---
print("\nPerforming hyperparameter tuning with GridSearchCV (Macro F1)...")
# Use StratifiedKFold for cross-validation
# SVM can be slow, so using fewer folds might be practical if needed.
cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV
# SVM's sklearn wrapper directly supports 'f1_macro'
grid_search = GridSearchCV(
    estimator=base_svm_model,
    param_grid=param_grid,
    scoring='f1_macro', # Optimize for Macro F1
    cv=cv_splitter,
    verbose=1,          # Show progress
    n_jobs=1            # Adjust based on system resources. -1 might not work well for SVM.
    # refit=True is default, which means the best estimator is trained on the full dataset
)

# Fit GridSearchCV
# Note: This step might take a considerable amount of time depending on the grid size and data.
grid_search.fit(X_train_full, y_train_full)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best Macro F1 Score (CV): {grid_search.best_score_:.4f}")

# --- 4. Get the Best Trained Model ---
# GridSearchCV automatically refits the best model on the entire training set
best_svm_model = grid_search.best_estimator_
print("\nBest SVM model selected.")

# --- 5. Find Optimal Threshold for Macro F1 Score (using training data) ---
# Although CV optimizes the model, we still need to find the best threshold
# for converting probabilities to class predictions.
print("\nFinding optimal threshold for Macro F1 score on training set...")
# Get predicted probabilities for the positive class (label 1) on the training set
# Using the best model which was refit on the full training set
train_proba = best_svm_model.predict_proba(X_train_full)[:, 1]

# Test thresholds from 0.3 to 0.7 in steps of 0.01
thresholds = np.arange(0.3, 0.71, 0.01)
f1_scores = []

for thresh in thresholds:
    # Convert probabilities to class predictions using current threshold
    train_preds = (train_proba >= thresh).astype(int)
    # Calculate Macro F1 score
    macro_f1 = f1_score(y_train_full, train_preds, average='macro')
    f1_scores.append(macro_f1)

# Find the threshold that gives the maximum Macro F1 score
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]
best_f1_score = f1_scores[optimal_threshold_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"Best Macro F1 Score on Training Set (with optimal threshold): {best_f1_score:.4f}")

# --- 6. Make Predictions on Test Set ---
print("\nMaking predictions on test set...")
# Get predicted probabilities for the positive class (label 1) on the test set
test_proba = best_svm_model.predict_proba(X_test)[:, 1]

# --- 7. Apply Optimal Threshold and Create Submission File ---
print(f"\nApplying optimal threshold ({optimal_threshold:.3f}) and creating submission file...")
# Convert test probabilities to class predictions using the OPTIMAL threshold
final_predictions = (test_proba >= optimal_threshold).astype(int)

print("\nCreating submission file: svm_tuned_submission.csv")
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': final_predictions
})
# Save to CSV without index
submission_df.to_csv('svm_tuned_submission.csv', index=False)
print("Submission file 'svm_tuned_submission.csv' created successfully!")

# --- 8. Optional: Compare with Default Threshold ---
# This shows the F1 score using the standard 0.5 threshold on the training set
default_preds = (train_proba > 0.5).astype(int)
default_f1 = f1_score(y_train_full, default_preds, average='macro')
print(f"\nFor comparison (Tuned SVM model):")
print(f"Macro F1 Score on Training Set (threshold=0.5): {default_f1:.4f}")
print(f"Macro F1 Score on Training Set (optimal threshold={optimal_threshold:.3f}): {best_f1_score:.4f}")
print(f"Best CV Macro F1 Score (before threshold tuning): {grid_search.best_score_:.4f}")

print("\nTuned SVM model execution completed.")
