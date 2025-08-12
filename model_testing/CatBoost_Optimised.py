import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Import the required boosting library
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore') # Suppress potential warnings for cleaner output

# --- Custom Scorer for Macro F1 ---
# Define the scoring function for GridSearchCV
macro_f1_scorer = make_scorer(f1_score, average='macro')

# --- 1. Load Data ---
print("Loading data...")
# Load training data
train_df = pd.read_csv('./data/train_tfidf_features.csv')
X_train_full = train_df.drop(['id', 'label'], axis=1) # Features (columns 0-4999)
y_train_full = train_df['label']                       # Labels

# Load test data
test_df = pd.read_csv('./data/test_tfidf_features.csv')
X_test = test_df.drop(['id'], axis=1) # Features (columns 0-4999)
test_ids = test_df['id']              # Test IDs for submission

print(f"Training set shape: {X_train_full.shape}, Labels shape: {y_train_full.shape}")
print(f"Test set shape: {X_test.shape}")

# --- 2. Define CatBoost Model and Hyperparameter Grid ---
print("\nDefining CatBoost model and hyperparameter grid...")
# Base CatBoost model (parameters that won't be searched are set here)
base_cat_model = CatBoostClassifier(
    verbose=0,            # Suppress output during training
    random_state=42,      # For reproducibility
    thread_count=-1,      # Use all available CPU cores
    # Consider iterations as part of tuning or set a reasonable fixed value
    # If tuning iterations, be careful as it can be time-consuming
    # iterations=1000 # Example fixed value
)

# Define hyperparameter grid for tuning
# Note: Using a smaller grid or fewer options can speed up the search
param_grid = {
    'iterations': [800, 1000], # Number of trees
    'depth': [5, 6],          # Depth of trees
    'learning_rate': [0.08, 0.1], # Step size shrinkage
    'subsample': [0.7, 0.8],     # Fraction of samples for stochastic gradient boosting
    # 'l2_leaf_reg': [1, 3, 5] # Example of another parameter to tune
}

# --- 3. Perform Hyperparameter Tuning with Cross-Validation ---
print("\nPerforming hyperparameter tuning with GridSearchCV (Macro F1)...")
# Use StratifiedKFold for cross-validation
cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced folds for speed

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=base_cat_model,
    param_grid=param_grid,
    scoring=macro_f1_scorer, # Optimize for Macro F1
    cv=cv_splitter,
    verbose=1,               # Show progress
    n_jobs=1,                # Adjust based on system resources, -1 can sometimes be slower
    # refit=True is default, which means the best estimator is trained on the full dataset
)

# Fit GridSearchCV
grid_search.fit(X_train_full, y_train_full)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best Macro F1 Score (CV): {grid_search.best_score_:.4f}")

# --- 4. Get the Best Trained Model ---
# GridSearchCV automatically refits the best model on the entire training set
best_cat_model = grid_search.best_estimator_
print("\nBest CatBoost model selected.")

# --- 5. Find Optimal Threshold for Macro F1 Score (using training data) ---
# Although CV optimizes the model, we still need to find the best threshold
# for converting probabilities to class predictions.
print("\nFinding optimal threshold for Macro F1 score on training set...")
# Get predicted probabilities for the positive class (label 1) on the training set
# Using the best model which was refit on the full training set
train_proba = best_cat_model.predict_proba(X_train_full)[:, 1]

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
test_proba = best_cat_model.predict_proba(X_test)[:, 1]

# --- 7. Apply Optimal Threshold and Create Submission File ---
print(f"\nApplying optimal threshold ({optimal_threshold:.3f}) and creating submission file...")
# Convert test probabilities to class predictions using the OPTIMAL threshold
final_predictions = (test_proba >= optimal_threshold).astype(int)

print("\nCreating submission file: catboost_tuned_submission.csv")
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': final_predictions
})
# Save to CSV without index
submission_df.to_csv('catboost_tuned_submission.csv', index=False)
print("Submission file 'catboost_tuned_submission.csv' created successfully!")

# --- 8. Optional: Compare with Default Threshold ---
# This shows the F1 score using the standard 0.5 threshold on the training set
default_preds = (train_proba > 0.5).astype(int)
default_f1 = f1_score(y_train_full, default_preds, average='macro')
print(f"\nFor comparison (Tuned CatBoost model):")
print(f"Macro F1 Score on Training Set (threshold=0.5): {default_f1:.4f}")
print(f"Macro F1 Score on Training Set (optimal threshold={optimal_threshold:.3f}): {best_f1_score:.4f}")
print(f"Best CV Macro F1 Score (before threshold tuning): {grid_search.best_score_:.4f}")

print("\nTuned CatBoost model execution completed.")
