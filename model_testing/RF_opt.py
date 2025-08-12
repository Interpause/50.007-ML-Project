import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Import the required ensemble library
from sklearn.ensemble import RandomForestClassifier
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

# --- 2. Define Random Forest Model and Hyperparameter Grid ---
print("\nDefining Random Forest model and hyperparameter grid...")
# Base Random Forest model (parameters that won't be searched are set here)
base_rf_model = RandomForestClassifier(
    random_state=42,      # For reproducibility
    n_jobs=-1             # Use all available CPU cores
    # Consider n_estimators as part of tuning or set a reasonable fixed value
    # n_estimators=800 # Example fixed value
)

# Define hyperparameter grid for tuning
# Note: Using a smaller grid or fewer options can speed up the search
param_grid = {
    'n_estimators': [600, 800],     # Number of trees
    'max_depth': [15, 20, None],    # Maximum depth of trees (None means nodes are expanded until all leaves are pure or until min_samples_split)
    'min_samples_split': [2, 5],    # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],     # Minimum number of samples required to be at a leaf node
    # 'max_features': ['sqrt', 'log2'] # Number of features to consider when looking for the best split
}

# --- 3. Perform Hyperparameter Tuning with Cross-Validation ---
print("\nPerforming hyperparameter tuning with GridSearchCV (Macro F1)...")
# Use StratifiedKFold for cross-validation
cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced folds for speed

# Initialize GridSearchCV
# RandomForest's sklearn wrapper directly supports 'f1_macro'
grid_search = GridSearchCV(
    estimator=base_rf_model,
    param_grid=param_grid,
    scoring='f1_macro', # Optimize for Macro F1
    cv=cv_splitter,
    verbose=1,          # Show progress
    n_jobs=1            # Adjust based on system resources (-1 can sometimes be slower)
    # refit=True is default, which means the best estimator is trained on the full dataset
)

# Fit GridSearchCV
grid_search.fit(X_train_full, y_train_full)

# --- 4. DISPLAY THE OPTIMAL HYPERPARAMETERS ---
print("\n" + "="*50)
print("OPTIMAL HYPERPARAMETERS FOUND:")
print("="*50)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Macro F1 Score (CV): {grid_search.best_score_:.4f}")
print("="*50)
# --- END DISPLAY SECTION ---

# --- 5. Get the Best Trained Model ---
# GridSearchCV automatically refits the best model on the entire training set
best_rf_model = grid_search.best_estimator_
print("\nBest Random Forest model selected.")

# --- 6. Find Optimal Threshold for Macro F1 Score (using training data) ---
# Although CV optimizes the model, we still need to find the best threshold
# for converting probabilities to class predictions.
print("\nFinding optimal threshold for Macro F1 score on training set...")
# Get predicted probabilities for the positive class (label 1) on the training set
# Using the best model which was refit on the full training set
train_proba = best_rf_model.predict_proba(X_train_full)[:, 1]

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

# --- 7. Make Predictions on Test Set ---
print("\nMaking predictions on test set...")
# Get predicted probabilities for the positive class (label 1) on the test set
test_proba = best_rf_model.predict_proba(X_test)[:, 1]

# --- 8. Apply Optimal Threshold and Create Submission File ---
print(f"\nApplying optimal threshold ({optimal_threshold:.3f}) and creating submission file...")
# Convert test probabilities to class predictions using the OPTIMAL threshold
final_predictions = (test_proba >= optimal_threshold).astype(int)

print("\nCreating submission file: rf_tuned_submission.csv")
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': final_predictions
})
# Save to CSV without index
submission_df.to_csv('rf_tuned_submission.csv', index=False)
print("Submission file 'rf_tuned_submission.csv' created successfully!")

# --- 9. Optional: Compare with Default Threshold ---
# This shows the F1 score using the standard 0.5 threshold on the training set
default_preds = (train_proba > 0.5).astype(int)
default_f1 = f1_score(y_train_full, default_preds, average='macro')
print(f"\nFor comparison (Tuned Random Forest model):")
print(f"Macro F1 Score on Training Set (threshold=0.5): {default_f1:.4f}")
print(f"Macro F1 Score on Training Set (optimal threshold={optimal_threshold:.3f}): {best_f1_score:.4f}")
print(f"Best CV Macro F1 Score (before threshold tuning): {grid_search.best_score_:.4f}")

print("\nTuned Random Forest model execution completed.")
