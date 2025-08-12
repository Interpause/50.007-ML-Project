import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA # <--- Import PCA ---
# Import the required libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier # Keep import for completeness, though not used
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore') # Suppress potential warnings for cleaner output

# --- 1. Load Data ---
print("Loading data...")
# Load training data (TF-IDF features)
# Make sure these files are in your working directory or provide the full path
train_df = pd.read_csv('train_tfidf_features.csv')
X_train_full = train_df.drop(['id', 'label'], axis=1) # Features (columns 0-4999)
y_train_full = train_df['label']                       # Labels

# Load test data
test_df = pd.read_csv('test_tfidf_features.csv')
X_test = test_df.drop(['id'], axis=1) # Features (columns 0-4999)
test_ids = test_df['id']              # Test IDs for submission

print(f"Training set shape: {X_train_full.shape}, Labels shape: {y_train_full.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Calculate class imbalance ratio for XGBoost ---
unique, counts = np.unique(y_train_full, return_counts=True)
print(f"Class distribution in training set: {dict(zip(unique, counts))}")

# Assuming binary classification: label 0 (majority/negative), label 1 (minority/positive)
if len(unique) == 2 and 0 in unique and 1 in unique:
    # Calculate scale_pos_weight: count of negative samples / count of positive samples
    scale_pos_weight_xgb = counts[0] / counts[1]
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight_xgb:.2f}")
else:
    print("Warning: Could not automatically determine class imbalance ratio for XGBoost.")
    print("Please check your label distribution or set scale_pos_weight manually if needed.")
    scale_pos_weight_xgb = 1.0 # Default if not imbalanced or not binary

# --- 2. Define Models with Provided Hyperparameters, Imbalance Handling (CPU Only) ---
# No cross-validation or hyperparameter search here.
print("\nDefining models with provided optimal hyperparameters, imbalance handling, and CPU only...")
# Note: Random Forest is removed, SVM is defined separately later
base_models = {
    # --- CatBoost (CPU Only) ---
    # Fix: Specify bootstrap_type when using subsample
    'cat': CatBoostClassifier(
        depth=6,
        iterations=1000,
        learning_rate=0.1,
        verbose=0,
        random_state=42,
        thread_count=-1,
        auto_class_weights='Balanced',
        # Removed task_type and devices for CPU
        bootstrap_type='Bernoulli', # <--- Add this line for compatibility with subsample
        subsample=0.8               # <--- Now this is compatible
    ),
    # --- LightGBM (CPU Only) ---
    'lgb': LGBMClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=1000,
        subsample=0.7,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
        # Removed device='gpu' for CPU
    ),
    # --- Naive Bayes (CPU Only) ---
    'nb':  MultinomialNB(alpha=0.2),
    # --- Logistic Regression (CPU Only) ---
    'lr':  LogisticRegression(
        C=1.0,
        penalty='l2',
        random_state=42,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'
    ),
    # --- XGBoost (CPU Only) ---
    'xgb': XGBClassifier(
        max_depth=6,
        learning_rate=0.0336,
        subsample=0.780,
        colsample_bynode=0.799,
        min_child_weight=1.32,
        reg_alpha=0.291,
        reg_lambda=2.684,
        gamma=0.373,
        n_estimators=100, # Default, iterations/hundreds is common
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_xgb
        # Removed tree_method='gpu_hist' and predictor='gpu_predictor' for CPU
    )
    # --- Random Forest (Removed) ---
    # --- SVM (CPU Only, with PCA) ---
    # Defined and handled separately below
}

# --- 3. Train Base Models (No Cross-Validation) ---
print("\nTraining base models (NO cross-validation)...")
# Dictionary to store the trained models (excluding SVM for now)
trained_models = {}
# Arrays to store predictions for meta-learner training
# Shape: (n_samples, n_models). We'll add SVM's column later.
train_meta_features = np.zeros((X_train_full.shape[0], len(base_models)))

model_names = list(base_models.keys())

# --- Train Non-SVM Models ---
print("--- Training Non-SVM Models ---")
for i, (name, model) in enumerate(base_models.items()):
    print(f"Training {name.upper()}...")
    model.fit(X_train_full, y_train_full)
    trained_models[name] = model

    if hasattr(model, "predict_proba"):
        train_pred_proba = model.predict_proba(X_train_full)[:, 1]
    else:
        print(f"Warning: {name} does not support predict_proba. Using decision_function or predict.")
        if hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_train_full)
            train_pred_proba = 1 / (1 + np.exp(-df_scores)) # Sigmoid
        else:
            train_pred_proba = model.predict(X_train_full)
    train_meta_features[:, i] = train_pred_proba

# --- Train SVM with PCA ---
print("\n--- Training SVM with PCA (500 components) ---")
# Define SVM with provided hyperparameters and imbalance handling
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

# Define and fit PCA transformer
n_components_svm = 500 # <--- Changed to 500 components ---
pca_svm = PCA(n_components=n_components_svm, random_state=42)
print(f"Fitting PCA for SVM to {n_components_svm} components...")
pca_svm.fit(X_train_full)
print(f"PCA for SVM fitted. Explained variance ratio sum: {np.sum(pca_svm.explained_variance_ratio_):.4f}")

# Transform training data
print("Transforming training data for SVM...")
X_train_pca_svm = pca_svm.transform(X_train_full)

# Train SVM on PCA-transformed data
print("Training SVM on PCA-transformed data...")
svm_model.fit(X_train_pca_svm, y_train_full)
# Store the trained SVM model and its PCA transformer
trained_models['svm'] = svm_model
pca_transformers = {'svm': pca_svm} # Store PCA transformers if needed later

# Get SVM predictions on PCA-transformed training data
if hasattr(svm_model, "predict_proba"):
    svm_train_pred_proba = svm_model.predict_proba(X_train_pca_svm)[:, 1]
else:
    if hasattr(svm_model, "decision_function"):
        df_scores = svm_model.decision_function(X_train_pca_svm)
        svm_train_pred_proba = 1 / (1 + np.exp(-df_scores)) # Sigmoid
    else:
        svm_train_pred_proba = svm_model.predict(X_train_pca_svm)

# Add SVM predictions to the meta-features array
train_meta_features = np.column_stack([train_meta_features, svm_train_pred_proba])
# Update model names list to include SVM
model_names.append('svm')

print("All base models trained and training meta-features collected.")

# --- 4. Train the Meta-Learner ---
print("\nTraining the meta-learner (Logistic Regression)...")
meta_learner = LogisticRegression(random_state=42, max_iter=1000)
meta_learner.fit(train_meta_features, y_train_full)

# --- 5. Make Predictions on Test Set using Base Models ---
print("\nMaking predictions on test set with base models...")
# Array to store test predictions from base models (meta-features for test set)
# Shape: (n_test_samples, n_models)
test_meta_features = np.zeros((X_test.shape[0], len(model_names) - 1)) # Exclude SVM initially

# --- Predict with Non-SVM Models on Test Set ---
print("--- Predicting with Non-SVM Models on Test Set ---")
for i, (name, model) in enumerate(base_models.items()):
    if hasattr(model, "predict_proba"):
        test_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_test)
            test_pred_proba = 1 / (1 + np.exp(-df_scores)) # Sigmoid
        else:
            test_pred_proba = model.predict(X_test)
    test_meta_features[:, i] = test_pred_proba

# --- Predict with SVM (PCA) on Test Set ---
print("\n--- Predicting with SVM (PCA) on Test Set ---")
# Transform test data using the fitted PCA transformer
print("Transforming test data for SVM...")
X_test_pca_svm = pca_transformers['svm'].transform(X_test) # Use the PCA fitted on training data

# Get SVM predictions on PCA-transformed test data
if hasattr(svm_model, "predict_proba"):
    svm_test_pred_proba = svm_model.predict_proba(X_test_pca_svm)[:, 1]
else:
    if hasattr(svm_model, "decision_function"):
        df_scores = svm_model.decision_function(X_test_pca_svm)
        svm_test_pred_proba = 1 / (1 + np.exp(-df_scores)) # Sigmoid
    else:
        svm_test_pred_proba = svm_model.predict(X_test_pca_svm)

# Add SVM predictions to the test meta-features array
test_meta_features = np.column_stack([test_meta_features, svm_test_pred_proba])

print("All test set meta-features collected.")

# --- 6. Make Final Prediction using Meta-Learner ---
print("\nMaking final prediction using the meta-learner...")
final_proba = meta_learner.predict_proba(test_meta_features)[:, 1]

# --- 7. Apply Optimal Thresholds (Individual Model Thresholds are NOT used here) ---
# Optimize threshold for the meta-learner's output
print("\nFinding optimal threshold for meta-learner's final prediction (Macro F1)...")
meta_train_proba = meta_learner.predict_proba(train_meta_features)[:, 1]

thresholds = np.arange(0.3, 0.71, 0.01)
f1_scores_meta = []
for thresh in thresholds:
    preds = (meta_train_proba >= thresh).astype(int)
    macro_f1 = f1_score(y_train_full, preds, average='macro')
    f1_scores_meta.append(macro_f1)

optimal_threshold_idx = np.argmax(f1_scores_meta)
optimal_threshold_meta = thresholds[optimal_threshold_idx]
best_f1_score_meta = f1_scores_meta[optimal_threshold_idx]

print(f"Optimal Threshold for Meta-Learner: {optimal_threshold_meta:.3f}")
print(f"Best Macro F1 Score on Training Set (Meta-Learner): {best_f1_score_meta:.4f}")

# --- 8. Apply Optimal Threshold and Create Submission File ---
print(f"\nApplying optimal threshold ({optimal_threshold_meta:.3f}) and creating submission file...")
final_predictions = (final_proba >= optimal_threshold_meta).astype(int)

print("\nCreating submission file: stacked_ensemble_svm_pca500_no_rf_cpu_submission.csv")
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': final_predictions
})
submission_filename = 'stacked_ensemble_svm_pca500_no_rf_cpu_submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file '{submission_filename}' created successfully!")

# --- 9. Optional: Evaluate Meta-Learner with Default Threshold (0.5) ---
default_meta_preds = (meta_train_proba > 0.5).astype(int)
default_meta_f1 = f1_score(y_train_full, default_meta_preds, average='macro')
print(f"\nFor comparison (Meta-Learner Performance):")
print(f"Macro F1 Score on Training Set (threshold=0.5): {default_meta_f1:.4f}")
print(f"Macro F1 Score on Training Set (optimal threshold={optimal_threshold_meta:.3f}): {best_f1_score_meta:.4f}")

print("\nLayered Ensemble (Stacking) with SVM PCA (500) and no RF (CPU Only) completed.")
