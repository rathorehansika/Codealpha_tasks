import os
import sys
# Force UTF-8 encoding for standard output to support printing emoji in Windows console/logs
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# 1. Load and explore the dataset
print("Step 1: Loading German Credit Dataset from OpenML...")
german_credit = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
df = german_credit.frame

# Map 'good' class to 1 (creditworthy) and 'bad' class to 0 (not creditworthy)
df['class'] = df['class'].map({'good': 1, 'bad': 0}).astype(int)

# 2. EDA Plots
print("\nStep 2: Performing EDA and generating plots...")
sns.set_theme(style="whitegrid")

# Plot 1: Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df, palette='viridis')
plt.title('Creditworthiness Class Distribution (1 = Creditworthy, 0 = Not Creditworthy)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/class_distribution.png', dpi=300)
plt.close()

# Plot 2: Correlation Heatmap
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('class') # Exclude target column
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300)
plt.close()

# Plot 3: Feature Distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data=df, x='duration', hue='class', kde=True, multiple='stack', ax=axes[0], palette='crest')
axes[0].set_title('Distribution of Duration by Creditworthiness')

sns.histplot(data=df, x='credit_amount', hue='class', kde=True, multiple='stack', ax=axes[1], palette='crest')
axes[1].set_title('Distribution of Credit Amount by Creditworthiness')

sns.histplot(data=df, x='age', hue='class', kde=True, multiple='stack', ax=axes[2], palette='crest')
axes[2].set_title('Distribution of Age by Creditworthiness')

plt.tight_layout()
plt.savefig('plots/feature_distributions.png', dpi=300)
plt.close()

# 3. Train-Test Split (Split first, then scale/SMOTE to prevent leakage)
print("\nStep 3: Train-Test Split...")
X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Preprocessing
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
feature_names = numerical_cols + encoded_cat_features
# Sanitize feature names for plotting
sanitized_feature_names = [f.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_') for f in feature_names]

# 4. SMOTE on training data only
print("\nStep 4: Applying SMOTE to training data only...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

# Convert training and test features to NumPy arrays to bypass XGBoost string/character restrictions
X_train_res_np = X_train_res.values if hasattr(X_train_res, 'values') else X_train_res
X_test_preprocessed_np = X_test_preprocessed.values if hasattr(X_test_preprocessed, 'values') else X_test_preprocessed

# Convert labels to NumPy arrays as well
y_train_res_np = y_train_res.values if hasattr(y_train_res, 'values') else y_train_res
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

# 5. Train Baseline Models
print("\nStep 5: Training Baseline Models (including XGBoost)...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_res_np, y_train_res_np)

# 6. Evaluate Baseline Models
print("\nStep 6: Evaluating Baseline Models...")
metrics_dict = {}

for name, model in models.items():
    y_pred = model.predict(X_test_preprocessed_np)
    y_prob = model.predict_proba(X_test_preprocessed_np)[:, 1]

    acc = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred)
    rec = recall_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    auc_score = roc_auc_score(y_test_np, y_prob)

    metrics_dict[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_score
    }

metrics_df = pd.DataFrame(metrics_dict).T
print(metrics_df.to_string())

# Plot 4: Confusion Matrices (1x4 layout)
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_preprocessed_np)
    cm = confusion_matrix(y_test_np, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'Confusion Matrix - {name}')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xticklabels(['Not Creditworthy', 'Creditworthy'])
    axes[idx].set_yticklabels(['Not Creditworthy', 'Creditworthy'])
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=300)
plt.close()

# Plot 5: ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_preprocessed_np)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_np, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=300)
plt.close()

# 7. GridSearchCV for Hyperparameter Tuning on Best Model
best_model_name = metrics_df['ROC-AUC'].idxmax()
print(f"\nStep 7: Hyperparameter Tuning on Best Model ({best_model_name})...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 8, 12, 16],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    base_estimator = RandomForestClassifier(random_state=42)
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    base_estimator = XGBClassifier(eval_metric='logloss', random_state=42)
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'saga']
    }
    base_estimator = LogisticRegression(random_state=42, max_iter=1000)
else:
    param_grid = {
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    base_estimator = DecisionTreeClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=base_estimator,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_res_np, y_train_res_np)
best_tuned_model = grid_search.best_estimator_

print(f"\nBest Parameters found: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Evaluate tuned model
y_pred_tuned = best_tuned_model.predict(X_test_preprocessed_np)
y_prob_tuned = best_tuned_model.predict_proba(X_test_preprocessed_np)[:, 1]

tuned_acc = accuracy_score(y_test_np, y_pred_tuned)
tuned_prec = precision_score(y_test_np, y_pred_tuned)
tuned_rec = recall_score(y_test_np, y_pred_tuned)
tuned_f1 = f1_score(y_test_np, y_pred_tuned)
tuned_auc = roc_auc_score(y_test_np, y_prob_tuned)

print("\nTuned Model Evaluation:")
print(f"Accuracy:  {tuned_acc:.4f}")
print(f"Precision: {tuned_prec:.4f}")
print(f"Recall:    {tuned_rec:.4f}")
print(f"F1-Score:  {tuned_f1:.4f}")
print(f"ROC-AUC:   {tuned_auc:.4f}")

# Plot 6: Feature Importances
print("\nStep 8: Plotting Feature Importances...")
if hasattr(best_tuned_model, 'feature_importances_'):
    importances = best_tuned_model.feature_importances_
    importance_title = "Feature Importances"
    xlabel = "Importance"
elif hasattr(best_tuned_model, 'coef_'):
    importances = np.abs(best_tuned_model.coef_[0])
    importance_title = "Absolute Coefficients (Magnitude)"
    xlabel = "Magnitude"
else:
    importances = None

if importances is not None:
    feat_imp_df = pd.DataFrame({
        'Feature': sanitized_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top 15 Features by {importance_title} ({best_model_name})')
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    print("Feature importance plot saved.")

# Save final pipeline
print("\nStep 9: Saving the Final Pipeline using joblib...")
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_tuned_model)
])
joblib.dump(final_pipeline, 'credit_scoring_model.joblib')
print("Model pipeline successfully saved.")

# Prompt 1 Verification: Sample Prediction Demo
print("\nStep 10: Running Sample Prediction Demo...")
# 1. Takes one raw sample from X_test (use index 0)
sample_raw = X_test.iloc[[0]]
actual_label = y_test.iloc[0]

# 2. Passes it through final_pipeline
pred_label = final_pipeline.predict(sample_raw)[0]
pred_prob = final_pipeline.predict_proba(sample_raw)[0][1]

# Mapping
label_map = {0: "Not Creditworthy", 1: "Creditworthy"}
actual_str = label_map[actual_label]
pred_str = label_map[pred_label]

# Confidence calculation
confidence = pred_prob if pred_label == 1 else (1 - pred_prob)

print("="*50)
print("DEMO 1: prediction on actual test sample")
print(f"Actual Label:    {actual_label} ({actual_str})")
print(f"Predicted Label: {pred_label} ({pred_str})")
print(f"Confidence:      {confidence:.1%}")
print("="*50)

# 5. Create a custom input dict
custom_input_dict = {
    'checking_status': 'no checking',
    'duration': 12,
    'credit_history': 'existing paid',
    'purpose': 'radio/tv',
    'credit_amount': 3000,
    'savings_status': 'unknown/ no savings account',
    'employment': '1<=X<4',
    'installment_commitment': 2,
    'personal_status': 'female div/dep/mar',
    'other_parties': 'none',
    'residence_since': 3,
    'property_magnitude': 'car',
    'age': 35,
    'other_payment_plans': 'none',
    'housing': 'own',
    'existing_credits': 1,
    'job': 'skilled',
    'num_dependents': 1,
    'own_telephone': 'yes',
    'foreign_worker': 'yes'
}

custom_df = pd.DataFrame([custom_input_dict])
custom_pred = final_pipeline.predict(custom_df)[0]
custom_prob = final_pipeline.predict_proba(custom_df)[0][1]

custom_pred_str = label_map[custom_pred]
custom_confidence = custom_prob if custom_pred == 1 else (1 - custom_prob)

print("DEMO 2: prediction on custom input")
print("Input features: age=35, credit_amount=3000, duration=12, checking_status='no checking'")
print(f"Predicted Label: {custom_pred} ({custom_pred_str})")
print(f"Confidence:      {custom_confidence:.1%}")
print("="*50)

# Prompt 3 Verification: Final Summary Comparison Table
print("\nStep 11: Displaying Comparison Summary...")
summary_data = []
for name in metrics_df.index:
    row = {
        'Model': name,
        'Accuracy': round(metrics_df.loc[name, 'Accuracy'], 4),
        'Precision': round(metrics_df.loc[name, 'Precision'], 4),
        'Recall': round(metrics_df.loc[name, 'Recall'], 4),
        'F1-Score': round(metrics_df.loc[name, 'F1-Score'], 4),
        'ROC-AUC': round(metrics_df.loc[name, 'ROC-AUC'], 4),
    }
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
best_auc = summary_df['ROC-AUC'].max()
summary_df['Best Model'] = summary_df['ROC-AUC'].apply(lambda x: '✅' if x == best_auc else '❌')

print(summary_df.to_string(index=False))
print(f"\nBest overall model: {best_model_name} with ROC-AUC of {best_auc:.4f}")
