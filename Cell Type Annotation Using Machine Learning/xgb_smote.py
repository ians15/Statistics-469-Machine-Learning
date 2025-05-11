#need following packages
# !pip install pyreadr pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

import pyreadr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Load and clean training data
def load_data(train_path="train.rds"):
    train_data = pyreadr.read_r(train_path)[None]
    train_data.columns = train_data.columns.str.replace('[- ]', '_', regex=True)

    
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(train_data.drop(['SampleID', 'CellType'], axis=1))
    X = pd.DataFrame(X, columns=train_data.drop(['SampleID', 'CellType'], axis=1).columns[selector.get_support()])
    y = train_data['CellType']
    return X, y, selector

X, y, selector = load_data()

#Encode and split
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Apply targeted SMOTE to rare classes
rare_classes = [4, 5]  # Adjust based on label distribution
minority_mask = np.isin(y_train, rare_classes)

smote = SMOTE(
    sampling_strategy={cls: 2 * sum(y_train == cls) for cls in rare_classes},
    random_state=42
)

X_smote, y_smote = smote.fit_resample(X_train[minority_mask], y_train[minority_mask])

#  Combine with majority data
X_train_balanced = np.vstack([X_train[~minority_mask], X_smote])
y_train_balanced = np.hstack([y_train[~minority_mask], y_smote])

X_train_balanced = pd.DataFrame(X_train_balanced, columns=X.columns)

# Compute class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_balanced)

# Train XGBoost
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    n_estimators=500,
    early_stopping_rounds=20,
    random_state=42,
    learning_rate=0.05
)

xgb_model.fit(
    X_train_balanced, y_train_balanced,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=10
)

#  Evaluation
val_preds = xgb_model.predict(X_val)

print("\n=== SMOTE XGBoost Evaluation ===")
print(f"Accuracy: {accuracy_score(y_val, val_preds):.3f}")
print(f"Macro F1: {f1_score(y_val, val_preds, average='macro'):.3f}")
print(f"Weighted F1: {f1_score(y_val, val_preds, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=label_encoder.classes_.astype(str)))

# Predict test data and save RDS
test_data = pyreadr.read_r("test.rds")[None]
test_data.columns = test_data.columns.str.replace('[- ]', '_', regex=True)
X_test = test_data.drop(['SampleID'], axis=1)
X_test = pd.DataFrame(selector.transform(X_test), columns=X.columns)

test_preds = xgb_model.predict(X_test)
pred_df = pd.DataFrame({'Predicted': test_preds})
pyreadr.write_rds("xgb_smote_predictions.rds", pred_df[['Predicted']])
print("Saved SMOTE test predictions to xgb_smote_predictions.rds")




# PCA plot of SMOTE-balanced data
def plot_pca(X, y, title):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=y, cmap=ListedColormap(plt.cm.tab10.colors),
                          alpha=0.6, s=10)
    plt.colorbar(scatter, label="Class")
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_pca(X_train_balanced, y_train_balanced, "PCA After Targeted SMOTE Oversampling")
