#need following packages
# !pip install pyreadr pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

import pyreadr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

# Load and clean training data
def load_data(train_path="train.rds"):
    train_data = pyreadr.read_r(train_path)[None]
    train_data.columns = train_data.columns.str.replace('[- ]', '_', regex=True)

    # Remove near-zero variance features
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(train_data.drop(['SampleID', 'CellType'], axis=1))
    X = pd.DataFrame(X, columns=train_data.drop(['SampleID', 'CellType'], axis=1).columns[selector.get_support()])

    y = train_data['CellType']
    return X, y, selector

X, y, selector = load_data()

#  Encode labels and split data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Train XGBoost
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    n_estimators=500,
    early_stopping_rounds=20,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=10
)

#Evaluate model
val_preds = xgb_model.predict(X_val)

print("\n=== Enhanced Evaluation ===")
print(f"Accuracy: {accuracy_score(y_val, val_preds):.3f}")
print(f"Macro F1: {f1_score(y_val, val_preds, average='macro'):.3f}")
print(f"Weighted F1: {f1_score(y_val, val_preds, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=label_encoder.classes_.astype(str)))

#Load test data, apply same transformation
test_data = pyreadr.read_r("test.rds")[None]
test_data.columns = test_data.columns.str.replace('[- ]', '_', regex=True)
X_test = test_data.drop(['SampleID'], axis=1)
X_test = pd.DataFrame(selector.transform(X_test), columns=X.columns)

# Predict and save as RDS
test_preds = xgb_model.predict(X_test)
pred_df = pd.DataFrame({'Predicted': test_preds})
pyreadr.write_rds("xgb_predictions.rds", pred_df[['Predicted']])

print("\n Saved test predictions to xgb_predictions.rds")


#  PCA Visualization 
def plot_pca(X, y, title, label_encoder):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=y, cmap='viridis', alpha=0.7)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cell Type')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot training data
plot_pca(X_train, y_train, "PCA After XGBoost", label_encoder)


