#need following packages
# !pip install pyreadr pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pyreadr

# Load and prepare training data
print("Loading training data...")
train = pyreadr.read_r("train.rds")[None]
train.columns = train.columns.str.replace('[- ]', '_', regex=True)

X = train.drop(['SampleID', 'CellType'], axis=1)
y = train['CellType']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize and reduce dimensionality
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Apply KMeans clustering
print("Fitting KMeans...")
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

# Map clusters to cell types using majority vote
cluster_to_type = defaultdict(list)
for label, true in zip(cluster_labels, y_train):
    cluster_to_type[label].append(true)

cluster_mapping = {
    c: max(set(labels), key=labels.count)
    for c, labels in cluster_to_type.items()
}

#Predict and evaluate
y_pred = np.array([cluster_mapping[c] for c in cluster_labels])

print("\n=== KMeans Classification Report ===")
print(classification_report(y_train, y_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_train, y_pred):.3f}")
print(f"Weighted F1: {f1_score(y_train, y_pred, average='weighted'):.3f}")

#Save predictions to RDS file
pred_df = pd.DataFrame({'Predicted': y_pred})
pyreadr.write_rds("kmeans_predictions.rds", pred_df[['Predicted']])
print("\nSaved predictions to kmeans_predictions.rds")

#PCA VISUAL CODE

# 1. Load and prepare data
print("Loading data for PCA visualization")
train_data = pyreadr.read_r("train.rds")[None]
train_data.columns = train_data.columns.str.replace('[- ]', '_', regex=True)

X_train = train_data.drop(['SampleID', 'CellType'], axis=1)
y_train = train_data['CellType']

# 2. Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train) 

# 3. Run PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Cluster with K-means
kmeans = KMeans(n_clusters=10, random_state=42)  # Using 10 clusters
cluster_labels = kmeans.fit_predict(X_scaled)  

# 5. Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
            c=cluster_labels, 
            cmap='tab10', 
            s=10, 
            alpha=0.6)

plt.title(f"K-means Clusters in 2D PCA Space")
plt.xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
plt.ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")

# Add legend
plt.legend(*scatter.legend_elements(),
           title="Clusters",
           bbox_to_anchor=(1.05, 1),
           loc='upper left')


plt.xlim([0, 800])
plt.ylim([-400, 200])

plt.tight_layout()
plt.show()