#need following packages
# !pip install pyreadr pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn



import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pyreadr

def robust_rf_baseline(train_path="train.rds", test_path="test.rds"):
    # Load and clean training data
    train = pyreadr.read_r(train_path)[None]
    train.columns = train.columns.str.replace('[- ]', '_', regex=True)

    # Split features and labels
    X = train.drop(['SampleID', 'CellType'], axis=1)
    y = train['CellType']

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train RF model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Evaluation on validation set
    print("\n=== Validation Performance for Baseline Random Forest ===")
    val_pred = rf.predict(X_val)
    print(classification_report(y_val, val_pred, zero_division=0))

    # PCA plot
    def plot_pca(X, y, title):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype('category').cat.codes,
                              alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Cell Type')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    plot_pca(X_train, y_train, "Training Data - PCA Projection")

    # --- Predict Test Labels and Save as RDS ---
    print("\nGenerating predictions for test data and saving to RDS...")

    test = pyreadr.read_r(test_path)[None]
    test.columns = test.columns.str.replace('[- ]', '_', regex=True)
    X_test = test.drop(['SampleID'], axis=1)

    predictions = rf.predict(X_test)

    output_df = pd.DataFrame({'Predicted': predictions})
    pyreadr.write_rds("rf_predictions.rds", output_df[['Predicted']])

    print("Saved predictions to rf_predictions.rds")

    return rf


model = robust_rf_baseline()


#To view first 10 predictions run the 3 lines below

#result = pyreadr.read_r("rf_predictions.rds")
#pred_df = result[None]
#print(pred_df.head(10))