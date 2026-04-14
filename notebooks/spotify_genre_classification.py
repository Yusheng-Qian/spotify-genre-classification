# =========================================
# Spotify Genre Classification
# =========================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# -----------------------------------------
# 2. Load dataset
# -----------------------------------------
df = pd.read_csv("../data/musicData.csv")

print("Dataset shape:", df.shape)
df.head()
# -----------------------------------------
# 3. Inspect dataset
# -----------------------------------------
print(df.info())
print("\nMissing values:\n")
print(df.isnull().sum())
# -----------------------------------------
# 4. Drop irrelevant columns
# -----------------------------------------
# Columns like Spotify ID, artist name, song name, and obtained date
# are not the main focus for this baseline classification model

drop_cols = ["spotify_id", "artist_name", "song_name", "obtained_date"]

# Only drop columns that actually exist
drop_cols = [col for col in drop_cols if col in df.columns]

df = df.drop(columns=drop_cols)

print("Remaining columns:", df.columns.tolist())
# -----------------------------------------
# 5. Handle target variable
# -----------------------------------------
target_col = "genre"   # make sure this matches your dataset column name

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# Encode target labels
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])

print("Encoded genres:", list(label_encoder.classes_))
# -----------------------------------------
# 6. Split features and target
# -----------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)
# -----------------------------------------
# 7. Identify numeric and categorical columns
# -----------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
# -----------------------------------------
# 8. Preprocessing pipeline
# -----------------------------------------
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
# -----------------------------------------
# 9. Train-test split
# IMPORTANT: stratify=y to avoid AUC errors
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
# -----------------------------------------
# 10. Baseline model: Logistic Regression
# -----------------------------------------
log_reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

log_reg_model.fit(X_train, y_train)

y_pred_lr = log_reg_model.predict(X_test)
y_prob_lr = log_reg_model.predict_proba(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr, multi_class="ovr")

print("Logistic Regression Accuracy:", round(acc_lr, 4))
print("Logistic Regression AUC:", round(auc_lr, 4))
# -----------------------------------------
# 11. Random Forest model
# -----------------------------------------
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class="ovr")

print("Random Forest Accuracy:", round(acc_rf, 4))
print("Random Forest AUC:", round(auc_rf, 4))
# -----------------------------------------
# 12. Classification report
# -----------------------------------------
print("Classification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
# -----------------------------------------
# 13. Confusion matrix
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# -----------------------------------------
# 14. PCA visualization (optional but strong)
# -----------------------------------------
# For visualization only, use numeric columns
X_numeric = df[numeric_features].copy()

# Impute + scale
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_numeric_imputed = imputer.fit_transform(X_numeric)
X_numeric_scaled = scaler.fit_transform(X_numeric_imputed)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_numeric_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "genre": label_encoder.inverse_transform(y)
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="genre", alpha=0.6, s=40)
plt.title("PCA Visualization of Spotify Genres")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
# -----------------------------------------
# 15. Save best model
# -----------------------------------------
import joblib

joblib.dump(rf_model, "../model/spotify_genre_classifier.pkl")
print("Model saved successfully.")
