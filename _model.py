# ------------------------------------------
#  Complete Model Development
# Predictive Water Contamination Mapping
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("MLPROJECTFINALDATA.csv")
print("Dataset shape:", df.shape)

# -----------------------------
# Define features & targets
# -----------------------------
targets_reg = [c for c in ["EC", "NO3", "F"] if c in df.columns]
target_clf = "hotspot_label"

exclude = {"sn", "location_raw", "sample_date"}
features = [c for c in df.columns
            if c not in exclude and c not in targets_reg and c != target_clf]
features = df[features].select_dtypes(include=[np.number]).columns.tolist()

# Fill missing values
for f in features:
    df[f] = df[f].fillna(df[f].median())

df[target_clf] = df[target_clf].astype(int)

# Split base sets
X = df[features].values
y = df[target_clf].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
#  1. CLASSIFICATION MODELS (Random Forest, Logistic Regression, SVM)
# ---------------------------------------------------------------------
print("\n=== Classification Models ===")

models_clf = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="rbf", probability=True)
}

results_clf = {}

for name, model in models_clf.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results_clf[name] = {"Accuracy": acc, "F1": f1}

    print(f"\n{name} → Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print(classification_report(y_test, y_pred))

    # ROC-AUC (multi-class)
    y_prob = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    print(f"ROC-AUC: {roc_auc:.3f}")

# Confusion Matrix for best model (Random Forest)
best_clf = models_clf["RandomForest"]
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Random Forest Classifier")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_RF.png", dpi=300)
plt.show()

# Feature Importance
feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": best_clf.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(10))
plt.title("Top 10 Feature Importances (Classifier)")
plt.tight_layout()
plt.savefig("feature_importance_RF.png", dpi=300)
plt.show()

# -----------------------------
# K-Fold Cross Validation
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_clf, X, y, cv=kf, scoring="accuracy")
print("\n5-Fold Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores).round(3))

# ---------------------------------------------------------------------
#  2. REGRESSION MODELS (Random Forest, SVR, XGBoost)
# ---------------------------------------------------------------------
print("\n=== Regression Models ===")
results_reg = []

for targ in targets_reg:
    df_reg = df[df[targ].notna()]
    Xr = df_reg[features].values
    yr = df_reg[targ].values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    models_reg = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR": SVR(kernel="rbf"),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    }

    for name, model in models_reg.items():
        model.fit(Xr_train, yr_train)
        yr_pred = model.predict(Xr_test)
        rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
        mae = mean_absolute_error(yr_test, yr_pred)
        r2 = r2_score(yr_test, yr_pred)
        results_reg.append([targ, name, rmse, mae, r2])

        print(f"\n{targ} – {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

        # Scatter Plot
        plt.figure(figsize=(5,5))
        plt.scatter(yr_test, yr_pred, alpha=0.7)
        plt.plot([min(yr_test), max(yr_test)], [min(yr_test), max(yr_test)], "r--")
        plt.xlabel("Actual"); plt.ylabel("Predicted")
        plt.title(f"{targ} – {name} (R²={r2:.2f})")
        plt.tight_layout()
        plt.savefig(f"{targ}_{name}_regression.png", dpi=300)
        plt.show()

# Save regression summary
pd.DataFrame(results_reg, columns=["Target", "Model", "RMSE", "MAE", "R2"]).to_csv("regression_summary.csv", index=False)

print("\nAll results and plots saved successfully ")
