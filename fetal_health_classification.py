# =============================================================================
# Fetal Health Classification from Cardiotocography (CTG) Data
# Team Members: Surianandhan Sridhar, Pattan Sameera Hussainy
# Models: Logistic Regression, XGBoost, MLP (Neural Network)
# Dataset: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("fetal_health.csv")

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nClass Distribution:\n{df['fetal_health'].value_counts()}")
print(f"\nStatistical Summary:\n{df.describe()}")

# Class label mapping
label_map = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
df["fetal_health_label"] = df["fetal_health"].map(label_map)

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

colors = ["#2ecc71", "#f39c12", "#e74c3c"]

# Figure 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
class_counts = df["fetal_health_label"].value_counts()

axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor="black")
axes[0].set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Fetal Health Status")
axes[0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

axes[1].pie(class_counts.values, labels=class_counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Class Distribution (Proportion)", fontsize=14, fontweight="bold")

plt.suptitle("Figure 1: Fetal Health Class Distribution", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig1_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_class_distribution.png")

# Figure 2: Feature Correlation Heatmap
plt.figure(figsize=(16, 12))
corr = df.drop(columns=["fetal_health", "fetal_health_label"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
            linewidths=0.5, square=True)
plt.title("Figure 2: Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_correlation_heatmap.png")

# Figure 3: Key Feature Distributions by Class
key_features = [
    "baseline value", "accelerations", "fetal_movement",
    "uterine_contractions", "abnormal_short_term_variability",
    "mean_value_of_long_term_variability"
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    for label, color in zip(["Normal", "Suspect", "Pathological"], colors):
        subset = df[df["fetal_health_label"] == label][feat]
        axes[i].hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="none")
    axes[i].set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].legend(fontsize=8)

plt.suptitle("Figure 3: Key Feature Distributions by Fetal Health Class",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig3_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_feature_distributions.png")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

X = df.drop(columns=["fetal_health", "fetal_health_label"])
y = df["fetal_health"].astype(int) - 1  # 0=Normal, 1=Suspect, 2=Pathological

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")

# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Training Models")
print("=" * 60)

results = {}

# 4.1 Logistic Regression
print("\n[1/3] Logistic Regression ...")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0,
                         solver="lbfgs", multi_class="multinomial")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results["Logistic Regression"] = {
    "model": lr, "y_pred": y_pred_lr,
    "accuracy": accuracy_score(y_test, y_pred_lr),
    "report": classification_report(y_test, y_pred_lr,
                target_names=["Normal", "Suspect", "Pathological"], output_dict=True),
    "cm": confusion_matrix(y_test, y_pred_lr), "scaled": True,
}

# 4.2 XGBoost
print("[2/3] XGBoost ...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     eval_metric="mlogloss", random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results["XGBoost"] = {
    "model": xgb, "y_pred": y_pred_xgb,
    "accuracy": accuracy_score(y_test, y_pred_xgb),
    "report": classification_report(y_test, y_pred_xgb,
                target_names=["Normal", "Suspect", "Pathological"], output_dict=True),
    "cm": confusion_matrix(y_test, y_pred_xgb), "scaled": False,
}

# 4.3 MLP Neural Network
print("[3/3] MLP Neural Network ...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     solver="adam", alpha=0.001, max_iter=500,
                     random_state=42, early_stopping=True, validation_fraction=0.1)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
results["MLP Neural Network"] = {
    "model": mlp, "y_pred": y_pred_mlp,
    "accuracy": accuracy_score(y_test, y_pred_mlp),
    "report": classification_report(y_test, y_pred_mlp,
                target_names=["Normal", "Suspect", "Pathological"], output_dict=True),
    "cm": confusion_matrix(y_test, y_pred_mlp), "scaled": True,
}

# =============================================================================
# 5. RESULTS & VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Results & Visualizations")
print("=" * 60)

class_names = ["Normal", "Suspect", "Pathological"]

for name, res in results.items():
    print(f"\n-- {name} --")
    print(f"  Accuracy: {res['accuracy']:.4f}")
    print(classification_report(y_test, res["y_pred"], target_names=class_names))

# Figure 4: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
bar_colors = ["#3498db", "#e67e22", "#9b59b6"]

for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray")
    ax.set_title(f"{name}\n(Acc: {res['accuracy']:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

plt.suptitle("Figure 4: Confusion Matrices — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_confusion_matrices.png")

# Figure 5: Model Accuracy Comparison
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] for n in model_names]

plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, accuracies, color=bar_colors, edgecolor="black", width=0.5)
plt.ylim(0.7, 1.0)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Figure 5: Model Accuracy Comparison", fontsize=14, fontweight="bold")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{acc:.4f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("fig5_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_accuracy_comparison.png")

# Figure 6: Per-class F1 Score Comparison
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(class_names))
width = 0.25

for i, (name, color) in enumerate(zip(model_names, bar_colors)):
    f1s = [results[name]["report"][c]["f1-score"] for c in class_names]
    ax.bar(x + i * width, f1s, width, label=name, color=color, alpha=0.85, edgecolor="black")

ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Figure 6: Per-Class F1-Score Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fig6_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig6_f1_comparison.png")

# Figure 7: MLP Training Loss Curve
plt.figure(figsize=(9, 5))
plt.plot(mlp.loss_curve_, color="#9b59b6", linewidth=2, label="Training Loss")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Figure 7: MLP Neural Network Training Loss Curve", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fig7_mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig7_mlp_loss_curve.png")

# Figure 8: XGBoost Feature Importance
importances = pd.Series(xgb.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()

plt.figure(figsize=(10, 7))
top15.plot(kind="barh", color="#e67e22", edgecolor="black")
plt.xlabel("Feature Importance Score", fontsize=12)
plt.title("Figure 8: XGBoost Top-15 Feature Importances", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig8_feature_importance.png")

# Figure 9: Cross-Validation Box Plot
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, res in results.items():
    X_use = X_train_scaled if res["scaled"] else X_train.values
    scores = cross_val_score(res["model"], X_use, y_train, cv=cv, scoring="accuracy")
    cv_scores[name] = scores
    print(f"  {name} CV: {scores.mean():.4f} +/- {scores.std():.4f}")

plt.figure(figsize=(9, 5))
plt.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True,
            boxprops=dict(facecolor="#d0e8ff"), medianprops=dict(color="red", linewidth=2))
plt.ylabel("5-Fold CV Accuracy", fontsize=12)
plt.title("Figure 9: Cross-Validation Accuracy (5-Fold)", fontsize=14, fontweight="bold")
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig("fig9_cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig9_cross_validation.png")

# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Summary Table")
print("=" * 60)

summary_rows = []
for name, res in results.items():
    r = res["report"]
    summary_rows.append({
        "Model"    : name,
        "Accuracy" : f"{res['accuracy']:.4f}",
        "Precision": f"{r['weighted avg']['precision']:.4f}",
        "Recall"   : f"{r['weighted avg']['recall']:.4f}",
        "F1-Score" : f"{r['weighted avg']['f1-score']:.4f}",
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{summary_df.to_string(index=False)}")
summary_df.to_csv("model_summary.csv", index=False)
print("\n  Saved model_summary.csv")
print("\n" + "=" * 60)
print("ALL DONE! All figures saved successfully.")
print("=" * 60)
