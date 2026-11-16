
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# Load dataset
df = pd.read_csv("data.csv")
print("Dataset shape:", df.shape)

# Preprocessing
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df = df.dropna(subset=["diagnosis"])

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"].map({"M": 1, "B": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM models
kernels = ["linear", "poly", "rbf", "sigmoid"]
results = []

for kernel in kernels:
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=1)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Kernel": kernel,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {kernel.upper()} SVM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Display results
results_df = pd.DataFrame(results)
print("\nSVM Kernel Comparison:")
print(results_df.to_string(index=False))

# Performance comparison chart
results_df.set_index("Kernel").plot(
    kind="bar", figsize=(10, 6)
)
plt.title("SVM Kernel Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Best kernel
best = results_df.loc[results_df["F1 Score"].idxmax()]
print("\nBest Kernel:", best["Kernel"])
print("Best F1 Score:", round(best["F1 Score"], 4))
