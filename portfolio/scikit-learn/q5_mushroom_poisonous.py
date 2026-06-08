import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat",
]

df = pd.read_csv("../data/agaricus-lepiota.data", header=None, names=columns)
print(f"Shape: {df.shape}")
print(f"Classes: {df['class'].value_counts().to_dict()}  (e=edible, p=poisonous)")
print(f"Missing values (marked '?'):\n{(df == '?').sum()[df.apply(lambda c: (c == '?').any())]}")

df = df.replace("?", np.nan)

X = df.drop("class", axis=1)
y = (df["class"] == "p").astype(int)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cv_scores = cross_val_score(model, X_encoded, y, cv=5)
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

importances = model.feature_importances_
feature_names = columns[1:]
idx = np.argsort(importances)[-10:]
axes[0].barh(np.array(feature_names)[idx], importances[idx], color="firebrick")
axes[0].set_title("Top 10 Feature Importances")
axes[0].set_xlabel("Importance")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Edible", "Poisonous"],
    ax=axes[1], colorbar=False
)
axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("q5_mushroom_poisonous.png", dpi=150)
plt.show()
print("Plot saved to q5_mushroom_poisonous.png")
