import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("../data/Fish.csv")
print(df.head())
print(df.dtypes)
print(f"Shape: {df.shape}")

X = df.drop("Weight", axis=1)
y = df["Weight"]

numeric_features = ["Length1", "Length2", "Length3", "Height", "Width"]
categorical_features = ["Species"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
print(f"MAE: {mae:.2f} g | R²: {r2:.4f} | 5-fold CV R²: {cv_r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
lims = [min(y_test.min(), y_pred.min()) - 50, max(y_test.max(), y_pred.max()) + 50]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual Weight (g)")
axes[0].set_ylabel("Predicted Weight (g)")
axes[0].set_title(f"Fish Weight Prediction\nR²={r2:.4f}, MAE={mae:.1f} g")
axes[0].legend()

species = df["Species"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(species)))
for sp, col in zip(species, colors):
    mask = df["Species"] == sp
    axes[1].scatter(df.loc[mask, "Length1"], df.loc[mask, "Weight"], label=sp, color=col, alpha=0.7)
axes[1].set_xlabel("Length1 (cm)")
axes[1].set_ylabel("Weight (g)")
axes[1].set_title("Weight vs Length1 by Species")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("q3_fish_weight.png", dpi=150)
plt.show()
print("Plot saved to q3_fish_weight.png")
