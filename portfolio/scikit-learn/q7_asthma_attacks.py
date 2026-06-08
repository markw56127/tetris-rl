import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not available, falling back to sklearn")

df = pd.read_csv("../data/asthma.csv")
print(df.head())
print(df.dtypes)
print(f"\nAttack count distribution:\n{df['attack'].describe()}")

df["gender_enc"] = (df["gender"] == "male").astype(int)
df["res_inf_enc"] = (df["res_inf"] == "yes").astype(int)

if HAS_STATSMODELS:
    poisson_model = smf.glm(
        formula="attack ~ gender_enc + res_inf_enc + ghq12",
        data=df,
        family=sm.families.Poisson()
    ).fit()
    print("\nPoisson GLM Summary:")
    print(poisson_model.summary())

    nb_model = smf.glm(
        formula="attack ~ gender_enc + res_inf_enc + ghq12",
        data=df,
        family=sm.families.NegativeBinomial()
    ).fit()
    print("\nNegative Binomial GLM Summary:")
    print(nb_model.summary())

    df["poisson_pred"] = poisson_model.predict(df)
    df["nb_pred"] = nb_model.predict(df)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(df["attack"], df["poisson_pred"], alpha=0.5, label="Poisson")
    axes[0].scatter(df["attack"], df["nb_pred"], alpha=0.5, label="Neg. Binomial", color="orange")
    lim = df["attack"].max() + 1
    axes[0].plot([0, lim], [0, lim], "r--")
    axes[0].set_xlabel("Actual Attacks")
    axes[0].set_ylabel("Predicted Attacks")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].boxplot(
        [df.loc[df["gender"] == g, "attack"] for g in ["female", "male"]],
        labels=["Female", "Male"]
    )
    axes[1].set_ylabel("Attack Count")
    axes[1].set_title("Attacks by Gender")

    axes[2].scatter(df["ghq12"], df["attack"], alpha=0.5, c=df["res_inf_enc"], cmap="coolwarm")
    axes[2].set_xlabel("GHQ-12 Score")
    axes[2].set_ylabel("Attack Count")
    axes[2].set_title("Attacks vs GHQ-12\n(red=res. inf., blue=no res. inf.)")

else:
    from sklearn.ensemble import GradientBoostingRegressor
    X = df[["gender_enc", "res_inf_enc", "ghq12"]]
    y = df["attack"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    lim = max(y_test.max(), y_pred.max()) + 1
    axes[0].plot([0, lim], [0, lim], "r--")
    axes[0].set_xlabel("Actual Attacks")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Predicted vs Actual (MAE={mae:.2f})")

    axes[1].scatter(df["ghq12"], df["attack"], alpha=0.5, c=df["res_inf_enc"], cmap="coolwarm")
    axes[1].set_xlabel("GHQ-12 Score")
    axes[1].set_ylabel("Attack Count")
    axes[1].set_title("Attacks vs GHQ-12")

plt.tight_layout()
plt.savefig("q7_asthma_attacks.png", dpi=150)
plt.show()
print("Plot saved to q7_asthma_attacks.png")
