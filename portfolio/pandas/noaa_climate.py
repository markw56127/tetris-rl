import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load & clean ──────────────────────────────────────────────────────────────

raw = pd.read_csv("../data/4310265.csv")
raw["DATE"] = pd.to_datetime(raw["DATE"])
raw = raw.sort_values("DATE").reset_index(drop=True)

# Dataset contains San Diego International Airport
city = "SAN DIEGO INTERNATIONAL AIRPORT, CA US"
df = raw[raw["NAME"] == city].copy()

print(f"Station : {city}")
print(f"Rows    : {len(df)}")
print(f"Date range: {df['DATE'].min().date()} – {df['DATE'].max().date()}")
print(df[["DATE", "TMAX", "TMIN", "PRCP"]].head())
print()

# Split by year
df_2021 = df[df["DATE"].dt.year == 2021].copy()
df_2022 = df[df["DATE"].dt.year == 2022].copy()

# ── Q1: Daily TMAX & TMIN for 2022 ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_2022["DATE"], df_2022["TMAX"], color="tomato",    linewidth=1.0, label="TMAX (°F)")
ax.plot(df_2022["DATE"], df_2022["TMIN"], color="steelblue", linewidth=1.0, label="TMIN (°F)")
ax.fill_between(df_2022["DATE"], df_2022["TMIN"], df_2022["TMAX"],
                alpha=0.15, color="purple", label="Daily range")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_xlabel("Month (2022)")
ax.set_ylabel("Temperature (°F)")
ax.set_title("Daily High & Low Temperature – San Diego 2022")
ax.legend()
plt.tight_layout()
plt.savefig("noaa_temp_2022.png", dpi=120)
plt.show()

# ── Q2: Average monthly rainfall for 2022 ────────────────────────────────────

monthly_prcp_2022 = (
    df_2022.groupby(df_2022["DATE"].dt.month)["PRCP"]
    .mean()
    .rename_axis("Month")
)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(monthly_prcp_2022.index, monthly_prcp_2022.values, color="cornflowerblue",
       tick_label=[month_labels[m - 1] for m in monthly_prcp_2022.index])
ax.set_xlabel("Month")
ax.set_ylabel("Average Daily Precipitation (inches)")
ax.set_title("Average Monthly Precipitation – San Diego 2022")
plt.tight_layout()
plt.savefig("noaa_prcp_2022.png", dpi=120)
plt.show()

# ── Q3: Compare 2021 vs 2022 ─────────────────────────────────────────────────

# Monthly averages for each year
def monthly_avg(d):
    return d.groupby(d["DATE"].dt.month)[["TMAX", "TMIN", "PRCP"]].mean()

mo_2021 = monthly_avg(df_2021)
mo_2022 = monthly_avg(df_2022)

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# Temperature comparison
ax1 = axes[0]
ax1.plot(mo_2021.index, mo_2021["TMAX"], "o--", color="tomato",    alpha=0.6, label="TMAX 2021")
ax1.plot(mo_2022.index, mo_2022["TMAX"], "o-",  color="tomato",               label="TMAX 2022")
ax1.plot(mo_2021.index, mo_2021["TMIN"], "s--", color="steelblue", alpha=0.6, label="TMIN 2021")
ax1.plot(mo_2022.index, mo_2022["TMIN"], "s-",  color="steelblue",            label="TMIN 2022")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_labels)
ax1.set_ylabel("Temperature (°F)")
ax1.set_title("Monthly Average Temperature – San Diego: 2021 vs 2022")
ax1.legend(ncol=2)

# Precipitation comparison
ax2 = axes[1]
x = list(range(1, 13))
width = 0.35
ax2.bar([i - width / 2 for i in x], mo_2021["PRCP"], width=width,
        color="cornflowerblue", alpha=0.7, label="2021")
ax2.bar([i + width / 2 for i in x], mo_2022["PRCP"], width=width,
        color="mediumseagreen", alpha=0.7, label="2022")
ax2.set_xticks(x)
ax2.set_xticklabels(month_labels)
ax2.set_ylabel("Avg Daily Precipitation (inches)")
ax2.set_title("Monthly Average Precipitation – San Diego: 2021 vs 2022")
ax2.legend()

plt.tight_layout()
plt.savefig("noaa_2021_vs_2022.png", dpi=120)
plt.show()

# Print summary stats
print("=== 2021 Monthly Averages ===")
print(mo_2021.round(2))
print()
print("=== 2022 Monthly Averages ===")
print(mo_2022.round(2))
print()

# Year-level summary comparison
print("=== Year-level Summary ===")
summary = pd.DataFrame({
    "2021": df_2021[["TMAX", "TMIN", "PRCP"]].mean(),
    "2022": df_2022[["TMAX", "TMIN", "PRCP"]].mean(),
}).round(2)
print(summary)
