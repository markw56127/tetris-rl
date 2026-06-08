import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("../data/athlete_events.csv")
print(f"Shape: {df.shape}")
print(df.head(3))
print()

THRESHOLD = 5  # minimum non-null entries to keep a column

# ── Helper: mean-height pivot, then drop sparse columns ──────────────────────

def height_table(data, sex, season):
    """Pivot: rows = Event, columns = Year, values = mean Height."""
    subset = data[(data["Sex"] == sex) & (data["Season"] == season)].copy()
    pivot = (
        subset.groupby(["Event", "Year"])["Height"]
        .mean()
        .unstack("Year")
    )
    # Drop columns (years) with fewer than THRESHOLD non-null entries
    pivot = pivot.dropna(axis=1, thresh=THRESHOLD)
    return pivot


# ── Q1 & Q2: Height tables by sex and season ─────────────────────────────────

print("Building height pivot tables …")
female_summer = height_table(df, "F", "Summer")
female_winter = height_table(df, "F", "Winter")
male_summer   = height_table(df, "M", "Summer")
male_winter   = height_table(df, "M", "Winter")

print(f"Female Summer events × years: {female_summer.shape}")
print(f"Female Winter events × years: {female_winter.shape}")
print(f"Male   Summer events × years: {male_summer.shape}")
print(f"Male   Winter events × years: {male_winter.shape}")
print()

# ── Q3: Highest variance events and trend plots ───────────────────────────────

def top_variance_events(pivot, n=10):
    """Return events with highest variance across years."""
    return pivot.var(axis=1).sort_values(ascending=False).head(n)


print("=== Top 10 Female Summer events by height variance ===")
print(top_variance_events(female_summer))
print()
print("=== Top 10 Male Summer events by height variance ===")
print(top_variance_events(male_summer))
print()

# Plot mean height over time for a curated set of interesting summer events
# (events that span many years and show real trends)
INTERESTING_EVENTS = [
    "Basketball Men's Basketball",
    "Gymnastics Women's Individual All-Around",
    "Volleyball Women's Volleyball",
    "Wrestling Freestyle Men's Featherweight",
    "Swimming Men's 100 metres Freestyle",
]

fig, axes = plt.subplots(len(INTERESTING_EVENTS), 1, figsize=(12, 3 * len(INTERESTING_EVENTS)))

for ax, event in zip(axes, INTERESTING_EVENTS):
    sex = "M" if "Men" in event or "men" not in event.lower() else "F"
    # pick the right table
    src = male_summer if sex == "M" else female_summer
    if event not in src.index:
        ax.set_visible(False)
        continue
    row = src.loc[event].dropna()
    ax.plot(row.index, row.values, marker="o", linewidth=1.5)
    ax.set_title(event, fontsize=9)
    ax.set_ylabel("Mean Height (cm)")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(
    "Mean Athlete Height Over Time – Selected Summer Olympic Events\n"
    "(Basketball heights have risen steadily; gymnastics shows a decline in female height)",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig("olympics_height_trends.png", dpi=120)
plt.show()

# ── Q4: Gold medals per country per year ──────────────────────────────────────

golds = df[df["Medal"] == "Gold"].copy()
gold_table = (
    golds.groupby(["NOC", "Year"])
    .size()
    .unstack("Year")
    .fillna(0)
    .astype(int)
)
print("=== Gold Medals per Country per Year (top 10 countries, all-time) ===")
top_gold_countries = gold_table.sum(axis=1).sort_values(ascending=False).head(10).index
print(gold_table.loc[top_gold_countries].to_string())
print()

# ── Q5: Average age per year per country ─────────────────────────────────────

age_table = (
    df.dropna(subset=["Age"])
    .groupby(["NOC", "Year"])["Age"]
    .mean()
    .round(1)
    .unstack("Year")
)
print("=== Average Athlete Age per Country per Year (sample) ===")
print(age_table.head(10).to_string())
print()

# ── Q6: Average age of athletes per games over time ──────────────────────────

avg_age_by_year = (
    df.dropna(subset=["Age"])
    .groupby(["Year", "Season"])["Age"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))
for season, grp in avg_age_by_year.groupby("Season"):
    ax.plot(grp["Year"], grp["Age"], marker="o", label=season, linewidth=1.5)

ax.set_xlabel("Year")
ax.set_ylabel("Average Age")
ax.set_title("Average Age of Olympic Athletes Over Time (Summer vs Winter Games)")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
plt.tight_layout()
plt.savefig("olympics_avg_age.png", dpi=120)
plt.show()
