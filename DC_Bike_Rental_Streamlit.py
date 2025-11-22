import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DC Bike Rental Dashboard", layout="wide")

st.title("🚲 Washington D.C. Bike Rental Data Dashboard")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # rename seasons
    season_map = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
    df['season_name'] = df['season'].map(season_map)

    # day period
    def get_period(h):
        if 0 <= h < 6:
            return "Night"
        elif 6 <= h < 12:
            return "Morning"
        elif 12 <= h < 18:
            return "Afternoon"
        else:
            return "Evening"

    df['day_period'] = df['hour'].apply(get_period)
    return df

df = load_data()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")
year_filter = st.sidebar.multiselect("Select Year", df["year"].unique(), default=df["year"].unique())
season_filter = st.sidebar.multiselect("Select Season", df["season_name"].unique(), default=df["season_name"].unique())

df_filtered = df[(df["year"].isin(year_filter)) & (df["season_name"].isin(season_filter))]

st.write("### Dataset Preview")
st.dataframe(df_filtered.head())

# ========================================================
# 1️⃣ DISTRIBUTION PLOTS (Assignment II – Q1 & Q2)
# ========================================================

st.header("📊 Numerical Column Distributions")

num_cols = ["temp", "atemp", "humidity", "windspeed", "count"]

for col in num_cols:
    st.subheader(f"Distribution of **{col}**")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df_filtered[col], bins=30)
    ax.set_title(f"Histogram of {col}")
    plt.tight_layout(pad=2)     # 🔥 FIXED SPACING
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(df_filtered[col], vert=True)
    ax.set_title(f"Boxplot of {col}")
    plt.tight_layout(pad=2)     # 🔥 FIXED SPACING
    st.pyplot(fig)


# ========================================================
# 2️⃣ MEAN COUNT: Working vs Non-working Days (Q3)
# ========================================================

st.header("🏢 Working vs Non-Working Day Rentals")

mean_working = df_filtered.groupby("workingday")["count"].mean()

fig, ax = plt.subplots(figsize=(7,5))
ax.bar(["Non-Working", "Working"], mean_working)
ax.set_title("Mean Rentals by Working Day Status")
plt.tight_layout(pad=2)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 3️⃣ MEAN COUNT BY MONTH (Q4)
# ========================================================

st.header("📅 Mean Rentals by Month (Both Years Combined)")

month_mean = df_filtered.groupby("month")["count"].mean()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(month_mean.index, month_mean.values, marker="o")
ax.set_title("Mean Monthly Rentals")
ax.set_xlabel("Month")
ax.set_ylabel("Average Count")
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ------------------ 2011 vs 2012 (multi-panel) --------------------
st.subheader("2011 vs 2012 Monthly Rentals")

df_2011 = df[df["year"] == 2011].groupby("month")["count"].mean()
df_2012 = df[df["year"] == 2012].groupby("month")["count"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14,5))

axes[0].plot(df_2011.index, df_2011.values, marker="o")
axes[0].set_title("2011 Monthly Mean Rentals")

axes[1].plot(df_2012.index, df_2012.values, marker="o")
axes[1].set_title("2012 Monthly Mean Rentals")

plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 4️⃣ MEAN & 95% CI FOR WEATHER (Q6)
# ========================================================

st.header("⛅ Weather Effect on Rentals (Mean + 95% CI)")

weather_stats = df_filtered.groupby("weather")["count"].agg(["mean", "std", "count"])
weather_stats["ci95"] = 1.96 * (weather_stats["std"] / weather_stats["count"]**0.5)

fig, ax = plt.subplots(figsize=(10,5))
ax.errorbar(weather_stats.index, weather_stats["mean"], yerr=weather_stats["ci95"], fmt="o-")
ax.set_title("Mean Rentals by Weather with 95% CI")
ax.set_xlabel("Weather Category")
ax.set_ylabel("Count")
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 5️⃣ HOURLY RENTALS (Q7)
# ========================================================

st.header("⏰ Hourly Rental Pattern")

hour_mean = df_filtered.groupby("hour")["count"].mean()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(hour_mean.index, hour_mean.values, marker="o")
ax.set_title("Mean Rentals vs Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("Average Count")
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 6️⃣ HOURLY RENTALS BY DAY OF WEEK (Q8)
# ========================================================

st.header("📆 Hourly Rentals by Day of Week")

days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig, ax = plt.subplots(figsize=(12,6))
for d in range(7):
    group = df_filtered[df_filtered["dayofweek"] == d].groupby("hour")["count"].mean()
    ax.plot(group.index, group.values, label=days[d])

ax.set_title("Hourly Rentals by Day of Week")
ax.legend()
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 7️⃣ HOURLY RENTALS BY SEASON (Q9 Multi-panel)
# ========================================================

st.header("🍂 Seasonal Hourly Rental Trend")

fig, axes = plt.subplots(2, 2, figsize=(14,10))

seasons = df["season_name"].unique()
pos = 0
for i in range(2):
    for j in range(2):
        if pos >= len(seasons):
            break
        season = seasons[pos]
        subset = df[df["season_name"] == season].groupby("hour")["count"].mean()
        axes[i][j].plot(subset.index, subset.values)
        axes[i][j].set_title(f"{season}")
        pos += 1

plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 8️⃣ DAY PERIOD 95% CI (Q10)
# ========================================================

st.header("🌙 Day Period Rental Pattern (Mean + CI)")

period_stats = df_filtered.groupby("day_period")["count"].agg(["mean", "std", "count"])
period_stats["ci95"] = 1.96 * (period_stats["std"] / period_stats["count"]**0.5)

fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(period_stats.index, period_stats["mean"], yerr=period_stats["ci95"], fmt="o-")
ax.set_title("Mean Rentals by Day Period (with CI)")
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


# ========================================================
# 9️⃣ CORRELATION HEATMAP (Q11)
# ========================================================

st.header("📌 Correlation Heatmap")

corr = df_filtered[["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]].corr()

fig, ax = plt.subplots(figsize=(8,6))
cax = ax.imshow(corr, cmap="coolwarm", interpolation="nearest")
ax.set_title("Correlation Matrix")
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.tight_layout(pad=3)        # 🔥 FIXED SPACING
st.pyplot(fig)


st.success("All Assignment II visualizations loaded successfully!")
