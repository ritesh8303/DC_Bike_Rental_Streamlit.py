import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
#                LOAD DATA
# ============================================
@st.cache_data
def load_data(path="train.csv"):
    df = pd.read_csv(path)

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # New columns
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour

    # Season names
    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    # Day period
    def period(h):
        if 0 <= h < 6:
            return "night"
        elif 6 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        return "evening"

    df["day_period"] = df["hour"].apply(period)

    # Weather names
    weather_map = {
        1: "Clear / Few Clouds",
        2: "Mist + Cloudy",
        3: "Light Snow / Rain",
        4: "Heavy Rain / Thunderstorm"
    }
    df["weather_name"] = df["weather"].map(weather_map)

    return df


# ============================================
#              PLOTTING FUNCTIONS
# ============================================
def histograms(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(7, 3 * len(num_cols)))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f"Histogram: {col}")
    return fig


def boxplots(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(7, 3 * len(num_cols)))
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot: {col}")
    return fig


def mean_working(df):
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="workingday", y="count", estimator=np.mean, ax=ax)
    ax.set_xticklabels(["Non-working", "Working"])
    ax.set_title("Mean Rentals: Working vs Non-working")
    return fig


def monthly_mean(df):
    fig, ax = plt.subplots()
    tmp = df.groupby("month")["count"].mean()
    sns.lineplot(x=tmp.index, y=tmp.values, marker="o", ax=ax)
    ax.set_title("Mean Rentals by Month")
    return fig


def monthly_year_split(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for i, y in enumerate([2011, 2012]):
        tmp = df[df["year"] == y].groupby("month")["count"].mean()
        sns.lineplot(x=tmp.index, y=tmp.values, marker="o", ax=axes[i])
        axes[i].set_title(f"Mean Rentals - {y}")
    return fig


def weather_ci(df):
    fig, ax = plt.subplots()
    sns.pointplot(data=df, x="weather_name", y="count", ci=95, ax=ax)
    plt.xticks(rotation=20)
    ax.set_title("Mean Rentals by Weather (95% CI)")
    return fig


def hourly_pattern(df):
    fig, ax = plt.subplots()
    tmp = df.groupby("hour")["count"].mean()
    sns.lineplot(x=tmp.index, y=tmp.values, marker="o", ax=ax)
    ax.set_title("Mean Rentals by Hour")
    return fig


def hourly_by_weekday(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="hour", y="count", hue="day_of_week", estimator="mean")
    ax.set_title("Hourly Rentals by Day of Week")
    return fig


def season_hourly(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="hour", y="count", hue="season_name", estimator="mean")
    ax.set_title("Hourly Rentals by Season")
    return fig


def dayperiod_ci(df):
    fig, ax = plt.subplots()
    sns.pointplot(data=df, x="day_period", y="count", ci=95, ax=ax,
                  order=["night", "morning", "afternoon", "evening"])
    ax.set_title("Mean Rentals by Day Period (95% CI)")
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


# ============================================
#                STREAMLIT APP
# ============================================
def main():

    st.title("🚴 Washington D.C. Bike Rentals Dashboard")
    st.write("Interactive dashboard for Assignment 1, 2, and 3.")

    # Load data
    if not os.path.exists("train.csv"):
        st.error("train.csv not found. Please upload it to the same folder.")
        return

    df = load_data("train.csv")

    # Sidebar filters
    st.sidebar.header("Filters")
    year_filter = st.sidebar.multiselect("Select Year", [2011, 2012], default=[2011, 2012])
    season_filter = st.sidebar.multiselect("Select Season", ["spring", "summer", "fall", "winter"],
                                           default=["spring", "summer", "fall", "winter"])
    hour_filter = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

    df_filtered = df[
        (df["year"].isin(year_filter)) &
        (df["season_name"].isin(season_filter)) &
        (df["hour"].between(hour_filter[0], hour_filter[1]))
    ]

    # ================= Assignment 1 Outputs =================
    st.header("📘 Assignment I — Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.subheader("Total Casual & Registered Rentals per Year")
    st.write(df.groupby("year")[["casual", "registered"]].sum())

    st.subheader("Mean Rentals by Season")
    st.write(df.groupby("season_name")["count"].mean().sort_values(ascending=False))

    st.subheader("Monthly Totals per Year")
    st.write(df.groupby(["year", "month"])["count"].sum())

    st.subheader("Correlation with Rental Count")
    st.write(df.select_dtypes(include=[np.number]).corr()["count"].sort_values(ascending=False))

    st.subheader("Pivot: Day Period vs Working Day")
    st.write(pd.pivot_table(df, values="count", index="day_period", columns="workingday", aggfunc="mean"))

    # ================= Assignment 2 Visualizations =================
    st.header("📊 Assignment II — Visualizations")

    st.pyplot(histograms(df_filtered))
    st.pyplot(boxplots(df_filtered))
    st.pyplot(mean_working(df_filtered))
    st.pyplot(monthly_mean(df_filtered))
    st.pyplot(monthly_year_split(df_filtered))
    st.pyplot(weather_ci(df_filtered))
    st.pyplot(hourly_pattern(df_filtered))
    st.pyplot(hourly_by_weekday(df_filtered))
    st.pyplot(season_hourly(df_filtered))
    st.pyplot(dayperiod_ci(df_filtered))
    st.pyplot(correlation_heatmap(df_filtered))

    # ================= Assignment 3 Dashboard =================
    st.header("🧭 Assignment III — Interactive Exploration")

    grouping = st.selectbox("Group by:", ["hour", "day_period", "month", "season_name", "day_of_week"])
    agg = df_filtered.groupby(grouping)["count"].mean().sort_values(ascending=False)
    st.write(agg)
    st.line_chart(agg)


if __name__ == "__main__":
    main()
