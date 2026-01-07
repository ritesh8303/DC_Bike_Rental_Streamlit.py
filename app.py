import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bike Sharing – Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour

    bins = [0, 6, 12, 18, 24]
    labels = ["night", "morning", "afternoon", "evening"]
    df["day_period"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False)

    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    return df

df = load_data()

st.title("Washington D.C. Bike Sharing – Interactive Dashboard")
st.caption("Dashboard summarizing Assignments I & II on the Kaggle bike-sharing-demand data.")

# --------- SIDEBAR FILTERS (≥3 widgets) ---------
with st.sidebar:
    st.header("Filters")

    year_filter = st.multiselect(
        "Year",
        options=sorted(df["year"].unique()),
        default=sorted(df["year"].unique())
    )

    season_filter = st.multiselect(
        "Season",
        options=sorted(df["season_name"].dropna().unique()),
        default=sorted(df["season_name"].dropna().unique())
    )

    workingday_filter = st.selectbox(
        "Working day filter",
        options=["All days", "Working days only", "Non-working days only"]
    )

    plot_choice = st.selectbox(
        "Select plot",
        [
            "Histogram of rentals",
            "Monthly mean rentals",
            "Hourly pattern by day of week",
            "Weather vs rentals (mean + 95% CI)",
            "Period of day vs rentals (95% CI)",
            "Correlation heatmap",
        ]
    )

# --------- APPLY FILTERS ---------
filtered = df[df["year"].isin(year_filter)]
filtered = filtered[filtered["season_name"].isin(season_filter)]

if workingday_filter == "Working days only":
    filtered = filtered[filtered["workingday"] == 1]
elif workingday_filter == "Non-working days only":
    filtered = filtered[filtered["workingday"] == 0]

st.markdown(f"**Filtered rows:** {len(filtered):,} (out of {len(df):,})")

sns.set_theme(style="whitegrid")

# --------- PLOTS (reuse your notebook code here) ---------

if plot_choice == "Histogram of rentals":
    st.subheader("Distribution of hourly rentals (count)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=filtered, x="count", kde=True, ax=ax)
    ax.set_xlabel("Hourly rentals (count)")
    st.pyplot(fig)

elif plot_choice == "Monthly mean rentals":
    st.subheader("Mean hourly rentals by month")
    month_means = (
        filtered.groupby("month")["count"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=month_means, x="month", y="count", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean hourly rentals")
    st.pyplot(fig)

elif plot_choice == "Hourly pattern by day of week":
    st.subheader("Hourly rentals by day of week")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=filtered,
        x="hour",
        y="count",
        hue="dayofweek",
        estimator="mean",
        ci=None,
        marker="o",
        ax=ax
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean hourly rentals")
    ax.legend(title="Day of week (0=Mon)")
    st.pyplot(fig)

elif plot_choice == "Weather vs rentals (mean + 95% CI)":
    st.subheader("Mean rentals by weather category (95% CI)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.pointplot(
        data=filtered,
        x="weather",
        y="count",
        ci=95,
        join=False,
        capsize=0.2,
        ax=ax
    )
    ax.set_xlabel("Weather category (1–4)")
    ax.set_ylabel("Mean hourly rentals")
    st.pyplot(fig)

elif plot_choice == "Period of day vs rentals (95% CI)":
    st.subheader("Mean rentals by period of day and workingday (95% CI)")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.pointplot(
        data=filtered,
        x="day_period",
        y="count",
        hue="workingday",
        ci=95,
        dodge=0.3,
        capsize=0.2,
        ax=ax
    )
    ax.set_xlabel("Period of day")
    ax.set_ylabel("Mean hourly rentals")
    ax.legend(title="Working day (0/1)")
    st.pyplot(fig)

elif plot_choice == "Correlation heatmap":
    st.subheader("Correlation between numerical variables")
    num_cols = ["temp", "atemp", "humidity", "windspeed",
                "casual", "registered", "count"]
    corr = filtered[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    st.pyplot(fig)
