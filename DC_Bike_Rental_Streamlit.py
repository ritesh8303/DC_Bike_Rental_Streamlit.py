import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DC Bike Rental Dashboard", layout="wide")
st.title("🚲 Washington D.C. Bike Rentals Dashboard")

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
hour_filter = st.sidebar.slider("Hour Range", 0, 23, (0,23))

df_filtered = df[(df["year"].isin(year_filter)) & 
                 (df["season_name"].isin(season_filter)) &
                 (df["hour"].between(hour_filter[0], hour_filter[1]))]

# ---------------------------
# Key Metrics
# ---------------------------
st.header("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", int(df_filtered["count"].sum()))
col2.metric("Peak Month", df_filtered.groupby("month")["count"].sum().idxmax())
col3.metric("Peak Hour", df_filtered.groupby("hour")["count"].mean().idxmax())

# ---------------------------
# Collapsible EDA Section
# ---------------------------
with st.expander("View Dataset Summary & Pivot Table"):
    st.subheader("Dataset Preview")
    st.dataframe(df_filtered.head())

    st.subheader("Total Casual & Registered Rentals per Year")
    st.write(df_filtered.groupby("year")[["casual","registered"]].sum())

    st.subheader("Pivot: Day Period vs Working Day (Mean Rentals)")
    st.write(pd.pivot_table(df_filtered, values="count", index="day_period", columns="workingday", aggfunc="mean"))

# ---------------------------
# Interactive Plot Selector
# ---------------------------
st.header("📈 Visualizations")
plot_options = [
    "Mean Rentals by Season",
    "Hourly Rentals",
    "Mean Rentals by Month",
    "Hourly Rentals by Day of Week",
    "Weather Effect on Rentals (Mean + 95% CI)",
    "Correlation Heatmap"
]
selected_plot = st.selectbox("Select Plot to Display", plot_options)

# ---------------------------
# Plot Functions
# ---------------------------
def plot_mean_by_season(df):
    mean_season = df.groupby("season_name")["count"].mean()
    fig, ax = plt.subplots(figsize=(7,4))
    mean_season.plot(kind="bar", ax=ax)
    ax.set_title("Mean Rentals by Season")
    plt.tight_layout(pad=2)
    st.pyplot(fig)

def plot_hourly(df):
    hourly_mean = df.groupby("hour")["count"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hourly_mean.index, hourly_mean.values, marker="o")
    ax.set_title("Mean Rentals by Hour")
    plt.tight_layout(pad=2)
    st.pyplot(fig)

def plot_monthly(df):
    month_mean = df.groupby("month")["count"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(month_mean.index, month_mean.values, marker="o")
    ax.set_title("Mean Rentals by Month")
    plt.tight_layout(pad=2)
    st.pyplot(fig)

def plot_hourly_by_weekday(df):
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig, ax = plt.subplots(figsize=(10,5))
    for d in range(7):
        group = df[df["dayofweek"]==d].groupby("hour")["count"].mean()
        ax.plot(group.index, group.values, label=days[d])
    ax.set_title("Hourly Rentals by Day of Week")
    ax.legend()
    plt.tight_layout(pad=2)
    st.pyplot(fig)

def plot_weather_ci(df):
    weather_stats = df.groupby("weather")["count"].agg(["mean","std","count"])
    weather_stats["ci95"] = 1.96 * (weather_stats["std"]/weather_stats["count"]**0.5)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.errorbar(weather_stats.index, weather_stats["mean"], yerr=weather_stats["ci95"], fmt="o-")
    ax.set_title("Weather Effect on Rentals (Mean + 95% CI)")
    plt.tight_layout(pad=2)
    st.pyplot(fig)

def plot_corr_heatmap(df):
    corr = df[["temp","atemp","humidity","windspeed","casual","registered","count"]].corr()
    fig, ax = plt.subplots(figsize=(7,5))
    cax = ax.imshow(corr, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout(pad=2)
    st.pyplot(fig)

# ---------------------------
# Display Selected Plot
# ---------------------------
if selected_plot == "Mean Rentals by Season":
    plot_mean_by_season(df_filtered)
elif selected_plot == "Hourly Rentals":
    plot_hourly(df_filtered)
elif selected_plot == "Mean Rentals by Month":
    plot_monthly(df_filtered)
elif selected_plot == "Hourly Rentals by Day of Week":
    plot_hourly_by_weekday(df_filtered)
elif selected_plot == "Weather Effect on Rentals (Mean + 95% CI)":
    plot_weather_ci(df_filtered)
elif selected_plot == "Correlation Heatmap":
    plot_corr_heatmap(df_filtered)

st.success("✅ Dashboard ready for interactive exploration!")
