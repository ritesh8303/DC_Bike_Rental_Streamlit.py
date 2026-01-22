🚲 DC Bike Rental Demand Forecaster

Streamlit app predicting hourly Capital Bikeshare rentals (cnt) from UCI dataset. Regression ML with weather/time features. RMSE ~145 (XGBoost); insights: temp/hour drive peaks.
Features

EDA: Trends by season/weather (Plotly).

Predict: Sidebar sliders for inputs.

Model viz: Feature importance, residuals.

Quick Start
git clone https://github.com/ritesh8303/dc-bike-rental
cd dc-bike-rental
pip install -r requirements.txt
streamlit run DC_Bike_Rental_Streamlit.py

Dataset: hour.csv (UCI: temp, hum, windspeed, casual, registered, cnt).

Performance

Model	RMSE	R²
Linear	220	0.72
RF	160	0.85
XGBoost	145	0.89
Tech: Streamlit, XGBoost, Plotly, Pandas.

Contributions open!
