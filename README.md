# heat-pump-ml

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heat_pump_data_extended.csv")

df = load_data()

# Split data
X = df[['Evaporator_Temp', 'Condenser_Temp', 'Power_Input']]
y = df['COP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Heat Pump COP Prediction Using Machine Learning")

st.write("This app predicts the *Coefficient of Performance (COP)* of a heat pump based on its operating conditions.")

# User Input
evap_temp = st.number_input("Evaporator Temperature (°C)", min_value=-10.0, max_value=15.0, value=0.0)
cond_temp = st.number_input("Condenser Temperature (°C)", min_value=25.0, max_value=60.0, value=40.0)
power_input = st.number_input("Power Input (kW)", min_value=1.0, max_value=10.0, value=3.0)

# Predict COP
if st.button("Predict COP"):
    user_input = np.array([[evap_temp, cond_temp, power_input]])
    predicted_cop = model.predict(user_input)[0]
    st.success(f"Predicted COP: {predicted_cop:.2f}")

    # Predictions for monitoring
    y_pred = model.predict(X_test)

    # Performance Visualization
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_xlabel("Actual COP")
    ax.set_ylabel("Predicted COP")
    ax.set_title("Actual vs. Predicted COP")
    st.pyplot(fig)

# Display sample dataset
if st.checkbox("Show Sample Data"):
    st.dataframe(df.head())
