
import streamlit as st
import joblib
import pandas as pd

# Load Linear Regression model
model = joblib.load("LinearRegression_proj_1.pkl")

st.title("🚗 Used Car Price Prediction")

# User inputs
year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2026,
    value=2022
)

km = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=25000
)

engine = st.number_input(
    "Engine CC",
    min_value=500,
    max_value=5000,
    value=1500
)

owners = st.number_input(
    "Previous Owners",
    min_value=1,
    max_value=5,
    value=1
)

brand = st.selectbox(
    "Brand",
    ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Ford", "Mahindra"]
)

fuel = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

if st.button("Predict Price"):

    # Create input DataFrame
    new_car = pd.DataFrame({
        "Year": [year],
        "Kilometers_Driven": [km],
        "Engine_CC": [engine],
        "Previous_Owners": [owners]
    })

    # Add encoded columns
    for col in model.feature_names_in_:
        if col not in new_car.columns:
            new_car[col] = 0

    # Set selected Brand
    brand_col = "Brand_" + brand

    if brand_col in new_car.columns:
        new_car[brand_col] = 1

    # Set selected Fuel Type
    fuel_col = "Fuel_Type_" + fuel

    if fuel_col in new_car.columns:
        new_car[fuel_col] = 1

    # Set selected Transmission
    transmission_col = "Transmission_" + transmission

    if transmission_col in new_car.columns:
        new_car[transmission_col] = 1

    # Arrange columns in the same order as training
    new_car = new_car[model.feature_names_in_]

    # Prediction
    prediction = model.predict(new_car)

    st.success(
        f"Predicted Car Price: ₹{prediction[0]:.2f} Lakh"
    )
