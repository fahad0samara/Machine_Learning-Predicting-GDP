import streamlit as st
import pandas as pd
import joblib

# Load preprocessed data and models
df = pd.read_csv('cleaned_countries_data.csv')
gb_model = joblib.load('best_gb_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Features used in the models
required_features = ['latitude', 'longitude', 'agricultural_land', 'forest_area', 'land_area',
                     'rural_land', 'urban_land', 'central_government_debt_pct_gdp', 'expense_pct_gdp',
                     'inflation', 'self_employed_pct', 'tax_revenue_pct_gdp', 'unemployment_pct',
                     'vulnerable_employment_pct', 'electricity_access_pct', 'alternative_nuclear_energy_pct',
                     'urban_rural_ratio', 'gdp_per_land_area']

# Verify if required features are present
available_features = [feature for feature in required_features if feature in df.columns]
st.write("Available Features:", available_features)

# Check if required features are available
if all(feature in available_features for feature in required_features):
    # Allow user input for features
    feature_values = {feature: st.number_input(f"{feature}", value=0.0) for feature in available_features}
    
    # Convert feature values to DataFrame
    features_df = pd.DataFrame([feature_values], columns=available_features)
    
    # Debug: Print the features used for prediction
    st.write("Features for Prediction:", features_df)
    
    # Make predictions
    if st.button('Predict'):
        try:
            # Make predictions
            gb_pred = gb_model.predict(features_df)[0]
            rf_pred = rf_model.predict(features_df)[0]
            xgb_pred = xgb_model.predict(features_df)[0]
            
            st.write(f"Gradient Boosting Prediction: ${gb_pred:.2f}")
            st.write(f"Random Forest Prediction: ${rf_pred:.2f}")
            st.write(f"XGBoost Prediction: ${xgb_pred:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("The required features are not available in the dataset.")
