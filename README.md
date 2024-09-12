

# Predicting GDP with Machine Learning

## Project Overview

This project aims to predict the Gross Domestic Product (GDP) of countries using machine learning algorithms. By leveraging different regression models, this project analyzes various country-level features to build predictive models for GDP. The primary models used are Gradient Boosting, Random Forest, and XGBoost.

### Objectives

1. **Data Preprocessing:** Clean and preprocess country-level data to make it suitable for training machine learning models.
2. **Model Training:** Train and evaluate different regression models to predict GDP based on various features.
3. **Model Evaluation:** Assess model performance using metrics such as Mean Squared Error (MSE) and R-squared score.
4. **Feature Analysis:** Identify and visualize the most important features affecting GDP predictions.
5. **Regional Analysis:** Analyze GDP variations across different regions and explore correlations with other economic indicators.

## Dataset

- **Countries.csv:** Original dataset containing raw country-level data.
- **cleaned_countries_data.csv:** Cleaned version of the dataset with missing values and outliers handled.
- **preprocessed_countries.csv:** Dataset after feature engineering and scaling, ready for model training.

## Models

1. **Gradient Boosting Regressor:**
   - **Best Parameters:** `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}`
   - **Performance:** Evaluated using MSE and R-squared score. Includes feature importance visualization.

2. **Random Forest Regressor:**
   - **Pre-trained Model:** Saved as `random_forest_model.pkl`
   - **Performance:** Evaluated and compared with other models.

3. **XGBoost Regressor:**
   - **Pre-trained Model:** Saved as `xgboost_model.pkl`
   - **Performance:** Evaluated and compared with other models.

## Usage

### Running the Streamlit App

1. Navigate to the project directory where `app.py` is located.
2. Run the Streamlit app with the following command:

   ```bash
   streamlit run app.py
   ```

   This will launch a local server where you can view the app in your browser. The app displays the Mean Squared Error (MSE) values for the Random Forest and XGBoost models.

### Training and Analyzing Models

1. Open the Jupyter notebook `predicting-gdp-with-machine-learning.ipynb` to explore data analysis and model training.
2. The notebook includes:

   - **Data Cleaning:** Steps to clean and preprocess the raw dataset.
   - **Model Training:** Code for training Gradient Boosting, Random Forest, and XGBoost models.
   - **Model Evaluation:** Metrics and plots for evaluating model performance.
   - **Feature Importance:** Visualization of the top features impacting GDP predictions.
   - **Regional Analysis:** Analysis of GDP by different regions and correlations with other economic indicators.

## Files

- `app.py`: Streamlit application for model performance visualization.
- `best_gb_model.pkl`: Saved Gradient Boosting model with the best parameters.
- `random_forest_model.pkl`: Saved Random Forest model.
- `xgboost_model.pkl`: Saved XGBoost model.
- `cleaned_countries_data.csv`: Cleaned dataset.
- `Countries.csv`: Raw dataset.
- `preprocessed_countries.csv`: Preprocessed dataset.
- `predicting-gdp-with-machine-learning.ipynb`: Jupyter notebook for model training and analysis.
- `requirements.txt`: List of Python dependencies required for the project.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the interactive web application framework.
- [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/) for the machine learning libraries.

