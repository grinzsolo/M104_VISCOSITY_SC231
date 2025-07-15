
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Viscosity Prediction App using Gradient Boosting")

uploaded_file = st.file_uploader("Upload CSV file for training", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df.head())

    # Select features
    features = ['IMP_Flash_Point', 'IMP_Sp.Gr.@_6060_deg.F', 'Viscosity']
    df = df[features].dropna()

    # Remove outliers
    def remove_outliers_iqr(dataframe, columns):
        df_clean = dataframe.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean

    df_cleaned = remove_outliers_iqr(df, features)
    st.write("Data after Outlier Removal", df_cleaned.head())

    # Define X, y
    X = df_cleaned[['IMP_Flash_Point', 'IMP_Sp.Gr.@_6060_deg.F']]
    y = df_cleaned['Viscosity']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with Grid Search
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Model Evaluation")
    st.write("Best Parameters:", grid_search.best_params_)
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"RMSE: {rmse:.4f}")

    # Prediction form
    st.subheader("Predict Viscosity from New Input")
    flash_point = st.number_input("IMP_Flash_Point", min_value=0.0, value=100.0)
    sp_gravity = st.number_input("IMP_Sp.Gr.@_6060_deg.F", min_value=0.0, value=0.85)

    if st.button("Predict Viscosity"):
        new_data = pd.DataFrame([[flash_point, sp_gravity]], columns=['IMP_Flash_Point', 'IMP_Sp.Gr.@_6060_deg.F'])
        prediction = best_model.predict(new_data)
        st.success(f"Predicted Viscosity: {prediction[0]:.4f}")
