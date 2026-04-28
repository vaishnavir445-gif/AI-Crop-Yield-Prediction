import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

st.title("🌾 AI-Based Crop Yield Prediction & Optimization")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.write(df.head())

    required_cols = ['Crop','State','Season','Area','Annual_Rainfall','Fertilizer','Pesticide','Yield']

    if all(col in df.columns for col in required_cols):

        # Encoding categorical data
        le_crop = LabelEncoder()
        le_state = LabelEncoder()
        le_season = LabelEncoder()

        df['Crop'] = le_crop.fit_transform(df['Crop'])
        df['State'] = le_state.fit_transform(df['State'])
        df['Season'] = le_season.fit_transform(df['Season'])

        # Features
        X = df[['Crop','State','Season','Area','Annual_Rainfall','Fertilizer','Pesticide']]
        y = df['Yield']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Accuracy
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        st.write(f"### Model Accuracy (R²): {r2:.2f}")

        st.write("---")
        st.write("### Enter Input Values")

        # UI inputs
        crop_input = st.selectbox("Crop", le_crop.classes_)
        state_input = st.selectbox("State", le_state.classes_)
        season_input = st.selectbox("Season", le_season.classes_)

        area = st.number_input("Area (hectares)", value=100.0)
        rainfall = st.number_input("Annual Rainfall (mm)", value=1000.0)
        fertilizer = st.number_input("Fertilizer (kg)", value=100.0)
        pesticide = st.number_input("Pesticide (kg)", value=10.0)

        # Prediction
        if st.button("Predict Yield"):

            input_data = np.array([[
                le_crop.transform([crop_input])[0],
                le_state.transform([state_input])[0],
                le_season.transform([season_input])[0],
                area, rainfall, fertilizer, pesticide
            ]])

            prediction = model.predict(input_data)

            st.success(f"🌾 Predicted Yield: {prediction[0]:.2f}")

            # Suggestions
            st.write("### 📌 Suggestions")

            if rainfall < 500:
                st.warning("Low rainfall → Consider irrigation")
            elif rainfall > 2000:
                st.warning("High rainfall → Risk of flooding")

            if fertilizer < 50:
                st.warning("Low fertilizer usage → May reduce yield")

            if pesticide < 5:
                st.warning("Low pesticide usage → Risk of crop damage")

            if prediction[0] > 3:
                st.success("Good yield expected!")

        # Optimization
        st.write("---")
        if st.button("Find Optimal Conditions"):

            best_yield = -1
            best_values = None

            for rain in range(500, 2001, 200):
                for fert in range(50, 301, 50):
                    for pest in range(5, 51, 5):

                        input_data = np.array([[
                            le_crop.transform([crop_input])[0],
                            le_state.transform([state_input])[0],
                            le_season.transform([season_input])[0],
                            area, rain, fert, pest
                        ]])

                        pred = model.predict(input_data)[0]

                        if pred > best_yield:
                            best_yield = pred
                            best_values = (rain, fert, pest)

            st.success(f"🌟 Optimal Yield: {best_yield:.2f}")
            st.write(f"Best Rainfall: {best_values[0]} mm")
            st.write(f"Best Fertilizer: {best_values[1]} kg")
            st.write(f"Best Pesticide: {best_values[2]} kg")

    else:
        st.error("Dataset format incorrect")

else:
    st.info("Upload dataset to begin")