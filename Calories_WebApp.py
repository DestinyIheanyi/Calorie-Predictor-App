import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load("calories_trained_model.sav")

# App Config
st.set_page_config(page_title="Calories Burnt Predictor", layout="centered")

# Title
st.title("🔥 Calories Burnt Predictor")
st.write("Estimate how many calories you've burnt based on your activity and body metrics.")

# Session storage for visualizing predictions
if "history" not in st.session_state:
    st.session_state["history"] = []

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 80, 25)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
duration = st.slider("Exercise Duration (minutes)", 1, 180, 30)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1, value=37.0)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)

gender_binary = 1 if gender == "Male" else 0

# Features
features = np.array([[age, height, weight, duration, body_temp, heart_rate, gender_binary]])

# Predict
if st.button("Predict Calories Burnt"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Calories Burnt: **{prediction:.2f}** 🔥")

    # Save to history
    st.session_state["history"].append({
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "body_temp": body_temp,
        "heart_rate": heart_rate,
        "Calories Burnt": prediction
    })

# Show history and graph
if st.session_state["history"]:
    st.subheader("📊 Prediction History")
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history)

    # Plot line chart of predictions
    st.line_chart(df_history["Calories Burnt"], use_container_width=True)

    # Optional: Activity-based bar chart
    #fig, ax = plt.subplots()
   # df_history.groupby("Calories Burnt").mean().plot(kind="bar", ax=ax, color="orange")
   # ax.set_ylabel("Avg Calories Burnt")
    #ax.set_title("🔥 Avg Calories Burnt per Activity Type")
    #st.pyplot(fig)
