import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the machine learning model
with open(r'C:\Users\conne\OneDrive\Documents\GitHub\data_Analysis_project\Pandas_Project\trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the input fields
st.title("Prediction Application")

name = st.text_input("Name")
marks = st.number_input("Marks", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
half_marks = st.slider("Half Marks", min_value=0, max_value=50)
male_female = st.radio("Male or Female", ("Male", "Female"))

# Convert gender and male_female to binary values if necessary
gender_binary = 1 if gender == "Male" else 0
male_female_binary = 1 if male_female == "Male" else 0

# Prepare the input data for the model
input_data = np.array([[marks, gender_binary, half_marks, male_female_binary]])

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The prediction for {name} is: {prediction[0]}")

    # Plotting the prediction
    fig, ax = plt.subplots()
    ax.bar(["Prediction"], prediction)
    st.pyplot(fig)
