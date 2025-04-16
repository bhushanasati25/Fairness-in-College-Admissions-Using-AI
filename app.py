import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your saved model (adjust the filename and method if needed)
@st.cache_resource
def load_model():
    # if you saved using joblib, use joblib.load; if you used pickle, use pickle.load
    model = joblib.load("college_admissions_model.pkl")
    return model

model = load_model()

# Streamlit app title
st.title("College Admissions Selectivity Predictor")
st.write("This application uses a fairness-aware model to predict if an institution is selective.")

# Input fields for a new observation
st.sidebar.header("Input Features")
def get_input_features():
    # For illustration, we'll use SAT_AVG as our input feature.
    # You can extend this function to include more features.
    sat_avg = st.sidebar.number_input("Average SAT Score", value=1000, min_value=400, max_value=1600, step=10)
    data = {
        "SAT_AVG": sat_avg
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_features = get_input_features()
st.subheader("Input Data")
st.write(input_features)

# Make prediction and display result
prediction = model.predict(input_features)
if prediction[0] == 1:
    st.success("The model predicts that the institution is **selective**.")
else:
    st.info("The model predicts that the institution is **non-selective**.")

# (Optional) Display additional information
st.write("Note: This is a demo version. To deploy this application in production, consider using Docker or a cloud service.")

# If you want to add fairness metrics or other components, you could allow file upload and compute these metrics on new datasets.

