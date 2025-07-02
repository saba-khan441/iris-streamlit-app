
import streamlit as st
import pickle
import numpy as np

# Load model (no folder needed since it's in the same folder)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Classifier")
st.write("Enter features to classify the flower:")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {prediction}")
