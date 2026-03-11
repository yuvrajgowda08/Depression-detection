import streamlit as st
import joblib

# load trained model
model = joblib.load("outputs/tfidf_lr.joblib")

st.title("Depression Detection from Text")

st.write("Type a message and the model will predict depression risk.")

text = st.text_area("Enter text")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([text])[0]

        if prediction == 1:
            st.error("⚠ Depression Risk Detected")
        else:
            st.success("No Depression Risk Detected")