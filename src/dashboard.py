import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from ollama_explainer import explain_prediction
from chatbot import chatbot_reply
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="MindWatch AI",
    page_icon="💙",
    layout="wide"
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="main-title">MindWatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your AI companion for emotional wellness</div>', unsafe_allow_html=True)

model = joblib.load("outputs/tfidf_lr.joblib")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Emotion","📊 Mental Health Trends","🤖 AI Assistant"])


with tab1:

    st.markdown("### Share how you're feeling")

    text = st.text_area(
    "Enter your message",
    placeholder="Type how you feel...",
    label_visibility="collapsed"
)

if st.button("Analyze Emotion"):

    prediction = model.predict([text])[0]

    if prediction == 1:
        st.error("⚠ Possible Depression Risk")
    else:
        st.success("✓ Emotion looks Healthy")

    explanation = explain_prediction(text, prediction)

    st.subheader("🧠 AI Explanation")
    st.write(explanation)


with tab2:

    df = pd.read_csv("data/processed/user_windows.csv")

    trend = df.groupby("window_id")["label"].mean().reset_index()

    fig = px.line(
        trend,
        x="window_id",
        y="label",
        markers=True,
        title="Depression Risk Trend Over Time"
    )

    st.plotly_chart(fig, width="stretch")


with tab3:

    user_msg = st.text_input("Talk with the assistant")

    if st.button("Send"):

        reply = chatbot_reply(user_msg)

        st.write(reply)
