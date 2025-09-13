import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Constants
MAX_FEATURES = 5000
MAXLEN = 200

# Load model
@st.cache_resource
def load_rnn_model():
    return load_model(r"C:\Users\vikra\OneDrive\Documents\rnn\imdb_rnn_model.h5")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open(r"C:\Users\vikra\OneDrive\Documents\rnn\tokenizer.json", "r") as f:
        tokenizer_json = f.read()  # read as string
    return tokenizer_from_json(tokenizer_json)


model = load_rnn_model()
tokenizer = load_tokenizer()

# Streamlit UI
st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict whether it's **positive** or **negative**.")

# Input from user
user_input = st.text_area("📝 Your Review", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        # Preprocess and predict
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAXLEN)

        prediction = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Output result
        st.subheader("🔍 Prediction")
        st.markdown(f"**Sentiment**: {sentiment}")
        st.markdown(f"**Confidence**: {confidence:.2f}")

