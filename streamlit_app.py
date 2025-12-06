import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# --------------------------
# Load Model + Vectorizers
# --------------------------

MODEL_PATH = "model.keras"
CV1_PATH = "cv1.pkl"   # headline vectorizer
CV2_PATH = "cv2.pkl"   # review body vectorizer

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load CountVectorizers
cv1 = joblib.load(CV1_PATH)
cv2 = joblib.load(CV2_PATH)

st.title("🛍 Amazon Review Sentiment Classifier")
st.write("Enter review headline & review body to classify as **Positive** or **Negative**.")


# Streamlit Input Fields

headline = st.text_input("📝 Review Headline")
review_body = st.text_area("✏ Review Body")


# Prediction Function

def preprocess_and_predict(headline_text, body_text):
    # Convert to vector using saved vectorizers
    vec1 = cv1.transform([headline_text])
    vec2 = cv2.transform([body_text])

    # Combine both vectors
    from scipy.sparse import hstack
    combined = hstack([vec1, vec2])

    # Convert sparse to dense for keras
    combined = combined.toarray()

    # Predict
    pred = model.predict(combined)[0][0]

    sentiment = "Positive 😀" if pred >= 0.75 else "Negative 😕"
    probability = float(pred)

    return sentiment, probability


# Predict Button

if st.button("Predict Sentiment"):
    if headline.strip() == "" or review_body.strip() == "":
        st.warning("⚠ Please enter BOTH headline and review body.")
    else:
        sentiment, prob = preprocess_and_predict(headline, review_body)

        st.success(f"### ⭐ Sentiment: **{sentiment}**")
        st.info(f"### 📊 Model confidence: **{prob:.3f}**")

