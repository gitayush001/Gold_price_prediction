import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Gold Price Prediction", page_icon="🏆", layout="centered")

st.title("🏆 Gold Price Prediction App")
st.write("Enter market values below to predict the gold price using your trained ML model.")


@st.cache_resource
def load_model():
    """Load trained pickle model from the project folder."""
    possible_model_files = [
        "gold_price.pkl",
        "gold_price_model.pkl",
        "model.pkl",
        "Gold_Price.pkl",
    ]

    for file_name in possible_model_files:
        model_path = Path(file_name)
        if model_path.exists():
            with open(model_path, "rb") as file:
                return pickle.load(file), file_name

    st.error(
        "Model file not found. Please keep your trained pickle file in the same folder as app.py "
        "and name it gold_price.pkl"
    )
    st.stop()


model, loaded_file = load_model()
st.success(f"Model loaded successfully: {loaded_file}")

st.subheader("Input Features")

spx = st.number_input("SPX Value", min_value=0.0, value=1447.16, step=0.01)
uso = st.number_input("USO Value", min_value=0.0, value=78.47, step=0.01)
slv = st.number_input("SLV Value", min_value=0.0, value=15.18, step=0.01)
eur_usd = st.number_input("EUR/USD Value", min_value=0.0, value=1.47, step=0.0001, format="%.4f")

if st.button("Predict Gold Price"):
    input_data = pd.DataFrame(
        [[spx, uso, slv, eur_usd]],
        columns=["SPX", "USO", "SLV", "EUR/USD"],
    )

    prediction = model.predict(input_data)
    predicted_price = float(np.ravel(prediction)[0])

    st.subheader("Prediction Result")
    st.success(f"Predicted Gold Price: {predicted_price:.2f}")

st.markdown("---")
st.caption("Make sure app.py and gold_price.pkl are in the same folder before deploying on Streamlit Cloud.")
