import base64
import os

import requests
import streamlit as st

# Set this to your deployed backend URL, e.g. https://<you>-leaflens-api.hf.space
API_URL = os.environ.get("CNN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="LeafLens", page_icon="🍃", layout="centered")

st.title("🍃 LeafLens")
st.caption("Upload a potato leaf photo to check for early blight, late blight, or a healthy leaf.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL, language="text")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        st.success(f"Connected — {health['architecture']}, trained={health['trained']}")
    except requests.exceptions.RequestException:
        st.error("Can't reach the API. Check CNN_API_URL.")

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded leaf", use_container_width=True)

    if st.button("Classify leaf", type="primary"):
        with st.spinner("Running the model..."):
            raw = uploaded.getvalue()
            b64 = base64.b64encode(raw).decode("utf-8")
            data_url = f"data:{uploaded.type};base64,{b64}"

            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"imageDataUrl": data_url},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.subheader(result["summary"])

        if not result["trained"]:
            st.warning("No trained checkpoint is loaded on the backend — this is a placeholder result.")

        cols = st.columns(3)
        cols[0].metric("Plant", result["plant"])
        cols[1].metric("Diagnosis", result["disease"])
        cols[2].metric("Confidence", result["confidence"])

        st.markdown("**Top predictions**")
        for pred in result["topK"]:
            st.progress(pred["probability"], text=f"{pred['label']} — {pred['probability']:.1%}")

        tabs = st.tabs(["Symptoms", "Causes", "Treatment", "Prevention"])
        for tab, key in zip(tabs, ["symptoms", "causes", "treatment", "prevention"]):
            with tab:
                for item in result[key]:
                    st.markdown(f"- {item}")
