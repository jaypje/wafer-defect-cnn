"""
Streamlit demo for the wafer defect pattern classifier.

Run with:
    streamlit run app.py

Expects a trained model at ./wafer_defect_cnn.keras (saved by the notebook) and, optionally,
data/wm811k_processed.npz to populate the "try a sample wafer" picker.
"""

import numpy as np
import streamlit as st
import tensorflow as tf

VALID_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random",
                  "Scratch", "Near-full", "none"]

ROOT_CAUSE = {
    "Edge-Ring": "Edge-handling / edge-bead removal / chamber edge effects -> route to edge-handling equipment engineer",
    "Center": "Non-uniform deposition/etch across wafer radius -> route to process uniformity check",
    "Donut": "Radial process non-uniformity (mid-radius) -> route to process uniformity check",
    "Edge-Loc": "Localized edge defect (particle/contact issue near edge) -> inspect edge handling + particle counts",
    "Loc": "Localized defect cluster, not edge-specific -> investigate specific process step for that lot",
    "Scratch": "Physical scratch during handling -> route to handling/robotics/logistics",
    "Random": "Particle contamination, no systematic cause -> route to yield/particle control",
    "Near-full": "Catastrophic process excursion -> immediate escalation",
    "none": "No actionable pattern -> no action; normal yield loss",
}

CONFIDENCE_THRESHOLD = 0.90  # matches the operating point chosen in the notebook


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("wafer_defect_cnn.keras")


@st.cache_data
def load_sample_wafers():
    """Optional: lets the user pick a real test-set example instead of uploading a file."""
    try:
        data = np.load("data/wm811k_processed.npz")
        return data["X_test"], data["y_test"]
    except FileNotFoundError:
        return None, None


def predict(model, wafer_map):
    """wafer_map: (64, 64) or (64, 64, 1) array, values already preprocessed as in training."""
    x = np.asarray(wafer_map, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., np.newaxis]
    x = x[np.newaxis, ...]  # add batch dim
    proba = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    return VALID_CLASSES[pred_idx], float(proba[pred_idx]), proba


def render_wafer(wafer_map, size=280):
    """Render a wafer map array as a simple grayscale image for display."""
    img = np.squeeze(wafer_map)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    st.image(img, width=size, clamp=True)


st.set_page_config(page_title="Wafer Defect Classifier", layout="centered")
st.title("Wafer Defect Pattern Classifier")
st.caption(
    "CNN trained on WM-811K. Classifies a wafer's pass/fail spatial pattern into one of 9 "
    "failure signatures and maps it to a likely engineering action."
)

model = load_model()
X_test, y_test = load_sample_wafers()

st.subheader("1. Choose a wafer map")
source = st.radio("Source", ["Pick a sample from the test set", "Upload a .npy file"], horizontal=True)

wafer_map = None
true_label = None

if source == "Pick a sample from the test set":
    if X_test is None:
        st.warning("data/wm811k_processed.npz not found — place it there to use sample wafers, "
                    "or switch to 'Upload a .npy file'.")
    else:
        idx = st.slider("Test-set index", 0, len(X_test) - 1, 0)
        wafer_map = X_test[idx]
        true_label = VALID_CLASSES[int(y_test[idx])]
else:
    uploaded = st.file_uploader("Upload a preprocessed 64x64 wafer map (.npy)", type=["npy"])
    if uploaded is not None:
        wafer_map = np.load(uploaded)

if wafer_map is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Wafer map")
        render_wafer(wafer_map)
        if true_label is not None:
            st.caption(f"True label: **{true_label}**")

    with col2:
        st.subheader("3. Prediction")
        pred_class, confidence, proba = predict(model, wafer_map)
        st.metric("Predicted class", pred_class, f"{confidence:.1%} confidence")

        if confidence >= CONFIDENCE_THRESHOLD:
            st.success(f"Confidence >= {CONFIDENCE_THRESHOLD:.0%} threshold -> auto-classify")
        else:
            st.warning(f"Confidence below {CONFIDENCE_THRESHOLD:.0%} threshold -> route to manual review")

        st.markdown(f"**Suggested action:** {ROOT_CAUSE[pred_class]}")

    st.subheader("4. Full probability breakdown")
    proba_dict = {cls: float(p) for cls, p in zip(VALID_CLASSES, proba)}
    st.bar_chart(proba_dict)
else:
    st.info("Choose or upload a wafer map above to see a prediction.")

st.divider()
st.caption(
    "This is a portfolio/demo app, not a production tool. See the notebook for full methodology, "
    "error analysis, and the confidence-threshold study behind the 0.90 operating point used here."
)
