import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Flower Classification AI",
    page_icon="🌸",
    layout="centered"
)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "flower_model_tl.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )


model = load_model()


# -----------------------------
# Constants
# -----------------------------

IMG_SIZE = (180, 180)

class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]


# -----------------------------
# Prediction function
# -----------------------------

def predict_flower(image):

    img = image.resize(IMG_SIZE)

    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(
        img_array,
        verbose=0
    )

    scores = predictions[0]

    predicted_index = np.argmax(scores)

    predicted_class = class_names[predicted_index]

    confidence = 100 * np.max(scores)

    return predicted_class, confidence, scores


# -----------------------------
# UI
# -----------------------------

st.title("🌸 Flower Classification AI")

st.write(
    "Upload an image and let the AI identify the flower."
)

st.divider()


uploaded_file = st.file_uploader(
    "📷 Upload a flower image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    predicted_class, confidence, scores = predict_flower(image)

    st.subheader("🌼 Prediction")

    st.success(
        f"Predicted Flower: **{predicted_class.title()}**"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.divider()

    st.subheader("🏆 Top 3 Predictions")

    top_3_indices = np.argsort(scores)[-3:][::-1]

    for rank, index in enumerate(top_3_indices, start=1):

        flower = class_names[index]
        percentage = scores[index] * 100

        st.write(
            f"**#{rank} {flower.title()}** — {percentage:.2f}%"
        )

        st.progress(float(scores[index]))   

    if confidence < 50:

        st.warning(
            "⚠️ The model is not very confident about this prediction."
        )


# -----------------------------
# Model information
# -----------------------------

with st.expander("🤖 About the Model"):

    st.write("**Model:** MobileNetV2 Transfer Learning")

    st.write("**Number of Classes:** 5")

    st.write("**Input Size:** 180 × 180 pixels")

    st.write("**Validation Accuracy:** ~90.6%")