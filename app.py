import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("flower_model.keras")


# -----------------------------
# Settings
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

    # Resize image
    img = image.resize(IMG_SIZE)

    # Convert image into numbers
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get probabilities
    scores = predictions[0]

    # Find class with highest probability
    predicted_class = class_names[np.argmax(scores)]

    # Get highest probability
    confidence = 100 * np.max(scores)

    return predicted_class, confidence, scores


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌸 Flower Classification AI")

st.write("Upload a flower image and let the CNN identify it.")


# Upload image
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)


# If image is uploaded
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image")


    # Get prediction
    predicted_class, confidence, scores = predict_flower(image)


    # Display main prediction
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")


    # Display probabilities
    st.divider()

    st.subheader("Prediction Probabilities")

    for flower, score in zip(class_names, scores):

        st.write(f"{flower}: {score * 100:.2f}%")

        st.progress(float(score))


    # Low confidence warning
    if confidence < 50:

        st.warning(
            "The model is not very confident about this prediction."
        )