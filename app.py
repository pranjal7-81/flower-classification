import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load MobileNetV2 transfer learning model
model = tf.keras.models.load_model(
    "flower_model_tl.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False
)


IMG_SIZE = (180, 180)

class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]


def predict_flower(image):

    # Resize image
    img = image.resize(IMG_SIZE)

    # Convert PIL image → array
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get probabilities
    scores = predictions[0]

    # Find highest probability
    predicted_index = np.argmax(scores)

    predicted_class = class_names[predicted_index]

    # Convert to percentage
    confidence = 100 * np.max(scores)

    return predicted_class, confidence, scores


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌸 Flower Classification AI")

st.write(
    "Upload a flower image and let the MobileNetV2 model identify it."
)


uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image"
    )

    # Prediction
    predicted_class, confidence, scores = predict_flower(image)

    # Show result
    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.divider()

    # Show probabilities
    st.subheader("Prediction Probabilities")

    for flower, score in zip(class_names, scores):

        st.write(
            f"{flower}: {score * 100:.2f}%"
        )

        st.progress(float(score))


    # Low confidence warning
    if confidence < 50:

        st.warning(
            "The model is not very confident about this prediction."
        )