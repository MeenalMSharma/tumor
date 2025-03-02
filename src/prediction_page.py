import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model (update path if needed)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/tumor_detection_model.h5")  # Update with your model path
    return model

model = load_model()

# Class labels (Update these based on your dataset)
CLASS_NAMES = ["No Tumor", "Tumor Detected"]

def predict(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    class_idx = np.argmax(prediction)  # Get the class with highest probability
    confidence = np.max(prediction)  # Get confidence score

    return CLASS_NAMES[class_idx], confidence

def main():
    st.title("🧠 Tumor Detection")
    st.write("Upload an MRI scan to detect if a tumor is present.")

    uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            label, confidence = predict(image)
            st.write(f"### **Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2%}")

if __name__ == "__main__":
    main()
