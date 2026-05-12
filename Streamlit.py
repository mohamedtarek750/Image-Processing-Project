import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Breast Cancer Image Classification",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Model Paths
# =========================
MODEL_PATHS = {
    "Scratch CNN": r"D:\Downloads\Image Preprocessing Project\CNN_model.h5",
    "ResNet50": r"D:\Downloads\Image Preprocessing Project\ResNet50_model.h5",
    "DenseNet169": r"D:\Downloads\Image Preprocessing Project\DenseNet169_model.h5",
}

# Important: this must match the class order used during training.
# flow_from_directory usually sorts folder names alphabetically.
CLASS_NAMES = ["Cancer", "Non-Cancer"]
IMAGE_SIZE = (224, 224)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e6eaf0;
        text-align: center;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e6eaf0;
        background-color: #ffffff;
        box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }
    .warning-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #fff7e6;
        border: 1px solid #ffd591;
        color: #5c3b00;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Helper Functions
# =========================
@st.cache_resource
def load_selected_model(model_path):
    """Load a Keras .h5 model once and cache it."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # compile=False avoids many old H5 loss/metric compatibility issues.
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(uploaded_image):
    """Resize and normalize the uploaded image exactly like training."""
    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch


def predict_image(model, img_batch):
    """Return predicted label, confidence, and class probabilities."""
    predictions = model.predict(img_batch)

    # Expected shape: (1, 2)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_label, confidence, probabilities


def render_probability_chart(probabilities):
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probabilities
    })

    st.bar_chart(
        df.set_index("Class"),
        use_container_width=True
    )


# =========================
# Sidebar
# =========================
st.sidebar.title("Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose prediction model",
    list(MODEL_PATHS.keys()),
    index=2
)

selected_model_path = MODEL_PATHS[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.write("**Selected model path:**")
st.sidebar.code(selected_model_path, language="text")

st.sidebar.markdown("---")
st.sidebar.write("**Required input size:**")
st.sidebar.code("224 x 224 RGB", language="text")

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard uses the same preprocessing used during training: image resizing to 224x224 and pixel rescaling by 1/255."
)


# =========================
# Main Page
# =========================
st.markdown('<p class="main-title">Breast Cancer Image Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a breast cancer image, choose a trained model, and get the predicted class with confidence score.</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="warning-box">
    <b>Note:</b> This dashboard is for academic image classification only. It is not a medical diagnosis tool.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.subheader("1. Upload Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        original_image, processed_image = preprocess_image(uploaded_file)

        st.image(
            original_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.info("Image uploaded and preprocessed successfully.")
    else:
        st.info("Please upload an image to start prediction.")


with col_result:
    st.subheader("2. Prediction Result")

    if uploaded_file is not None:
        try:
            with st.spinner(f"Loading {selected_model_name} model..."):
                model = load_selected_model(selected_model_path)

            with st.spinner("Running prediction..."):
                predicted_label, confidence, probabilities = predict_image(model, processed_image)

            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted Class", predicted_label)

            with result_col2:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.subheader("3. Class Probabilities")

            probability_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": [f"{p * 100:.2f}%" for p in probabilities]
            })

            st.dataframe(probability_df, use_container_width=True, hide_index=True)
            render_probability_chart(probabilities)

            st.write("")
            st.subheader("4. Model Used")
            st.success(f"Prediction completed using: {selected_model_name}")

        except FileNotFoundError as error:
            st.error(str(error))
            st.warning("Check that the .h5 model file exists in the exact path shown in the sidebar.")

        except Exception as error:
            st.error("An error happened while loading the model or making the prediction.")
            st.exception(error)

    else:
        st.warning("Upload an image first to show prediction results.")


# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Project: Breast Cancer Image Classification | Models: Scratch CNN, ResNet50, DenseNet169"
)
