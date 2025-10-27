"""
Streamlit Web Application for Plant Classification
"""

import io
import time
from PIL import Image
import requests
import streamlit as st
import torch

from src.config import settings
from src.training.model import load_model
from src.utils.helpers import preprocess_image, image_to_tensor


# Page config
st.set_page_config(
    page_title="Plant Classifier",
    page_icon="🌼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.dandelion {
    background-color: #fff3cd;
    border: 2px solid #ffc107;
}
.grass {
    background-color: #d1ecf1;
    border: 2px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌼 Dandelion vs Grass Classifier")
st.markdown("Upload an image to classify it as **Dandelion** or **Grass**")

# Sidebar
st.sidebar.header("Settings")
use_api = st.sidebar.checkbox("Use API", value=False)

if use_api:
    api_url = st.sidebar.text_input("API URL", value=f"http://localhost:{settings.API_PORT}")
else:
    st.sidebar.info("Using local model")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application uses deep learning to classify images of plants.
- **Model**: ResNet18
- **Classes**: Dandelion, Grass
""")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.header("🎯 Prediction Result")
    
    if uploaded_file is not None:
        with st.spinner("Classifying..."):
            start_time = time.time()
            
            try:
                if use_api:
                    # Use API
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{api_url}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                        prediction_time = result["prediction_time"]
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.stop()
                else:
                    # Use local model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model_path = settings.MODELS_DIR / f"{settings.MODEL_NAME}_best.pth"
                    
                    if not model_path.exists():
                        st.error("Model not found! Please train the model first.")
                        st.stop()
                    
                    model = load_model(str(model_path), device)
                    
                    # Preprocess
                    processed_image = preprocess_image(image, target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
                    img_tensor = image_to_tensor(processed_image).unsqueeze(0).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence_tensor, predicted = torch.max(probabilities, 1)
                    
                    class_names = ["dandelion", "grass"]
                    predicted_class = class_names[predicted.item()]
                    confidence = confidence_tensor.item()
                    prediction_time = time.time() - start_time
                
                # Display results
                st.markdown(f"<p class='big-font'>Prediction: {predicted_class.upper()}</p>", unsafe_allow_html=True)
                
                # Result box with color
                box_class = "dandelion" if predicted_class == "dandelion" else "grass"
                st.markdown(f"""
                <div class='result-box {box_class}'>
                    <h3>{'🌼 Dandelion' if predicted_class == 'dandelion' else '🌱 Grass'}</h3>
                    <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
                    <p><strong>Time:</strong> {prediction_time:.3f}s</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.progress(confidence)
                
                # Additional info
                with st.expander("See details"):
                    st.write(f"**Model**: ResNet18")
                    st.write(f"**Image size**: {settings.IMAGE_SIZE}x{settings.IMAGE_SIZE}")
                    st.write(f"**Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
                    
                    # Show both class probabilities
                    st.write("**Class probabilities:**")
                    if use_api:
                        st.write(f"- Dandelion: {confidence * 100 if predicted_class == 'dandelion' else (1-confidence) * 100:.2f}%")
                        st.write(f"- Grass: {confidence * 100 if predicted_class == 'grass' else (1-confidence) * 100:.2f}%")
                    else:
                        probs = probabilities[0].cpu().numpy()
                        st.write(f"- Dandelion: {probs[0] * 100:.2f}%")
                        st.write(f"- Grass: {probs[1] * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>MLOps Project - Dandelion vs Grass Classification</p>
    <p>Built with ❤️ using Streamlit & PyTorch</p>
</div>
""", unsafe_allow_html=True)
