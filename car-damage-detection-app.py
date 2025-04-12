import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Car Damage Detection",
    page_icon="🚗",
    layout="wide"
)

# Define constants
img_width, img_height = 150, 150
class_names = ["01-minor", "02-moderate", "03-severe"]
display_names = ["Minor", "Moderate", "Severe"]

# Function to load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('damage_classification_model.h5')
        return model
    except:
        st.error("Model file not found. Please make sure 'damage_classification_model.h5' is in the same directory as this app.")
        return None

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_damage(model, img_array):
    prediction = model.predict(img_array)
    pred_class_idx = np.argmax(prediction[0])
    pred_class = class_names[pred_class_idx]
    display_name = display_names[pred_class_idx]
    confidence = float(prediction[0][pred_class_idx])
    return pred_class, display_name, confidence, prediction[0]

# Function to create prediction visualization
def create_prediction_visualization(predictions):
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(display_names, predictions, color=['#8BC34A', '#FFC107', '#F44336'])
    
    # Find the highest prediction and highlight it
    max_idx = np.argmax(predictions)
    bars[max_idx].set_color('darkblue')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence')
    ax.set_title('Damage Severity Prediction')
    
    # Add percentage labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{predictions[i]:.1%}', ha='center', va='bottom')
    
    # Convert plot to image for Streamlit
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# Main app functionality
def main():
    st.title("Car Damage Severity Detection")
    st.write("Upload an image of a damaged car to classify the severity of the damage")
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Analyze Damage"):
            with st.spinner("Analyzing damage severity..."):
                # Preprocess the image
                img_array = preprocess_image(image)
                
                # Make prediction
                pred_class, display_name, confidence, predictions = predict_damage(model, img_array)
                
                with col2:
                    # Display results
                    st.subheader("Prediction Results")
                    st.markdown(f"**Damage Severity:** {display_name}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Display prediction visualization
                    prediction_chart = create_prediction_visualization(predictions)
                    st.image(prediction_chart, use_column_width=True)
                
                # Additional information
                st.subheader("Damage Categories")
                st.markdown("""
                - **Minor Damage**: Scratches, small dents, or minimal damage that doesn't affect functionality
                - **Moderate Damage**: Larger dents, damaged bumpers, or cosmetic parts that require repair
                - **Severe Damage**: Structural damage, major component damage, or safety concerns
                """)
                
                # Disclaimer
                st.info("Note: This is an automated assessment and should be verified by a professional.")

# Run the app
if __name__ == "__main__":
    main()
