import gradio as gr
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

print("Loading model and classes...")
# Load the trained model
model = load_model('plant_disease_model.keras')

# Load the saved LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# --- THE PREDICTION ENGINE ---
def predict_image(img):
    # Ensure OpenCV can read it and resize
    img = cv2.resize(img, (128, 128))
    
    # Normalize
    img = img / 255.0
    
    # Expand dims (Make it a batch of 1)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Get the name of the disease from the loaded encoder
    disease_name = encoder.classes_[class_index]
    
    return f"{disease_name} with confidence ({confidence * 100:.2f}%)"

# --- THE WEB INTERFACE ---
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🌿 LeafLens: Plant Disease Classifier",
    description="Upload a picture of a plant leaf to detect disease using a Deep Learning CNN."
)

if __name__ == "__main__":
    # Launch the web app
    interface.launch(debug=True, share=True)