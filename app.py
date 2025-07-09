import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import gdown
from PIL import Image

# Set Streamlit Config
st.set_page_config(page_title="WeeDiTech - Unified AgriCare", layout="wide")

# ======================== Load Plant Disease Model ===========================
keras_url = 'https://drive.google.com/uc?id=1NkDmqQpP6ezNW80qanxRSs8YG_0crda3'
weights_url = 'https://drive.google.com/uc?id=1ia6HeDgEnsf81kq3ZcM3xBwQmzwGMDvU'

if not os.path.exists('trained_model.keras'):
    st.info("📥 Downloading trained_model.keras from Google Drive...")
    gdown.download(keras_url, 'trained_model.keras', quiet=False)

if not os.path.exists('crop_weed_detection.weights'):
    st.info("📥 Downloading crop_weed_detection.weights from Google Drive...")
    gdown.download(weights_url, 'crop_weed_detection.weights', quiet=False)
# Load the trained model
@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model('trained_model.keras')

plant_model = load_plant_model()

# Class names for plant disease detection
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Remedies for plant diseases
disease_remedies = {
    "Apple___Apple_scab": "Apply fungicides like Mancozeb and Captan. Prune infected leaves.",
    "Apple___Black_rot": "Use copper-based fungicides. Remove and destroy infected fruit.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper trees. Apply fungicides in early spring.",
    "Apple___healthy": "No disease detected. Keep monitoring your plant.",
    "Blueberry___healthy": "No disease detected. Maintain proper watering and sunlight.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicides. Ensure good air circulation.",
    "Cherry_(including_sour)___healthy": "No disease detected. Keep monitoring your plant.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids and rotate crops regularly.",
    "Corn_(maize)___Common_rust_": "Plant rust-resistant corn varieties. Apply fungicides if necessary.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant varieties and apply foliar fungicides.",
    "Corn_(maize)___healthy": "No disease detected. Keep monitoring your plant.",
    "Grape___Black_rot": "Remove infected parts and apply fungicides like Myclobutanil.",
    "Grape___Esca_(Black_Measles)": "Prune infected vines and use balanced fertilization.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicides and avoid overhead watering.",
    "Grape___healthy": "No disease detected. Keep monitoring your plant.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control psyllids with insecticides and remove infected trees.",
    "Peach___Bacterial_spot": "Apply copper sprays and ensure proper air circulation.",
    "Peach___healthy": "No disease detected. Maintain regular care.",
    "Pepper,_bell___Bacterial_spot": "Use copper-based sprays and remove infected leaves.",
    "Pepper,_bell___healthy": "No disease detected. Keep monitoring your plant.",
    "Potato___Early_blight": "Use fungicides like Chlorothalonil and practice crop rotation.",
    "Potato___Late_blight": "Destroy infected plants and avoid overhead watering.",
    "Potato___healthy": "No disease detected. Maintain proper watering and fertilization.",
    "Raspberry___healthy": "No disease detected. Keep monitoring your plant.",
    "Soybean___healthy": "No disease detected. Ensure balanced soil nutrition.",
    "Squash___Powdery_mildew": "Apply sulfur or neem oil sprays. Ensure good air circulation.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and apply fungicides if necessary.",
    "Strawberry___healthy": "No disease detected. Maintain proper care.",
    "Tomato___Bacterial_spot": "Use copper-based sprays and remove infected leaves.",
    "Tomato___Early_blight": "Apply fungicides like Mancozeb and remove affected leaves.",
    "Tomato___Late_blight": "Destroy infected plants and avoid overhead watering.",
    "Tomato___Leaf_Mold": "Improve air circulation and use fungicidal sprays.",
    "Tomato___Septoria_leaf_spot": "Apply chlorothalonil-based fungicides and prune lower leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil or insecticidal soap to control mites.",
    "Tomato___Target_Spot": "Apply copper-based fungicides and remove affected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies as they spread the virus.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect gardening tools.",
    "Tomato___healthy": "No disease detected. Maintain regular care."
}

# Remedies for YOLO Crop & Weed Detection
yolo_remedies = {
    "crop": "✅ Maintain proper irrigation and use organic fertilizers.",
    "weed": "⚠️ Apply selective herbicides or manual weeding to remove weeds."
}

# ===================== Streamlit Sidebar =========================
st.sidebar.markdown("### 📋 Navigation")
app_mode = st.sidebar.selectbox("Choose a section", [
    "Home", "About", "Plant Disease Recognition", "Crop & Weed Detection"
])

# ===================== Home =========================
if app_mode == "Home":
    st.markdown("<h1 style='color:#2E8B57;'>🌱 WeeDiTech - PLANT DISEASE & WEED RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
    st.image("home_page(1).jpeg", use_column_width=True)
    st.markdown("""
    <div style='font-size: 16px;'>
    Welcome to the <b>WeeDiTech - Plant Disease Recognition System</b>! 🌿🔍<br><br>
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.<br>
    <b>Together, let's protect our crops and ensure a healthier harvest!</b>
    </div>

    <h3 style='color:#4682B4;'>How It Works</h3>
    <ol>
        <li><b>Upload Image:</b> Go to the <b>Disease Recognition</b> page and upload an image of a plant with suspected diseases.</li>
        <li><b>Analysis:</b> Our system will process the image using advanced algorithms to identify potential diseases.</li>
        <li><b>Results:</b> View the results and recommendations for further action.</li>
    </ol>

    <h3 style='color:#4682B4;'>Why Choose Us?</h3>
    <ul>
        <li><b>Accuracy:</b> Our system utilizes state-of-the-art machine learning techniques for accurate disease and weed detection.</li>
        <li><b>User-Friendly:</b> Simple and intuitive interface for seamless user experience.</li>
        <li><b>Fast and Efficient:</b> Receive results in seconds, allowing for quick decision-making.</li>
    </ul>

    <h3 style='color:#4682B4;'>Get Started</h3>
    Click on the <b>Plant Disease Recognition</b> or <b>Crop & Weed Detection</b> page in the sidebar to upload an image and experience the power of our recognition system!

    <h3 style='color:#4682B4;'>About Us</h3>
    Learn more about the project, our team, and our goals on the <b>About</b> page.
    """, unsafe_allow_html=True)

# ===================== About =========================
elif app_mode == "About":
    st.markdown("<h2 style='color:#4682B4;'>About the Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    <b>Dataset Info:</b><br>
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.<br>
    It consists of about <b>87K RGB images</b> of healthy and diseased crop leaves categorized into <b>38 different classes</b>. <br>
    The total dataset is divided into an <b>80/20 ratio</b> of training and validation set preserving the directory structure.<br>
    A new directory containing <b>33 test images</b> is created later for prediction purposes.<br>

    <b>Content:</b><br>
    - Train (70295 images)<br>
    - Valid (17572 images)<br>
    - Test (33 images)<br>

    <b>Team Name: Greennovators</b><br>
    - Saksham Singla<br>
    - Shagun Sharma<br>
    - Gargi Tokas<br>
    - Ritika Sanghwani
    """, unsafe_allow_html=True)

# ===================== Plant Disease Recognition =========================
elif app_mode == "Plant Disease Recognition":
    st.markdown("<h2 style='color:#32CD32;'>📷 Plant Disease Recognition</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("Upload a plant leaf image:", type=["jpg", "jpeg", "png"])

    if test_image:
        image = Image.open(test_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                image_resized = image.resize((128, 128))
                input_arr = np.expand_dims(np.array(image_resized) / 255.0, axis=0)
                prediction = plant_model.predict(input_arr)
                result_index = np.argmax(prediction)
                predicted_disease = class_names[result_index]
                remedy = disease_remedies.get(predicted_disease, "No remedy available.")

                st.success(f"🧪 Predicted Disease: **{predicted_disease}**")
                st.info(f"💡 Suggested Remedy: {remedy}")

# ===================== Crop & Weed Detection =========================
elif app_mode == "Crop & Weed Detection":
    st.markdown("<h2 style='color:#8A2BE2;'>🌾 Crop & Weed Detection using YOLO</h2>", unsafe_allow_html=True)

    base_path = r'C:\Users\Saksham Singla\OneDrive\Desktop\Innovera\Detection\testing'
    labelsPath = os.path.join(base_path, 'obj.names')
    weightsPath = os.path.join(base_path, 'crop_weed_detection.weights')
    configPath = os.path.join(base_path, 'crop_weed.cfg')

    for path in [labelsPath, weightsPath, configPath]:
        if not os.path.exists(path):
            st.error(f"Missing file: {path}")
            st.stop()

    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    uploaded_file = st.file_uploader("Upload a crop/field image:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        (H, W) = image_cv.shape[:2]

        blob = cv2.dnn.blobFromImage(image_cv, 1 / 255.0, (512, 512), swapRB=True, crop=False)
        net.setInput(blob)

        ln = net.getUnconnectedOutLayersNames() if hasattr(net, 'getUnconnectedOutLayersNames') else [
            net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        layerOutputs = net.forward(ln)

        boxes, confidences, classIDs = [], [], []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        detected_classes = set()

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                label = LABELS[classIDs[i]].lower()
                detected_classes.add(label)
                cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image_cv, f"{label}: {confidences[i]:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Detection Output", use_column_width=True)

        st.subheader("💡 Remedy Suggestions")
        if detected_classes:
            for cls in detected_classes:
                remedy = yolo_remedies.get(cls, "No remedy available.")
                st.write(f"**{cls.capitalize()}**: {remedy}")
        else:
            st.warning("No crops or weeds detected.")