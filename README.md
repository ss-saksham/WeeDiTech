# 🌿 WeeDiTech - Unified Plant Disease & Weed Detection System

**WeeDiTech** is a smart agricultural assistant that helps farmers and researchers detect plant diseases using deep learning and identify crops vs. weeds using YOLO object detection — all from a simple web interface powered by Streamlit.

![Banner](home_page(1).jpeg)

---

## 🚀 Features

- 🌱 **Plant Disease Recognition** with a TensorFlow model trained on 38 plant classes
- 🌾 **YOLOv4 Crop & Weed Detection** using real-time object detection
- 💡 **Remedy Suggestions** after detection
- 📷 Upload an image and get instant AI-powered results
- 🧠 Streamlit UI with sidebar navigation and smooth layout

---

## 🧰 Tech Stack

| Component       | Technology         |
|----------------|--------------------|
| Web Framework   | Streamlit          |
| ML Framework    | TensorFlow / Keras |
| Object Detection| OpenCV + YOLO      |
| Preprocessing   | NumPy, Pillow      |

---

## 🖥️ Local Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/WeeDiTech.git
cd WeeDiTech
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
WeeDiTech/
├── app.py                       # Main Streamlit app
├── crop_weed.cfg                # YOLO config
├── obj.names                    # YOLO class labels
├── home_page(1).jpeg            # Banner image
├── requirements.txt             # Required Python libraries
└── README.md                    # Project overview
```

---

## 📊 Dataset Information

- 📸 87,000+ RGB images of crop leaves
- 🏷️ 38 categories (healthy and diseased)
- 🔄 80/20 Train-Validation split
- 🧠 YOLO trained separately on crop/weed annotations

---

## 👨‍💻 Team Greennovators

- 🌟 Saksham Singla  
- 🌟 Shagun Sharma  
- 🌟 Gargi Tokas  
- 🌟 Ritika Sanghwani

---

## 📃 License

This project is licensed under the **MIT License** — feel free to use, modify, and share!
