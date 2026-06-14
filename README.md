```markdown
# 🌿 LeafLens: Plant Disease Detection System

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📌 Project Overview
LeafLens is an AI-powered botanical assistant that leverages a Convolutional Neural Network (CNN) to detect plant diseases from leaf images. Built entirely from scratch using TensorFlow and Keras, this model classifies leaf images into **15 distinct health and disease categories**. 

The project includes a fully interactive web interface built with **Gradio**, allowing users to upload a photo of a leaf and receive an instant AI diagnosis along with a confidence score.

---

## 🚀 Features
- **Custom Deep Learning Architecture:** Built from scratch without pre-trained weights.
- **High Accuracy:** Achieved **~96% Training Accuracy** and **~92% Validation Accuracy**.
- **Memory-Optimized Processing:** Implements efficient OpenCV image resizing and explicit garbage collection to prevent memory overflow during training.
- **Interactive Web UI:** Drag-and-drop Gradio interface for real-time predictions.

---

## 📊 Dataset Preparation (Important!)
This project uses the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle. 

**⚠️ Note on Data Structure:** The original Kaggle zip file contains a nested duplicate folder structure (`PlantVillage` inside `plantvillage`). For the training script to work correctly, you must flatten the directory so that the 15 disease class folders sit directly inside a folder named `plant_data`.

### Setup Instructions:
1. Download the dataset from Kaggle and extract it.
2. Create a folder named `plant_data` in the root of this project.
3. Move the 15 class folders (e.g., `Tomato_healthy`, `Potato___Early_blight`, etc.) directly into the `plant_data` folder.

Your final folder structure should look exactly like this:
```text
leaflens/
│
├── plant_data/
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   ├── Potato___Early_blight/
│   ├── ... (15 folders total)
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
|
└── app.py

```

---

## 🧠 Model Architecture

The custom CNN processes `128x128` RGB images and extracts hierarchical features using the following architecture:

1. **Input Layer:** `(128, 128, 3)`
2. **Convolutional Block 1:** `Conv2D (32 filters)` + `MaxPooling2D`
3. **Convolutional Block 2:** `Conv2D (64 filters)` + `MaxPooling2D`
4. **Convolutional Block 3:** `Conv2D (128 filters)` + `MaxPooling2D`
5. **Fully Connected Block:** `Flatten()` + `Dense (128 neurons)` + `Dropout (0.3)`
6. **Output Layer:** `Dense (15 neurons)` with `Softmax` activation

---

## 📈 Performance Results

After training for 15 epochs using the Adam optimizer and Categorical Crossentropy:

* **Training Accuracy:** ~96%
* **Validation Accuracy:** ~92%
* **Loss:** Converged smoothly without severe overfitting thanks to Dropout regularization.

*(Optional: Add screenshots of your Matplotlib accuracy/loss graphs here)*

---

## 🛠 Tech Stack

* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy, Scikit-Learn (`LabelEncoder`, `train_test_split`)
* **Deployment:** Gradio
* **Visualization:** Matplotlib, Seaborn

---

## ▶️ How to Run Locally

### 1. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install tensorflow opencv-python numpy scikit-learn gradio matplotlib

```

### 2. Train the Model

Ensure your `plant_data` folder is set up correctly (see Dataset section), then run:

```bash
python train.py

```

*This will train the CNN and generate two files: `plant_disease_model.keras` (the AI model) and `label_encoder.pkl` (the disease names).*

### 3. Launch the Web App

Once training is complete, start the Gradio interface:

```bash
python app.py

```

Click the local URL generated in your terminal (usually `http://127.0.0.1:7860`) to open the app in your browser!

---

## 🤝 Acknowledgments

* Dataset provided by **PlantVillage** and uploaded to Kaggle by *emmarex*.
* Web interface powered by **Gradio**.

```

```