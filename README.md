# 🎗️ BCancerNet

## 📖 Project Overview
BCancerNet is a machine learning-based web application that predicts whether a tumor is **Malignant** or **Benign** based on tumor features. The app uses a **Neural Network model built with TensorFlow and Keras**, along with **Streamlit for a user-friendly web interface**. Users can upload a CSV file containing tumor data, visualize insights, and make real-time predictions.

## 📂 Folder Structure
```
BCancerNet/
├── Dataset/                 # Contains sample dataset
│   ├── BCancerPred.csv
├── Main/                    # Core application files
│   ├── Main.py               # Streamlit app script
│   ├── commandlineprompt.txt # Instructions for running
├── Screenshots/             # UI and results screenshots
│   ├── Input-Details.png
│   ├── Jupyter-Environment.png
│   ├── Result.png
│   ├── Web-Application-UI.png
├── README.md                # Project documentation
```

## 🚀 How to Use

### **1️⃣ Install Dependencies**
Make sure you have Python installed. Then, install the required libraries:
```bash
pip install numpy pandas streamlit scikit-learn tensorflow matplotlib
```

### **2️⃣ Run the Project**
Execute the **Main.py** script to launch the Streamlit web application:
```bash
streamlit run Main/Main.py
```

### **3️⃣ Features & Workflow**
1. **Upload CSV Dataset** – Users upload a CSV file containing tumor data.
2. **Model Training** – The app processes the dataset, trains a neural network model, and standardizes data.
3. **Analytics & Graphs** – Displays accuracy and loss plots in the sidebar.
4. **Prediction Input** – Users enter tumor feature values manually.
5. **Get Prediction** – The app classifies the tumor as **Malignant** or **Benign**.

## 📷 Screenshots
| **Input Details** | **Jupyter Notebook** |
|------------------|------------------|
| ![Input](Screenshots/Input-Details.png) | ![Jupyter](Screenshots/Jupyter-Environment.png) |

| **Prediction Result** | **Web Application UI** |
|------------------|------------------|
| ![Result](Screenshots/Result.png) | ![UI](Screenshots/Web-Application-UI.png) |

## 📜 License
This project is licensed under the Apache License 2.0.

---
🔬 **Predict with Confidence!** 🎉

