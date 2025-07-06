
# 🏥 AI-Powered Skin Cancer Detection System

This is an advanced **AI-powered web application** built using **Streamlit** and **TensorFlow** that enables users to analyze dermoscopic skin lesion images for early signs of **skin cancer**. The application uses a fine-tuned **Xception-based CNN model** and integrates **Google Gemini AI** to validate medical image inputs.

---

## 🚀 Features

- 🔐 **Login-based access with consent**  
- 🖼️ **Upload and validate dermoscopic images using Gemini AI**  
- 🧠 **Classifies 7 skin lesion types** (e.g., Melanoma, BCC, AKIEC, etc.)  
- 📊 **Displays confidence, urgency level, and recommended actions**  
- 📜 **Downloadable PDF report with prediction history**  
- 🧾 **Live analysis history and detailed medical resources**  
- 🎨 **Fully responsive UI with custom CSS animations**

---

## 📁 Project Structure

```
├── newapp.py                  # Main Streamlit application
├── requirements.txt           # Python dependencies
├── model_xception.h5          # Trained CNN model weights (not included here)
├── skin_cancer_recognition_HAM1000.ipynb  # Model training notebook
```

---

## 🔧 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/abhishek172003/skin-cancer-detector.git
   cd skin-cancer-detector
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained model file**  
   Place `model_xception.h5` in the root directory.

5. **Set your Google API key**  
   Create a `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

---

## ▶️ Run the App

```bash
streamlit run newapp.py
```

---

## 🧠 Model Overview

- **Base model:** Xception (pretrained on ImageNet)  
- **Custom layers:** Fully connected + dropout + softmax  
- **Classes predicted:**  
  - Actinic Keratosis (akiec)  
  - Basal Cell Carcinoma (bcc)  
  - Benign Keratosis (bkl)  
  - Melanoma (mel)  
  - Dermatofibroma (df)  
  - Melanocytic Nevus (nv)  
  - Vascular Lesions (vasc)

---

## 🧪 Dataset

Model was trained using the **[HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)** — a large collection of multi-source dermatoscopic images.

---

## ⚠️ Disclaimer

> This is an **AI-based screening tool**, **not a substitute for medical advice**. Always consult a qualified dermatologist for any diagnosis or treatment decisions.

---

## 📄 License

This project is under the **MIT License**.

---

## 🤝 Let's Connect

Feel free to reach out if you'd like to collaborate or discuss improvements!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abhishek-palve-652ba91b1)
