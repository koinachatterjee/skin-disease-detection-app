# 🔬 SkinSense AI — Skin Disease Detection & Monitoring App

> Major Project | AI InternsElite | Batch June 2026

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-orange)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green)

## 🌐 Live App
**[👉 Click here to open SkinSense AI](https://skin-disease-detection-app-xz8il5phavbdzhgjm9ba9t.streamlit.app)**

---

## 📌 Project Overview
SkinSense AI addresses the global dermatology crisis — **3 billion people** worldwide lack access to qualified dermatologists. This app uses deep learning to provide instant skin disease detection, severity assessment, treatment recommendations, and weekly condition monitoring.

---

## ✨ Features
| Feature | Description |
|---------|-------------|
| 🔬 AI Diagnosis | Upload a skin photo → CNN predicts disease in seconds |
| ⚠️ Severity Assessment | Low / Medium / High risk classification |
| 💊 Treatment Advice | Personalized recommendations for each disease |
| 🚨 Doctor Referral | Automatic alert for high-risk conditions |
| 📅 Weekly Monitoring | Track if condition is improving or worsening |
| 📊 Disease Analytics | Info cards for all 7 detectable diseases |

---

## 🧠 Model Details
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Dataset:** HAM10000 — 10,015 dermatoscopy images
- **Classes:** 7 skin disease categories
- **Accuracy:** >80% on test set
- **Format:** ONNX (for cross-platform deployment)

## 🦠 Diseases Detected
| Code | Disease | Risk |
|------|---------|------|
| MEL | Melanoma | 🔴 HIGH |
| NV | Melanocytic Nevi (Mole) | 🟢 LOW |
| BCC | Basal Cell Carcinoma | 🟡 MEDIUM |
| AKIEC | Actinic Keratoses | 🟡 MEDIUM |
| BKL | Benign Keratosis | 🟢 LOW |
| DF | Dermatofibroma | 🟢 LOW |
| VASC | Vascular Lesions | 🟢 LOW |

---

## 🛠️ Tech Stack
