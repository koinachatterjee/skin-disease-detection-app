App link:https://skin-disease-detection-app-xz8il5phavbdzhgjm9ba9t.streamlit.app/


The central problem addressed by this project is the lack of accessible, affordable, and timely
skin disease diagnosis for the majority of the global population

"Given a photograph of a skin lesion, can an AI system accurately classify the disease
category, assess its severity, recommend appropriate treatment, and track its progression over
time — all without requiring access to a specialist?"

Objectives

1 Train a CNN model on HAM10000 dataset for 7-class skin disease classification 
2 Achieve test accuracy above 80% using MobileNetV2 transfer learning 
3 Build a web app with image upload and real-time AI diagnosis Completed
4 Add severity scoring (Low/Medium/High) based on disease type and confidence 
5 Generate personalized treatment recommendations for each disease category 
6 Implement automatic doctor referral alerts for high-risk conditions 
7 Build a weekly monitoring timeline to track condition improvement/worsening 
8 Deploy the app publicly on Streamlit Cloud

Dataset — HAM10000
The HAM10000 (Human Against Machine with 10,000 training images) dataset is a large
collection of multi-source dermatoscopic images of common pigmented skin lesions. It was
published by Tschandl et al. (2018) and is widely used as the benchmark for skin lesion
classification research.

Web Application Features

Image Upload User: uploads a JPG/PNG photo of skin concern via drag-and-drop interface
AI Diagnosis : CNN model predicts disease class with confidence percentage
Top 3 Predictions : Shows top 3 most likely diseases with confidence bar chart
Severity Assessment : Classifies condition as Low/Medium/High risk based on disease type
Treatment Advice : Personalized treatment recommendations for each of 7 disease categories
Doctor Referral : Alert Automatic red warning card for HIGH risk conditions (MEL, BCC, AKIEC)
Weekly Monitoring Track : same skin spot over multiple weeks with trend analysis
Trend Detection App : detects if condition is Improving, Stable, or Worsening
3-Page Navigation : Home & Diagnosis, Weekly Monitoring, About & Disclaimer pages
Mobile Friendly : Responsive layout works on smartphones and tablets


Future Work
• Train on full HAM10000 dataset (10,015 images) with Colab Pro for higher accuracy
• Add real-time camera capture for instant smartphone diagnosis
• Integrate with telemedicine platforms for direct doctor consultation booking
• Extend to detect additional diseases including eczema and psoriasis
• Implement federated learning to train on patient data while preserving privacy
• Add explainability features (Grad-CAM heatmaps) to highlight suspicious regions
• Build a native mobile app for Android and iOS using Flutter
• Integrate with Electronic Health Records (EHR) systems
