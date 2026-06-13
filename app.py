import streamlit as st
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    import keras
import pickle
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from datetime import datetime
import os
import io
import base64

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.main-title    { font-size:2.2rem; font-weight:600; color:#185FA5; margin-bottom:0; }
.sub-title     { font-size:1rem; color:#666; margin-top:0; margin-bottom:2rem; }
.result-card   { background:#f0f6ff; border-left:4px solid #185FA5;
                 padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.warning-card  { background:#fff8e1; border-left:4px solid #BA7517;
                 padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.danger-card   { background:#ffebee; border-left:4px solid #D85A30;
                 padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.success-card  { background:#f0fff8; border-left:4px solid #1D9E75;
                 padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.metric-box    { background:#fff; border:1px solid #e0e0e0;
                 padding:1rem; border-radius:8px; text-align:center; }
.section-head  { font-size:1.2rem; font-weight:600; color:#1a1a2e; margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Disease info database ─────────────────────────────────────
DISEASE_INFO = {
    'MEL': {
        'name'     : 'Melanoma',
        'severity' : 'HIGH',
        'color'    : '#D85A30',
        'referral' : True,
        'treatment': [
            'Immediate dermatologist consultation required',
            'Surgical excision is the primary treatment',
            'May require sentinel lymph node biopsy',
            'Immunotherapy or targeted therapy if advanced',
            'Regular full-body skin checks every 3-6 months'
        ],
        'description': 'Melanoma is the most serious type of skin cancer. Early detection is critical for successful treatment.'
    },
    'NV': {
        'name'     : 'Melanocytic Nevi (Mole)',
        'severity' : 'LOW',
        'color'    : '#1D9E75',
        'referral' : False,
        'treatment': [
            'Generally benign and requires no treatment',
            'Monitor for changes in size, shape, or color',
            'Use ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution',
            'Apply sunscreen SPF 30+ daily',
            'Annual skin check recommended'
        ],
        'description': 'Common moles are usually harmless. Monitor for any changes using the ABCDE rule.'
    },
    'BCC': {
        'name'     : 'Basal Cell Carcinoma',
        'severity' : 'MEDIUM',
        'color'    : '#BA7517',
        'referral' : True,
        'treatment': [
            'Consult a dermatologist promptly',
            'Mohs surgery is the most effective treatment',
            'Radiation therapy for inoperable cases',
            'Topical treatments for superficial BCC',
            'Avoid sun exposure and use SPF 50+ sunscreen'
        ],
        'description': 'Most common skin cancer. Rarely spreads but needs prompt treatment to prevent local tissue damage.'
    },
    'AKIEC': {
        'name'     : 'Actinic Keratoses',
        'severity' : 'MEDIUM',
        'color'    : '#BA7517',
        'referral' : True,
        'treatment': [
            'See a dermatologist — may develop into cancer',
            'Cryotherapy (freezing) is common treatment',
            'Topical creams: 5-fluorouracil or imiquimod',
            'Photodynamic therapy (PDT)',
            'Daily sunscreen and sun-protective clothing'
        ],
        'description': 'Precancerous lesions caused by UV damage. Treatment prevents progression to squamous cell carcinoma.'
    },
    'BKL': {
        'name'     : 'Benign Keratosis',
        'severity' : 'LOW',
        'color'    : '#1D9E75',
        'referral' : False,
        'treatment': [
            'Usually harmless, no treatment necessary',
            'Can be removed for cosmetic reasons',
            'Cryotherapy or curettage if bothersome',
            'Keep skin moisturized',
            'Monitor for any rapid changes'
        ],
        'description': 'Non-cancerous skin growths that are very common with age. Usually no treatment needed.'
    },
    'DF': {
        'name'     : 'Dermatofibroma',
        'severity' : 'LOW',
        'color'    : '#1D9E75',
        'referral' : False,
        'treatment': [
            'Benign — no treatment required',
            'Surgical removal if causing discomfort',
            'Steroid injections may flatten the lesion',
            'Avoid trauma to the area',
            'Monitor for any size changes'
        ],
        'description': 'Common benign skin nodule. Harmless and usually requires no treatment.'
    },
    'VASC': {
        'name'     : 'Vascular Lesion',
        'severity' : 'LOW',
        'color'    : '#185FA5',
        'referral' : False,
        'treatment': [
            'Most are benign and need no treatment',
            'Laser therapy for cosmetic removal',
            'Pulsed dye laser is most effective',
            'Sclerotherapy for certain vascular lesions',
            'Consult dermatologist if growing rapidly'
        ],
        'description': 'Blood vessel abnormalities in the skin. Most are harmless birthmarks or acquired lesions.'
    }
}

IMG_SIZE = 224

# ─── Load model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    try:
        model = keras.models.load_model('best_skin_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('class_names.json', 'r') as f:
            class_info = json.load(f)
        return model, le, class_info
    except Exception as e:
        return None, None, None

# ─── Prediction function ───────────────────────────────────────
def predict_disease(image, model, le):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    preds = model.predict(img, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]
    results = []
    for idx in top_idx:
        code = le.classes_[idx]
        results.append({
            'code'      : code,
            'name'      : DISEASE_INFO[code]['name'],
            'confidence': float(preds[idx]) * 100,
            'severity'  : DISEASE_INFO[code]['severity'],
            'referral'  : DISEASE_INFO[code]['referral'],
            'color'     : DISEASE_INFO[code]['color'],
        })
    return results, preds, le.classes_

# ─── Monitoring history (session state) ────────────────────────
if 'monitoring_history' not in st.session_state:
    st.session_state.monitoring_history = []

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 SkinSense AI")
    st.markdown("*Skin Disease Detection & Monitoring*")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Home & Diagnosis",
        "📅 Weekly Monitoring",
        "ℹ️ About & Disclaimer"
    ])
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Dataset: HAM10000")
    st.markdown("- Architecture: MobileNetV2")
    st.markdown("- Classes: 7 skin conditions")
    st.divider()
    st.caption("⚠️ For educational use only. Always consult a qualified dermatologist.")

# ─── Load model ────────────────────────────────────────────────
model, le, class_info = load_model_and_labels()

# ════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & DIAGNOSIS
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home & Diagnosis":
    st.markdown('<p class="main-title">🔬 SkinSense AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a skin image to receive an AI-powered diagnosis, severity assessment, and treatment recommendations.</p>', unsafe_allow_html=True)

    if model is None:
        st.error("❌ Model files not found! Make sure these files are in the same folder as app.py:\n- best_skin_model.h5\n- label_encoder.pkl\n- class_names.json")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-head">📸 Upload Skin Image</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a clear, well-lit photo of the skin concern",
            type=['jpg','jpeg','png'],
            help="Best results with close-up, in-focus images in natural light"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)

            st.markdown('<p class="section-head">⚙️ Analysis Settings</p>', unsafe_allow_html=True)
            sensitivity = st.slider("Detection sensitivity", 0.5, 1.0, 0.7,
                                    help="Higher = more cautious referrals")
            add_to_monitor = st.checkbox("Add to weekly monitoring tracker")
            spot_name = ""
            if add_to_monitor:
                spot_name = st.text_input("Name this spot (e.g. 'Left arm mole')", "Spot 1")

            analyze = st.button("🔬 Analyse Image", type="primary", use_container_width=True)

            if analyze:
                with st.spinner("Analysing image..."):
                    results, all_preds, classes = predict_disease(image, model, le)

                top = results[0]

                with col2:
                    st.markdown('<p class="section-head">🩺 Diagnosis Result</p>', unsafe_allow_html=True)

                    # Main result card
                    sev_color = {'HIGH':'danger','MEDIUM':'warning','LOW':'success'}
                    card_class = sev_color.get(top['severity'], 'result')
                    st.markdown(f"""
                    <div class="{card_class}-card">
                        <h3 style="margin:0;color:{top['color']}">{top['name']}</h3>
                        <p style="margin:4px 0 0;font-size:0.9rem">Confidence: <b>{top['confidence']:.1f}%</b> &nbsp;|&nbsp; Severity: <b>{top['severity']}</b></p>
                        <p style="margin:8px 0 0;font-size:0.88rem;color:#444">{DISEASE_INFO[top['code']]['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Confidence", f"{top['confidence']:.1f}%")
                    m2.metric("Severity",   top['severity'])
                    m3.metric("Referral",   "YES ⚠️" if top['referral'] else "NO ✅")

                    # Doctor referral alert
                    if top['referral'] or top['confidence'] > sensitivity * 100:
                        st.markdown("""
                        <div class="danger-card">
                            <b>🚨 Doctor Referral Recommended</b><br>
                            This condition requires professional medical evaluation.
                            Please consult a certified dermatologist as soon as possible.
                        </div>
                        """, unsafe_allow_html=True)

                    # Top 3 predictions bar chart
                    st.markdown('<p class="section-head">📊 Top 3 Predictions</p>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    names  = [r['name'].split('(')[0].strip()[:20] for r in results]
                    confs  = [r['confidence'] for r in results]
                    colors = [r['color'] for r in results]
                    bars = ax.barh(names, confs, color=colors, alpha=0.85, edgecolor='white')
                    for bar, val in zip(bars, confs):
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                f'{val:.1f}%', va='center', fontsize=9)
                    ax.set_xlim(0, 110)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_title('Prediction Confidence')
                    ax.invert_yaxis()
                    fig.patch.set_alpha(0)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # All class probabilities
                    st.markdown('<p class="section-head">🔍 All Class Probabilities</p>', unsafe_allow_html=True)
                    prob_df = pd.DataFrame({
                        'Disease': [DISEASE_INFO[c]['name'] for c in classes],
                        'Probability (%)': (all_preds * 100).round(2)
                    }).sort_values('Probability (%)', ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                    # Treatment recommendations
                    st.markdown('<p class="section-head">💊 Treatment Recommendations</p>', unsafe_allow_html=True)
                    for tip in DISEASE_INFO[top['code']]['treatment']:
                        st.markdown(f"• {tip}")

                    # Add to monitoring
                    if add_to_monitor and spot_name:
                        st.session_state.monitoring_history.append({
                            'spot'      : spot_name,
                            'date'      : datetime.now().strftime('%Y-%m-%d'),
                            'disease'   : top['name'],
                            'code'      : top['code'],
                            'confidence': top['confidence'],
                            'severity'  : top['severity'],
                        })
                        st.success(f"✅ Added to monitoring tracker for '{spot_name}'")

# ════════════════════════════════════════════════════════════════
#  PAGE 2 — WEEKLY MONITORING
# ════════════════════════════════════════════════════════════════
elif page == "📅 Weekly Monitoring":
    st.markdown('<p class="main-title">📅 Weekly Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Track your skin condition over time and see if it is improving or worsening.</p>', unsafe_allow_html=True)

    if not st.session_state.monitoring_history:
        st.info("No monitoring data yet. Go to **Home & Diagnosis**, upload an image, check **'Add to weekly monitoring tracker'**, and analyse it.")
    else:
        history_df = pd.DataFrame(st.session_state.monitoring_history)
        spots = history_df['spot'].unique().tolist()
        selected_spot = st.selectbox("Select spot to view", spots)
        spot_df = history_df[history_df['spot'] == selected_spot].reset_index(drop=True)

        st.markdown(f'<p class="section-head">📈 Confidence Trend — {selected_spot}</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spot_df.index, spot_df['confidence'],
                marker='o', linewidth=2.5, color='#185FA5',
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(spot_df.index, spot_df['confidence'],
                        alpha=0.1, color='#185FA5')

        # Trend direction
        if len(spot_df) >= 2:
            trend = spot_df['confidence'].iloc[-1] - spot_df['confidence'].iloc[0]
            trend_text = f"{'⬆️ Worsening' if trend > 2 else '⬇️ Improving' if trend < -2 else '➡️ Stable'} ({trend:+.1f}%)"
            ax.set_title(f'Confidence Over Time — {trend_text}', fontweight='bold')
        else:
            ax.set_title('Confidence Over Time (need 2+ entries for trend)')

        ax.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Medium risk threshold')
        ax.axhline(y=85, color='red',    linestyle='--', linewidth=1, alpha=0.7, label='High risk threshold')
        ax.set_xlabel('Week / Visit')
        ax.set_ylabel('Model Confidence (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)

        # Status card
        latest = spot_df.iloc[-1]
        if len(spot_df) >= 2:
            trend = spot_df['confidence'].iloc[-1] - spot_df['confidence'].iloc[0]
            if trend > 2:
                st.markdown('<div class="danger-card">⚠️ <b>Condition appears to be worsening.</b> Please consult a dermatologist soon.</div>', unsafe_allow_html=True)
            elif trend < -2:
                st.markdown('<div class="success-card">✅ <b>Condition appears to be improving.</b> Continue monitoring weekly.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">➡️ <b>Condition is stable.</b> Continue weekly monitoring.</div>', unsafe_allow_html=True)

        # History table
        st.markdown('<p class="section-head">📋 Visit History</p>', unsafe_allow_html=True)
        st.dataframe(spot_df[['date','disease','confidence','severity']],
                     use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear monitoring history"):
            st.session_state.monitoring_history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════
#  PAGE 3 — ABOUT
# ════════════════════════════════════════════════════════════════
elif page == "ℹ️ About & Disclaimer":
    st.markdown('<p class="main-title">ℹ️ About SkinSense AI</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    SkinSense AI is a major project developed as part of the AI InternsElite program (Batch June 2026).
    It addresses the global shortage of dermatologists — **3 billion people** worldwide lack access to skin care specialists.

    ### How It Works
    1. User uploads a photo of a skin concern
    2. A CNN model (MobileNetV2) trained on **HAM10000** — 10,000+ dermatoscopy images — analyses it
    3. The app returns a diagnosis, confidence score, severity level, and treatment recommendations
    4. High-risk conditions trigger an automatic doctor referral alert
    5. Weekly monitoring tracks whether a condition improves or worsens over time

    ### Disease Categories Detected
    | Code | Disease | Risk Level |
    |------|---------|------------|
    | MEL  | Melanoma | 🔴 High |
    | NV   | Melanocytic Nevi | 🟢 Low |
    | BCC  | Basal Cell Carcinoma | 🟡 Medium |
    | AKIEC | Actinic Keratoses | 🟡 Medium |
    | BKL  | Benign Keratosis | 🟢 Low |
    | DF   | Dermatofibroma | 🟢 Low |
    | VASC | Vascular Lesions | 🟢 Low |

    ### Tech Stack
    - **Model:** MobileNetV2 (Transfer Learning) + TensorFlow/Keras
    - **Dataset:** HAM10000 (Harvard Dataverse)
    - **Web App:** Streamlit
    - **Image Processing:** OpenCV, PIL

    ### ⚠️ Medical Disclaimer
    > This application is built for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.

    ---
    **Developer:** Koina Chatterjee | Roll No: AIML-A6/9951 | Batch: June 2026
    """)
