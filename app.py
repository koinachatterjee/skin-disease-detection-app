import streamlit as st
import numpy as np
import pickle
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import gdown

# ─── Page config MUST be first streamlit call ──────────────────
st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Auto-download model from Google Drive ─────────────────────
MODEL_PATH = 'best_skin_model.h5'
GDRIVE_ID  = '1Tf8NvCJhPbbBADMKRJAG5VE9fmAmz_sM'

if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading AI model... please wait 1-2 mins'):
        gdown.download(
            f'https://drive.google.com/uc?id={GDRIVE_ID}',
            MODEL_PATH, quiet=False
        )

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.main-title   { font-size:2.2rem; font-weight:600; color:#185FA5; margin-bottom:0; }
.sub-title    { font-size:1rem; color:#666; margin-top:0; margin-bottom:2rem; }
.result-card  { background:#f0f6ff; border-left:4px solid #185FA5;
                padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.warning-card { background:#fff8e1; border-left:4px solid #BA7517;
                padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.danger-card  { background:#ffebee; border-left:4px solid #D85A30;
                padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.success-card { background:#f0fff8; border-left:4px solid #1D9E75;
                padding:1rem 1.5rem; border-radius:8px; margin:1rem 0; }
.section-head { font-size:1.2rem; font-weight:600; color:#1a1a2e; margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Disease database ──────────────────────────────────────────
DISEASE_INFO = {
    'MEL':   {'name':'Melanoma',              'severity':'HIGH',   'color':'#D85A30', 'referral':True,
              'description':'Most serious skin cancer. Early detection is critical.',
              'treatment':['Immediate dermatologist consultation required',
                           'Surgical excision is primary treatment',
                           'May require immunotherapy if advanced',
                           'Regular full-body skin checks every 3-6 months']},
    'NV':    {'name':'Melanocytic Nevi (Mole)','severity':'LOW',   'color':'#1D9E75', 'referral':False,
              'description':'Common moles are usually harmless.',
              'treatment':['Generally benign, no treatment needed',
                           'Monitor using ABCDE rule',
                           'Apply sunscreen SPF 30+ daily',
                           'Annual skin check recommended']},
    'BCC':   {'name':'Basal Cell Carcinoma',  'severity':'MEDIUM', 'color':'#BA7517', 'referral':True,
              'description':'Most common skin cancer. Rarely spreads but needs prompt treatment.',
              'treatment':['Consult a dermatologist promptly',
                           'Mohs surgery is most effective',
                           'Avoid sun exposure, use SPF 50+']},
    'AKIEC': {'name':'Actinic Keratoses',     'severity':'MEDIUM', 'color':'#BA7517', 'referral':True,
              'description':'Precancerous lesions caused by UV damage.',
              'treatment':['See dermatologist — may develop into cancer',
                           'Cryotherapy is common treatment',
                           'Daily sunscreen essential']},
    'BKL':   {'name':'Benign Keratosis',      'severity':'LOW',   'color':'#1D9E75', 'referral':False,
              'description':'Non-cancerous skin growths, very common with age.',
              'treatment':['Usually harmless, no treatment necessary',
                           'Can be removed for cosmetic reasons',
                           'Keep skin moisturized']},
    'DF':    {'name':'Dermatofibroma',        'severity':'LOW',   'color':'#1D9E75', 'referral':False,
              'description':'Common benign skin nodule. Harmless.',
              'treatment':['Benign — no treatment required',
                           'Surgical removal if causing discomfort']},
    'VASC':  {'name':'Vascular Lesion',       'severity':'LOW',   'color':'#185FA5', 'referral':False,
              'description':'Blood vessel abnormalities. Most are harmless.',
              'treatment':['Most are benign, no treatment needed',
                           'Laser therapy for cosmetic removal']},
}

IMG_SIZE = 224

# ─── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        # Install tensorflow at runtime to bypass Python version issues
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'tensorflow-cpu', '--quiet'
        ])
        import tensorflow as tf
        model = tf.keras.models.load_model('best_skin_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('class_names.json', 'r') as f:
            ci = json.load(f)
        return model, le, ci
    except Exception as e:
        st.error(f'Model load error: {e}')
        return None, None, None


# ─── Predict ───────────────────────────────────────────────────
def predict(image, model, le):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    preds = model.predict(img, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]
    results = []
    for idx in top3:
        code = le.classes_[idx]
        info = DISEASE_INFO.get(code, {})
        results.append({
            'code'      : code,
            'name'      : info.get('name', code),
            'confidence': float(preds[idx]) * 100,
            'severity'  : info.get('severity', 'LOW'),
            'referral'  : info.get('referral', False),
            'color'     : info.get('color', '#888'),
        })
    return results, preds, le.classes_

# ─── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 SkinSense AI")
    st.markdown("*Skin Disease Detection & Monitoring*")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Home & Diagnosis",
        "📅 Weekly Monitoring",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Dataset: HAM10000")
    st.markdown("- Architecture: MobileNetV2")
    st.markdown("- Classes: 7 skin conditions")
    st.divider()
    st.caption("For educational use only. Always consult a dermatologist.")

model, le, class_info = load_model()

# ════════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSIS
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home & Diagnosis":
    st.markdown('<p class="main-title">🔬 SkinSense AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a skin image for AI-powered diagnosis, severity assessment and treatment advice.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<p class="section-head">📸 Upload Skin Image</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a clear close-up photo", type=['jpg','jpeg','png'])

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
            add_monitor = st.checkbox("Add to weekly monitoring tracker")
            spot_name   = st.text_input("Spot name", "Spot 1") if add_monitor else ""
            run         = st.button("🔬 Analyse", type="primary", use_container_width=True)

            if run:
                if model is None:
                    st.error("Model not loaded. Please refresh the page.")
                else:
                    with st.spinner("Analysing..."):
                        results, all_preds, classes = predict(image, model, le)
                    top = results[0]
                    with col2:
                        st.markdown('<p class="section-head">🩺 Result</p>', unsafe_allow_html=True)
                        sev_map    = {'HIGH':'danger','MEDIUM':'warning','LOW':'success'}
                        card_class = sev_map.get(top['severity'], 'result')
                        info       = DISEASE_INFO.get(top['code'], {})
                        st.markdown(f"""
                        <div class="{card_class}-card">
                            <h3 style="margin:0;color:{top['color']}">{top['name']}</h3>
                            <p style="margin:4px 0 0">Confidence: <b>{top['confidence']:.1f}%</b> | Severity: <b>{top['severity']}</b></p>
                            <p style="margin:8px 0 0;font-size:0.88rem;color:#444">{info.get('description','')}</p>
                        </div>""", unsafe_allow_html=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Confidence", f"{top['confidence']:.1f}%")
                        m2.metric("Severity",   top['severity'])
                        m3.metric("Referral",   "YES ⚠️" if top['referral'] else "NO ✅")

                        if top['referral']:
                            st.markdown('<div class="danger-card">🚨 <b>Doctor Referral Recommended.</b> Please consult a certified dermatologist soon.</div>', unsafe_allow_html=True)

                        # Bar chart
                        fig, ax = plt.subplots(figsize=(7, 2.5))
                        names  = [r['name'][:22] for r in results]
                        confs  = [r['confidence'] for r in results]
                        colors = [r['color'] for r in results]
                        bars   = ax.barh(names, confs, color=colors, alpha=0.85)
                        for bar, val in zip(bars, confs):
                            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                    f'{val:.1f}%', va='center', fontsize=9)
                        ax.set_xlim(0, 110)
                        ax.set_xlabel('Confidence (%)')
                        ax.invert_yaxis()
                        fig.patch.set_alpha(0)
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.markdown('<p class="section-head">💊 Treatment Advice</p>', unsafe_allow_html=True)
                        for tip in info.get('treatment', []):
                            st.markdown(f"• {tip}")

                        if add_monitor and spot_name:
                            st.session_state.history.append({
                                'spot'      : spot_name,
                                'date'      : datetime.now().strftime('%Y-%m-%d'),
                                'disease'   : top['name'],
                                'confidence': top['confidence'],
                                'severity'  : top['severity'],
                            })
                            st.success(f"Added to monitoring for '{spot_name}'")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — MONITORING
# ════════════════════════════════════════════════════════════════
elif page == "📅 Weekly Monitoring":
    st.markdown('<p class="main-title">📅 Weekly Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Track your condition over time — is it improving or worsening?</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No data yet. Go to Home & Diagnosis, upload an image and check 'Add to weekly monitoring tracker'.")
    else:
        df      = pd.DataFrame(st.session_state.history)
        spot    = st.selectbox("Select spot", df['spot'].unique())
        spot_df = df[df['spot'] == spot].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spot_df.index, spot_df['confidence'], marker='o', linewidth=2.5,
                color='#185FA5', markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(spot_df.index, spot_df['confidence'], alpha=0.1, color='#185FA5')

        if len(spot_df) >= 2:
            trend = spot_df['confidence'].iloc[-1] - spot_df['confidence'].iloc[0]
            label = 'Worsening' if trend > 2 else 'Improving' if trend < -2 else 'Stable'
            ax.set_title(f'{label} ({trend:+.1f}%)', fontweight='bold')
        else:
            ax.set_title('Need 2+ entries to show trend')

        ax.axhline(70, color='orange', linestyle='--', linewidth=1, label='Medium risk')
        ax.axhline(85, color='red',    linestyle='--', linewidth=1, label='High risk')
        ax.set_ylabel('Confidence (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)

        if len(spot_df) >= 2:
            if trend > 2:
                st.markdown('<div class="danger-card">⚠️ Condition worsening — consult a dermatologist.</div>', unsafe_allow_html=True)
            elif trend < -2:
                st.markdown('<div class="success-card">✅ Condition improving — continue monitoring.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">➡️ Condition stable — continue weekly monitoring.</div>', unsafe_allow_html=True)

        st.dataframe(spot_df[['date','disease','confidence','severity']], use_container_width=True, hide_index=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="main-title">ℹ️ About SkinSense AI</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    SkinSense AI addresses the global shortage of dermatologists — **3 billion people** worldwide lack access to skin care specialists.

    ### How It Works
    1. User uploads a photo of a skin concern
    2. CNN model (MobileNetV2) trained on HAM10000 analyses it
    3. Returns diagnosis, confidence, severity and treatment advice
    4. High-risk conditions trigger automatic doctor referral alert
    5. Weekly monitoring tracks whether condition improves or worsens

    ### Disease Categories
    | Code | Disease | Risk |
    |------|---------|------|
    | MEL  | Melanoma | HIGH |
    | NV   | Melanocytic Nevi | LOW |
    | BCC  | Basal Cell Carcinoma | MEDIUM |
    | AKIEC | Actinic Keratoses | MEDIUM |
    | BKL  | Benign Keratosis | LOW |
    | DF   | Dermatofibroma | LOW |
    | VASC | Vascular Lesions | LOW |

    ### Tech Stack
    - Model: MobileNetV2 + TensorFlow/Keras
    - Dataset: HAM10000 (10,000+ dermatoscopy images)
    - App: Streamlit | Image: OpenCV, PIL

    > ⚠️ **For educational purposes only.** Always consult a qualified dermatologist.

    **Developer:** Koina Chatterjee | AIML-A6/9951 | Batch June 2026
    """)
