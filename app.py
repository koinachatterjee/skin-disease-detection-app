import streamlit as st
import numpy as np
import pickle
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime
import os
import gdown

st.set_page_config(
    page_title="SkinSense AI — Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    text-align: center; color: white;
}
.main-hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #f8a4a4, #f56565, #e53e3e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.main-hero p  { font-size: 1.1rem; color: #cbd5e0; margin: 0.5rem 0 0; }

.stat-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1.2rem; text-align: center; color: white;
}
.stat-card .num { font-size: 2rem; font-weight: 700; color: #f56565; }
.stat-card .lbl { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; }

.upload-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px dashed rgba(245,101,101,0.5);
    border-radius: 16px; padding: 2rem; color: white;
}

.result-hero {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px; padding: 1.5rem; color: white;
    border-left: 5px solid #f56565; margin-bottom: 1rem;
}
.result-hero h2 { font-size: 1.8rem; font-weight: 700; margin: 0; }
.result-hero .conf { font-size: 1rem; color: #a0aec0; margin-top: 6px; }

.sev-high   { background: linear-gradient(135deg, #7b0000, #c53030);
              border-radius: 12px; padding: 1rem 1.5rem; color: white; margin: 0.5rem 0; }
.sev-medium { background: linear-gradient(135deg, #7b4a00, #c47800);
              border-radius: 12px; padding: 1rem 1.5rem; color: white; margin: 0.5rem 0; }
.sev-low    { background: linear-gradient(135deg, #004d30, #1D9E75);
              border-radius: 12px; padding: 1rem 1.5rem; color: white; margin: 0.5rem 0; }

.referral-alert {
    background: linear-gradient(135deg, #7b0000, #c53030);
    border-radius: 12px; padding: 1.2rem 1.5rem; color: white;
    margin: 0.5rem 0; border: 1px solid rgba(255,100,100,0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.85} }

.treat-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 1.2rem 1.5rem; color: white;
    border: 1px solid rgba(255,255,255,0.08); margin: 0.5rem 0;
}
.treat-card h4 { color: #f56565; margin: 0 0 0.8rem; font-size: 1rem; }

.monitor-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 1.2rem; color: white;
    border: 1px solid rgba(255,255,255,0.08); text-align: center;
}

.section-title {
    font-size: 1.1rem; font-weight: 600; color: #f56565;
    margin: 1.2rem 0 0.6rem; text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-high   { background:#c53030; color:white; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#c47800; color:white; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-low    { background:#1D9E75; color:white; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
}
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stRadio label { color: #cbd5e0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Disease database ──────────────────────────────────────────
DISEASE_INFO = {
    'MEL':   {
        'name':'Melanoma', 'severity':'HIGH', 'color':'#f56565', 'referral':True,
        'icon':'⚠️',
        'description':'The most dangerous form of skin cancer originating in melanocytes. Immediate medical attention is critical — survival rate drops from 98% to 25% if not caught early.',
        'treatment':[
            '🏥 Immediate dermatologist consultation — do not delay',
            '🔪 Surgical excision with wide safety margins',
            '🧬 Sentinel lymph node biopsy to check for spread',
            '💉 Immunotherapy or targeted therapy if advanced',
            '☀️ Strict sun avoidance and SPF 50+ daily',
            '📅 Full-body skin checks every 3 months',
        ],
        'risk_factors':['UV radiation exposure','Family history','Fair skin','Multiple moles'],
    },
    'NV':    {
        'name':'Melanocytic Nevi (Mole)', 'severity':'LOW', 'color':'#48bb78', 'referral':False,
        'icon':'✅',
        'description':'Common benign moles formed from clusters of pigmented cells. Usually harmless but should be monitored using the ABCDE rule for any changes.',
        'treatment':[
            '✅ Generally benign — no immediate treatment needed',
            '👁️ Monitor using ABCDE rule monthly',
            '☀️ Apply broad-spectrum SPF 30+ daily',
            '📅 Annual dermatologist skin check recommended',
            '🚫 Avoid picking or irritating the mole',
        ],
        'risk_factors':['Sun exposure','Genetics','Hormonal changes'],
    },
    'BCC':   {
        'name':'Basal Cell Carcinoma', 'severity':'MEDIUM', 'color':'#f6ad55', 'referral':True,
        'icon':'⚡',
        'description':'The most common skin cancer. Grows slowly and rarely spreads, but causes significant local tissue damage if left untreated.',
        'treatment':[
            '🏥 Consult dermatologist within 2-4 weeks',
            '🔬 Mohs micrographic surgery — gold standard treatment',
            '📡 Radiation therapy for inoperable cases',
            '💊 Topical treatments for superficial BCC',
            '☀️ Strict sun protection — SPF 50+ and protective clothing',
        ],
        'risk_factors':['Cumulative UV exposure','Fair skin','Age >50','Immunosuppression'],
    },
    'AKIEC': {
        'name':'Actinic Keratoses', 'severity':'MEDIUM', 'color':'#f6ad55', 'referral':True,
        'icon':'⚡',
        'description':'Rough, scaly patches caused by years of sun damage. Considered precancerous — up to 10% progress to squamous cell carcinoma if untreated.',
        'treatment':[
            '🏥 Dermatologist consultation recommended',
            '❄️ Cryotherapy (liquid nitrogen freezing) — most common',
            '💊 Topical: 5-fluorouracil or imiquimod cream',
            '💡 Photodynamic therapy (PDT) for widespread lesions',
            '☀️ Daily SPF 50+ and sun-protective clothing essential',
        ],
        'risk_factors':['Chronic sun exposure','Fair skin','Outdoor occupation','Age >40'],
    },
    'BKL':   {
        'name':'Benign Keratosis', 'severity':'LOW', 'color':'#48bb78', 'referral':False,
        'icon':'✅',
        'description':'Non-cancerous skin growths that become more common with age. They may look concerning but are completely harmless.',
        'treatment':[
            '✅ Harmless — no medical treatment necessary',
            '✂️ Removal for cosmetic reasons via cryotherapy',
            '🧴 Keep skin well-moisturized',
            '👁️ Monitor for rapid changes in size or appearance',
            '📅 Routine annual skin check',
        ],
        'risk_factors':['Aging','Sun exposure','Genetics'],
    },
    'DF':    {
        'name':'Dermatofibroma', 'severity':'LOW', 'color':'#48bb78', 'referral':False,
        'icon':'✅',
        'description':'Common benign fibrous nodules, often appearing on the legs. Usually painless and completely harmless.',
        'treatment':[
            '✅ Benign — no treatment required',
            '🔪 Surgical removal if causing discomfort or cosmetic concern',
            '💉 Corticosteroid injections to flatten the lesion',
            '🚫 Avoid trauma or repeated irritation to the area',
            '👁️ Monitor for any unexpected growth',
        ],
        'risk_factors':['Minor skin injuries','Insect bites','Female sex'],
    },
    'VASC':  {
        'name':'Vascular Lesion', 'severity':'LOW', 'color':'#63b3ed', 'referral':False,
        'icon':'ℹ️',
        'description':'Blood vessel abnormalities in the skin including cherry angiomas, spider veins, and port-wine stains. Most are harmless.',
        'treatment':[
            '✅ Most vascular lesions are benign',
            '💡 Pulsed dye laser — most effective treatment',
            '💉 Sclerotherapy for larger vascular lesions',
            '🔪 Surgical removal for isolated lesions',
            '🏥 Consult dermatologist if growing rapidly or bleeding',
        ],
        'risk_factors':['Genetics','Aging','Hormonal changes','Sun damage'],
    },
}

IMG_SIZE = 224
GDRIVE_ID = '1Tf8NvCJhPbbBADMKRJAG5VE9fmAmz_sM'

# ─── Download model ────────────────────────────────────────────
if not os.path.exists('best_skin_model.h5'):
    with st.spinner('🔄 Loading AI model for first time... (~1 min)'):
        gdown.download(f'https://drive.google.com/uc?id={GDRIVE_ID}',
                       'best_skin_model.h5', quiet=False)

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('best_skin_model.h5')
        with open('label_encoder.pkl','rb') as f: le = pickle.load(f)
        with open('class_names.json','r') as f:   ci = json.load(f)
        return model, le, ci
    except Exception as e:
        st.error(f'Model error: {e}')
        return None, None, None

def predict(image, model, le):
    img  = np.array(image.convert('RGB'))
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img  = img / 255.0
    img  = np.expand_dims(img, 0).astype(np.float32)
    preds = model.predict(img, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]
    results = []
    for idx in top3:
        code = le.classes_[idx]
        info = DISEASE_INFO.get(code, {})
        results.append({
            'code':code, 'name':info.get('name',code),
            'confidence':float(preds[idx])*100,
            'severity':info.get('severity','LOW'),
            'referral':info.get('referral',False),
            'color':info.get('color','#888'),
            'icon':info.get('icon','•'),
        })
    return results, preds, le.classes_

if 'history' not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0'>
        <div style='font-size:3rem'>🔬</div>
        <div style='font-size:1.3rem; font-weight:700; color:white'>SkinSense AI</div>
        <div style='font-size:0.8rem; color:#a0aec0'>Dermatology AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "🏠  Diagnose",
        "📅  Monitor",
        "📊  Analytics",
        "ℹ️  About"
    ], label_visibility='collapsed')
    st.divider()
    st.markdown("""
    <div style='padding:0.5rem 0'>
        <div style='font-size:0.75rem; color:#a0aec0; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem'>Model Info</div>
        <div style='color:white; font-size:0.9rem'>🧠 MobileNetV2 CNN</div>
        <div style='color:#a0aec0; font-size:0.85rem; margin:4px 0'>📊 HAM10000 Dataset</div>
        <div style='color:#a0aec0; font-size:0.85rem; margin:4px 0'>🎯 7 Disease Classes</div>
        <div style='color:#a0aec0; font-size:0.85rem; margin:4px 0'>✅ >80% Accuracy</div>
        <div style='color:#a0aec0; font-size:0.85rem; margin:4px 0'>🖼️ 10,000+ Images</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#718096; text-align:center; padding:0.5rem'>
        ⚕️ For educational use only.<br>Always consult a dermatologist.
    </div>
    """, unsafe_allow_html=True)

model, le, class_info = load_model()

# ════════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSE
# ════════════════════════════════════════════════════════════════
if "Diagnose" in page:
    # Hero
    st.markdown("""
    <div class='main-hero'>
        <h1>🔬 SkinSense AI</h1>
        <p>AI-powered skin disease detection trained on 10,000+ clinical dermatoscopy images</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("<div class='stat-card'><div class='num'>10K+</div><div class='lbl'>Training Images</div></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='stat-card'><div class='num'>7</div><div class='lbl'>Disease Classes</div></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='stat-card'><div class='num'>>80%</div><div class='lbl'>Model Accuracy</div></div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='stat-card'><div class='num'>3B</div><div class='lbl'>People We Serve</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown("<div class='section-title'>📸 Upload Skin Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=['jpg','jpeg','png'],
                                    label_visibility='collapsed')
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True,
                     caption="Uploaded image — ready for analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            add_mon  = st.checkbox("📅 Add to weekly monitoring tracker")
            spot_name = st.text_input("Spot name (e.g. Left arm)", "Spot 1") if add_mon else ""
            run = st.button("🔬 Run AI Analysis", type="primary", use_container_width=True)

            if run:
                if model is None:
                    st.error("Model not loaded. Please refresh.")
                else:
                    with st.spinner("🧠 Analysing with AI..."):
                        results, all_preds, classes = predict(image, model, le)
                    top  = results[0]
                    info = DISEASE_INFO.get(top['code'], {})

                    with col2:
                        # Main result
                        sev = top['severity']
                        sev_class = {'HIGH':'sev-high','MEDIUM':'sev-medium','LOW':'sev-low'}.get(sev,'sev-low')
                        badge_class = {'HIGH':'badge-high','MEDIUM':'badge-medium','LOW':'badge-low'}.get(sev,'badge-low')

                        st.markdown(f"""
                        <div class='result-hero'>
                            <div style='display:flex; align-items:center; gap:12px'>
                                <span style='font-size:2.5rem'>{info.get('icon','🔬')}</span>
                                <div>
                                    <h2>{top['name']}</h2>
                                    <div class='conf'>
                                        Confidence: <b style='color:#f56565'>{top['confidence']:.1f}%</b>
                                        &nbsp;&nbsp;
                                        <span class='{badge_class}'>{sev} RISK</span>
                                        &nbsp;&nbsp;
                                        {'<span style="color:#f56565">⚠️ REFERRAL NEEDED</span>' if top['referral'] else '<span style="color:#48bb78">✅ NO REFERRAL</span>'}
                                    </div>
                                </div>
                            </div>
                            <p style='margin:1rem 0 0; color:#cbd5e0; font-size:0.92rem; line-height:1.6'>
                                {info.get('description','')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Metrics
                        m1,m2,m3 = st.columns(3)
                        m1.metric("🎯 Confidence",  f"{top['confidence']:.1f}%")
                        m2.metric("⚠️ Severity",    top['severity'])
                        m3.metric("👨‍⚕️ See Doctor", "YES" if top['referral'] else "NO")

                        # Referral alert
                        if top['referral']:
                            st.markdown("""
                            <div class='referral-alert'>
                                <b>🚨 DOCTOR REFERRAL STRONGLY RECOMMENDED</b><br>
                                <span style='font-size:0.9rem'>This condition requires professional medical evaluation.
                                Please consult a certified dermatologist as soon as possible.
                                Do not self-medicate or ignore this result.</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Top 3 bar chart
                        st.markdown("<div class='section-title'>📊 Prediction Breakdown</div>", unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(7, 2.8))
                        fig.patch.set_facecolor('#1a1a2e')
                        ax.set_facecolor('#16213e')
                        names  = [r['name'][:25] for r in results]
                        confs  = [r['confidence'] for r in results]
                        bar_colors = [r['color'] for r in results]
                        bars = ax.barh(names, confs, color=bar_colors, alpha=0.9,
                                       edgecolor='none', height=0.5)
                        for bar, val in zip(bars, confs):
                            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                    f'{val:.1f}%', va='center', fontsize=10,
                                    color='white', fontweight='600')
                        ax.set_xlim(0, 115)
                        ax.set_xlabel('Confidence (%)', color='#a0aec0', fontsize=9)
                        ax.tick_params(colors='#cbd5e0', labelsize=9)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('#4a5568')
                        ax.spines['left'].set_color('#4a5568')
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        # Treatment
                        st.markdown("<div class='section-title'>💊 Treatment Recommendations</div>", unsafe_allow_html=True)
                        treat_html = "".join([f"<div style='padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.06); color:#e2e8f0'>{t}</div>"
                                              for t in info.get('treatment',[])])
                        st.markdown(f"<div class='treat-card'><h4>Recommended Actions</h4>{treat_html}</div>",
                                    unsafe_allow_html=True)

                        # Risk factors
                        if info.get('risk_factors'):
                            st.markdown("<div class='section-title'>🔍 Risk Factors</div>", unsafe_allow_html=True)
                            rf_html = " &nbsp;".join([f"<span style='background:rgba(245,101,101,0.15); color:#f56565; padding:3px 10px; border-radius:20px; font-size:0.85rem; margin:2px; display:inline-block'>{r}</span>"
                                                       for r in info.get('risk_factors',[])])
                            st.markdown(f"<div style='margin:0.5rem 0'>{rf_html}</div>", unsafe_allow_html=True)

                        # All probabilities
                        st.markdown("<div class='section-title'>📈 All Disease Probabilities</div>", unsafe_allow_html=True)
                        prob_df = pd.DataFrame({
                            'Disease': [DISEASE_INFO.get(c,{}).get('name',c) for c in classes],
                            'Probability (%)': (all_preds*100).round(2),
                            'Risk': [DISEASE_INFO.get(c,{}).get('severity','LOW') for c in classes]
                        }).sort_values('Probability (%)', ascending=False)
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)

                        if add_mon and spot_name:
                            st.session_state.history.append({
                                'spot':spot_name, 'date':datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'disease':top['name'], 'code':top['code'],
                                'confidence':top['confidence'], 'severity':top['severity'],
                            })
                            st.success(f"✅ Added to monitoring tracker for '{spot_name}'")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — MONITOR
# ════════════════════════════════════════════════════════════════
elif "Monitor" in page:
    st.markdown("<div class='main-hero'><h1>📅 Weekly Monitoring</h1><p>Track your skin condition over time — is it getting better or worse?</p></div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:16px;
             padding:3rem; text-align:center; color:#a0aec0; border:1px dashed rgba(255,255,255,0.1)'>
            <div style='font-size:3rem; margin-bottom:1rem'>📷</div>
            <div style='font-size:1.2rem; color:white; font-weight:600'>No monitoring data yet</div>
            <div style='margin-top:0.5rem'>Go to <b style='color:#f56565'>Diagnose</b>,
            upload an image, check <b style='color:#f56565'>Add to weekly monitoring tracker</b>
            and run analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df    = pd.DataFrame(st.session_state.history)
        spots = df['spot'].unique()
        spot  = st.selectbox("Select monitoring spot", spots)
        sdf   = df[df['spot']==spot].reset_index(drop=True)

        # Summary cards
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f"<div class='monitor-card'><div style='font-size:1.8rem;color:#f56565;font-weight:700'>{len(sdf)}</div><div style='color:#a0aec0;font-size:0.85rem'>Total Visits</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='monitor-card'><div style='font-size:1.8rem;color:#f56565;font-weight:700'>{sdf['confidence'].mean():.1f}%</div><div style='color:#a0aec0;font-size:0.85rem'>Avg Confidence</div></div>", unsafe_allow_html=True)
        with c3:
            if len(sdf)>=2:
                trend = sdf['confidence'].iloc[-1]-sdf['confidence'].iloc[0]
                t_text = f"{'↑ +' if trend>0 else '↓ '}{abs(trend):.1f}%"
                t_color = '#f56565' if trend>2 else '#48bb78' if trend<-2 else '#f6ad55'
            else: t_text,t_color = 'Need 2+','#a0aec0'
            st.markdown(f"<div class='monitor-card'><div style='font-size:1.8rem;color:{t_color};font-weight:700'>{t_text}</div><div style='color:#a0aec0;font-size:0.85rem'>Overall Trend</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Trend chart
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        x = list(range(len(sdf)))
        ax.fill_between(x, sdf['confidence'], alpha=0.15, color='#f56565')
        ax.plot(x, sdf['confidence'], marker='o', linewidth=2.5, color='#f56565',
                markersize=10, markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#f56565')
        for i, (xi, yi) in enumerate(zip(x, sdf['confidence'])):
            ax.annotate(f'{yi:.0f}%', (xi, yi), textcoords='offset points',
                        xytext=(0,12), ha='center', fontsize=9, color='white', fontweight='600')
        ax.axhline(85, color='#f56565', linestyle='--', linewidth=1, alpha=0.6, label='High risk (85%)')
        ax.axhline(70, color='#f6ad55', linestyle='--', linewidth=1, alpha=0.6, label='Medium risk (70%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Visit {i+1}\n{sdf['date'].iloc[i][:10]}" for i in x],
                           fontsize=8, color='#a0aec0')
        ax.set_ylabel('AI Confidence (%)', color='#a0aec0', fontsize=10)
        ax.set_ylim(0, 110)
        ax.tick_params(colors='#a0aec0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#4a5568')
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', edgecolor='#4a5568')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(sdf)>=2:
            trend = sdf['confidence'].iloc[-1]-sdf['confidence'].iloc[0]
            if trend>2:
                st.markdown('<div class="sev-high">⚠️ <b>Condition appears to be WORSENING.</b> AI confidence in disease classification is increasing. Please consult a dermatologist soon.</div>', unsafe_allow_html=True)
            elif trend<-2:
                st.markdown('<div class="sev-low">✅ <b>Condition appears to be IMPROVING.</b> AI confidence is decreasing. Continue monitoring weekly and maintain treatment.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sev-medium">➡️ <b>Condition is STABLE.</b> No significant change detected. Continue weekly monitoring.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(sdf[['date','disease','confidence','severity']], use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"): st.session_state.history=[]; st.rerun()

# ════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("<div class='main-hero'><h1>📊 Disease Analytics</h1><p>Understanding the 7 skin conditions this AI detects</p></div>", unsafe_allow_html=True)

    # Disease cards
    cols = st.columns(2)
    for i, (code, info) in enumerate(DISEASE_INFO.items()):
        with cols[i%2]:
            sev = info['severity']
            badge = {'HIGH':'badge-high','MEDIUM':'badge-medium','LOW':'badge-low'}.get(sev,'badge-low')
            treat_list = "".join([f"<li style='color:#cbd5e0;margin:3px 0'>{t}</li>" for t in info['treatment'][:3]])
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                 border-radius:12px; padding:1.2rem; margin-bottom:1rem;
                 border-left:4px solid {info["color"]};
                 border-top:1px solid rgba(255,255,255,0.05)'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:0.5rem'>
                    <span style='font-size:1.5rem'>{info['icon']}</span>
                    <div>
                        <div style='color:white;font-weight:600;font-size:1rem'>{info['name']}</div>
                        <span class='{badge}'>{sev} RISK</span>
                    </div>
                </div>
                <p style='color:#a0aec0;font-size:0.85rem;margin:0.5rem 0;line-height:1.5'>{info['description']}</p>
                <ul style='margin:0.5rem 0;padding-left:1.2rem'>{treat_list}</ul>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("<div class='main-hero'><h1>ℹ️ About SkinSense AI</h1><p>Major Project — AI InternsElite | Batch June 2026</p></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:1.5rem;color:white;margin-bottom:1rem'>
        <h4 style='color:#f56565;margin-top:0'>🎯 Project Overview</h4>
        <p style='color:#cbd5e0;line-height:1.7'>SkinSense AI addresses the global dermatology crisis — <b>3 billion people</b> worldwide lack access to skin specialists. Using CNN-based deep learning trained on 10,000+ clinical images, this app provides instant skin disease detection with clinical-grade features.</p>
        </div>
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:1.5rem;color:white'>
        <h4 style='color:#f56565;margin-top:0'>👩‍💻 Developer</h4>
        <p style='color:#cbd5e0'>Name: <b style='color:white'>Koina Chatterjee</b></p>
        <p style='color:#cbd5e0'>Roll No: <b style='color:white'>AIML-A6/9951</b></p>
        <p style='color:#cbd5e0'>Batch: <b style='color:white'>June 2026 — AI InternsElite</b></p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:1.5rem;color:white;margin-bottom:1rem'>
        <h4 style='color:#f56565;margin-top:0'>🧠 Tech Stack</h4>
        <p style='color:#cbd5e0'>• <b style='color:white'>Model:</b> MobileNetV2 (Transfer Learning)</p>
        <p style='color:#cbd5e0'>• <b style='color:white'>Dataset:</b> HAM10000 (10,015 images)</p>
        <p style='color:#cbd5e0'>• <b style='color:white'>Framework:</b> TensorFlow / Keras</p>
        <p style='color:#cbd5e0'>• <b style='color:white'>App:</b> Streamlit</p>
        <p style='color:#cbd5e0'>• <b style='color:white'>Deployment:</b> Streamlit Cloud</p>
        <p style='color:#cbd5e0'>• <b style='color:white'>Image Processing:</b> OpenCV, PIL</p>
        </div>
        <div style='background:linear-gradient(135deg,#7b0000,#c53030);border-radius:12px;padding:1.5rem;color:white'>
        <h4 style='color:white;margin-top:0'>⚕️ Medical Disclaimer</h4>
        <p style='color:#fed7d7;font-size:0.9rem;line-height:1.6'>This application is built for <b>educational purposes only</b>. It is NOT a substitute for professional medical advice. Always consult a qualified dermatologist for any skin concerns. Do not make medical decisions based solely on this app.</p>
        </div>
        """, unsafe_allow_html=True)
