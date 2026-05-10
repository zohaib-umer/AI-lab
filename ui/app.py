# =========================================================
# STREAMLIT FRONTEND
# Intelligent Reading Comprehension System
# =========================================================

import streamlit as st
import pandas as pd
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import RCInferenceSystem

st.set_page_config(page_title="EduAI Comprehension", layout="wide", page_icon="🧠")

# =========================================================
# SESSION STATE
# =========================================================

for key, val in {
    'hints_used': 0,
    'answer_revealed': False,
    'article': "",
    'question': "",
    'selected_option': None,
    'answer_checked': False,
    'active_options': [],
    'active_correct': "",
    'label_map': {},
    'text_to_label': {},
    'prediction_result': None,
    'hints': [],
    'answer_text': "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, #1e3a8a 0%, transparent 25%),
                radial-gradient(circle at bottom right, #7c3aed 0%, transparent 25%),
                linear-gradient(135deg, #0f172a 0%, #111827 100%);
    color: white;
}
.main-title {
    text-align: center; font-size: 3.5rem; font-weight: 700;
    background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.subtitle { text-align: center; color: #cbd5e1; font-size: 1.1rem; margin-bottom: 30px; }
.glass-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px; padding: 25px; backdrop-filter: blur(18px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 25px;
}

/* ── Hide ONLY option buttons via their wrapper class ── */


.stButton > button {
    width: 100%; background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white; border: none; border-radius: 14px; padding: 12px;
    font-weight: 600; transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px); box-shadow: 0px 10px 25px rgba(124,58,237,0.5);
}

.success-msg {
    background: rgba(16,185,129,0.15); border-left: 5px solid #10b981;
    padding: 15px; border-radius: 10px; color: #d1fae5; margin-top: 12px;
}
.error-msg {
    background: rgba(239,68,68,0.15); border-left: 5px solid #ef4444;
    padding: 15px; border-radius: 10px; color: #fecaca; margin-top: 12px;
}
.hint-box {
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3);
    padding: 15px; border-radius: 12px; margin-top: 10px; color: #fde68a;
}
.prediction-box {
    background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.25);
    border-radius: 14px; padding: 16px; margin-top: 6px;
}
.pred-label-badge {
    width: 36px; height: 36px; border-radius: 50%;
    background: #818cf8; color: white;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1rem;
}
.pred-label-text { font-size: 0.85rem; color: #94a3b8; margin-bottom: 2px; }
.pred-value-text { font-size: 0.95rem; color: #e2e8f0; font-weight: 500; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06); border-radius: 18px; padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}
/* ========================================== */
/* RADIO OPTIONS PREMIUM */
/* ========================================== */

div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 18px;
    border-radius: 14px;
    margin-bottom: 10px;
    transition: all 0.25s ease;
}

div[role="radiogroup"] > label:hover {
    border: 1px solid #818cf8;
    background: rgba(129,140,248,0.12);
    transform: translateY(-2px);
}

div[role="radiogroup"] label[data-checked="true"] {
    border: 2px solid #818cf8 !important;
    background: rgba(129,140,248,0.18) !important;
}
</style>
""", unsafe_allow_html=True)
# =========================================================
# SYSTEM LOADING
# =========================================================

@st.cache_resource
def load_system():
    return RCInferenceSystem()

sys_ai = load_system()

@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv('data/raw/train.csv')
    except:
        return None


def option_card_html(label, text, state="default"):
    """
    state: 'default' | 'selected' | 'correct' | 'wrong' | 'dimmed'
    """
    styles = {
        "default": {
            "bg": "rgba(255,255,255,0.04)",
            "border": "1.5px solid rgba(255,255,255,0.10)",
            "color": "#e2e8f0",
            "badge_bg": "rgba(255,255,255,0.08)",
            "badge_color": "#94a3b8",
            "badge_border": "1.5px solid rgba(255,255,255,0.15)",
        },
        "selected": {
            "bg": "rgba(129,140,248,0.18)",
            "border": "2px solid #818cf8",
            "color": "#c7d2fe",
            "badge_bg": "#818cf8",
            "badge_color": "white",
            "badge_border": "2px solid #818cf8",
        },
        "correct": {
            "bg": "rgba(16,185,129,0.15)",
            "border": "2px solid #10b981",
            "color": "#d1fae5",
            "badge_bg": "#10b981",
            "badge_color": "white",
            "badge_border": "2px solid #10b981",
        },
        "wrong": {
            "bg": "rgba(239,68,68,0.12)",
            "border": "2px solid #ef4444",
            "color": "#fecaca",
            "badge_bg": "#ef4444",
            "badge_color": "white",
            "badge_border": "2px solid #ef4444",
        },
        "dimmed": {
            "bg": "rgba(255,255,255,0.02)",
            "border": "1.5px solid rgba(255,255,255,0.05)",
            "color": "#475569",
            "badge_bg": "rgba(255,255,255,0.04)",
            "badge_color": "#475569",
            "badge_border": "1.5px solid rgba(255,255,255,0.06)",
        },
    }
    s = styles.get(state, styles["default"])
    return f"""
    <div style="
        display:flex; align-items:center; gap:14px;
        padding:14px 18px; border-radius:14px;
        border:{s['border']}; background:{s['bg']};
        color:{s['color']}; font-size:0.95rem;
        margin-bottom:8px; transition: all 0.2s ease;
        cursor: pointer;
    ">
        <div style="
            width:32px; height:32px; border-radius:50%;
            background:{s['badge_bg']}; color:{s['badge_color']};
            border:{s['badge_border']};
            display:flex; align-items:center; justify-content:center;
            font-weight:700; font-size:0.85rem; flex-shrink:0;
        ">{label}</div>
        <span>{text}</span>
    </div>
    """

def option_card_html(label, text, state="default"):
    styles = {
        "default": {
            "bg": "rgba(255,255,255,0.04)",
            "border": "1.5px solid rgba(255,255,255,0.10)",
            "color": "#e2e8f0",
            "badge_bg": "rgba(255,255,255,0.08)",
            "badge_color": "#94a3b8",
            "badge_border": "1.5px solid rgba(255,255,255,0.15)",
        },
        "selected": {
            "bg": "rgba(129,140,248,0.18)",
            "border": "2px solid #818cf8",
            "color": "#c7d2fe",
            "badge_bg": "#818cf8",
            "badge_color": "white",
            "badge_border": "2px solid #818cf8",
        },
        "correct": {
            "bg": "rgba(16,185,129,0.15)",
            "border": "2px solid #10b981",
            "color": "#d1fae5",
            "badge_bg": "#10b981",
            "badge_color": "white",
            "badge_border": "2px solid #10b981",
        },
        "wrong": {
            "bg": "rgba(239,68,68,0.12)",
            "border": "2px solid #ef4444",
            "color": "#fecaca",
            "badge_bg": "#ef4444",
            "badge_color": "white",
            "badge_border": "2px solid #ef4444",
        },
        "dimmed": {
            "bg": "rgba(255,255,255,0.02)",
            "border": "1.5px solid rgba(255,255,255,0.05)",
            "color": "#475569",
            "badge_bg": "rgba(255,255,255,0.04)",
            "badge_color": "#475569",
            "badge_border": "1.5px solid rgba(255,255,255,0.06)",
        },
    }
    s = styles.get(state, styles["default"])
    return f"""
    <div style="
        display:flex; align-items:center; gap:14px;
        padding:14px 18px; border-radius:14px;
        border:{s['border']}; background:{s['bg']};
        color:{s['color']}; font-size:0.95rem;
        margin-bottom:0px; cursor:pointer;
    ">
        <div style="
            width:32px; height:32px; border-radius:50%;
            background:{s['badge_bg']}; color:{s['badge_color']};
            border:{s['badge_border']};
            display:flex; align-items:center; justify-content:center;
            font-weight:700; font-size:0.85rem; flex-shrink:0;
        ">{label}</div>
        <span>{text}</span>
    </div>
    """


def render_options(options, correct, checked, text_to_label):

    labels = []

    for opt in options:
        label = text_to_label.get(opt, "?")
        labels.append(f"{label}. {opt}")

    selected = st.radio(
        "Choose an option",
        labels,
        label_visibility="collapsed"
    )

    selected_text = selected.split(". ", 1)[1]

    return selected_text

# =========================================================
# HEADER
# =========================================================

st.markdown('<h1 class="main-title">🧠 EduAI Comprehension</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Reading Comprehension and Quiz Generation using Machine Learning</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 Quiz System", "💡 Hint Engine", "📊 Analytics"])

# =========================================================
# TAB 1
# =========================================================

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📝 Reading Passage")

        if st.button("🎲 Load Random Article"):
            df = load_dataframe()
            if df is not None:
                sample = df.sample(1).iloc[0]
                st.session_state.update({
                    'article': sample['article'],
                    'question': sample['question'],
                    'answer_text': sample[sample['answer']],
                    'options_dict': {'A': sample['A'], 'B': sample['B'], 'C': sample['C'], 'D': sample['D']},
                    'hints_used': 0, 'answer_revealed': False,
                    'selected_option': None, 'answer_checked': False,
                    'active_options': [], 'label_map': {},
                    'text_to_label': {}, 'prediction_result': None,
                })
                st.rerun()

        article_text  = st.text_area("Article",  key="article",  height=280)
        question_text = st.text_input("Question", key="question")
        manual_answer = st.text_input("Correct Answer (Manual)",
                                      help="For custom articles, type the correct answer here.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Smart Quiz")

        if article_text and question_text:
            final_ans_text = st.session_state.get('answer_text') or manual_answer

            # Generate Quiz button
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("🚀 Generate Quiz"):
                if not final_ans_text:
                    st.warning("Please provide a correct answer for distractor generation.")
                else:
                    with st.spinner("Processing AI pipeline..."):
                        distractors = sys_ai.generate_distractors(article_text, final_ans_text)
                        options = [final_ans_text] + distractors
                        random.shuffle(options)
                        label_map     = {chr(65 + i): opt for i, opt in enumerate(options)}
                        text_to_label = {v: k for k, v in label_map.items()}
                        st.session_state.update({
                            'active_options': options,
                            'active_correct': final_ans_text,
                            'label_map': label_map,
                            'text_to_label': text_to_label,
                            'selected_option': None,
                            'answer_checked': False,
                            'prediction_result': sys_ai.verify_answer(
                                article_text, question_text,
                                label_map
                            ),
                        })
            st.markdown('</div>', unsafe_allow_html=True)

            # Options
            if st.session_state['active_options']:
                st.markdown("### Select your answer")

                selected = render_options(
                    st.session_state['active_options'],
                    st.session_state['active_correct'],
                    st.session_state['answer_checked'],
                    st.session_state['text_to_label'],
                )

                # This replaces the old click handler
                st.session_state['selected_option'] = selected

                # Check Answer
                if st.session_state['selected_option'] and not st.session_state['answer_checked']:
                    sel_label = st.session_state['text_to_label'].get(st.session_state['selected_option'], "")
                    st.markdown('<div class="action-btn">', unsafe_allow_html=True)
                    if st.button(f"✅ Check Answer — Option {sel_label}"):
                        st.session_state['answer_checked'] = True
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                # Result
                if st.session_state['answer_checked']:
                    correct_label = st.session_state['text_to_label'].get(st.session_state['active_correct'], "")
                    if st.session_state['selected_option'] == st.session_state['active_correct']:
                        st.markdown(
                            f'<div class="success-msg">🎉 Correct! <strong>Option {correct_label}</strong> — {st.session_state["active_correct"]}</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="error-msg">❌ Incorrect. Correct answer: <strong>Option {correct_label}</strong> — {st.session_state["active_correct"]}</div>',
                            unsafe_allow_html=True)

                    st.markdown('<div class="action-btn">', unsafe_allow_html=True)
                    if st.button("🔄 Try Again"):
                        st.session_state['selected_option'] = None
                        st.session_state['answer_checked'] = False
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                # Model Prediction
                # Model Prediction
                with st.expander("🤖 Model Prediction (Internal Analysis)"):
                    res = st.session_state.get('prediction_result')
                    t2l = st.session_state.get('text_to_label', {})
                    if isinstance(res, dict):
                        pred_text  = res.get('Predicted_Text', 'N/A')
                        pred_label = t2l.get(pred_text, res.get('Predicted_Answer', 'N/A'))
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                                <div class="pred-label-badge">{pred_label}</div>
                                <div>
                                    <div class="pred-label-text">Predicted Label</div>
                                    <div class="pred-value-text">Option {pred_label}</div>
                                </div>
                            </div>
                            <div class="pred-label-text">Extracted Text</div>
                            <div class="pred-value-text">{pred_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- NEW: Show a Bar Chart of the Confidence Scores! ---
                        conf_scores = res.get('Confidence_Scores')
                        if conf_scores:
                            st.write("**Model Confidence by Option:**")
                            # Convert dictionary to DataFrame for Streamlit bar chart
                            chart_data = pd.DataFrame(
                                list(conf_scores.values()), 
                                index=list(conf_scores.keys()), 
                                columns=["Probability"]
                            )
                            st.bar_chart(chart_data)
                            
                    elif res:
                        st.info(f"Predicted Label: {str(res)}")
                    else:
                        st.info("Generate a quiz to see model predictions.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💡 Intelligent Hint Engine")

    if st.button("Generate Hints"):
        if st.session_state['article'] and st.session_state['question']:
            with st.spinner("Extracting clues..."):
                st.session_state.update({
                    'hints': sys_ai.generate_hints(st.session_state['article'], st.session_state['question']),
                    'hints_used': 0,
                    'answer_revealed': False,
                })
        else:
            st.warning("Load or enter an article and question first.")

    if st.session_state.get('hints'):
        for i in range(st.session_state['hints_used']):
            st.markdown(
                f'<div class="hint-box"><strong>Clue {i+1}:</strong> {st.session_state["hints"][i]}</div>',
                unsafe_allow_html=True)

        if st.session_state['hints_used'] < 3:
            if st.button(f"Reveal Hint {st.session_state['hints_used'] + 1}"):
                st.session_state['hints_used'] += 1
                st.rerun()
        elif not st.session_state['answer_revealed']:
            if st.button("🚨 Reveal Final Answer"):
                st.session_state['answer_revealed'] = True
                st.rerun()

        if st.session_state.get('answer_revealed'):
            final_reveal  = st.session_state.get('answer_text', "")
            correct_label = st.session_state.get('text_to_label', {}).get(final_reveal, "")
            label_str = f"Option {correct_label} — " if correct_label else ""
            st.markdown(
                f'<div class="success-msg">✅ Correct Answer: <strong>{label_str}{final_reveal}</strong></div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble Accuracy", "51.00%")
    c2.metric("Macro F1 Score", "0.475")
    c3.metric("Avg Latency", "1.1s")
    st.markdown("---")
    st.write("**Architecture:** Soft-Voting Ensemble (Logistic Regression + Naive Bayes)")
    st.write("**Feature Engineering:** TF-IDF Vectorization (Max 5000 Features)")
    st.markdown('</div>', unsafe_allow_html=True)