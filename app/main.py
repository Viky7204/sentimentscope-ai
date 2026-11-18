"""
SentimentScope AI - Streamlit Application
"""

import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import SentimentPredictor


st.set_page_config(
    page_title="SentimentScope AI",
    page_icon="💬",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        color: #1a365d;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .result-card {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .positive { background: #f0fff4; border: 2px solid #68d391; }
    .negative { background: #fff5f5; border: 2px solid #fc8181; }
    .neutral  { background: #f7fafc; border: 2px solid #a0aec0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = "models/sentiment_model.joblib"
    if not os.path.exists(path):
        return None
    return SentimentPredictor(path)


def main():
    st.markdown('<p class="main-title">💬 SentimentScope AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Analyze the sentiment of any review or text instantly</p>',
        unsafe_allow_html=True
    )

    predictor = load_model()

    if predictor is None:
        st.error("Model not found!")
        st.info("Please train the model first by running this command:")
        st.code("python app/train.py", language="bash")
        st.stop()

    # Input
    st.markdown("### Enter your text")
    user_text = st.text_area(
        label="Review or any text",
        height=160,
        placeholder="Example: This product is amazing! Completely worth the money..."
    )

    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict(user_text)

            if result["label"] == "Unknown":
                st.error(result.get("message", "Could not process the text."))
            else:
                # Color mapping
                css_map = {
                    "Positive": "positive",
                    "Negative": "negative",
                    "Neutral": "neutral"
                }
                color_map = {
                    "Positive": "#276749",
                    "Negative": "#c53030",
                    "Neutral": "#4a5568"
                }

                css_class = css_map.get(result["label"], "neutral")
                color = color_map.get(result["label"], "#4a5568")

                st.markdown(f"""
                <div class="result-card {css_class}">
                    <h1 style="margin:0; font-size:2.8rem;">{result['emoji']}</h1>
                    <h2 style="color:{color}; margin:0.4rem 0;">{result['label']}</h2>
                    <p style="font-size:1.25rem; margin:0;">Confidence: <b>{result['confidence']}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Probability Breakdown")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("😞 Negative", f"{result['proba_negative']}%")
                with c2:
                    st.metric("😐 Neutral", f"{result['proba_neutral']}%")
                with c3:
                    st.metric("😊 Positive", f"{result['proba_positive']}%")

                with st.expander("View cleaned text used by model"):
                    st.code(result["cleaned_text"])

    # Quick examples
    st.markdown("---")
    st.markdown("### Try quick examples")

    examples = {
        "Positive example": "This product is amazing! Totally worth the money. Highly recommended.",
        "Negative example": "Worst experience ever. The product broke in two days. Complete waste of money.",
        "Neutral example": "It's okay. Nothing special but it does the basic job."
    }

    cols = st.columns(3)
    for idx, (title, text) in enumerate(examples.items()):
        with cols[idx]:
            if st.button(title, use_container_width=True):
                st.session_state["example"] = text
                st.rerun()

    if "example" in st.session_state:
        st.text_area("Loaded example", value=st.session_state["example"], height=100)

    # Footer
    st.markdown("---")
    st.caption("Fresher-level AI project • TF-IDF + Logistic Regression • Built for portfolio & interviews")


if __name__ == "__main__":
    main()
