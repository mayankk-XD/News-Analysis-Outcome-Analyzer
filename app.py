import streamlit as st
from retrieval.retrieve import search

from models.sentiment import get_sentiment
from models.outcome import predict_outcome
from models.llm_explainer import generate_llm_explanation

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="News Intelligence AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CLEAN THEME ----------------
st.markdown("""
<style>

/* Background */
body, .stApp {
    background-color: #0b1220;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* Header */
h1, h2, h3 {
    color: #f9fafb;
    font-weight: 600;
}

/* Input box */
.stTextInput > div > div > input {
    background-color: #111827;
    color: white;
    border-radius: 10px;
    border: 1px solid #374151;
}

/* Button */
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton button:hover {
    background-color: #1d4ed8;
}

/* Card */
.card {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Metrics box */
.metric {
    background-color: #0f172a;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #1f2937;
    text-align: center;
}

.small {
    color: #9ca3af;
    font-size: 13px;
}

hr {
    border: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("News Intelligence Dashboard")
st.caption("AI-powered sentiment • impact analysis • reasoning engine")

# ---------------- INPUT ----------------
query = st.text_input("Search global news insights...")

# ---------------- SEARCH ----------------
if st.button("Analyze"):

    if not query.strip():
        st.warning("Enter a query")
        st.stop()

    # =====================================
    # FETCH ONLY 2 RESULTS
    # =====================================

    results = search(query, top_k=2)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        st.error("No results found")
        st.stop()

    # =====================================
    # LOOP RESULTS
    # =====================================

    for i, (doc, meta) in enumerate(zip(docs, metas)):

        sentiment = get_sentiment(doc)
        impact = predict_outcome(doc)

        # =====================================
        # GEMINI CALL
        # =====================================

        try:

            explanation = generate_llm_explanation(
                doc,
                sentiment,
                impact,
                None
            )

        except:

            explanation = """
⚠️ AI explanation unavailable.

Possible reasons:
- Gemini quota exceeded
- API rate limit reached

Showing local analysis only.
"""

        # =====================================
        # SENTIMENT BADGE
        # =====================================

        if sentiment.lower() == "positive":
            badge = "🟢 Positive"

        elif sentiment.lower() == "negative":
            badge = "🔴 Negative"

        else:
            badge = "🟡 Neutral"

        # =====================================
        # CARD UI
        # =====================================

        st.markdown(f"""
        <div class="card">

        <h3>📰 News Result {i+1}</h3>

        <p style="color:#cbd5e1; line-height:1.5;">
        {doc[:400]}...
        </p>

        <hr>

        <p>
        <b>{badge}</b>
        &nbsp; | &nbsp;
        <b>📊 Impact:</b> {impact}
        </p>

        <p class="small">
        🌐 Source: {meta.get('source','N/A')}
        </p>

        </div>
        """, unsafe_allow_html=True)

        # =====================================
        # AI ANALYSIS BOX FOR BOTH
        # =====================================

        with st.expander("🧠 AI Analysis Reasoning"):
            st.write(explanation)

        st.write("")