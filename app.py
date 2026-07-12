import json
import re
import string

import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

# ---------- Setup: NLTK stopwords + stemmer (cached, with offline fallback) ----------
FALLBACK_STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "to", "of", "in",
    "for", "with", "this", "that", "are", "was", "were", "be", "been", "it",
    "as", "by", "do", "does", "how", "what", "can", "i", "my", "me"
}


@st.cache_resource
def load_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            return FALLBACK_STOPWORDS, SnowballStemmer("english")
    try:
        return set(stopwords.words("english")), SnowballStemmer("english")
    except Exception:
        return FALLBACK_STOPWORDS, SnowballStemmer("english")


STOPWORDS, STEMMER = load_nltk_resources()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ---------- Load FAQs and build the TF-IDF index ----------
@st.cache_data
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def build_index(_faqs):
    # Combine each question with its optional keywords so the TF-IDF corpus
    # captures more of the ways a user might phrase the same thing.
    corpus_texts = []
    for item in _faqs:
        text = item["question"] + " " + " ".join(item.get("keywords", []))
        corpus_texts.append(text)
    processed = [preprocess(t) for t in corpus_texts]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed)
    return vectorizer, matrix


def get_answer(user_query, faqs, vectorizer, matrix, threshold: float = 0.2):
    processed_query = preprocess(user_query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return (
            "I'm not totally sure I understood that. Could you rephrase, "
            "or reach out to support directly?",
            best_score,
            None,
        )
    return faqs[best_idx]["answer"], best_score, faqs[best_idx]["question"]


# ---------- UI ----------
st.title("💬 FAQ Chatbot")
st.caption("Ask a question about HimDrive — bookings, payments, and more.")

faqs = load_faqs()
vectorizer, matrix = build_index(faqs)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the HimDrive support bot. Ask me anything below."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer, score, matched_q = get_answer(prompt, faqs, vectorizer, matrix)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if matched_q:
            st.caption(f"Matched: '{matched_q}'  ·  similarity {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot matches your question against a set of FAQs using "
        "TF-IDF + cosine similarity — no external API calls, so it deploys "
        "instantly on Streamlit Cloud."
    )
    st.write(f"**{len(faqs)}** FAQs loaded.")
    threshold_note = st.slider(
        "Match confidence threshold", 0.0, 1.0, 0.2, 0.05,
        help="Lower = more lenient matching, higher = stricter matching."
    )
    if threshold_note != 0.2:
        st.caption("Note: adjust the `threshold` default inside app.py to make this persist.")
    if st.button("Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()
