# 💬 FAQ Chatbot — CodeAlpha Task 2

A lightweight FAQ chatbot that matches user questions to the closest FAQ using TF-IDF and cosine similarity, served through a Streamlit chat interface. No external APIs, no GPU, no heavy models — just fast, deterministic text matching.

**🔗 Live App:** [codealphafaqchatbot-mzbjqfpwv9bg3jc7nw4eat.streamlit.app](https://codealphafaqchatbot-mzbjqfpwv9bg3jc7nw4eat.streamlit.app/)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core implementation |
| UI / App framework | [Streamlit](https://streamlit.io/) | Chat interface, session state, hosting on Streamlit Community Cloud |
| NLP preprocessing | [NLTK](https://www.nltk.org/) | Stopword removal, `SnowballStemmer` for word stemming |
| Matching engine | [scikit-learn](https://scikit-learn.org/) | `TfidfVectorizer` + `cosine_similarity` for intent matching |
| Numerical ops | [NumPy](https://numpy.org/) | Vector similarity computations |
| Data storage | JSON (`faqs.json`) | FAQ question/answer/keyword pairs — no database required |
| Hosting | [Streamlit Community Cloud](https://share.streamlit.io/) | Free, GitHub-connected deployment |

**How matching works:** each FAQ is enriched with related keywords, combined into a corpus, and vectorized with TF-IDF. A user's question is preprocessed the same way (lowercased, punctuation stripped, stopwords removed, stemmed) and compared against the corpus using cosine similarity. The highest-scoring FAQ above a confidence threshold is returned as the answer.

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit app (UI + matching logic)
├── faqs.json                # FAQ dataset — question, answer, keywords
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Theme configuration
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/swasxtik/codealpha_faq_chatbot.git
cd codealpha_faq_chatbot
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done — see live link above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repo/branch, and set **Main file path** to `app.py`.
4. Click **Deploy** — `requirements.txt` is installed automatically, no `packages.txt` needed.

---

## ✨ Customizing

- Swap `faqs.json` for your own question/answer/keyword sets — no code changes needed.
- Tune the match confidence `threshold` inside `get_answer()` in `app.py` if the bot is too strict or too lenient.

---

## 📌 About

Built as part of the [CodeAlpha](https://www.codealpha.tech) Artificial Intelligence internship — Task 2: Chatbot for FAQs.
