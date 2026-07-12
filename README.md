# FAQ Chatbot (CodeAlpha Task 2)

A lightweight FAQ chatbot using TF-IDF + cosine similarity to match user questions
to the closest FAQ, served through a Streamlit chat UI. No external APIs, no GPU,
no heavy models — deploys cleanly on Streamlit Community Cloud.

## Files

```
faq-chatbot/
├── app.py                  # Streamlit app
├── faqs.json                # FAQ dataset (swap in your own topic here)
├── requirements.txt         # Pinned dependencies
├── .streamlit/config.toml   # Theme config
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repo named `CodeAlpha_FAQChatbot` and push these files
   (folder structure must be flattened into the repo root — `app.py` and
   `faqs.json` should sit at the top level, not inside `faq-chatbot/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repo/branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically —
   no system packages (`packages.txt`) are needed for this project.
5. First load will download NLTK's `stopwords` corpus once (cached afterward via
   `st.cache_resource`); this happens automatically and needs no manual step.

## Customizing

- Swap `faqs.json` for your own question/answer pairs — no code changes needed.
- Adjust the match confidence `threshold` in `get_answer()` inside `app.py` if
  the bot is too strict or too lenient about what counts as a match.

## Why this task deploys easily

Unlike music generation (needs a trained model file + heavier ML libraries) or
object detection (needs OpenCV system libraries, webcam/video handling, and
CPU/GPU-specific code), this app only uses `scikit-learn` and `nltk` — both
pure-Python-friendly, small, and fast to cold-start on Streamlit Cloud's free tier.
