"""
CrisisLens — Disaster Tweet Classifier
Streamlit frontend for the NLP disaster-tweet classification model.
"""
import os
import re
import json
import joblib
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CrisisLens — Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered",
)

# ------------------------------------------------------------------
# NLTK setup (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_nltk():
    for pkg in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ------------------------------------------------------------------
# Load model + vectorizer (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    try:
        with open("model/metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None
    return model, vectorizer, metrics

model, vectorizer, metrics = load_artifacts()

def predict(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    return int(pred), float(proba), cleaned

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🚨 CrisisLens")
    st.caption("NLP-powered disaster tweet classifier")
    st.markdown("---")
    if metrics:
        best = metrics.get("best_model", "N/A")
        st.markdown(f"**Model in use:** {best}")
        m = metrics["results"][best]
        st.metric("F1-score", f"{m['f1']:.3f}")
        st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        st.metric("Recall", f"{m['recall']:.3f}")
        st.metric("Precision", f"{m['precision']:.3f}")
    st.markdown("---")
    st.markdown(
        "Built with **TF-IDF + Logistic Regression** on a labeled "
        "tweet dataset. Enter a tweet or upload a CSV to classify."
    )

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
st.title("🚨 CrisisLens: Disaster Tweet Classifier")
st.write(
    "Classify whether a tweet describes a **real disaster** or is **unrelated**, "
    "using a trained NLP + Machine Learning pipeline."
)

tab1, tab2, tab3 = st.tabs(["✍️ Single Tweet", "📄 Batch (CSV) Prediction", "📊 Insights"])

# ---------------- Tab 1: Single tweet ----------------
with tab1:
    st.subheader("Classify a single tweet")

    example_tweets = [
        "Type your own tweet...",
        "Massive earthquake hits the coast, thousands evacuated from their homes",
        "I'm having a great time at the beach with my friends today!",
        "Wildfire spreading rapidly near the residential area, evacuation ordered",
        "This new song is absolutely fire, on repeat all day",
    ]
    choice = st.selectbox("Try an example or write your own below:", example_tweets)
    default_text = "" if choice == example_tweets[0] else choice

    user_text = st.text_area("Tweet text", value=default_text, height=100,
                              placeholder="e.g. Forest fire spreading near downtown, residents told to evacuate immediately")

    if st.button("🔍 Classify Tweet", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            pred, proba, cleaned = predict(user_text)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.error("🚨 **DISASTER TWEET**")
                else:
                    st.success("✅ **NOT A DISASTER**")
            with col2:
                st.metric("Disaster probability", f"{proba*100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))
            with st.expander("See cleaned/preprocessed text"):
                st.code(cleaned if cleaned else "(empty after cleaning)")

# ---------------- Tab 2: Batch prediction ----------------
with tab2:
    st.subheader("Classify a CSV of tweets")
    st.caption("Upload a CSV with a column named **text** (or **tweet**).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            batch_df = None

        if batch_df is not None:
            text_col = None
            for candidate in ["text", "tweet", "Text", "Tweet"]:
                if candidate in batch_df.columns:
                    text_col = candidate
                    break

            if text_col is None:
                st.error("No 'text' or 'tweet' column found in the uploaded CSV.")
            else:
                with st.spinner("Classifying..."):
                    cleaned_texts = batch_df[text_col].astype(str).apply(clean_text)
                    vecs = vectorizer.transform(cleaned_texts)
                    preds = model.predict(vecs)
                    probas = model.predict_proba(vecs)[:, 1]
                    batch_df["prediction"] = ["Disaster" if p == 1 else "Not Disaster" for p in preds]
                    batch_df["disaster_probability"] = probas.round(3)

                st.success(f"Classified {len(batch_df)} tweets.")
                st.dataframe(batch_df, use_container_width=True)

                n_disaster = int((preds == 1).sum())
                col1, col2 = st.columns(2)
                col1.metric("Predicted disasters", n_disaster)
                col2.metric("Predicted non-disasters", len(batch_df) - n_disaster)

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv_out,
                    file_name="classified_tweets.csv",
                    mime="text/csv",
                )

# ---------------- Tab 3: Insights (graphs from the notebook) ----------------
with tab3:
    st.subheader("Dataset & Model Insights")
    st.caption("These graphs are generated in the training notebook and saved to the `assets/` folder.")

    assets_dir = "assets"
    if not os.path.isdir(assets_dir):
        st.warning(
            "No `assets/` folder found next to app.py. Run the notebook "
            "`CrisisLens_Disaster_Tweet_Classification.ipynb` once to generate the graphs."
        )
    else:
        gallery = [
            ("class_distribution.png", "Class Distribution", "Disaster vs. Not-Disaster tweet counts."),
            ("top_keywords.png", "Top Keywords in Disaster Tweets", "Most frequent keywords among real disaster tweets."),
            ("text_length_dist.png", "Tweet Length Distribution", "Character-length distribution by class."),
            ("wordcloud_disaster.png", "Word Cloud — Disaster Tweets", "Most common cleaned words in disaster tweets."),
            ("wordcloud_nondisaster.png", "Word Cloud — Non-Disaster Tweets", "Most common cleaned words in non-disaster tweets."),
            ("model_comparison.png", "Model Comparison", "Accuracy / Precision / Recall / F1 across all 4 trained models."),
            ("roc_curve.png", "ROC Curves", "ROC curve and AUC for every trained model."),
            ("confusion_matrix.png", "Confusion Matrix", "Confusion matrix for the final selected model."),
        ]
        for filename, title, caption in gallery:
            path = os.path.join(assets_dir, filename)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, caption=caption, use_container_width=True)
                st.markdown("---")

        if metrics:
            st.markdown("**Full metrics (all models)**")
            st.dataframe(pd.DataFrame(metrics["results"]).T.round(4), use_container_width=True)

st.markdown("---")
st.caption("CrisisLens · NLP Disaster Tweet Classification Project")
