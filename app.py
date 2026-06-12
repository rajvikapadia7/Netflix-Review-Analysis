import gradio as gr
import pickle
import pandas as pd
import numpy as np
import re
import os
from groq import Groq
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",  quiet=True)

# Load model
with open("sentiment_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model  = bundle["model"]
tfidf  = bundle["tfidf"]
scaler = bundle["scaler"]
df     = pd.read_csv("netflix_reviews_clean.csv")

# Text cleaning
stop_words = set(stopwords.words("english"))
custom_stopwords = [
    "netflix", "app", "show", "watch", "movie", "series", "use", "used",
    "really", "already", "still", "get", "one", "also", "would", "ill",
    "ive", "im", "thats", "us"
]
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Groq summariser
def groq_summarise(reviews_text: str, sentiment_label: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ GROQ_API_KEY secret not set. Go to HF Space Settings → Repository secrets and add it."
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are a business analyst reviewing customer feedback for Netflix.

Here are {sentiment_label.lower()} from the Netflix Google Play Store:

{reviews_text}

Provide a structured analysis with:
1. **Top 5 Themes** — the most common topics mentioned (1-2 sentences each)
2. **Most Urgent Issues** — what needs immediate attention (if negative) or what to keep doing (if positive)
3. **One Key Business Recommendation** — a single actionable insight

Be concise and business-focused."""

        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return message.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"

# Predict function
def predict_sentiment(review_text):
    if not review_text.strip():
        return "⚠️ Please enter a review.", ""

    cleaned       = clean_text(review_text)
    tfidf_vec     = tfidf.transform([cleaned])
    length_scaled = scaler.transform([[len(review_text.split())]])
    X_input       = hstack([tfidf_vec, csr_matrix(length_scaled)])

    prediction = model.predict(X_input)[0]
    decision   = model.decision_function(X_input)[0]
    confidence = 1 / (1 + np.exp(-abs(decision)))

    label  = "✅ Positive Review" if prediction == 1 else "❌ Negative Review"
    result = f"{label}  —  Confidence: {confidence:.1%}"
    cleaned_display = f"Cleaned text seen by model: {cleaned}"

    return result, cleaned_display

# Summarise function
def summarise_reviews(sentiment_choice, n_reviews):
    label  = 0 if sentiment_choice == "Negative Reviews" else 1
    pool   = df[df["is_positive_review"] == label]["review"].dropna()
    sample = pool.sample(min(int(n_reviews), len(pool)), random_state=42).tolist()

    reviews_text = "\n".join([f"- {r}" for r in sample])
    return groq_summarise(reviews_text, sentiment_choice)

# Gradio UI
with gr.Blocks(title="Netflix Review Analyser") as demo:

    gr.Markdown(
        """
        # 🎬 Netflix Review Analyser
        Sentiment classifier trained on **142K Google Play Store reviews**.
        Built with Logistic Regression + TF-IDF · F1 Macro: 0.84
        """
    )

    with gr.Tab("🔍 Sentiment Predictor"):
        gr.Markdown("Enter any Netflix review and the model will classify it as positive or negative.")

        with gr.Row():
            review_input = gr.Textbox(
                label="Enter a review",
                placeholder="e.g. The app keeps crashing and I lost my watchlist...",
                lines=4
            )

        predict_btn    = gr.Button("Predict Sentiment", variant="primary")
        result_output  = gr.Textbox(label="Prediction", interactive=False)
        cleaned_output = gr.Textbox(label="Cleaned text seen by model", interactive=False)

        predict_btn.click(
            fn=predict_sentiment,
            inputs=review_input,
            outputs=[result_output, cleaned_output]
        )

        gr.Markdown("**Try these examples:**")
        gr.Examples(
            examples=[
                ["Absolutely love the content, so many great shows and movies. Worth every penny!"],
                ["App crashes every time I open it. Terrible experience, cancelling my subscription."],
                ["Good selection but the interface is slow and clunky on my phone."],
            ],
            inputs=review_input
        )

    with gr.Tab("🤖 AI Review Summarizer"):
        gr.Markdown(
            "Select a sentiment and number of reviews — AI will summarize the key themes.  \n"
            "Powered by **Groq · Llama 3.3 70B**."
        )

        with gr.Row():
            sentiment_dropdown = gr.Dropdown(
                choices=["Negative Reviews", "Positive Reviews"],
                value="Negative Reviews",
                label="Which reviews to summarize?"
            )
            n_slider = gr.Slider(
                minimum=10, maximum=50, value=20, step=5,
                label="Number of reviews to sample"
            )

        summarise_btn  = gr.Button("Generate AI Summary", variant="primary")
        summary_output = gr.Markdown(label="AI Summary")

        summarise_btn.click(
            fn=summarise_reviews,
            inputs=[sentiment_dropdown, n_slider],
            outputs=summary_output
        )

    gr.Markdown(
        f"Dataset: {len(df):,} Netflix reviews · "
        "Model: Logistic Regression · "
        "F1 Macro: 0.84"
    )

demo.launch()
