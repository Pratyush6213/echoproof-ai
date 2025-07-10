
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load tokenizer and model from Hugging Face (no sentence-transformers used)
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_hf_model()

# Embed a sentence using BERT
def embed_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token as embedding
    return outputs.last_hidden_state[:, 0, :]

# Compute average similarity among sentences
def detect_echo_chamber_hf(messages, threshold=0.7):
    embeddings = [embed_sentence(msg) for msg in messages]
    embeddings = torch.stack(embeddings).squeeze(1)
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    n = len(messages)
    high_similarity = (sim_matrix > threshold).sum().item() - n  # exclude self-similarity
    score = high_similarity / (n * (n - 1))
    return score

# Streamlit UI
st.title("🗣️ Echo Chamber Detector (Alt Version)")

text_input = st.text_area("Paste group conversation (one message per line):", height=200)
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)

if st.button("Analyze"):
    if text_input.strip():
        messages = [msg.strip() for msg in text_input.strip().split("\n") if msg.strip()]
        score = detect_echo_chamber_hf(messages, threshold)
        if score > 0.5:
            st.error(f"🚨 Strong echo chamber detected! Score: {score:.2f}")
        elif score > 0.3:
            st.warning(f"⚠️ Moderate echo chamber detected. Score: {score:.2f}")
        else:
            st.success(f"✅ Balanced conversation. Score: {score:.2f}")
    else:
        st.warning("Please enter at least 2 sentences.")
