
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import streamlit as st
import torchaudio
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_bias_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_counter_model():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_length=100)

model = load_sentence_model()
bias_model = load_bias_model()
counter_model = load_counter_model()

# Echo Chamber detection logic
def detect_echo_chamber(messages, threshold=0.7):
    embeddings = model.encode(messages)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    high_similarity = (sim_matrix > threshold).sum().item() - len(messages)
    score = high_similarity / (len(messages) * (len(messages) - 1))
    return score

# Transcribe audio to text
def transcribe_audio(audio_path):
    try:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        with torch.inference_mode():
            emissions, _ = model(waveform)
        decoder = torchaudio.transforms.GreedyCTCDecoder(blank=0)
        transcript = decoder(emissions)
        return transcript
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="EchoProof AI", layout="wide")
st.title("🔍 EchoProof AI - Unified Bias & Echo Detector")

tabs = st.tabs([
    "🗣️ Echo Chamber",
    "📰 News Bias",
    "🧠 Counter Opinion",
    "🔉 Voice Input",
    "📊 Visualization (Demo)"
])

with tabs[0]:
    st.header("Echo Chamber Detection")
    conversation = st.text_area("Paste group conversation (one line per message):", height=200)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
    if st.button("Analyze Echo Chamber"):
        if conversation.strip():
            messages = conversation.strip().split("\n")
            score = detect_echo_chamber(messages, threshold)
            if score > 0.5:
                st.error(f"🚨 Strong echo chamber detected! Score: {score:.2f}")
            elif score > 0.3:
                st.warning(f"⚠️ Moderate echo chamber detected. Score: {score:.2f}")
            else:
                st.success(f"✅ Healthy conversation. Score: {score:.2f}")
        else:
            st.warning("Please paste some text.")

with tabs[1]:
    st.header("News Bias Detection")
    article = st.text_area("Paste a news headline or paragraph:", height=150)
    if st.button("Check Bias"):
        if article:
            result = bias_model(article)
            label = result[0]['label']
            score = result[0]['score']
            bias = "🔴 Left-Leaning" if label == "NEGATIVE" else "🟢 Right-Leaning" if label == "POSITIVE" else "⚪ Neutral"
            st.markdown(f"**Bias Detected:** {bias}")
            st.markdown(f"**Confidence:** {score:.2f}")

with tabs[2]:
    st.header("AI Counter Opinion Generator")
    user_text = st.text_area("Enter a statement to get an opposing view:")
    if st.button("Generate Counter Opinion"):
        if user_text.strip():
            with st.spinner("Thinking..."):
                response = counter_model(f"Provide a respectful counter opinion to: {user_text}")
                st.success(response[0]['generated_text'])
        else:
            st.warning("Please enter a statement.")

with tabs[3]:
    st.header("Transcribe Voice and Analyze")
    audio_file = st.file_uploader("Upload an audio file (.wav format preferred)", type=["wav", "mp3"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_file.read())
            transcript = transcribe_audio(temp.name)
            st.markdown("**Transcribed Text:**")
            st.code(transcript)
            os.remove(temp.name)

with tabs[4]:
    st.header("Visualization (Demo)")
    st.markdown("This section will show bias or echo trends over time. Demo graph below.")
    scores = [0.1, 0.3, 0.5, 0.6, 0.45, 0.2, 0.15]
    st.line_chart(scores)
