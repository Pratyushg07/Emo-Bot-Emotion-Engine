# app.py
import streamlit as st
from speech_io import record_audio, transcribe_audio
from nlp_pipeline import EmotionSentimentPipeline
from fsm_engine import EmotionFSM
from utils import format_prediction
import time
import os
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="EmoMind — FSM-Based Emotion Engine",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #fafafa;
    text-align: center;
}
.sidebar .sidebar-content {
    background: rgba(20, 20, 25, 0.8);
    border-right: 1px solid #2c2f36;
}
div[data-testid="stMetricValue"] {
    font-size: 2.5rem;
}
.result-box {
    background: rgba(40, 43, 52, 8);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid rgba(100, 255, 255, 0.2);
}
.mood-image {
    width: 150px;
    border-radius: 15px;
    margin: 10px auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🧠 EmoMind — FSM-Based Emotion Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;'>Speak into your mic and watch your AI's mood evolve through a Finite State Machine.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")
use_whisper = st.sidebar.checkbox("Prefer local Whisper (if installed)", value=True)
record_seconds = st.sidebar.slider("🎙️ Record seconds", 1, 8, 3)
auto_mode = st.sidebar.checkbox("Auto continuous mode (loop recording)", value=False)
model_name = st.sidebar.text_input("🤖 HF model (optional)", value="j-hartmann/emotion-english-distilroberta-base")

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_pipeline(model_name, prefer_whisper):
    with st.spinner("Loading Emotion Engine..."):
        pipe = EmotionSentimentPipeline(model_name=model_name)
        fsm = EmotionFSM()
    return pipe, fsm

pipe, fsm = load_pipeline(model_name, use_whisper)

# ---------- EMOTION IMAGE MAP ----------
IMAGE_MAP = {
    "Happy": "happy.jpeg",
    "Sad": "sad.jpeg",
    "Angry": "angry.jpeg",
    "Fearful": "fearfull.jpeg",
    "Surprised": "surprised.jpeg",
    "Curious": "curious.jpeg",
    "Neutral": "neutral.jpeg"
}

# ---------- TRACK PREVIOUS MOOD ----------
if "prev_mood" not in st.session_state:
    st.session_state["prev_mood"] = "Neutral"

# ---------- DISPLAY CURRENT + PREVIOUS MOOD ----------
mood_col, graph_col = st.columns([2, 3])

with mood_col:

    # Create placeholders for dynamic UI updates
    prev_mood_text_ph = st.empty()
    prev_img_ph = st.empty()
    current_mood_text_ph = st.empty()
    current_img_ph = st.empty()

    prev_mood = st.session_state.get("prev_mood", "Neutral")
    current_mood = fsm.state

    # Initial rendering
    prev_mood_text_ph.markdown(
        f"**Previous Mood:** {prev_mood}"
    )
    prev_img_ph.image(IMAGE_MAP.get(prev_mood), width=150)

    current_mood_text_ph.markdown(
        f"**Current Mood:** {current_mood}"
    )
    current_img_ph.image(IMAGE_MAP.get(current_mood), width=150)

with graph_col:
    graph_placeholder = st.empty()
    graph_placeholder.graphviz_chart(fsm.get_graphviz_source())

# ---------- INTERACTION BELOW GRAPH ----------
st.markdown("### 🎧 Interaction")
record_button = st.button("🎙️ Record once", use_container_width=True)

if auto_mode:
    st.info("Auto mode ON — recording repeatedly until stopped.")
    stop_auto = st.button("🛑 Stop auto mode")
else:
    stop_auto = False

# ---------- FUNCTION: AUDIO HANDLING ----------
def handle_audio_cycle():
    with st.spinner("🎤 Recording..."):
        audio_path = record_audio(duration=record_seconds)
    st.success("✅ Recorded!")

    with st.spinner("🧩 Transcribing..."):
        transcription = transcribe_audio(audio_path, prefer_whisper=use_whisper)

    st.markdown(f"**🗣️ Transcription:** _{transcription or '(no transcription)'}_")

    with st.spinner("🧠 Analyzing emotion + sentiment..."):
        emo_pred, sent_pred = pipe.predict(transcription)

    # Save previous mood before updating
    previous = fsm.state
    st.session_state["prev_mood"] = previous

    # FSM Update
    new_state = fsm.update_from_nlp(emo_pred, sent_pred)

    # Display results
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.write("**Emotion predictions:**", format_prediction(emo_pred))
    st.write("**Sentiment:**", format_prediction(sent_pred))
    st.markdown("</div>", unsafe_allow_html=True)

    # Update placeholders instead of creating new elements
    prev_mood_text_ph.markdown(f"**Previous Mood:** {st.session_state['prev_mood']}")
    prev_img_ph.image(IMAGE_MAP.get(st.session_state["prev_mood"]), width=150)

    current_mood_text_ph.markdown(f"**Current Mood:** {fsm.state}")
    current_img_ph.image(IMAGE_MAP.get(fsm.state), width=150)

    # Update FSM graph
    graph_placeholder.graphviz_chart(fsm.get_graphviz_source())

# ---------- EVENT HANDLERS ----------
if record_button:
    handle_audio_cycle()

if auto_mode:
    st.session_state.setdefault("auto_running", True)
    st.session_state["auto_running"] = True
if stop_auto:
    st.session_state["auto_running"] = False

if st.session_state.get("auto_running", False):
    try:
        while st.session_state.get("auto_running", True):
            handle_audio_cycle()
            time.sleep(0.5)
            st.experimental_rerun()
    except st.errors.ScriptRunnerStoppedException:
        pass

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#777;font-size:0.9em;'>
💡 Tip: If Whisper or Gemini fails, EmoMind auto-switches to VADER for robust fallback.<br>
Built with ❤️ using Streamlit + Transformers + FSM logic.
</p>
""", unsafe_allow_html=True)
