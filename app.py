# app.py
import streamlit as st
from speech_io import record_audio, transcribe_audio
from nlp_pipeline import EmotionSentimentPipeline
from fsm_engine import EmotionFSM
from utils import format_prediction
import time


st.set_page_config(page_title="EmoMind - FSM Emotion Engine", layout="centered")

st.title("EmoMind — FSM-Based Emotion Engine")
st.write("Speak into your microphone and watch the bot's mood update via an FSM. Uses transformer-based emotion+sentiment (with fallback).")

# Sidebar controls
st.sidebar.header("Controls")
use_whisper = st.sidebar.checkbox("Prefer local Whisper (if installed)", value=True)
record_seconds = st.sidebar.slider("Record seconds", 1, 8, 3)
auto_mode = st.sidebar.checkbox("Auto continuous mode (records repeatedly)", value=False)
model_name = st.sidebar.text_input("HF model (optional)", value="j-hartmann/emotion-english-distilroberta-base")

# Load NLP pipeline (cached)
@st.cache_resource
def load_pipeline(model_name, prefer_whisper):
    st.sidebar.text("Loading models... this may take a bit.")
    pipe = EmotionSentimentPipeline(model_name=model_name)
    fsm = EmotionFSM()
    return pipe, fsm

pipe, fsm = load_pipeline(model_name, use_whisper)

# Visualization area
st.subheader("Bot Mood")
col1, col2 = st.columns([2,3])
with col1:
    mood_placeholder = st.empty()
    st.write("Current FSM State:")
    state_text = st.empty()
with col2:
    graph_placeholder = st.empty()

# Controls
record_button = st.button("Record once")
if auto_mode:
    st.info("Auto mode ON — app will record repeatedly until you stop it.")
    stop_auto = st.button("Stop auto mode")
else:
    stop_auto = False

# show initial FSM graph
graph_placeholder.graphviz_chart(fsm.get_graphviz_source())
state_text.markdown(f"**{fsm.state}**")

# Main loop / single record
def handle_audio_cycle():
    with st.spinner("Recording..."):
        audio_path = record_audio(duration=record_seconds)
    st.success("Recorded.")
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_path, prefer_whisper=use_whisper)
    st.write("**Transcription:**", transcription or "_(no transcription)_")

    # NLP inference
    with st.spinner("Running emotion & sentiment analysis..."):
        emo_pred, sent_pred = pipe.predict(transcription)
    st.write("**Emotion predictions:**", format_prediction(emo_pred))
    st.write("**Sentiment:**", format_prediction(sent_pred))

    # decide incoming scoring for FSM transitions: combine emotion and sentiment
    # we'll use highest-scoring emotion label and sentiment label to decide state transitions
    new_state = fsm.update_from_nlp(emo_pred, sent_pred)
    state_text.markdown(f"**{fsm.state}**")
    mood_placeholder.metric(label="Current Mood", value=fsm.state)
    graph_placeholder.graphviz_chart(fsm.get_graphviz_source())

# event handlers
if record_button:
    handle_audio_cycle()

if auto_mode:
    st.session_state.setdefault("auto_running", True)
    st.session_state["auto_running"] = True

if stop_auto:
    st.session_state["auto_running"] = False

# auto loop (non-blocking-ish)
if st.session_state.get("auto_running", False):
    try:
        while st.session_state.get("auto_running", True):
            handle_audio_cycle()
            # small delay
            time.sleep(0.5)
            # allow Streamlit to rerun/check for user interactions
            st.experimental_rerun()
    except st.errors.ScriptRunnerStoppedException:
        pass

st.markdown("---")
st.markdown("**Notes & Troubleshooting**")
st.markdown("""
- If Whisper isn't installed or fails, the app will fallback to Google SpeechRecognition (internet needed).
- Transformer models are downloaded from HuggingFace; first run will be slow (model download).
- For faster runs, consider using a lighter model or VADER fallback (already included).
""")
