import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓")


@st.cache_resource
def load_everything():
    """Load models once; Streamlit reuses them on every rerun."""
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    encoder = load_model(MODEL_DIR / "encoder.keras", compile=False)
    decoder = load_model(MODEL_DIR / "decoder.keras", compile=False)
    return config, tokenizer, encoder, decoder


config, tokenizer, encoder, decoder = load_everything()
MAX_Q = config["max_question_len"]
MAX_A = config["max_answer_len"]
START_ID = tokenizer.word_index["startseq"]
END_ID = tokenizer.word_index["endseq"]
OOV_ID = tokenizer.word_index.get("<OOV>")


def get_reply(question: str) -> str:
    # 1. Turn the question into padded token ids
    seq = tokenizer.texts_to_sequences([question])
    seq = pad_sequences(seq, maxlen=MAX_Q, padding="post", truncating="post")

    # 2. Encoder gives the initial states for the decoder
    h, c = encoder.predict(seq, verbose=0)

    # 3. Decoder generates one word at a time (greedy decoding)
    token = np.array([[START_ID]])
    words = []
    for _ in range(MAX_A):
        probs, h, c = decoder.predict([token, h, c], verbose=0)
        next_id = int(np.argmax(probs[0, -1, :]))
        if next_id == END_ID:
            break
        if next_id != OOV_ID:
            words.append(tokenizer.index_word.get(next_id, ""))
        token = np.array([[next_id]])

    reply = " ".join(w for w in words if w).strip()
    return reply.capitalize() if reply else "Sorry, I couldn't understand that."


# ---------------- UI ----------------
st.title("🎓 College FAQ Chatbot")
st.caption("Ask about admission, fees, exams, hostel, placements, library...")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        reply = get_reply(prompt)
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
