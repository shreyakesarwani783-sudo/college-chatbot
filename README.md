# 🎓 College FAQ Chatbot

A chatbot that answers common college questions (admission, fees, exams, hostel, placements, library, etc.) using a deep learning **sequence-to-sequence (Seq2Seq) LSTM** model. The app is built with **Streamlit**.

🔗 **Live demo:** https://college-faq-bot.streamlit.app

## How it works
1. **Tokenizer** converts the user's question into numbers (vocabulary of 146 words).
2. **Encoder** (Embedding + LSTM, 128 units) reads the question and compresses it into a context (hidden state).
3. **Decoder** (Embedding + LSTM + Dense softmax) starts with `startseq` and predicts the answer one word at a time until it predicts `endseq`.
4. **Streamlit** shows this as a simple chat window.

## Tech stack
- Python, TensorFlow / Keras (LSTM Encoder-Decoder)
- Streamlit (web app and deployment)
- NumPy

## Project structure
```
college_chatbot/
├── app.py              # Streamlit app + inference code
├── requirements.txt    # Dependencies
├── backup.py           # Makes a zip backup of the project
└── model/
    ├── encoder.keras
    ├── decoder.keras
    ├── tokenizer.pkl
    └── config.json
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model details
| Item | Value |
|---|---|
| Architecture | Seq2Seq LSTM (Encoder-Decoder) |
| Latent dimension | 128 |
| Embedding size | 64 |
| Vocabulary size | 146 |
| Max question length | 7 words |
| Max answer length | 19 words |
| Decoding | Greedy (argmax) |

## Limitations and future work
- Small vocabulary, so it works best on questions close to the training data.
- Future: more training data, attention mechanism, or a Transformer model.