## AI-Powered Mental Health Chatbot

An advanced AI-based chatbot designed to offer emotional support and simulate human-like conversations using **NLP**, **Voice Interaction**, and **Sentiment Analysis**. Built using **DialoGPT**, **Streamlit**, and **Google APIs**.

---

## Features

- 💬 Conversational AI with **DialoGPT**
- 🎯 Emotion-aware replies using predefined & semantic responses
- 📈 Sentiment detection using HuggingFace pipeline
- 🧠 Smart Response Layer via **Sentence Embeddings** + **Cosine Similarity**
- 🎙️ Voice input support (`.wav` only)
- 🔊 Text-to-Speech with `gTTS`
- 🖼️ imported  `DeepFace` to extend the feature of Chatbot in future to enable the facial input through webcam but not used here

---

##  Tech Stack

| Feature             | Library / Model                     |
|---------------------|--------------------------------------|
| Language Model      | `microsoft/DialoGPT-medium` (Transformers) |
| Sentiment Analysis  | `pipeline("sentiment-analysis")`     |
| Semantic Match      | `sentence-transformers` + `sklearn`  |
| Voice Input         | `speech_recognition` (Google API)    |
| Voice Output        | `gTTS`                                |
| Frontend Interface  | `Streamlit`                          |
| Deep Learning Core  | `PyTorch`                            |

---

##  Setup Instructions

```bash
git clone https://github.com/your-username/medical_chatbot.git
cd medical_chatbot
python -m venv chatbot-env
chatbot-env\Scripts\activate         # Windows
pip install -r requirements.txt
streamlit run chatbot1.py
