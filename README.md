# 🧠 Voice AI Teacher

A **voice-based AI assistant** built with **LangChain**, **Ollama**, and **Whisper**.  
You can **speak to it**, and it will **listen, understand, answer, and speak back** — all locally and for free.

---

## ⚙️ Features
- 🎙️ **Voice input** with Whisper (speech-to-text)  
- 🧠 **Local reasoning** using Ollama models (e.g. Llama 3, Mistral, Phi)  
- 🔊 **Speech output** using gTTS (text-to-speech)  
- 📊 **Visual explanations** with Matplotlib  
- 💬 **Offline and private**, no API keys needed  

---

## 🧾 Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
langchain
langchain-community
langchain-core
ollama
openai-whisper
gtts
sounddevice
numpy
playsound
matplotlib
```

Then pull a model for Ollama:
```bash
ollama pull llama3.2:3b
```

---

## 🚀 Run

Start Ollama:
```bash
ollama serve
```

Then:
```bash
python main.py
```

---

## 🪄 Notes
- Works fully **offline**  
- No **API keys** or **internet** required  
- You can easily expand it into a real **teaching or study assistant**
