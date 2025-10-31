# 🎓 Virtual Teacher (Free AI Voice Assistant)

A **local, offline AI teacher** built with **LangChain**, **Ollama**, and **Whisper**.  
You can **talk to it with your voice**, and it will **think, answer, speak back, and even draw graphs** when needed — all for free.

---

## ⚙️ Features
- 🎙️ Voice-to-text with **Whisper** (OpenAI’s free STT model)  
- 🧠 Local reasoning with **Ollama** (e.g. Mistral, Phi, Qwen)  
- 🔊 Speech output with **gTTS**  
- 📊 Auto-generated graphs using **matplotlib**  
- 💬 Built using **LangChain** for structured AI workflow  

---

## 🧾 Requirements

### 1. Install dependencies
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

### 2. Install Ollama & pull a model
```bash
ollama pull mistral
```

### 3. (Optional) Faster Whisper
```bash
pip install faster-whisper
```

---

## 🚀 Run the App

Start Ollama:
```bash
ollama serve
```

Then run the virtual teacher:
```bash
python main.py
```

---

## 🧠 Example

You say:  
> “Explain me how neural networks learn.”

The AI replies with speech and a graph showing error reduction during learning.

---

## 🪄 Notes
- Works fully offline  
- No API keys or paid services  
- Can be extended with **LangGraph**, **memory**, or **multi-turn conversation**
