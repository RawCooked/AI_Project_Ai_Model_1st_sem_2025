# 🎙️ AI Voice Agent & Prescription Assistant (LangChain, functional)

A local, privacy-first voice assistant that listens, understands, calls tools via LangChain, and can generate prescription PDFs — all with a simple, functional (non-OOP) code style. Whisper handles speech-to-text; a configurable OpenAI-compatible endpoint powers reasoning.

—

## ✨ Highlights
- 🎧 Hands-free voice capture with Voice Activity Detection
- 🧠 On-device Whisper transcription
- 🧰 LangChain tools (search, Wikipedia, math, notes, save, prescriptions)
- 💊 Quick and full prescription generation to PDF
- 🔒 Local-first workflow; uses your own OpenAI-compatible backend
- 🗂️ Conversation history saved to JSON

—

## 📦 Tech Stack
- Python 3.10+
- openai-whisper (transcription)
- sounddevice (microphone I/O)
- LangChain (functional chain + tools)
- ChatOpenAI (OpenAI-compatible, custom base_url)
- reportlab (PDF)

—

## 📁 Project Structure
- README.md — this file
- Requirements.txt — Python dependencies
- voice_agent_system.py — main app (run this)
- env/ — optional local venv (if you created one)

—

## 🧾 Requirements
Create a virtual environment (recommended):

```
python -m venv env
./env/Scripts/activate  # Windows
# source env/bin/activate  # macOS/Linux
```

Install dependencies:
```
pip install -r Requirements.txt
```

—

## 🔐 Configure LLM Endpoint (.env)
Create a .env file in the project root:
```
ESPRIT_API_KEY=your_api_key_here
ESPRIT_BASE_URL=https://tokenfactory.esprit.tn/api/v1
LLM_MODEL=hosted_vllm/Llama-3.1-70B-Instruct
```
Notes:
- Any OpenAI-compatible endpoint can be used by changing ESPRIT_BASE_URL.
- Model name must match what your endpoint exposes.

—

## 🚀 Run
```
python voice_agent_system.py
```
Speak after “Listening…”. The app stops recording after short silence, transcribes with Whisper, and routes your text through a LangChain tool-aware agent.

—

## 🧪 Built-in Tools (ask naturally; the model may decide to use them)
- search_web — quick web snippets
- get_wikipedia_info — short summary + URL
- calculate — basic calculator
- get_current_datetime — date/time
- create_note — saves a note file
- save_to_file — writes any content to a .txt file
- create_prescription — PDF with patient/meds/notes
- quick_prescription — presets for common conditions

—

## ⚕️ Safety
- For medical intents, the agent should ask for a real patient name and remind users to consult a licensed professional. The PDF is for educational/informational purposes only.

—

## ⚙️ Config Knobs (via env or edit constants)
- SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RECORDING_TIME
- WHISPER_MODEL (tiny, base, …)

—

## 🧩 Troubleshooting
- Microphone access: ensure your terminal/app has mic permissions.
- Certificate issues: if your endpoint has strict TLS, ensure valid certs; otherwise use a trusted HTTPS endpoint.
- Whisper speed: pick a smaller WHISPER_MODEL for slower machines.

—

## 📜 License
Educational/demonstration use. See repository terms or contact the author for details.
