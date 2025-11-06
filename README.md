# 🎙️ AI Voice Agent & Prescription Assistant

A local, privacy-first voice assistant that listens, understands, uses tools (search, Wikipedia, math, notes, files), and can generate modern prescription PDFs. Built around Whisper for speech-to-text and a configurable LLM backend.

—

## ✨ Highlights
- 🎧 Voice capture with automatic Voice Activity Detection (hands-free)
- 🧠 On-device transcription via Whisper
- 🤝 Tool-aware agent: web search, Wikipedia, math, notes, file save
- 💊 Medical helpers: quick prescriptions for common conditions + full PDF generator
- 🔒 Local-first workflow; configurable LLM endpoint (no public keys required)
- 🗂️ Conversation logging to JSON with tool call history

—

## 📦 Tech Stack
- Python 3.10+
- Whisper (openai-whisper)
- SoundDevice (mic input)
- ReportLab (PDF generation)
- OpenAI-compatible Chat Completions client (points to your school/hosted LLM)
- httpx (client)

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

If you plan to use the PDF prescription feature, ReportLab will be installed from Requirements.txt. Microphone support uses sounddevice (no extra driver on most systems).

—

## 🔐 Configure LLM Endpoint (.env)
This project talks to a hosted, OpenAI-compatible LLM (school/organization endpoint). Create a .env file in the project root:

```
ESPRIT_API_KEY=your_api_key_here
ESPRIT_BASE_URL=https://tokenfactory.esprit.tn/api/v1
LLM_MODEL=hosted_vllm/Llama-3.1-70B-Instruct
```

Notes:
- ESPRIT_BASE_URL can be changed to any OpenAI-compatible endpoint.
- LLM_MODEL is the model name exposed by your endpoint.

—

## 🚀 Run
Start the application:
```
python voice_agent_system.py
```
Then speak after the “Listening…” prompt. The app auto-stops when it detects a short silence and processes your request.

—

## 🗣️ Example Voice Prompts
- “My name is Sarah and I have a headache”
- “What time is it?”
- “Search for the latest AI news”
- “Create a prescription for John with cold”

—

## 🧪 Tools You Can Ask For
- search_web — quick web snippets
- get_wikipedia_info — short summary + page URL
- calculate — basic calculator
- get_current_datetime — date/time
- create_note — saves a note file
- save_to_file — writes any content to a .txt file
- create_prescription — full PDF with patient/meds/notes
- quick_prescription — fast preset for common conditions

—

## ⚕️ Prescription Safety
- The agent asks for the patient name before generating medical outputs.
- Output is for informational/educational purposes only. Always consult a licensed physician.

—

## ⚙️ Config Knobs (voice and whisper)
You can adjust these in Config inside voice_agent_system.py:
- SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION
- MAX_RECORDING_TIME
- WHISPER_MODEL (e.g., "tiny", "base", …)

—

## 🧩 Troubleshooting
- No mic detected: ensure system microphone permissions are granted for your terminal.
- SSL issues when calling your endpoint: the client is configured to skip certificate verification; consider providing a valid cert in production.
- Whisper model too slow on your machine: switch to a smaller WHISPER_MODEL (e.g., tiny, base).

—

## 📜 License
Educational/demonstration use. See repository terms or contact the author for details.
