import numpy as np
import whisper
import json
import os
from scipy.io.wavfile import write
import sounddevice as sd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# =========================
# 0️⃣ CHECK AUDIO DEVICES
# =========================
def list_audio_devices():
    print("📱 Available audio devices:")
    print(sd.query_devices())
    print()

# =========================
# 1️⃣ RECORD AUDIO
# =========================
def record_audio(filename="audio_input.wav", duration=6, samplerate=16000):
    try:
        # List devices first
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        print(f"🎤 Using input device: {devices[default_input]['name']}")
        print("🎙️ Speak now...")
        
        recording = sd.rec(
            int(duration * samplerate), 
            samplerate=samplerate, 
            channels=1, 
            dtype='float32',
            device=default_input
        )
        sd.wait()
        
        audio_data = np.int16(recording * 32767)
        write(filename, samplerate, audio_data)
        print("✅ Recording saved:", filename)
        return filename
        
    except Exception as e:
        print(f"❌ Audio recording error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure your microphone is connected")
        print("2. Check Windows Sound Settings > Input")
        print("3. Try running: sd.query_devices() to see available devices")
        raise

# =========================
# 2️⃣ SPEECH → TEXT (Whisper)
# =========================
def transcribe_audio(filename):
    print("🧠 Transcribing...")
    model = whisper.load_model("tiny")  # "base" for better accuracy
    result = model.transcribe(filename)
    text = result["text"]
    print(f"🗣️ You said: {text}")
    return text

# =========================
# 3️⃣ INITIALIZE LLM (Ollama)
# =========================
llm = OllamaLLM(model="llama3.2:3b")  # Updated to new import

prompt = PromptTemplate.from_template("""
You are a patient and kind virtual teacher.
Answer the user's question in a clear and structured way.
If relevant, include short code examples or explanations.
Keep it concise.

Return JSON only:
{{
"speech": "what you would say out loud",
"graph_code": "optional python code (matplotlib), leave empty if none"
}}

User question: {question}
""")

# =========================
# 4️⃣ PROCESS QUESTION
# =========================
def get_ai_response(question):
    print("🤖 Thinking...")
    formatted_prompt = prompt.format(question=question)
    response = llm.invoke(formatted_prompt)
    print("🧩 Raw response:", response)
    try:
        parsed = json.loads(response)
    except:
        parsed = {"speech": response, "graph_code": ""}
    return parsed

# =========================
# 5️⃣ EXECUTE GRAPH CODE (optional)
# =========================
def execute_graph_code(code):
    if code and len(code.strip()) > 0:
        print("📊 Executing graph code...")
        try:
            exec(code)
        except Exception as e:
            print(f"❌ Error executing graph code: {e}")

# =========================
# 6️⃣ MAIN LOOP
# =========================
if __name__ == "__main__":
    # First, list available audio devices
    list_audio_devices()
    
    try:
        filename = record_audio(duration=7)
        question = transcribe_audio(filename)
        ai_reply = get_ai_response(question)

        print(f"\n🧠 AI says: {ai_reply['speech']}\n")
        execute_graph_code(ai_reply.get("graph_code", ""))
    
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")