import numpy as np
import whisper
import json
import os
from scipy.io.wavfile import write
import sounddevice as sd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from datetime import datetime
import threading
import queue

# =========================
# CONFIGURATION
# =========================
class Config:
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01  # Adjust based on your mic sensitivity
    SILENCE_DURATION = 2.0    # Seconds of silence to stop recording
    MAX_RECORDING_TIME = 30   # Maximum recording duration
    WHISPER_MODEL = "tiny"    # "base" for better accuracy, "tiny" for speed
    LLM_MODEL = "llama3.2:3b"

# =========================
# CONVERSATION CONTEXT MANAGER
# =========================
class ConversationManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_message(self, role, content):
        """Add a message to conversation history"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.history = self.history[-self.max_history * 2:]
    
    def get_context(self):
        """Format conversation history for the prompt"""
        if not self.history:
            return "This is the start of the conversation."
        
        context = "Previous conversation:\n"
        for msg in self.history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context
    
    def save_session(self):
        """Save conversation to file"""
        filename = f"conversation_{self.session_id}.json"
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 Conversation saved to {filename}")

# =========================
# VOICE ACTIVITY DETECTION
# =========================
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, silence_threshold=0.01, silence_duration=2.0):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block during recording"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def calculate_energy(self, audio_chunk):
        """Calculate audio energy (volume)"""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def record_until_silence(self, max_duration=30):
        """Record audio until silence is detected"""
        print("🎤 Listening... (speak now)")
        
        recorded_chunks = []
        silence_chunks = 0
        silence_threshold_chunks = int(self.silence_duration * self.sample_rate / 1024)
        
        with sd.InputStream(samplerate=self.sample_rate, 
                          channels=1, 
                          callback=self.audio_callback,
                          blocksize=1024):
            
            start_time = datetime.now()
            speech_detected = False
            
            while True:
                # Check max duration
                if (datetime.now() - start_time).total_seconds() > max_duration:
                    print("⏱️ Max recording time reached")
                    break
                
                # Get audio chunk
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    recorded_chunks.append(chunk)
                    
                    # Calculate energy
                    energy = self.calculate_energy(chunk)
                    
                    # Detect speech
                    if energy > self.silence_threshold:
                        silence_chunks = 0
                        if not speech_detected:
                            print("🗣️ Speech detected...")
                            speech_detected = True
                    else:
                        if speech_detected:  # Only count silence after speech
                            silence_chunks += 1
                    
                    # Stop if silence detected after speech
                    if speech_detected and silence_chunks > silence_threshold_chunks:
                        print("🤫 Silence detected, processing...")
                        break
                        
                except queue.Empty:
                    continue
        
        if not recorded_chunks:
            return None
        
        # Combine all chunks
        audio_data = np.concatenate(recorded_chunks, axis=0)
        return audio_data

# =========================
# AUDIO PROCESSING
# =========================
def save_audio(audio_data, filename="temp_audio.wav", sample_rate=16000):
    """Save audio data to WAV file"""
    audio_int16 = np.int16(audio_data * 32767)
    write(filename, sample_rate, audio_int16)
    return filename

def transcribe_audio(filename, model):
    """Transcribe audio using Whisper"""
    print("🧠 Transcribing...")
    result = model.transcribe(filename)
    text = result["text"].strip()
    return text

# =========================
# LLM INTEGRATION
# =========================
class AIAssistant:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant in a voice conversation.
Be natural, conversational, and concise in your responses.
Keep answers brief and to the point for voice interaction.

{context}

Current user message: {question}

Respond naturally as if speaking. Keep it under 3 sentences unless more detail is requested.
""")
    
    def get_response(self, question, context):
        """Get AI response with context"""
        print("🤖 Thinking...")
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        response = self.llm.invoke(formatted_prompt)
        return response.strip()

# =========================
# MAIN CONVERSATION LOOP
# =========================
def main():
    print("=" * 50)
    print("🎙️  AI VOICE CONVERSATION SYSTEM")
    print("=" * 50)
    print("📋 Controls:")
    print("   - Speak naturally, system will detect when you stop")
    print("   - Press Ctrl+C to end conversation")
    print("=" * 50)
    print()
    
    # Initialize components
    whisper_model = whisper.load_model(Config.WHISPER_MODEL)
    vad = VoiceActivityDetector(
        sample_rate=Config.SAMPLE_RATE,
        silence_threshold=Config.SILENCE_THRESHOLD,
        silence_duration=Config.SILENCE_DURATION
    )
    conversation = ConversationManager(max_history=10)
    assistant = AIAssistant(model_name=Config.LLM_MODEL)
    
    print("✅ System ready!\n")
    
    # Conversation loop
    turn_number = 1
    try:
        while True:
            print(f"\n{'─' * 50}")
            print(f"Turn {turn_number}")
            print(f"{'─' * 50}")
            
            # Record audio with voice activity detection
            audio_data = vad.record_until_silence(max_duration=Config.MAX_RECORDING_TIME)
            
            if audio_data is None or len(audio_data) < Config.SAMPLE_RATE * 0.5:
                print("⚠️ No speech detected, trying again...")
                continue
            
            # Save and transcribe
            audio_file = save_audio(audio_data, sample_rate=Config.SAMPLE_RATE)
            user_text = transcribe_audio(audio_file, whisper_model)
            
            if not user_text or len(user_text.strip()) < 2:
                print("⚠️ Could not understand, please try again...")
                continue
            
            print(f"👤 You: {user_text}")
            
            # Add to conversation history
            conversation.add_message("user", user_text)
            
            # Get AI response with context
            context = conversation.get_context()
            ai_response = assistant.get_response(user_text, context)
            
            # Add AI response to history
            conversation.add_message("assistant", ai_response)
            
            print(f"🤖 AI: {ai_response}")
            
            # TODO: Add text-to-speech here if desired
            # speak(ai_response)
            
            turn_number += 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Ending conversation...")
        conversation.save_session()
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conversation.save_session()

# =========================
# FUTURE ENHANCEMENTS PLACEHOLDER
# =========================
"""
🔮 FUTURE FEATURES TO ADD:

1. RAG System:
   - Add vector database (ChromaDB, FAISS)
   - Document ingestion pipeline
   - Semantic search for context retrieval
   
2. MCP Server Integration:
   - Tool calling framework
   - Function definitions for LLM
   - Execution sandboxing
   
3. Text-to-Speech:
   - Add TTS engine (pyttsx3, edge-tts, or ElevenLabs)
   - Natural voice output
   
4. Multi-language Support:
   - Language detection
   - Multilingual Whisper models
   
5. Memory System:
   - Long-term memory storage
   - User preferences
   - Conversation summaries
"""

if __name__ == "__main__":
    main()