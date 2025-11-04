import numpy as np
import whisper
import json
import os
from scipy.io.wavfile import write
import sounddevice as sd
from datetime import datetime
import queue
import httpx
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# CONFIGURATION
# =========================
class Config:
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 2.0
    MAX_RECORDING_TIME = 30
    WHISPER_MODEL = "tiny"
    
    # School LLM API Configuration
    ESPRIT_API_KEY = os.getenv("ESPRIT_API_KEY")
    ESPRIT_BASE_URL = os.getenv("ESPRIT_BASE_URL", "https://tokenfactory.esprit.tn/api")
    LLM_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"

# =========================
# TOOL IMPLEMENTATIONS
# =========================

TOOLS_DICT = {}

def register_tool(func):
    """Decorator to register tools"""
    TOOLS_DICT[func.__name__] = func
    return func

@register_tool
def search_web(query: str) -> str:
    """Search the web for information using DuckDuckGo"""
    try:
        import requests
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        abstract = data.get('AbstractText', '')
        related = data.get('RelatedTopics', [])
        
        result = []
        if abstract:
            result.append(f"Summary: {abstract}")
        
        if related:
            for item in related[:3]:
                if isinstance(item, dict) and 'Text' in item:
                    result.append(f"- {item['Text']}")
        
        return "\n".join(result) if result else "No relevant results found for this query."
    except Exception as e:
        return f"Search error: {str(e)}"

@register_tool
def get_wikipedia_info(topic: str) -> str:
    """Get detailed information from Wikipedia about a topic"""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=4, auto_suggest=True)
        page = wikipedia.page(topic, auto_suggest=True)
        
        result = f"Wikipedia - {page.title}:\n\n{summary}\n\nURL: {page.url}"
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple topics found. Please be more specific. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{topic}'"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

@register_tool
def save_to_file(content: str, filename: str = None) -> str:
    """Save content to a text file"""
    try:
        if filename is None:
            filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        abs_path = os.path.abspath(filename)
        return f"✅ Content successfully saved to: {abs_path}"
    except Exception as e:
        return f"Save error: {str(e)}"

@register_tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@register_tool
def get_current_datetime() -> str:
    """Get the current date and time"""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M:%S %p")
    return f"Current date: {date_str}\nCurrent time: {time_str}"

@register_tool
def create_note(title: str, content: str) -> str:
    """Create a quick note with a title"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}_{title.replace(' ', '_')}.txt"
        
        full_content = f"Title: {title}\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return f"✅ Note '{title}' saved to {filename}"
    except Exception as e:
        return f"Error creating note: {str(e)}"

# =========================
# TOOLS DESCRIPTION FOR LLM
# =========================
def get_tools_description():
    """Generate tools description for the LLM"""
    tools_desc = []
    for name, func in TOOLS_DICT.items():
        tools_desc.append(f"- {name}: {func.__doc__}")
    return "\n".join(tools_desc)

# =========================
# CONVERSATION MANAGER
# =========================
class ConversationManager:
    def __init__(self, max_history=10):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tool_calls = []
        self.full_history = []
        self.max_history = max_history
        self.messages = []
        
    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.messages.append({"role": role, "content": content})
        # Keep only last max_history messages (plus system)
        if len(self.messages) > self.max_history * 2 + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]
    
    def add_interaction(self, user_message: str, ai_response: str, tool_calls: List[str] = None):
        """Add a complete interaction to history"""
        interaction = {
            "user": user_message,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        if tool_calls:
            interaction["tools_used"] = tool_calls
        
        self.full_history.append(interaction)
    
    def add_tool_call(self, tool_name: str, parameters: str, result: str):
        """Track tool usage"""
        self.tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def save_session(self):
        """Save conversation and tool calls to file"""
        session_data = {
            "session_id": self.session_id,
            "history": self.full_history,
            "tool_calls": self.tool_calls,
            "total_turns": len(self.full_history)
        }
        filename = f"conversation_{self.session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
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
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def calculate_energy(self, audio_chunk):
        return np.sqrt(np.mean(audio_chunk**2))
    
    def record_until_silence(self, max_duration=30):
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
                if (datetime.now() - start_time).total_seconds() > max_duration:
                    print("⏱️ Max recording time reached")
                    break
                
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    recorded_chunks.append(chunk)
                    
                    energy = self.calculate_energy(chunk)
                    
                    if energy > self.silence_threshold:
                        silence_chunks = 0
                        if not speech_detected:
                            print("🗣️ Speech detected...")
                            speech_detected = True
                    else:
                        if speech_detected:
                            silence_chunks += 1
                    
                    if speech_detected and silence_chunks > silence_threshold_chunks:
                        print("🤫 Silence detected, processing...")
                        break
                        
                except queue.Empty:
                    continue
        
        if not recorded_chunks:
            return None
        
        audio_data = np.concatenate(recorded_chunks, axis=0)
        return audio_data

# =========================
# AUDIO PROCESSING
# =========================
def save_audio(audio_data, filename="temp_audio.wav", sample_rate=16000):
    audio_int16 = np.int16(audio_data * 32767)
    write(filename, sample_rate, audio_int16)
    return filename

def transcribe_audio(filename, model):
    print("🧠 Transcribing...")
    result = model.transcribe(filename, language="en")
    text = result["text"].strip()
    return text

# =========================
# SIMPLE AI AGENT (NO LANGCHAIN)
# =========================
class SimpleVoiceAgent:
    def __init__(self):
        # Initialize OpenAI client with ESPRIT API
        http_client = httpx.Client(verify=False)
        
        self.client = OpenAI(
            api_key=Config.ESPRIT_API_KEY,
            base_url=Config.ESPRIT_BASE_URL,
            http_client=http_client
        )
        
        self.system_prompt = f"""You are a helpful AI voice assistant with access to various tools.

You are conversing through voice, so keep your responses:
- Natural and conversational
- Concise (2-3 sentences unless detail is requested)
- Clear and easy to understand when spoken aloud

Available tools:
{get_tools_description()}

When you need to use a tool, respond EXACTLY in this format:
TOOL: tool_name
ARGS: argument1, argument2, ...

For example:
- To search: TOOL: search_web\nARGS: artificial intelligence news
- To calculate: TOOL: calculate\nARGS: 25 * 17 + 100
- To get time: TOOL: get_current_datetime\nARGS: 

After using a tool, you'll receive the result and can provide a natural response.

If you don't need a tool, just respond naturally to the user."""
    
    def parse_tool_call(self, response: str) -> Optional[tuple]:
        """Parse tool call from LLM response"""
        if "TOOL:" in response and "ARGS:" in response:
            lines = response.strip().split('\n')
            tool_name = None
            args = None
            
            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("ARGS:"):
                    args = line.replace("ARGS:", "").strip()
            
            if tool_name:
                return (tool_name, args)
        return None
    
    def execute_tool(self, tool_name: str, args: str) -> str:
        """Execute a tool with given arguments"""
        if tool_name not in TOOLS_DICT:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            tool_func = TOOLS_DICT[tool_name]
            
            # Parse arguments
            if not args or args.strip() == "":
                result = tool_func()
            elif ',' in args:
                # Multiple arguments
                arg_list = [arg.strip() for arg in args.split(',')]
                result = tool_func(*arg_list)
            else:
                # Single argument
                result = tool_func(args.strip())
            
            return result
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def process_query(self, user_query: str, conversation: ConversationManager) -> tuple:
        """Process user query"""
        print("🤖 Processing query...")
        
        try:
            # Add user message
            conversation.add_message("user", user_query)
            
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}] + conversation.messages
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            tools_used = []
            
            # Check if LLM wants to use a tool
            tool_call = self.parse_tool_call(ai_response)
            
            if tool_call:
                tool_name, args = tool_call
                print(f"🔧 Using tool: {tool_name}({args})")
                
                # Execute tool
                tool_result = self.execute_tool(tool_name, args)
                tools_used.append(tool_name)
                
                # Track tool call
                conversation.add_tool_call(tool_name, args, tool_result)
                
                # Get final response with tool result
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nPlease provide a natural response to the user based on this result."})
                
                final_response = self.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                
                ai_response = final_response.choices[0].message.content
            
            # Add assistant message
            conversation.add_message("assistant", ai_response)
            
            return ai_response, tools_used
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error: {str(e)}", []

# =========================
# MAIN CONVERSATION LOOP
# =========================
def main():
    print("=" * 60)
    print("🎙️  AI VOICE AGENT - SIMPLE & WORKING VERSION")
    print("=" * 60)
    print("🔧 Powered by: Direct OpenAI API + ESPRIT Llama 3.1 70B")
    print("=" * 60)
    print("📋 Controls:")
    print("   - Speak naturally, system detects when you stop")
    print("   - Agent has tools: search, Wikipedia, save, calculate, etc.")
    print("   - Press Ctrl+C to end conversation")
    print("=" * 60)
    print()
    
    # Check API key
    if not Config.ESPRIT_API_KEY:
        print("❌ ERROR: ESPRIT_API_KEY not found in .env file!")
        print("Please create a .env file with your API key.")
        return
    
    # Initialize components
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model(Config.WHISPER_MODEL)
    
    print("🔄 Initializing AI agent...")
    agent = SimpleVoiceAgent()
    
    vad = VoiceActivityDetector(
        sample_rate=Config.SAMPLE_RATE,
        silence_threshold=Config.SILENCE_THRESHOLD,
        silence_duration=Config.SILENCE_DURATION
    )
    
    conversation = ConversationManager(max_history=10)
    
    print("✅ System ready!\n")
    print("💡 Try asking:")
    print("   - 'Search for the latest news about AI'")
    print("   - 'Tell me about quantum computing'")
    print("   - 'What time is it?'")
    print("   - 'Calculate 156 times 23 plus 789'")
    print("   - 'Save this conversation summary'\n")
    
    turn_number = 1
    try:
        while True:
            print(f"\n{'─' * 60}")
            print(f"Turn {turn_number}")
            print(f"{'─' * 60}")
            
            # Record audio
            audio_data = vad.record_until_silence(max_duration=Config.MAX_RECORDING_TIME)
            
            if audio_data is None or len(audio_data) < Config.SAMPLE_RATE * 0.5:
                print("⚠️ No speech detected, trying again...")
                continue
            
            # Transcribe
            audio_file = save_audio(audio_data, sample_rate=Config.SAMPLE_RATE)
            user_text = transcribe_audio(audio_file, whisper_model)
            
            if not user_text or len(user_text.strip()) < 2:
                print("⚠️ Could not understand, please try again...")
                continue
            
            print(f"\n👤 You: {user_text}")
            
            # Get AI response
            ai_response, tools_used = agent.process_query(user_text, conversation)
            
            # Add to conversation history
            conversation.add_interaction(user_text, ai_response, tools_used)
            
            print(f"\n🤖 AI: {ai_response}")
            if tools_used:
                print(f"🔧 Tools used: {', '.join(tools_used)}")
            
            turn_number += 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Ending conversation...")
        conversation.save_session()
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conversation.save_session()

if __name__ == "__main__":
    main()