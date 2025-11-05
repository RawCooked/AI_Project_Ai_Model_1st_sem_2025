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
    ESPRIT_BASE_URL = os.getenv("ESPRIT_BASE_URL", "https://tokenfactory.esprit.tn/api/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "hosted_vllm/Llama-3.1-70B-Instruct")

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
    """Search the web for current information"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('a', class_='result__snippet', limit=3):
            text = result.get_text(strip=True)
            if text and len(text) > 20:
                results.append(f"• {text}")
        
        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        
        return f"Limited information found for '{query}'. Try rephrasing."
    except Exception as e:
        return f"Search unavailable. Error: {str(e)}"

@register_tool
def get_wikipedia_info(topic: str) -> str:
    """Get detailed information from Wikipedia"""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=4, auto_suggest=True)
        page = wikipedia.page(topic, auto_suggest=True)
        return f"Wikipedia - {page.title}:\n\n{summary}\n\nURL: {page.url}"
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
        return f"✅ Saved to: {os.path.abspath(filename)}"
    except Exception as e:
        return f"Error: {str(e)}"

@register_tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@register_tool
def get_current_datetime() -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Date: {now.strftime('%A, %B %d, %Y')}\nTime: {now.strftime('%I:%M:%S %p')}"

@register_tool
def create_note(title: str, content: str) -> str:
    """Create a quick note"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}_{title.replace(' ', '_')}.txt"
        full_content = f"Title: {title}\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        return f"✅ Note '{title}' saved to {filename}"
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# PRESCRIPTION TOOLS
# =========================

@register_tool
def create_prescription(patient_info: str, medications: str, diagnosis: str = "", notes: str = "") -> str:
    """Create medical prescription PDF. Format: patient_info, medications, diagnosis, notes"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        
        # Parse patient info
        patient_parts = [p.strip() for p in patient_info.split(',')]
        patient_name = patient_parts[0] if len(patient_parts) > 0 else "Patient"
        patient_age = patient_parts[1] if len(patient_parts) > 1 else "N/A"
        patient_gender = patient_parts[2] if len(patient_parts) > 2 else "N/A"
        
        # Parse medications
        med_list = []
        if medications:
            for med in medications.split(';'):
                med = med.strip()
                if med:
                    parts = med.split(':')
                    med_list.append({
                        'name': parts[0].strip() if len(parts) > 0 else '',
                        'dosage': parts[1].strip() if len(parts) > 1 else '',
                        'frequency': parts[2].strip() if len(parts) > 2 else '',
                        'duration': parts[3].strip() if len(parts) > 3 else ''
                    })
        
        # Generate filename - clean the patient name
        clean_name = patient_name.replace(' ', '_').replace('"', '').replace("'", '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prescription_{clean_name}_{timestamp}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
            fontSize=24, textColor=colors.HexColor('#1a5490'), spaceAfter=30,
            alignment=TA_CENTER, fontName='Helvetica-Bold')
        
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
            fontSize=14, textColor=colors.HexColor('#2c5aa0'), spaceAfter=12,
            spaceBefore=12, fontName='Helvetica-Bold')
        
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
            fontSize=11, spaceAfter=6)
        
        # Header
        elements.append(Paragraph("℞ MEDICAL PRESCRIPTION", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Doctor info
        doctor_data = [
            ['Dr. AI Medical Assistant', ''],
            ['Medical AI System', 'Date: ' + datetime.now().strftime("%B %d, %Y")],
            ['License: AI-ASSISTANT-001', 'Time: ' + datetime.now().strftime("%I:%M %p")]
        ]
        
        doctor_table = Table(doctor_data, colWidths=[3.5*inch, 3*inch])
        doctor_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        elements.append(doctor_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Information
        elements.append(Paragraph("PATIENT INFORMATION", heading_style))
        
        patient_data = [
            ['Name:', patient_name, 'Age:', patient_age],
            ['Gender:', patient_gender, 'Date:', datetime.now().strftime("%m/%d/%Y")]
        ]
        
        patient_table = Table(patient_data, colWidths=[1*inch, 2.5*inch, 0.8*inch, 2.2*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Diagnosis
        if diagnosis:
            elements.append(Paragraph("DIAGNOSIS", heading_style))
            elements.append(Paragraph(diagnosis, normal_style))
            elements.append(Spacer(1, 0.15*inch))
        
        # Medications
        elements.append(Paragraph("PRESCRIPTION", heading_style))
        
        if med_list:
            med_data = [['#', 'Medication', 'Dosage', 'Frequency', 'Duration']]
            for idx, med in enumerate(med_list, 1):
                med_data.append([str(idx), med['name'], med['dosage'], 
                               med['frequency'], med['duration']])
            
            med_table = Table(med_data, colWidths=[0.4*inch, 2.2*inch, 1.3*inch, 1.5*inch, 1.1*inch])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ]))
            elements.append(med_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Notes
        if notes:
            elements.append(Paragraph("INSTRUCTIONS & NOTES", heading_style))
            elements.append(Paragraph(notes, normal_style))
            elements.append(Spacer(1, 0.2*inch))
        
        # Disclaimer
        elements.append(Spacer(1, 0.3*inch))
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#666666'), alignment=TA_CENTER)
        
        elements.append(Paragraph(
            "⚠️ AI-generated prescription for informational purposes only. Consult a licensed healthcare professional.",
            disclaimer_style))
        
        # Signature
        elements.append(Spacer(1, 0.4*inch))
        sig_style = ParagraphStyle('Signature', parent=styles['Normal'], 
            fontSize=10, alignment=TA_RIGHT)
        elements.append(Paragraph("_____________________________", sig_style))
        elements.append(Paragraph("AI Medical Assistant", sig_style))
        
        doc.build(elements)
        
        return f"✅ Prescription created!\n📄 {os.path.abspath(filename)}\n👤 {patient_name}\n💊 {len(med_list)} medications"
        
    except ImportError:
        return "❌ Install reportlab: pip install reportlab"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@register_tool
def quick_prescription(patient_name: str, condition: str) -> str:
    """Quick prescription for common conditions. Example: patient_name, condition"""
    try:
        prescriptions = {
            'common cold': ('Acetaminophen:500mg:Every 6 hours:5 days; Vitamin C:1000mg:Once daily:7 days',
                          'Rest, drink 8-10 glasses fluids daily. Return if fever >102°F.'),
            'flu': ('Oseltamivir:75mg:Twice daily:5 days; Acetaminophen:500mg:Every 6 hours:7 days',
                   'Isolation 5 days. Rest and fluids. Return if breathing difficulties.'),
            'headache': ('Ibuprofen:400mg:Every 8 hours as needed:3 days',
                        'Avoid bright lights. Stay hydrated.'),
            'fever': ('Acetaminophen:500mg:Every 6 hours:5 days',
                     'Monitor temp every 4 hours. Return if >103°F.'),
            'cough': ('Dextromethorphan:10ml:Every 6 hours:5 days; Ambroxol:30mg:3x daily:7 days',
                     'Stay hydrated. Use humidifier. Avoid smoking.'),
            'sore throat': ('Amoxicillin:500mg:3x daily:7 days; Throat lozenges:As needed:Every 3-4h:7 days',
                           'Warm salt water gargles. Complete antibiotic course.'),
            'allergies': ('Cetirizine:10mg:Once daily:14 days; Nasal spray:2 sprays:Once daily:14 days',
                         'Avoid allergens. May cause drowsiness.'),
            'back pain': ('Ibuprofen:600mg:3x daily with food:7 days; Muscle relaxant:10mg:Bedtime:5 days',
                         'Hot/cold packs. Gentle stretching. Avoid heavy lifting.'),
            'stomach pain': ('Omeprazole:20mg:Before breakfast:14 days; Antacid:As directed:As needed:14 days',
                            'Small frequent meals. Avoid spicy/fatty foods.'),
        }
        
        condition_lower = condition.lower().strip()
        matched = None
        
        for key in prescriptions:
            if key in condition_lower or condition_lower in key:
                matched = key
                break
        
        if matched:
            meds, notes = prescriptions[matched]
            diagnosis = matched.title()
        else:
            meds = "General medication:As directed:As needed:7 days"
            notes = "Follow instructions. Return if symptoms worsen."
            diagnosis = condition.title()
        
        return create_prescription(f"{patient_name}, N/A, N/A", meds, diagnosis, notes)
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# TOOLS DESCRIPTION
# =========================
def get_tools_description():
    """Generate tools description for LLM"""
    return "\n".join([f"- {name}: {func.__doc__}" for name, func in TOOLS_DICT.items()])

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
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history * 2 + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]
    
    def add_interaction(self, user_message: str, ai_response: str, tool_calls: List[str] = None):
        interaction = {
            "user": user_message,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        if tool_calls:
            interaction["tools_used"] = tool_calls
        self.full_history.append(interaction)
    
    def add_tool_call(self, tool_name: str, parameters: str, result: str):
        self.tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def save_session(self):
        session_data = {
            "session_id": self.session_id,
            "history": self.full_history,
            "tool_calls": self.tool_calls,
            "total_turns": len(self.full_history)
        }
        filename = f"conversation_{self.session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filename}")

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
        print("🎤 Listening...")
        
        recorded_chunks = []
        silence_chunks = 0
        silence_threshold_chunks = int(self.silence_duration * self.sample_rate / 1024)
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                          callback=self.audio_callback, blocksize=1024):
            start_time = datetime.now()
            speech_detected = False
            
            while True:
                if (datetime.now() - start_time).total_seconds() > max_duration:
                    print("⏱️ Max time reached")
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
                        print("🤫 Processing...")
                        break
                except queue.Empty:
                    continue
        
        if not recorded_chunks:
            return None
        return np.concatenate(recorded_chunks, axis=0)

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
    return result["text"].strip()

# =========================
# AI AGENT
# =========================
class SimpleVoiceAgent:
    def __init__(self):
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(
            api_key=Config.ESPRIT_API_KEY,
            base_url=Config.ESPRIT_BASE_URL,
            http_client=http_client
        )
        
        self.system_prompt = f"""You are a helpful AI voice assistant with tools.

Keep responses natural, conversational, concise (2-3 sentences unless detail requested).

Available tools:
{get_tools_description()}

When you need a tool, use this EXACT format:
TOOL: tool_name
ARGS: arguments

TOOL GUIDE:
- search_web: ARGS: "search query"
- get_wikipedia_info: ARGS: "topic"
- calculate: ARGS: "math expression"
- get_current_datetime: ARGS: (leave blank)
- create_note: ARGS: "title, content"
- save_to_file: ARGS: "content"
- create_prescription: ARGS: "Name Age Gender, Med:Dose:Freq:Days; Med2:..., Diagnosis, Notes"
- quick_prescription: ARGS: "Patient Name, condition"

MEDICAL MODE - IMPORTANT NAME HANDLING:
When a user mentions symptoms or requests a prescription:
1. First check if you know their name from the conversation history
2. If you know their name, use it: ARGS: "Their Actual Name, condition"
3. If you DON'T know their name, ASK for it before creating prescription
4. NEVER use placeholder names like "Your Name", "Patient", etc.

For symptoms, use quick_prescription for: cold, flu, headache, fever, cough, allergies, back pain, stomach pain
Always remind user to consult real doctor.

After tool use, provide natural spoken response that acknowledges the user by name if known."""
    
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
            return f"Tool '{tool_name}' not found"
        
        try:
            tool_func = TOOLS_DICT[tool_name]
            import inspect
            sig = inspect.signature(tool_func)
            param_count = len(sig.parameters)
            
            # Clean up args - remove surrounding quotes if present
            args = args.strip()
            if args.startswith('"') and args.endswith('"'):
                args = args[1:-1]
            if args.startswith("'") and args.endswith("'"):
                args = args[1:-1]
            
            if not args or args.strip() == "":
                result = tool_func()
            elif param_count == 1:
                result = tool_func(args.strip())
            else:
                # Split by comma and clean each argument
                arg_list = [arg.strip().strip('"').strip("'") for arg in args.split(',')]
                result = tool_func(*arg_list)
            
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Tool error: {str(e)}"
    
    def extract_patient_name(self, conversation: ConversationManager) -> Optional[str]:
        """Extract patient name from conversation history"""
        for msg in conversation.messages:
            if msg["role"] == "user":
                text = msg["content"].lower()
                # Pattern: "my name is X" or "I'm X" or "I am X"
                if "my name is" in text:
                    parts = text.split("my name is")
                    if len(parts) > 1:
                        name = parts[1].strip().split()[0]
                        return name.title()
                elif text.startswith("i'm ") or text.startswith("i am "):
                    words = text.replace("i'm ", "").replace("i am ", "").strip().split()
                    if words and len(words[0]) > 1:
                        return words[0].title()
        return None
    
    def process_query(self, user_query: str, conversation: ConversationManager) -> tuple:
        """Process user query and generate response"""
        print("🤖 Processing...")
        
        # Check if this is a medical request and we need a name
        medical_keywords = ['headache', 'pain', 'sick', 'ill', 'fever', 'cough', 'cold', 'flu', 
                           'prescription', 'medicine', 'hurt', 'sore', 'allerg']
        
        if any(keyword in user_query.lower() for keyword in medical_keywords):
            patient_name = self.extract_patient_name(conversation)
            if patient_name:
                # Add name context to help the AI
                user_query = f"[Patient name from history: {patient_name}] {user_query}"
        
        try:
            conversation.add_message("user", user_query)
            messages = [{"role": "system", "content": self.system_prompt}] + conversation.messages
            
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            tools_used = []
            
            tool_call = self.parse_tool_call(ai_response)
            
            if tool_call:
                tool_name, args = tool_call
                print(f"🔧 Using: {tool_name}({args})")
                
                tool_result = self.execute_tool(tool_name, args)
                tools_used.append(tool_name)
                conversation.add_tool_call(tool_name, args, tool_result)
                
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nProvide natural response."})
                
                final_response = self.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                
                ai_response = final_response.choices[0].message.content
            
            conversation.add_message("assistant", ai_response)
            return ai_response, tools_used
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", []

# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("🎙️  AI VOICE AGENT WITH PRESCRIPTION SYSTEM")
    print("=" * 60)
    print("✨ Features: Search, Wikipedia, Calculate, Save, Notes")
    print("💊 Medical: Create prescriptions for common conditions")
    print("=" * 60)
    print()
    
    if not Config.ESPRIT_API_KEY:
        print("❌ ESPRIT_API_KEY not found in .env!")
        return
    
    print("🔄 Loading Whisper...")
    whisper_model = whisper.load_model(Config.WHISPER_MODEL)
    
    print("🔄 Initializing agent...")
    agent = SimpleVoiceAgent()
    
    vad = VoiceActivityDetector(
        sample_rate=Config.SAMPLE_RATE,
        silence_threshold=Config.SILENCE_THRESHOLD,
        silence_duration=Config.SILENCE_DURATION
    )
    
    conversation = ConversationManager(max_history=10)
    
    print("✅ Ready!\n")
    print("💡 Try:")
    print("   - 'My name is [Your Name] and I have a headache'")
    print("   - 'What time is it?'")
    print("   - 'Search for AI news'")
    print("   - 'Create prescription for John with cold'\n")
    
    turn = 1
    try:
        while True:
            print(f"\n{'─' * 60}")
            print(f"Turn {turn}")
            print(f"{'─' * 60}")
            
            audio_data = vad.record_until_silence(max_duration=Config.MAX_RECORDING_TIME)
            
            if audio_data is None or len(audio_data) < Config.SAMPLE_RATE * 0.5:
                print("⚠️ No speech, trying again...")
                continue
            
            audio_file = save_audio(audio_data, sample_rate=Config.SAMPLE_RATE)
            user_text = transcribe_audio(audio_file, whisper_model)
            
            if not user_text or len(user_text.strip()) < 2:
                print("⚠️ Could not understand...")
                continue
            
            print(f"\n👤 You: {user_text}")
            
            ai_response, tools_used = agent.process_query(user_text, conversation)
            conversation.add_interaction(user_text, ai_response, tools_used)
            
            print(f"\n🤖 AI: {ai_response}")
            if tools_used:
                print(f"🔧 Tools: {', '.join(tools_used)}")
            
            turn += 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Ending...")
        conversation.save_session()
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conversation.save_session()

if __name__ == "__main__":
    main()